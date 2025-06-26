from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torchaudio
import os
import tempfile
import soundfile as sf
from werkzeug.utils import secure_filename
import json
import subprocess
import re
from typing import List, Tuple, Optional

# 导入原有模型和处理函数
from funasr import AutoModel

# 新增导入：纠错相关
import pypinyin
from pypinyin import lazy_pinyin, Style, pinyin
from Levenshtein import distance as levenshtein_distance

app = Flask(__name__)
CORS(app)

# 全局变量定义
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'mp4'}
ALLOWED_TEXT_EXTENSIONS = {'txt'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型（仅在应用启动时加载一次）
print("正在加载模型...")
model = AutoModel(model="./models/iic/SenseVoiceSmall",
                  vad_model="./models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                  vad_kwargs={"max_single_segment_time": 10000},
                  device="cuda:1"
                  )
print("模型加载完成!")

# 从原代码复制必要的函数和字典
emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷", }


# ===================== 新增：古诗文纠错功能 =====================
class PunctuationPreserver:
    """
    标点符号位置保持器
    """
    def __init__(self):
        self.punctuation_map = []  # [(汉字索引, 标点符号)]
        self.chinese_chars = []  # 纯汉字列表

    def extract_punctuation(self, text: str) -> str:
        """
        提取标点符号位置并返回纯汉字文本
        """
        self.punctuation_map = []
        self.chinese_chars = []

        chinese_count = 0
        i = 0

        while i < len(text):
            char = text[i]
            if re.match(r'[\u4e00-\u9fa5\d]', char):  # 汉字
                self.chinese_chars.append(char)
                chinese_count += 1
            else:  # 标点符号或其他字符
                if char.strip():  # 非空格字符
                    self.punctuation_map.append((chinese_count, char))
            i += 1

        return ''.join(self.chinese_chars)

    def restore_punctuation(self, corrected_chars: str, alignment_map: List[int] = None) -> str:
        """
        将标点符号重新插入到纠正后的文本中
        """
        if not self.punctuation_map:
            return corrected_chars

        result = list(corrected_chars)

        # 如果没有对齐映射，使用简单的比例映射
        if alignment_map is None:
            alignment_map = self._create_proportion_mapping(len(self.chinese_chars), len(corrected_chars))

        # 按位置倒序插入标点符号（避免插入位置偏移）
        for old_pos, punct in sorted(self.punctuation_map, reverse=True):
            # 计算新位置
            if old_pos < len(alignment_map):
                new_pos = alignment_map[old_pos]
            else:
                # 超出范围时按比例计算
                new_pos = min(int(old_pos * len(corrected_chars) / len(self.chinese_chars)), len(corrected_chars))

            # 确保位置有效
            new_pos = max(0, min(new_pos, len(result)))
            result.insert(new_pos, punct)

        final_text = ''.join(result)
        return final_text

    def _create_proportion_mapping(self, old_len: int, new_len: int) -> List[int]:
        """
        创建基于比例的位置映射
        """
        if old_len == 0:
            return []

        mapping = []
        for i in range(old_len + 1):  # +1 为了处理末尾位置
            new_pos = int(i * new_len / old_len)
            mapping.append(new_pos)

        return mapping


def load_target_text_from_file(file_path: str) -> str:
    """
    从文件加载目标文本（古诗文）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 移除标点符号和空格，只保留汉字
            content = re.sub(r'[^\u4e00-\u9fa5]', '', content)
            return content
    except FileNotFoundError:
        return ""
    except Exception as e:
        return ""


def load_target_text_from_string(text: str) -> str:
    """
    从字符串加载目标文本（古诗文）
    """
    if not text:
        return ""
    # 移除标点符号和空格，只保留汉字
    content = re.sub(r'[^\u4e00-\u9fa5]', '', text.strip())
    return content


def parse_pinyin(pinyin_str: str) -> Tuple[str, str, str]:
    """
    解析拼音，提取声母、韵母、声调
    """
    # 提取声调（数字）
    tone_match = re.search(r'(\d)$', pinyin_str)
    tone = tone_match.group(1) if tone_match else '0'

    # 移除声调得到声韵母
    base_pinyin = re.sub(r'\d$', '', pinyin_str)

    # 定义声母
    initials = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
                'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']

    # 按长度排序，优先匹配长的声母（如zh, ch, sh）
    initials.sort(key=len, reverse=True)

    initial = ''
    final = base_pinyin

    for init in initials:
        if base_pinyin.startswith(init):
            initial = init
            final = base_pinyin[len(init):]
            break

    return initial, final, tone


def pinyin_similarity(p1: str, p2: str) -> float:
    """
    改进的拼音相似度算法，针对古诗文ASR错误特点优化
    """
    if p1 == p2:
        return 1.0

    # 解析拼音
    initial1, final1, tone1 = parse_pinyin(p1)
    initial2, final2, tone2 = parse_pinyin(p2)

    score = 0.0

    # 1. 声母匹配
    if initial1 == initial2:
        score += 0.5
    else:
        # 声母相似度计算（处理相似声母）
        score += calculate_initial_similarity(initial1, initial2)

    # 2. 韵母匹配
    if final1 == final2:
        score += 0.5
    else:
        # 韵母相似度计算（处理ou/u等相似韵母）
        score += calculate_final_similarity(final1, final2)

    # 3. 声调匹配
    tone_score = calculate_tone_similarity(tone1, tone2)
    score += tone_score

    return min(score, 1.0)  # 确保不超过1.0


def calculate_final_similarity(f1: str, f2: str) -> float:
    """计算韵母相似度"""
    # 相似韵母组
    similar_finals = [
        {'ou', 'u'},  # 仇-书
        {'an', 'ang'},  # 类似鼻音
        {'en', 'eng'},
        {'in', 'ing'},
        {'ao', 'ou'},  # 开口度相似
        {'ai', 'ei'},
        {'ia', 'ie'},
        {'ua', 'uo'},
    ]

    for group in similar_finals:
        if f1 in group and f2 in group:
            return 0.3  # 相似韵母给予中等分数

    return 0.0


def calculate_initial_similarity(i1: str, i2: str) -> float:
    """计算声母相似度"""
    # 相似声母组（按发音位置和方式分组）
    similar_initials = [
        {'j', 'q', 'x'},  # 舌面音
        {'z', 'c', 's'},  # 舌尖前音
        {'zh', 'ch', 'sh'},  # 舌尖后音
        {'d', 't', 'n', 'l'},  # 舌尖中音
        {'g', 'k', 'h'},  # 舌根音
        {'b', 'p', 'm'},  # 双唇音
        {'f', 'h'},  # 摩擦音
        {'l', 'r'},  # 边音-颤音（隆-荣）
        {'j', 'x'},  # 久-修的情况
        {'ch', 'sh'},  # 仇-书的情况
        {'z', 'zh'},
        {'c', 'ch'},
        {'s', 'sh'},
    ]

    # 处理声母丢失的情况（如壕-凹）
    if (i1 == '' and i2 != '') or (i1 != '' and i2 == ''):
        return 0.15  # 声母丢失给予较低分数

    for group in similar_initials:
        if i1 in group and i2 in group:
            return 0.2  # 相似声母给予中等分数

    return 0.0


def calculate_tone_similarity(t1: str, t2: str) -> float:
    """计算声调相似度"""
    if t1 == t2:
        return 0.1  # 声调完全匹配

    # 声调相似度矩阵（基于音高变化相似性）
    tone_similarity_matrix = {
        ('1', '2'): 0.08,  # 一声-二声相对容易混淆
        ('2', '1'): 0.08,
        ('3', '4'): 0.08,  # 三声-四声
        ('4', '3'): 0.08,
        ('1', '3'): 0.03,  # 其他组合给予更低分数
        ('1', '4'): 0.03,
        ('2', '3'): 0.03,
        ('2', '4'): 0.03,
        ('3', '1'): 0.03,
        ('4', '1'): 0.03,
        ('3', '2'): 0.03,
        ('4', '2'): 0.03,
    }

    return tone_similarity_matrix.get((t1, t2), 0.0)


def sequence_alignment(asr_text: str, target_text: str, threshold: float = 0.4) -> Tuple[str, List[int]]:
    """
    改进的动态规划对齐算法
    能够区分同音字错误和真正的漏背
    支持多音字的最佳拼音匹配
    """
    if not target_text or not asr_text:
        return asr_text, list(range(len(asr_text) + 1))

    asr_chars = list(asr_text)
    target_chars = list(target_text)

    asr_pinyin = lazy_pinyin(asr_text, style=Style.TONE3)
    # 获取多音字的所有读音
    target_pinyin_all = pinyin(target_text, style=Style.TONE3, heteronym=True)
    # print(f"target_pinyin_all: {target_pinyin_all}")

    def get_best_pinyin_similarity(asr_py, target_py_list):
        """从目标字符的多个拼音中选择与ASR拼音最相似的，返回最高相似度和对应拼音"""
        best_sim = 0
        best_pinyin = target_py_list[0]  # 默认使用第一个
        for py in target_py_list:
            sim = pinyin_similarity(asr_py, py)
            if sim > best_sim:
                best_sim = sim
                best_pinyin = py
        return best_sim, best_pinyin

    # 为了后续函数使用，我们需要构建一个优化的target_pinyin
    # 基于与ASR的整体相似度来选择最佳拼音组合
    target_pinyin = []
    for i, py_list in enumerate(target_pinyin_all):
        if i < len(asr_pinyin):
            _, best_py = get_best_pinyin_similarity(asr_pinyin[i], py_list)
            target_pinyin.append(best_py)
        else:
            target_pinyin.append(py_list[0])  # 使用默认读音

    # print(f"optimized target_pinyin: {target_pinyin}")

    m, n = len(asr_chars), len(target_chars)

    # 创建相似度矩阵和决策矩阵
    similarity_matrix = np.zeros((m + 1, n + 1))
    decision_matrix = [[None for _ in range(n + 1)] for _ in range(m + 1)]

    # 动态规划填充矩阵
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # 使用多音字中最相似的拼音计算相似度
            sim, _ = get_best_pinyin_similarity(asr_pinyin[i - 1], target_pinyin_all[j - 1])

            match_score = similarity_matrix[i - 1][j - 1] + sim
            insert_score = similarity_matrix[i][j - 1] + 0.3
            delete_score = similarity_matrix[i - 1][j] + 0.3

            if match_score >= insert_score and match_score >= delete_score:
                similarity_matrix[i][j] = match_score
                decision_matrix[i][j] = ('match', sim)
            elif insert_score >= delete_score:
                similarity_matrix[i][j] = insert_score
                decision_matrix[i][j] = ('insert', 0)
            else:
                similarity_matrix[i][j] = delete_score
                decision_matrix[i][j] = ('delete', 0)

    # 回溯并智能处理插入操作
    operations = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and decision_matrix[i][j] is not None:
            op, sim = decision_matrix[i][j]
            if op == 'match':
                if sim >= threshold and asr_chars[i - 1] != target_chars[j - 1]:
                    operations.append(('replace', target_chars[j - 1], sim))
                else:
                    operations.append(('keep', asr_chars[i - 1], sim))
                i -= 1
                j -= 1
            elif op == 'insert':
                should_insert = is_valid_insertion(target_chars[j - 1], asr_chars, target_chars,
                                                   asr_pinyin, target_pinyin, i, j)
                if should_insert:
                    operations.append(('insert', target_chars[j - 1], 0))
                j -= 1
            elif op == 'delete':
                operations.append(('keep_extra', asr_chars[i - 1], 0))
                i -= 1
        elif i > 0:
            operations.append(('keep_extra', asr_chars[i - 1], 0))
            i -= 1
        elif j > 0:
            should_insert = is_valid_insertion(target_chars[j - 1], asr_chars, target_chars,
                                               asr_pinyin, target_pinyin, i, j)
            if should_insert:
                operations.append(('insert', target_chars[j - 1], 0))
            j -= 1

    operations.reverse()

    corrected_chars = []
    alignment_map = [0] * (m + 1)
    old_idx = 0
    new_idx = 0

    #print(f"\n--- 文本纠错操作过程 ---")
    #print(f"原文本: {asr_text}")

    for op, char, sim in operations:
        if op == 'keep':
            alignment_map[old_idx] = new_idx
            corrected_chars.append(char)
            #print(f"保持: '{char}' (位置{old_idx}→{new_idx})")
            old_idx += 1
            new_idx += 1
        elif op == 'replace':
            alignment_map[old_idx] = new_idx
            corrected_chars.append(char)
            #print(f"替换: '{asr_chars[old_idx]}' → '{char}' (位置{old_idx}→{new_idx}, 相似度:{sim:.3f})")
            old_idx += 1
            new_idx += 1
        elif op == 'keep_extra':
            alignment_map[old_idx] = new_idx
            corrected_chars.append(char)
            #print(f"保留: '{char}' (ASR额外识别, 位置{old_idx}→{new_idx})")
            old_idx += 1
            new_idx += 1
        elif op == 'insert':
            corrected_chars.append(char)
            #print(f"插入: '{char}' (目标位置{new_idx})")
            new_idx += 1

    alignment_map[m] = len(corrected_chars)
    corrected_text = ''.join(corrected_chars)

    #print(f"结果文本: {corrected_text}")
    #print(f"--- 操作完成 ---\n")

    return corrected_text, alignment_map


def is_valid_insertion(target_char: str, asr_chars: List[str], target_chars: List[str],
                       asr_pinyin: List[str], target_pinyin: List[str],
                       current_i: int, current_j: int) -> bool:
    """
    判断插入操作是否有效
    """
    return False


def simple_pinyin_correction(asr_text: str, target_text: str, preserver: PunctuationPreserver) -> str:
    """
    拼音纠错
    """
    if not target_text:
        return asr_text

    # 提取纯汉字和数字(123)进行比较
    clean_asr = preserver.extract_punctuation(asr_text)

    # 计算拼音级别相似度
    asr_pinyin = ' '.join(lazy_pinyin(clean_asr))
    target_pinyin = ' '.join(lazy_pinyin(target_text))

    # 计算字符串编辑距离
    distance = levenshtein_distance(asr_pinyin, target_pinyin)
    max_len = max(len(asr_pinyin), len(target_pinyin))
    similarity = 1 - (distance / max_len) if max_len > 0 else 0

    if similarity >= 0.3:
        corrected_chars, alignment_map = sequence_alignment(clean_asr, target_text)
    else:
        corrected_chars = clean_asr
        alignment_map = list(range(len(clean_asr) + 1))

    # 恢复标点符号
    final_text = preserver.restore_punctuation(corrected_chars, alignment_map)
    return final_text, similarity


def correct_with_target_text(asr_text: str, target_text: str = None, target_file_path: str = None) -> str:
    """
    基于目标文本的古诗文纠错（保持标点符号位置）
    支持直接传入文本或文件路径
    """
    # 优先使用直接传入的文本，否则从文件加载
    if target_text:
        loaded_target_text = load_target_text_from_string(target_text)
    elif target_file_path:
        loaded_target_text = load_target_text_from_file(target_file_path)
    else:
        return asr_text

    if not loaded_target_text:
        return asr_text

    # 创建标点符号保持器
    preserver = PunctuationPreserver()
    # 进行纠错（包含标点符号处理）
    corrected_text, similarity = simple_pinyin_correction(asr_text, loaded_target_text, preserver)
    return corrected_text, similarity


def has_audio_stream(file_path):
    """检查文件是否包含音频流"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', file_path
        ], capture_output=True, text=True, check=True)

        streams = json.loads(result.stdout)
        audio_streams = [s for s in streams['streams'] if s['codec_type'] == 'audio']
        return len(audio_streams) > 0
    except:
        return False


def extract_plain_text(s):
    # 定义所有需要删除的符号和表情（覆盖所有字典的键和值）
    symbols_to_remove = {
        *emo_dict.keys(),
        *emo_dict.values(),
        *event_dict.keys(),
        *event_dict.values(),
        *emoji_dict.keys(),
        *emoji_dict.values(),
        *lang_dict.keys(),
        *emo_set,
        *event_set
    }
    symbols_to_remove = {symbol for symbol in symbols_to_remove if symbol}
    # 按符号长度降序排序（优先处理长组合符号，如 "<|nospeech|><|Event_UNK|>"）
    sorted_symbols = sorted(symbols_to_remove, key=lambda x: len(x), reverse=True)
    for symbol in sorted_symbols:
        s = s.replace(symbol, "")
    s = ' '.join(s.split())
    return s


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_text_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_TEXT_EXTENSIONS


def process_audio(audio_path, language="auto", target_text=None, target_file_path=None):
    try:
        # 修改：添加 "ancient zh" 映射到 "zh"
        language_abbr = {"auto": "auto", "zh": "zh", "ancient zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko",
                         "nospeech": "nospeech"}
        selected_language = language_abbr.get(language, "auto")

        merge_vad = True

        # 检查文件是否包含音频流
        if not has_audio_stream(audio_path):
            return "未识别到文本"

        text = model.generate(input=audio_path,
                              cache={},
                              language=selected_language,
                              use_itn=True,
                              batch_size_s=300,
                              merge_vad=merge_vad)

        text = text[0]["text"]
        text_final = extract_plain_text(text)

        # 修改：文本正则化（仅在古代中文模式下进行）
        if language == "ancient zh" or language == "zh":
            try:
                from tn.chinese.normalizer import Normalizer
                normalizer = Normalizer(overwrite_cache=True, full_to_half=False, remove_erhua=False,
                                        remove_interjections=False, traditional_to_simple=False)
                text_final = normalizer.normalize(text_final)
            except ImportError:
                pass  # 如果没有tn库，跳过正则化

        # 修改：仅在古代中文模式且提供了目标文本或文件时进行纠错
        similarity = 0.0
        correction_enabled = False
        if (language == "ancient zh" or language=="zh") and (target_text or (target_file_path and os.path.exists(target_file_path))):
            text_final, similarity = correct_with_target_text(text_final, target_text, target_file_path)
            if similarity > 0.3:
                correction_enabled = True

        return {
            "text": text_final,
            "language": language,
            "correction_enabled": correction_enabled,
            "similarity": similarity
        }
    except Exception as e:
        raise e


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "服务运行正常"})


@app.route('/recognize', methods=['POST'])
def recognize_speech():
    try:
        # 检查是否有文件部分
        if 'audio' not in request.files:
            return jsonify({"error": "没有上传音频文件"}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "未选择文件"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": f"不支持的文件格式。支持的格式: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        # 获取语言参数，默认为自动
        language = request.form.get('language', 'auto')

        # 获取target_string参数（可选）
        target_string = request.form.get('target_string', None)

        # 获取target_file参数（可选，文件上传）
        target_file_path = None
        if 'target_file' in request.files:
            target_file = request.files['target_file']
            if target_file.filename != '' and allowed_text_file(target_file.filename):
                # 保存上传的目标文件
                target_filename = secure_filename(target_file.filename)
                target_file_path = os.path.join(UPLOAD_FOLDER, f"target_{target_filename}")
                target_file.save(target_file_path)

        # 保存上传的音频文件
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        try:
            # 处理音频文件
            result = process_audio(temp_path, language, target_string, target_file_path)

            # 返回结果
            response_data = {
                "success": True,
                "language": result["language"],
                "text": result["text"],
            }

            # 如果启用了纠错，添加额外信息
            if result["language"] == "ancient zh" or result["language"] == "zh":
                response_data["correction_enabled"] = result["correction_enabled"]
                response_data["similarity"] = result["similarity"]

            return jsonify(response_data)
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # 清理目标文件
            if target_file_path and os.path.exists(target_file_path):
                os.remove(target_file_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 为前端开发提供一个简单的上传表单
@app.route('/', methods=['GET'])
def index():
    return '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能语音识别系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            padding: 30px;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            color: #333;
            font-size: 2.2em;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header p {
            color: #666;
            font-size: 1em;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            align-items: start;
        }

        .left-panel {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
        }

        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .section-title {
            color: #495057;
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .file-upload-area {
            border: 2px dashed #667eea;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            margin-bottom: 20px;
        }

        .file-upload-area:hover {
            border-color: #764ba2;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            transform: translateY(-1px);
        }

        .file-upload-area.dragover {
            border-color: #28a745;
            background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(102, 126, 234, 0.1) 100%);
        }

        .file-upload-icon {
            font-size: 2em;
            color: #667eea;
            margin-bottom: 10px;
        }

        .file-upload-text {
            color: #333;
            font-size: 1em;
            margin-bottom: 5px;
            font-weight: 500;
        }

        .file-upload-hint {
            color: #666;
            font-size: 0.8em;
        }

        #audioFile {
            display: none;
        }

        .selected-file {
            background: #e8f5e8;
            border: 2px solid #28a745;
            color: #155724;
        }

        .media-player {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            display: none;
        }

        .media-player h4 {
            margin-bottom: 15px;
            color: #333;
            font-size: 1.1em;
        }

        .media-player audio,
        .media-player video {
            width: 100%;
            border-radius: 8px;
        }

        .media-player video {
            max-height: 200px;
            object-fit: contain;
            background: #000;
        }

        .options-panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1);
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 0.95em;
        }

        select, textarea {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 0.95em;
            transition: all 0.3s ease;
            background: white;
            font-family: inherit;
        }

        select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        textarea {
            resize: vertical;
            min-height: 80px;
        }

        .target-file-upload {
            position: relative;
            display: inline-block;
            cursor: pointer;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 12px 20px;
            transition: all 0.3s ease;
            width: 100%;
            text-align: center;
        }

        .target-file-upload:hover {
            background: linear-gradient(135deg, #e9ecef, #dee2e6);
            border-color: #667eea;
        }

        .target-file-upload input[type="file"] {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }

        .submit-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 50px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            position: relative;
            overflow: hidden;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        }

        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }

        .spinner {
            width: 35px;
            height: 35px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .result {
            background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(102, 126, 234, 0.05) 100%);
            border-radius: 15px;
            border-left: 5px solid #28a745;
            padding: 25px;
            display: none;
            grid-column: 1 / -1;
            margin-top: 20px;
        }

        .result h3 {
            color: #155724;
            margin-bottom: 15px;
            font-size: 1.3em;
        }

        .result-text {
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #c3e6cb;
            font-size: 1em;
            line-height: 1.6;
            color: #333;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 200px;
            overflow-y: auto;
        }

        .result-meta {
            margin-top: 15px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .meta-item {
            background: rgba(255, 255, 255, 0.8);
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            color: #666;
        }

        .error {
            background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
            border-radius: 15px;
            border-left: 5px solid #dc3545;
            color: #721c24;
            padding: 20px;
            display: none;
            grid-column: 1 / -1;
            margin-top: 20px;
        }

        .progress-bar {
            width: 100%;
            height: 4px;
            background-color: #e9ecef;
            border-radius: 2px;
            overflow: hidden;
            margin-top: 10px;
            display: none;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 2px;
            transition: width 0.3s ease;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }


        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }

            .main-content {
                grid-template-columns: 1fr;
                gap: 20px;
            }

            .header h1 {
                font-size: 1.8em;
            }

            .result {
                grid-column: auto;
            }

            .error {
                grid-column: auto;
            }

            .result-meta {
                flex-direction: column;
                gap: 8px;
            }
        }

        .advanced-hint {
            font-size: 0.8em;
            color: #6c757d;
            margin-top: 5px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 智能语音识别</h1>
            <p>支持多语言识别与古诗文智能纠错</p>
        </div>

        <form id="speechForm" enctype="multipart/form-data">
            <div class="main-content">
                <!-- 左侧面板：文件上传和播放 -->
                <div class="left-panel">
                    <h3 class="section-title">📁 媒体文件</h3>

                    <div class="file-upload-area" id="fileUploadArea">
                        <div class="file-upload-icon">🎵</div>
                        <div class="file-upload-text">点击或拖拽上传</div>
                        <div class="file-upload-hint">音频/视频文件</div>
                        <input type="file" id="audioFile" name="audio" accept=".wav,.mp3,.flac,.ogg,.m4a,.mp4,.avi,.mov,.wmv" required>
                    </div>


                    <div class="media-player" id="mediaPlayer">
                        <h4>📽️ 预览播放</h4>
                        <div id="playerContainer"></div>
                    </div>
                </div>

                <!-- 右侧面板：设置选项 -->
                <div class="right-panel">
                    <div class="options-panel">
                        <h3 class="section-title">⚙️ 识别设置</h3>

                        <div class="form-group">
                            <label for="language">识别语言</label>
                            <select id="language" name="language">
                                <option value="auto">自动检测</option>
                                <option value="zh">中文</option>
                                <option value="ancient zh">古代中文（古诗文）</option>
                                <option value="en">英文</option>
                                <option value="yue">粤语</option>
                                <option value="ja">日语</option>
                                <option value="ko">韩语</option>
                            </select>
                        </div>

                        <div class="form-group">
                            <label for="targetString">目标文本（可选）</label>
                            <textarea id="targetString" name="target_string" placeholder="输入要对照的原文进行智能纠错..."></textarea>
                            <div class="advanced-hint">💡 仅对中文/古代中文有效</div>
                        </div>

                        <div class="form-group">
                            <label for="targetFile">或上传文本文件</label>
                            <div class="target-file-upload">
                                <input type="file" id="targetFile" name="target_file" accept=".txt">
                                <span>📄 选择 TXT 文件</span>
                            </div>
                        </div>

                        <button type="submit" class="submit-btn" id="submitBtn">
                            🚀 开始识别
                        </button>

                        <div class="progress-bar" id="progressBar">
                            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                        </div>

                        <div class="loading" id="loading">
                            <div class="spinner"></div>
                            <p>正在处理，请稍候...</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 结果区域 -->
            <div class="result" id="result">
                <h3>🎯 识别结果</h3>
                <div class="result-text" id="resultText"></div>
                <div class="result-meta" id="resultMeta"></div>
            </div>

            <div class="error" id="error">
                <h4>❌ 错误信息</h4>
                <p id="errorText"></p>
            </div>
        </form>
    </div>

    <script>
        // API基础URL
        const API_BASE_URL = 'http://localhost:5001';

        const form = document.getElementById('speechForm');
        const fileUploadArea = document.getElementById('fileUploadArea');
        const audioFileInput = document.getElementById('audioFile');
        const targetFileInput = document.getElementById('targetFile');
        const submitBtn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const error = document.getElementById('error');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        const mediaPlayer = document.getElementById('mediaPlayer');
        const playerContainer = document.getElementById('playerContainer');

        // 支持的媒体格式
        const audioFormats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a'];
        const videoFormats = ['.mp4', '.avi', '.mov', '.wmv'];

        // 文件拖拽上传
        fileUploadArea.addEventListener('click', () => audioFileInput.click());

        fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadArea.classList.add('dragover');
        });

        fileUploadArea.addEventListener('dragleave', () => {
            fileUploadArea.classList.remove('dragover');
        });

        fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                audioFileInput.files = files;
                updateFileDisplay(files[0]);
            }
        });

        audioFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                updateFileDisplay(e.target.files[0]);
            }
        });

        targetFileInput.addEventListener('change', (e) => {
            const span = e.target.nextElementSibling;
            if (e.target.files.length > 0) {
                span.textContent = `📄 ${e.target.files[0].name}`;
            } else {
                span.textContent = '📄 选择 TXT 文件';
            }
        });

        function updateFileDisplay(file) {
            fileUploadArea.classList.add('selected-file');

            // 更新上传区域显示
            const iconElement = fileUploadArea.querySelector('.file-upload-icon');
            const textElement = fileUploadArea.querySelector('.file-upload-text');
            const hintElement = fileUploadArea.querySelector('.file-upload-hint');

            if (iconElement && textElement && hintElement) {
                iconElement.textContent = '✅';
                textElement.textContent = file.name;
                hintElement.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
            }

            // 显示文件信息
            const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
            const fileType = audioFormats.includes(fileExtension) ? 'audio' : 
                           videoFormats.includes(fileExtension) ? 'video' : 'unknown';

            // 创建媒体播放器
            createMediaPlayer(file, fileType);
        }

        function createMediaPlayer(file, fileType) {
            if (fileType === 'unknown') {
                mediaPlayer.style.display = 'none';
                return;
            }

            const fileURL = URL.createObjectURL(file);

            if (fileType === 'audio') {
                playerContainer.innerHTML = `
                    <audio controls controlsList="nodownload">
                        <source src="${fileURL}" type="${file.type}">
                        您的浏览器不支持音频播放。
                    </audio>
                `;
            } else if (fileType === 'video') {
                playerContainer.innerHTML = `
                    <video controls controlsList="nodownload">
                        <source src="${fileURL}" type="${file.type}">
                        您的浏览器不支持视频播放。
                    </video>
                `;
            }

            mediaPlayer.style.display = 'block';

            // 清理旧的URL对象
            setTimeout(() => URL.revokeObjectURL(fileURL), 60000);
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            // 隐藏之前的结果和错误
            result.style.display = 'none';
            error.style.display = 'none';

            // 显示加载状态
            loading.style.display = 'block';
            progressBar.style.display = 'block';
            submitBtn.disabled = true;

            // 动画进度条
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 8;
                if (progress > 90) progress = 90;
                progressFill.style.width = progress + '%';
            }, 300);

            try {
                const formData = new FormData(form);

                const response = await fetch(`${API_BASE_URL}/recognize`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                // 完成进度条
                clearInterval(progressInterval);
                progressFill.style.width = '100%';

                if (data.success) {
                    // 显示结果
                    document.getElementById('resultText').textContent = data.text || '未识别到文本';

                    // 显示元数据
                    const metaContainer = document.getElementById('resultMeta');
                    metaContainer.innerHTML = '';

                    const languageSpan = document.createElement('span');
                    languageSpan.className = 'meta-item';
                    languageSpan.textContent = `语言: ${data.language}`;
                    metaContainer.appendChild(languageSpan);

                    if (data.correction_enabled !== undefined) {
                        const correctionSpan = document.createElement('span');
                        correctionSpan.className = 'meta-item';
                        correctionSpan.textContent = `纠错: ${data.correction_enabled ? '已启用' : '未启用'}`;
                        metaContainer.appendChild(correctionSpan);
                    }

                    if (data.similarity !== undefined) {
                        const similaritySpan = document.createElement('span');
                        similaritySpan.className = 'meta-item';
                        similaritySpan.textContent = `相似度: ${(data.similarity * 100).toFixed(1)}%`;
                        metaContainer.appendChild(similaritySpan);
                    }

                    result.style.display = 'block';
                    result.scrollIntoView({ behavior: 'smooth' });
                } else {
                    throw new Error(data.error || '识别失败');
                }
            } catch (err) {
                document.getElementById('errorText').textContent = err.message;
                error.style.display = 'block';
                error.scrollIntoView({ behavior: 'smooth' });
            } finally {
                // 隐藏加载状态
                loading.style.display = 'none';
                progressBar.style.display = 'none';
                progressFill.style.width = '0%';
                submitBtn.disabled = false;
                clearInterval(progressInterval);
            }
        });
    </script>
</body>
</html>
    '''

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)