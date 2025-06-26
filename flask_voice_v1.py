from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torchaudio
import os
import tempfile
import soundfile as sf
from werkzeug.utils import secure_filename
import subprocess
import json

# 导入原有模型和处理函数
from funasr import AutoModel

app = Flask(__name__)
CORS(app)

# 全局变量定义
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'mp4', 'mov'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型（仅在应用启动时加载一次）
print("正在加载模型...")
model = AutoModel(model="./models/iic/SenseVoiceSmall",
                  vad_model="./models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                  vad_kwargs={"max_single_segment_time": 30000},
                  trust_remote_code=True,
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


def process_audio(audio_path, language="auto"):
    try:
        language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko",
                         "nospeech": "nospeech"}
        selected_language = language_abbr.get(language, "auto")

        merge_vad = True
        print(f"处理音频: {audio_path}, 语言: {selected_language}")

        if not has_audio_stream(audio_path):
            print(f"文件 {audio_path} 不包含音频流，返回默认结果“未识别到文本”")
            return "未识别到文本"

        text = model.generate(input=audio_path,
                              cache={},
                              language=selected_language,
                              use_itn=True,
                              batch_size_s=60,
                              merge_vad=merge_vad)

        text = text[0]["text"]
        # print('原始文本: ', text)
        text_final = extract_plain_text(text)
        # print('处理后文本: ', text_final)

        return text_final
    except Exception as e:
        print(f"处理音频时出错: {str(e)}")
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

        # 保存上传的文件
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        try:
            # 处理音频文件
            result_text = process_audio(temp_path, language)

            # 返回结果
            return jsonify({
                "success": True,
                "text": result_text
            })
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

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
    <title>语音识别系统</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        select, button {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #4285f4;
            color: white;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #3367d6;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-left: 4px solid #4285f4;
            background-color: #e8f0fe;
        }
        .error {
            color: #d32f2f;
            background-color: #ffebee;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }
        .progress {
            display: none;
            margin-top: 15px;
        }
        #recordButton {
            background-color: #ea4335;
        }
        #recordButton.recording {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .audio-controls {
            margin-top: 10px;
        }
        #fileInfo {
            margin-top: 5px;
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>语音识别系统</h1>
        
        <div class="form-group">
            <label for="audioFile">选择音频文件:</label>
            <input type="file" id="audioFile" accept="audio/*,video/mp4" />
            <div id="fileInfo"></div>
        </div>
        
        <div class="form-group">
            <label for="language">选择语言:</label>
            <select id="language">
                <option value="auto">自动检测</option>
                <option value="zh">中文</option>
                <option value="en">英文</option>
                <option value="yue">粤语</option>
                <option value="ja">日语</option>
                <option value="ko">韩语</option>
            </select>
        </div>
        
        <div class="form-group">
            <button id="uploadButton">上传并识别</button>
            <button id="recordButton">录制音频</button>
        </div>
        
        <div class="audio-controls" style="display: none;">
            <audio id="audioPreview" controls></audio>
        </div>
        
        <div class="progress">
            <p>正在处理，请稍候...</p>
        </div>
        
        <div id="result" class="result" style="display: none;">
            <h3>识别结果:</h3>
            <p id="resultText"></p>
        </div>
        
        <div id="error" class="error" style="display: none;"></div>
    </div>

    <script>
        // 服务器API地址
        const API_URL = 'http://43.242.96.20:5001/recognize';
        
        // DOM元素
        const audioFileInput = document.getElementById('audioFile');
        const languageSelect = document.getElementById('language');
        const uploadButton = document.getElementById('uploadButton');
        const recordButton = document.getElementById('recordButton');
        const fileInfo = document.getElementById('fileInfo');
        const audioPreview = document.getElementById('audioPreview');
        const audioControls = document.querySelector('.audio-controls');
        const progressDiv = document.querySelector('.progress');
        const resultDiv = document.getElementById('result');
        const resultText = document.getElementById('resultText');
        const errorDiv = document.getElementById('error');
        
        // 录音相关变量
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        
        // 文件选择事件
        audioFileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                fileInfo.textContent = `已选择: ${file.name} (${formatFileSize(file.size)})`;
                
                // 创建音频预览
                const url = URL.createObjectURL(file);
                audioPreview.src = url;
                audioControls.style.display = 'block';
            } else {
                fileInfo.textContent = '';
                audioControls.style.display = 'none';
            }
        });
        
        // 上传并识别
        uploadButton.addEventListener('click', function() {
            const file = audioFileInput.files[0];
            if (!file) {
                showError('请先选择音频文件');
                return;
            }
            
            uploadAudio(file);
        });
        
        // 录制音频
        recordButton.addEventListener('click', function() {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });
        
        // 开始录音
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (e) => {
                    audioChunks.push(e.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPreview.src = audioUrl;
                    audioControls.style.display = 'block';
                    
                    // 创建File对象，可以直接上传
                    const audioFile = new File([audioBlob], "recorded_audio.wav", { 
                        type: 'audio/wav',
                        lastModified: new Date().getTime()
                    });
                    
                    // 更新文件信息
                    fileInfo.textContent = `已录制: recorded_audio.wav (${formatFileSize(audioFile.size)})`;
                    
                    // 替换文件输入
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(audioFile);
                    audioFileInput.files = dataTransfer.files;
                };
                
                mediaRecorder.start();
                isRecording = true;
                recordButton.textContent = '停止录制';
                recordButton.classList.add('recording');
                
            } catch (error) {
                showError('无法访问麦克风: ' + error.message);
            }
        }
        
        // 停止录音
        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                recordButton.textContent = '录制音频';
                recordButton.classList.remove('recording');
                
                // 停止所有媒体轨道
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        }
        
        // 上传音频文件到服务器
        function uploadAudio(file) {
            // 隐藏之前的结果和错误
            resultDiv.style.display = 'none';
            errorDiv.style.display = 'none';
            
            // 显示进度提示
            progressDiv.style.display = 'block';
            
            // 禁用按钮
            uploadButton.disabled = true;
            
            // 创建FormData对象
            const formData = new FormData();
            formData.append('audio', file);
            formData.append('language', languageSelect.value);
            
            // 发送请求
            fetch(API_URL, {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('服务器响应错误: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                // 隐藏进度提示
                progressDiv.style.display = 'none';
                
                // 启用按钮
                uploadButton.disabled = false;
                
                if (data.success) {
                    // 显示结果
                    resultText.textContent = data.text || '未识别到文本';
                    resultDiv.style.display = 'block';
                } else {
                    showError(data.error || '识别失败，未知错误');
                }
            })
            .catch(error => {
                // 隐藏进度提示
                progressDiv.style.display = 'none';
                
                // 启用按钮
                uploadButton.disabled = false;
                
                showError('请求错误: ' + error.message);
            });
        }
        
        // 显示错误消息
        function showError(message) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
        
        // 格式化文件大小
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
    </script>
</body>
</html>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
