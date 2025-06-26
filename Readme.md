简体中文 | [English](./Readme_en.md)

# SenseAlign ASR - 基于SenseVoice和目标文本对齐的ASR系统

<div align="center">
    <img src="image\SenseAlign_logo.svg" alt="logo" style="zoom:800%;" />
</div>


![SenseAlign](https://img.shields.io/badge/SenseAlign-ASR%E7%B3%BB%E7%BB%9F-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![Flask](https://img.shields.io/badge/Flask-2.0+-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

**基于SenseVoice模型和目标文本对齐的ASR系统**

## 📖 项目简介

SenseAlign ASR是一个集成目标文本对齐纠错功能的高精度语音识别系统，具备以下核心能力：

- **[SenseVoiceSmall模型](https://github.com/FunAudioLLM/SenseVoice)**: 基于阿里达摩院开源的高精度多语言语音识别模型
- **目标文本对齐**: 通过与预设目标文本对比，实现智能纠错和内容验证
- **多语言支持**: 支持中文、英文、粤语、日语、韩语及自动语言检测
- **拼音级纠错**: 基于拼音相似度算法，精准识别和纠正发音错误
- **格式兼容**: 支持WAV、MP3、FLAC、OGG、M4A、MP4等多种音频视频格式

## 🔧 核心技术

### 目标文本对齐算法
系统采用**多层拼音相似度匹配**技术实现精准的文本对齐纠错：

1. **拼音解析**: 将汉字分解为声母+韵母+声调的完整语音特征
3. **相似度计算**: 处理常见发音混淆（ou/u, an/ang, en/eng等）
5. **动态规划对齐**: 使用动态规划算法实现最优字符序列对齐

## 🚀 快速开始

### 环境安装

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 模型文件准备

执行`download_models.py` 将以下模型文件存储于`./models/iic/`目录下：

- **[SenseVoiceSmall](https://www.modelscope.cn/models/iic/SenseVoiceSmall)**: 主要的多语言语音识别模型
- **[speech_fsmn_vad_zh-cn-16k-common-pytorch](https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)**: 语音活动检测(VAD)模型

### 启动服务

```bash
python flask_voice.py
```

启动成功后访问 `http://localhost:5001` 使用Web界面进行语音识别。

## 📡 API接口

### 语音识别与对齐接口

**端点**: `POST /recognize`

**请求参数**:

| 参数          | 类型   | 必填 | 说明                                    |
| ------------- | ------ | ---- | --------------------------------------- |
| audio         | file   | 是   | 音频/视频文件                           |
| language      | string | 否   | 识别语言 (默认: auto)                   |
| target_string | string | 否   | 目标对照文本，启用智能纠错              |
| target_file   | file   | 否   | 目标文本文件 (.txt格式)                 |

**支持的语言参数**:
- `auto`: 自动语言检测
- `zh`: 中文 (启用目标文本对齐)
- `en`: 英文
- `yue`: 粤语
- `ja`: 日语
- `ko`: 韩语

**请求示例**:

```bash
# 基于目标文本的对齐识别
curl -X POST http://localhost:5001/recognize \
  -F "audio=@speech.wav" \
  -F "language=zh" \
  -F "target_string=欢迎使用SenseAlign语音识别系统，这是一个高精度的ASR解决方案。"

# 多语言自动识别
curl -X POST http://localhost:5001/recognize \
  -F "audio=@meeting.mp3" \
  -F "language=auto"

# 使用目标文件进行对齐
curl -X POST http://localhost:5001/recognize \
  -F "audio=@presentation.wav" \
  -F "language=zh" \
  -F "target_file=@target_script.txt"
```

### **成功响应示例**:

```json
{
  "success": true,
  "language": "zh",
  "text": "欢迎使用SenseAlign语音识别系统，这是一个高精度的ASR解决方案。",
  "correction_enabled": true,
  "similarity": 0.92
}
```

## 📁 项目结构

```
SenseAlign-ASR/
├── flask_voice.py         # Flask主应用程序
├── requirements.txt       # Python依赖列表
├── models/                # 模型文件目录
│   └── iic/
│       ├── SenseVoiceSmall/
│       └── speech_fsmn_vad_zh-cn-16k-common-pytorch/
├── uploads/               # 临时文件存储目录
├── static/                # 静态资源文件
├── templates/             # Web界面模板
└── README.md             # 项目说明文档
```

## 🎨 识别结果展示

### Web界面展示

<div align="center">
    <img src="image/web_show.png" alt="web" />
</div>

### demo1（未对齐）

<div align="center">
    <img src="image/demo1.png" alt="web" />
</div>

### demo2（对齐）

<div align="center">
    <img src="image/demo2_1.png" alt="web"/>
</div>

<div align="center">
    <img src="image/demo2_2.png" alt="web"/>
</div>

## 🔧 技术支持

### 依赖项目
- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 语音识别核心框架
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 多语言语音识别模型
- [pypinyin](https://github.com/mozillazg/python-pinyin) - 中文拼音处理库
- [python-Levenshtein](https://github.com/ztane/python-Levenshtein) - 字符串编辑距离计算
