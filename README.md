# 🎤 Multilingual Subtitle and Traslation Generator

A tool for language learners. 

This pipeline generates **word-level karaoke-style subtitles** for videos, along with translations into your favourite languages.

If you choose a smaller translation model (1.8B vs the current 7B) and deactivate refinement you can run this on a toaster (possibly with as little as 4GB of VRAM).

---

## 🌍 Supported Languages

### Audio Source Languages (Qwen3-ASR-1.7B)
The pipeline natively aligns audio and generates timestamps for the following 30 languages:
```
INPUT_LANGUAGES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
    "mk": "Macedonian", "ro": "Romanian",
}
```

### Translation Target Languages (Hy-MT2-7B)
```
OUTPUT_LANGUAGES = {
    "zh": "Chinese", "en": "English", "fr": "French", "pt": "Portuguese",
    "es": "Spanish", "ja": "Japanese", "tr": "Turkish", "ru": "Russian",
    "ar": "Arabic", "ko": "Korean", "th": "Thai", "it": "Italian",
    "de": "German", "vi": "Vietnamese", "ms": "Malay", "id": "Indonesian",
    "tl": "Filipino", "hi": "Hindi", "zh-hant": "Traditional Chinese",
    "pl": "Polish", "cs": "Czech", "nl": "Dutch", "km": "Khmer",
    "my": "Burmese", "fa": "Persian", "gu": "Gujarati", "ur": "Urdu",
    "te": "Telugu", "mr": "Marathi", "he": "Hebrew", "bn": "Bengali",
    "ta": "Tamil", "uk": "Ukrainian", "bo": "Tibetan", "kk": "Kazakh",
    "mn": "Mongolian", "ug": "Uyghur", "yue": "Cantonese",
}
```

---

## ⚙️ Installation

### 1. Global Dependencies (Linux)

```
sudo pacman -Syu ffmpeg libass
```

### 2. The Main Pipeline Environment
This environment handles vocal isolation, transcription, refinement, translation and subtitle compilation.
This is the environment from were you will run script.

```
python3 -m venv main-env
source main-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. The Qwe3-ASR Transcription Environment
Qwen-ASR will soon be supported by transformers and creating a separate environment should no longer be required.
Do this on a different shell/tab.

```
python3 -m venv qwen-env
source qwen-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-qwen.txt
```

---

## 🚀 Usage

# Basic Usage (auto-detects language, translates to English)

```
python transcription.py input_track.mp4
```

### Optional Flags

**Display help for all options:**

```
python transcription.py -h
```

**Set Target (Translation) Language:**

```
python transcription.py input_track.mp4 --target-lang pt
```

**Set Source Audio Language (Overrides Auto-Detect):**

```
python transcription.py input_track.mp4 --source-lang de
```
