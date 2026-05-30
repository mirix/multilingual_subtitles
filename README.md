# 🎤 Multilingual Karaoke Subtitle Generator

This pipeline generates **word-level karaoke-style subtitles** for videos along with translations into your favourite languages. 

It utilizes an "Air-Gapped" architecture, separating the heavy acoustic transcription engine (NVIDIA NeMo Parakeet) from the translation engine (Tencent Hy-MT2) to prevent dependency conflicts and optimize VRAM allocation.

If you choose a smaller translation model (1.8B vs the current 8B) and deactivate refinement you can run this on a toaster.

---

## 🌍 Supported Languages

### Acoustic Source Languages (Parakeet TDT v3)
The pipeline natively aligns audio and generates timestamps for the following 25 languages:
```
INPUT_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "bg": "Bulgarian", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "et": "Estonian", "fi": "Finnish", "el": "Greek",
    "hu": "Hungarian", "it": "Italian", "lv": "Latvian", "lt": "Lithuanian",
    "mt": "Maltese", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish", "ru": "Russian",
    "uk": "Ukrainian",
}
```

### Translation Target Languages (Tencent Hy-MT2)
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

### 2. The Main Pipeline Environment (NeMo / Audio Processing)
This environment handles vocal isolation, transcription, and subtitle compilation.

```
python3 -m venv venv_main
source venv_main/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 🚀 Usage

# Basic Usage (Auto-detects language, translates to English)

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

**Set Source (Acoustic) Language (Overrides Auto-Detect):**

```
python transcription.py input_track.mp4 --source-lang de
```
