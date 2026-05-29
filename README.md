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
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "bn": "Bengali", "id": "Indonesian", "tr": "Turkish", "vi": "Vietnamese",
    "pl": "Polish", "nl": "Dutch", "th": "Thai", "cs": "Czech",
    "sv": "Swedish", "ro": "Romanian", "hu": "Hungarian", "da": "Danish",
    "fi": "Finnish", "el": "Greek", "uk": "Ukrainian", "bg": "Bulgarian",
    "sk": "Slovak", "hr": "Croatian", "lt": "Lithuanian", "sl": "Slovenian",
    "et": "Estonian", "lv": "Latvian", "sr": "Serbian", "ca": "Catalan",
    "ms": "Malay", "tl": "Tagalog", "ta": "Tamil", "te": "Telugu",
    "ml": "Malayalam", "mr": "Marathi", "ur": "Urdu", "fa": "Persian",
    "he": "Hebrew", "sw": "Swahili", "am": "Amharic", "yo": "Yoruba",
    "zu": "Zulu", "af": "Afrikaans", "is": "Icelandic", "ka": "Georgian",
    "hy": "Armenian", "km": "Khmer", "my": "Burmese", "ne": "Nepali",
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


### Step 2: Execute the Master Script (Terminal B)
Open a new terminal, activate your `venv_main` environment, and run the karaoke builder.

```
source venv_main/bin/activate
```

# Display help

```
python transcription.py -h
```

# Basic Usage (Auto-detects language, translates to English)

```
python transcription.py input_track.mp4
```

### Optional Flags

**Set Target (Translation) Language:**

```
python transcription.py input_track.mp4 --target-lang pt
```

**Set Source (Acoustic) Language (Overrides Auto-Detect):**

```
python transcription.py input_track.mp4 --source-lang de
```
