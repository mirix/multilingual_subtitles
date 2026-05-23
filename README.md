# 🎤 Multilingual Karaoke Subtitle Generator

This pipeline generates **word-level karaoke-style subtitles** for videos along with translations into your favourite languages. 

It utilizes an "Air-Gapped" architecture, separating the heavy acoustic transcription engine (NVIDIA NeMo Parakeet) from the translation engine (Tencent Hy-MT2) to prevent dependency conflicts and optimize VRAM allocation.

If you choose a smaller translation model (1.8B vs the current 8B) and deactivate refinement (Qwen) you can run this on a toaster.

---

## 🌍 Supported Languages

### Acoustic Source Languages (Parakeet TDT v3)
The pipeline natively aligns audio and generates timestamps for the following 25 languages:
`en`, `zh`, `es`, `fr`, `de`, `ru`, `it`, `pt`, `ja`, `ko`, `ar`, `nl`, `sv`, `pl`, `tr`, `hi`, `vi`, `id`, `cs`, `uk`, `el`, `ro`, `hu`, `da`, `fi`

### Translation Target Languages (Tencent Hy-MT2)
The translation engine supports all 25 source languages above, plus:
`th`, `ms`, `tl`, `km`, `lo`, `my`, `bg`, `no`

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

---

## 🎯 Features

- **Word-Level Sync:** Millisecond-accurate karaoke timing using NVIDIA's Token Duration Transducer (TDT).
- **Dual-Model Decoupling:** Runs ASR and LLM translation independently, preventing VRAM overflow.
- **Smart Punctuation & Pauses:** Intelligently splits subtitles based on natural grammar, instrumental breaks, and micro-pauses in the singer's breath.
- **Dynamic Resolution Scaling:** Automatically sizes karaoke fonts based on the source video's exact pixel dimensions via `ffprobe`.

---
