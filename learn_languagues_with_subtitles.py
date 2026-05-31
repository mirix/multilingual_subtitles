"""
Multilingual Language-Learning Subtitle Builder (Decoupled ASR + Forced Alignment)
==================================================================================

End-to-end pipeline that takes a video, isolates the vocals, transcribes them via
an Autoregressive ASR (Qwen3-ASR served over HTTP), refines the transcript with
an omni LLM (Gemma 4), forces alignment of the corrected text to the audio with
stable-ts, segments into subtitle lines with SaT, optionally translates, and
burns synchronized dual subtitles back onto the video.

Why a decoupled ASR server
--------------------------
Qwen3-ASR currently needs an older transformers; Gemma 4 audio needs a newer
one. Hosting Qwen behind its own OpenAI-compatible HTTP server (qwen-asr-serve
or vllm serve) keeps the two environments separate AND lets the server stay
warm across many video runs. See QWEN_ASR_SERVER_SETUP.md for setup.

Pipeline
--------
1.   Audio isolation (BS-RoFormer + Mel-RoFormer ensemble + de-reverb).
2.   Acoustic chunking (~25 s windows, max 30 s for Gemma's audio limit).
3.   ASR via Qwen3-ASR HTTP server (one request per chunk).
3a.  Whisper LID on the vocal track (Qwen also returns language; we use Whisper
     as the primary because it's already a small dedicated tool, and fall back
     to Qwen's reported language if Whisper fails).
4.   Multimodal refinement (Gemma 4 E4B): audio + draft -> clean paragraph.
5.   Forced alignment via stable-ts: the corrected paragraph is locked to the
     audio with Whisper's DTW, giving word-level timings.
6.   Semantic segmentation via SaT to split the clean paragraph into subtitle
     phrases, capped at MAX_PHRASE_WORDS for readability.
7.   Optional translation (Tencent Hunyuan-MT2) of the phrases.
8.   ASS subtitle generation (per-word karaoke timing) + ffmpeg compilation.
"""
from __future__ import annotations

import argparse
import base64
import difflib
import gc
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Callable, Optional
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch

# =====================================================================
# CONSTANTS
# =====================================================================
MASTER_SR = 44100          # working sample rate for isolation / output stems
ASR_SR = 16000             # sample rate fed to Qwen-ASR and Gemma audio
STFT_NFFT = 2048
STFT_HOP = 512
CHUNK_TARGET_SEC = 25.0    # target chunk length; MUST stay <= 30 (Gemma audio max)
CHUNK_MAX_SEC = 28.0       # hard cap; if a "natural" chunk would exceed this, force split
REFINER_CONTEXT_PAD = 0.2  # seconds of audio context around each refined chunk

# Sentence segmentation
SAT_MODEL = os.environ.get("SAT_MODEL", "sat-12l-sm")
SAT_THRESHOLD = float(os.environ.get("SAT_THRESHOLD", "0.20"))
MAX_PHRASE_WORDS = int(os.environ.get("MAX_PHRASE_WORDS", "10"))

# Translation
TRANSLATION_BATCH_SIZE = 40

# ASR server
DEFAULT_ASR_SERVER = os.environ.get("QWEN_ASR_SERVER", "http://127.0.0.1:8000")
ASR_REQUEST_TIMEOUT_SEC = 300
NUM_ASR_CANDIDATES = 3  # Number of diverse transcriptions to request from Qwen

# Attention backend. When unset (default 'auto'), each HF model load probes
# whether flash_attn is importable and uses 'flash_attention_2' if so, falling
# back to 'sdpa' (or the model's default) otherwise. Force a specific backend
# with USE_FLASH_ATTN='1' (require flash_attn_2) or '0' (disable) -- '1' raises
# loudly if the import fails so a typo in env doesn't silently degrade speed.
USE_FLASH_ATTN = os.environ.get("USE_FLASH_ATTN", "auto").lower()

MODEL_BS = "bs_roformer_vocals_resurrection_unwa.ckpt"
MODEL_MEL = "melband_roformer_big_beta5e.ckpt"
MODEL_DEREVERB = "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt"
# Workaround for an audio-separator download bug: requesting MODEL_DEREVERB by
# name fails to fetch a usable checkpoint, but the same weights download fine
# under this alias, which we then rename. See ensure_dereverb_model().
MODEL_DEREVERB_DOWNLOAD_ALIAS = "dereverb_echo_mbr_fused.ckpt"

# Persistent, shared model cache. This must NOT live inside the per-run work
# directory (which is deleted at the end of every run); otherwise audio-separator
# re-downloads every model on each invocation and overwrites any locally swapped
# checkpoint.
DEFAULT_MODEL_CACHE_DIR = os.environ.get(
    "AUDIO_SEPARATOR_MODEL_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "audio-separator-models"),
)

OMITTED = "[OMITTED_BY_LLM]"

# =====================================================================
# LOGGING
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("subtitle_builder")

# =====================================================================
# LANGUAGE TABLES
# =====================================================================
# Source-language list reflects Qwen3-ASR-1.7B's 30 supported languages.
# Lost vs the Parakeet pipeline: bg, hr, et, lv, lt, mt, sk, sl, uk.
# Gained: zh, yue, ar, ja, ko, vi, hi, id, ms, th, tr, fa, mk, fil.
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

# Full name table; broader than the source-language set on purpose so that
# Whisper LID codes (which may return e.g. "he", "uk", "no") still resolve to
# a readable language name for the prompt and translator.
LANGUAGE_NAMES = dict(INPUT_LANGUAGES)
LANGUAGE_NAMES.update({
    "bg": "Bulgarian", "hr": "Croatian", "et": "Estonian", "lv": "Latvian",
    "lt": "Lithuanian", "mt": "Maltese", "sk": "Slovak", "sl": "Slovenian",
    "uk": "Ukrainian", "he": "Hebrew", "no": "Norwegian", "ca": "Catalan",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "gu": "Gujarati", "ur": "Urdu", "tl": "Filipino",  # synonym of "fil"
})

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

# =====================================================================
# LAZY MODEL LOADING
# =====================================================================
_sat_model = None


def get_sat_model():
    """Lazily load and cache the SaT (Segment any Text) segmentation model."""
    global _sat_model
    if _sat_model is None:
        logger.info("Initializing SaT (Segment any Text) model: %s", SAT_MODEL)
        from wtpsplit import SaT
        _sat_model = SaT(SAT_MODEL)
    return _sat_model


def lang_name(code: str | None) -> str:
    if not code:
        return "its natively detected language"
    return LANGUAGE_NAMES.get(code.lower(), code.upper())

# Qwen reports languages as English NAMES; Whisper/stable-ts want ISO codes.
_NAME_TO_CODE = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}
_NAME_TO_CODE.update({"mandarin": "zh", "tagalog": "tl"})  # Qwen-name spellings

# Codes in the Qwen set that Whisper / stable-ts don't recognise.
_TO_WHISPER_CODE = {"fil": "tl"}  # Whisper uses ISO 'tl' for Tagalog

def normalize_lang_code(value: str | None) -> str | None:
    """ISO code or English name -> canonical lowercase ISO code."""
    if not value:
        return None
    v = value.strip().lower()
    if not v or v == "auto":
        return None
    if v in LANGUAGE_NAMES:
        return v
    return _NAME_TO_CODE.get(v, v)

def to_whisper_lang_code(value: str | None) -> str | None:
    code = normalize_lang_code(value)
    return None if code is None else _TO_WHISPER_CODE.get(code, code)
    
# =====================================================================
# UTILITIES
# =====================================================================
def flush_vram(stage: str = "") -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # torch.cuda.ipc_collect() WURDE ENTFERNT, UM DEADLOCKS ZU VERMEIDEN!
        if stage:
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            logger.info("VRAM [%s]: %.2f GB allocated, %.2f GB reserved", stage, allocated_gb, reserved_gb)


def best_compute_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def resolve_attn_implementation() -> str | None:
    """Decide which attention backend to ask transformers to use.

    Returns:
        - 'flash_attention_2' if flash_attn is importable AND we're on CUDA
          AND USE_FLASH_ATTN is 'auto' or '1';
        - 'sdpa' as a sensible fallback on CUDA;
        - None on CPU (let transformers pick the default).

    USE_FLASH_ATTN='1' forces flash-attn 2 and raises if unavailable, so a misconfig
    fails loudly rather than silently dropping back to a slower backend.
    """
    if not torch.cuda.is_available():
        if USE_FLASH_ATTN == "1":
            raise RuntimeError(
                "USE_FLASH_ATTN=1 but CUDA is unavailable; flash_attention_2 requires a GPU."
            )
        return None

    if USE_FLASH_ATTN == "0":
        return "sdpa"

    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError as exc:
        if USE_FLASH_ATTN == "1":
            raise RuntimeError(
                "USE_FLASH_ATTN=1 but flash_attn is not importable. "
                "Install with: pip install -U flash-attn --no-build-isolation"
            ) from exc
        logger.info("flash_attn not available; using 'sdpa' attention.")
        return "sdpa"


def check_external_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"Required tool(s) not found on PATH: {', '.join(missing)}. "
            "Install ffmpeg (which bundles ffprobe) and retry."
        )


def run_ffmpeg(cmd: list, *, action: str) -> None:
    """Run an ffmpeg command, surfacing the last lines of stderr on failure."""
    try:
        subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, check=True,
        )
    except KeyboardInterrupt:
        logger.warning("User interrupted! Halting ffmpeg (%s)...", action)
        raise
    except subprocess.CalledProcessError as exc:
        tail = "\n".join((exc.stderr or "").strip().splitlines()[-15:])
        logger.error(
            "ffmpeg failed during %s (exit code %s):\n%s",
            action, exc.returncode, tail or "<no stderr captured>",
        )
        raise RuntimeError(f"ffmpeg failed during {action}") from exc


def escape_for_ffmpeg_subtitles(path: str) -> str:
    """Escape a file path for use inside an ffmpeg ``subtitles``/``ass`` filter."""
    path = path.replace("\\", "/")
    for ch in (":", "[", "]", "'", ","):
        path = path.replace(ch, "\\" + ch)
    return path


def sanitize_ass_text(text: str) -> str:
    """Neutralise characters that would corrupt an ASS dialogue line.

    Braces start override blocks and a backslash followed by N/n/h is treated
    as a line break/space, even outside braces, so lyric content containing
    those characters could silently break rendering.
    """
    return (
        text.replace("\\", "/")
            .replace("{", "(")
            .replace("}", ")")
            .replace("\r", " ")
            .replace("\n", " ")
    )


def _make_phrase(words: list) -> dict:
    return {
        "text": " ".join(w["text"] for w in words).strip(),
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": words,
    }


def _split_overlong_phrase(words: list, max_words: int) -> list:
    """Recursively break a phrase that exceeds ``max_words`` at its largest
    internal pause. This is the safety net for any phrase that SaT leaves too
    long for a readable subtitle."""
    if max_words <= 0 or len(words) <= max_words:
        return [_make_phrase(words)] if words else []

    best_idx, best_score = None, None
    n = len(words)
    for i in range(n - 1):
        gap = words[i + 1]["start"] - words[i]["end"]
        # Prefer large gaps; tie-break toward the middle for balance.
        centrality = -abs((i + 1) - n / 2.0)
        score = (gap, centrality)
        if best_score is None or score > best_score:
            best_score, best_idx = score, i

    left = words[: best_idx + 1]
    right = words[best_idx + 1:]
    if not left or not right:
        mid = n // 2
        left, right = words[:mid], words[mid:]
    return _split_overlong_phrase(left, max_words) + _split_overlong_phrase(right, max_words)

# =====================================================================
# DYNAMIC SERVER MANAGEMENT
# =====================================================================
def is_server_running(url: str) -> bool:
    parsed = urlparse(url)
    health_endpoint = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}/v1/models"
    try:
        import requests
        res = requests.get(health_endpoint, timeout=2)
        return res.status_code == 200
    except Exception:
        return False

def start_qwen_server_dynamically(env_path: str, url: str) -> subprocess.Popen:
    parsed = urlparse(url)
    port = parsed.port or 8000
    executable = os.path.join(env_path, "bin", "qwen-asr-serve")
    
    if not os.path.isfile(executable):
        raise FileNotFoundError(f"Server executable not found at {executable}. Check --qwen-env-path.")

    logger.info("Booting isolated Qwen-ASR server on port %d...", port)
    
    cmd = [
        executable, 
        "Qwen/Qwen3-ASR-1.7B", 
        "--gpu-memory-utilization", "0.85", # Maximize VRAM for speed
        "--max-model-len", "4096",          # Conserve KV Cache
        "--dtype", "bfloat16",
        "--attention-backend", "FLASHINFER",
        "--port", str(port)
    ]
    
    # We pipe stderr to sys.stderr so you can see if vLLM crashes during boot!
    server_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=sys.stderr)
    
    ready = False
    for _ in range(120): 
        if is_server_running(url):
            ready = True
            break
        time.sleep(1)
            
    if not ready:
        server_process.terminate()
        raise RuntimeError("Qwen-ASR server failed to start within 120 seconds. Check the console errors above.")
        
    logger.info("Qwen-ASR server is online and ready.")
    return server_process

# =====================================================================
# PHASE 1: AUDIO ISOLATION
# =====================================================================
def extract_audio_from_video(video_path: str, output_audio_path: str) -> None:
    """Extract raw uncompressed stereo PCM audio from a video source."""
    logger.info("Extracting raw uncompressed audio from video source...")
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", str(MASTER_SR), "-ac", "2",
            output_audio_path,
        ],
        action="audio extraction",
    )


def run_isolation_inference(
    input_path: str,
    model_filename: str,
    output_key: str,
    output_dir: str,
    primary_stem: str = "Vocals",
    secondary_stem: str = "Instrumental",
    model_cache_dir: str = DEFAULT_MODEL_CACHE_DIR,
) -> str:
    """Run audio-separator with one model and return the primary-stem path."""
    logger.info("Running audio-separator with model: %s", model_filename)
    from audio_separator.separator import Separator

    os.makedirs(model_cache_dir, exist_ok=True)
    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=model_cache_dir,
        output_dir=output_dir,
        output_format="WAV",
        normalization_threshold=0.9,
        use_autocast=True,
    )
    try:
        separator.load_model(model_filename=model_filename)

        expected_primary = f"{primary_stem.lower()}_{output_key}"
        expected_secondary = f"{secondary_stem.lower()}_{output_key}"
        output_files = separator.separate(
            input_path,
            {primary_stem: expected_primary, secondary_stem: expected_secondary},
        )

        if not output_files:
            raise RuntimeError(
                f"Audio separator returned no output files for model '{model_filename}'"
            )

        # NOTE: the `expected_secondary not in f` guard matters for stems whose
        # names are substrings of one another (e.g. "dry" vs "no dry").
        primary = next(
            (f for f in output_files if expected_primary in f and expected_secondary not in f),
            None,
        )
        primary = primary or next(
            (f for f in output_files if expected_secondary not in f), None
        )
        primary = primary or output_files[0]

        return os.path.join(output_dir, primary)
    finally:
        del separator
        flush_vram()


def ensure_dereverb_model(model_cache_dir: str = DEFAULT_MODEL_CACHE_DIR) -> None:
    """Work around an audio-separator download bug: if MODEL_DEREVERB is missing
    in the cache, download the (working) alias and rename it."""
    expected = os.path.join(model_cache_dir, MODEL_DEREVERB)
    if os.path.exists(expected):
        return

    logger.info(
        "Dereverb checkpoint missing; applying audio-separator workaround "
        "(download %s, then rename to %s)...",
        MODEL_DEREVERB_DOWNLOAD_ALIAS, MODEL_DEREVERB,
    )
    from audio_separator.separator import Separator

    os.makedirs(model_cache_dir, exist_ok=True)
    separator = None
    try:
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_cache_dir,
            output_dir=model_cache_dir,
            output_format="WAV",
        )
        separator.load_model(model_filename=MODEL_DEREVERB_DOWNLOAD_ALIAS)
    finally:
        if separator is not None:
            del separator
        flush_vram()

    alias_path = os.path.join(model_cache_dir, MODEL_DEREVERB_DOWNLOAD_ALIAS)
    if os.path.exists(alias_path) and not os.path.exists(expected):
        os.replace(alias_path, expected)
        logger.info("Renamed %s -> %s", MODEL_DEREVERB_DOWNLOAD_ALIAS, MODEL_DEREVERB)
    elif not os.path.exists(expected):
        logger.warning(
            "Dereverb alias '%s' not found after download attempt; the dereverb "
            "stage may fail. Place '%s' in %s manually if needed.",
            MODEL_DEREVERB_DOWNLOAD_ALIAS, MODEL_DEREVERB, model_cache_dir,
        )


def ensemble_vocals_stft(file1: str, file2: str, output_path: str) -> None:
    """Blend two isolated vocal tracks via 50/50 STFT averaging."""
    logger.info("Blending isolated vocal components using STFT...")
    y1, sr1 = librosa.load(file1, sr=MASTER_SR, mono=False)
    y2, _ = librosa.load(file2, sr=MASTER_SR, mono=False)

    if y1.ndim == 1:
        y1 = y1[np.newaxis, :]
    if y2.ndim == 1:
        y2 = y2[np.newaxis, :]
    channels = min(y1.shape[0], y2.shape[0])
    min_len = min(y1.shape[-1], y2.shape[-1])
    y1, y2 = y1[:channels, :min_len], y2[:channels, :min_len]

    ensembled_channels = []
    for ch in range(channels):
        stft1 = librosa.stft(y1[ch], n_fft=STFT_NFFT, hop_length=STFT_HOP)
        stft2 = librosa.stft(y2[ch], n_fft=STFT_NFFT, hop_length=STFT_HOP)
        ensembled_channels.append(
            librosa.istft(0.5 * stft1 + 0.5 * stft2, hop_length=STFT_HOP)
        )

    final_audio = np.stack(ensembled_channels, axis=0)
    peak = float(np.max(np.abs(final_audio))) if final_audio.size else 0.0
    if peak > 1.0:
        final_audio = final_audio / peak

    out = final_audio[0] if channels == 1 else final_audio.T
    sf.write(output_path, out, sr1)
    
def extract_vocals_for_pipeline(
    input_path: str, dense_path: str, gated_path: str,
    top_db: float = 25.0,
    max_silence_sec: float = 3.0,
    min_voice_sec: float = 0.3,
    pad_sec: float = 1.2,
) -> tuple[Callable[[float], float], list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Returns:
        (dense_to_original, voice_intervals, dense_voice_intervals)
        - dense_to_original: callable mapping dense-audio seconds → original-audio seconds
        - voice_intervals: list of (start_sec, end_sec) in the original timeline
        - dense_voice_intervals: list of (start_sec, end_sec) in the dense timeline
          (includes the 1.2s padding). Used to bypass double-cropping in Phase 2.
    """
    logger.info(
        "Processing vocals (pad=%.1fs, top_db=%.0f, max_silence=%.1fs)...",
        pad_sec, top_db, max_silence_sec,
    )
    y, sr = librosa.load(input_path, sr=ASR_SR)
    n_samples = len(y)

    raw_intervals = librosa.effects.split(y, top_db=top_db)

    if len(raw_intervals) == 0:
        sf.write(dense_path, y, sr)
        sf.write(gated_path, np.zeros_like(y), sr)
        return lambda t: t, [], []

    min_voice_samples = int(min_voice_sec * sr)
    valid_intervals = [
        (s, e) for s, e in raw_intervals
        if (e - s) >= min_voice_samples
    ]
    if not valid_intervals:
        sf.write(dense_path, y, sr)
        sf.write(gated_path, y, sr)
        return lambda t: t, [], []

    pad_samples = int(pad_sec * sr)
    padded_intervals = [
        (max(0, s - pad_samples), min(n_samples, e + pad_samples))
        for s, e in valid_intervals
    ]

    merged_intervals = [padded_intervals[0]]
    for s, e in padded_intervals[1:]:
        last_s, last_e = merged_intervals[-1]
        if s <= last_e:
            merged_intervals[-1] = (last_s, max(last_e, e))
        else:
            merged_intervals.append((s, e))

    dense_chunks: list[np.ndarray] = []
    map_chunks:   list[np.ndarray] = []
    gated_y = np.zeros_like(y)
    last_end = 0
    max_sil_samples = int(max_silence_sec * sr)

    # Track dense intervals (in seconds) to return
    dense_voice_intervals_sec = []
    dense_time = 0.0

    for start, end in merged_intervals:
        silence_len = start - last_end

        if silence_len > 0:
            if last_end == 0:
                kept_silence = min(silence_len, int(0.1 * sr))
            else:
                kept_silence = min(silence_len, max_sil_samples)

            dense_chunks.append(np.zeros(kept_silence, dtype=y.dtype))

            if kept_silence > 1:
                map_chunks.append(
                    np.linspace(last_end, start, kept_silence, dtype=np.int64)
                )
            elif kept_silence == 1:
                map_chunks.append(np.array([start], dtype=np.int64))
            
            dense_time += kept_silence / sr

        # Voice segment
        dense_chunks.append(y[start:end])
        map_chunks.append(np.arange(start, end, dtype=np.int64))
        
        # Record this interval in dense-audio seconds
        dense_voice_intervals_sec.append((dense_time, dense_time + (end - start) / sr))
        dense_time += (end - start) / sr
        last_end = end

        gated_y[start:end] = y[start:end]

    trailing = n_samples - last_end
    if trailing > 0:
        kept_trail = min(trailing, int(0.1 * sr))
        dense_chunks.append(np.zeros(kept_trail, dtype=y.dtype))
        map_chunks.append(
            np.linspace(last_end, n_samples, kept_trail, dtype=np.int64)
        )

    final_dense = np.concatenate(dense_chunks)
    time_map = np.concatenate(map_chunks)
    total_dense_samples = len(time_map)

    sf.write(dense_path, final_dense, sr)
    sf.write(gated_path, gated_y, sr)

    def dense_to_original(dense_time_sec: float) -> float:
        if dense_time_sec <= 0.0:
            return 0.0
        sample_idx = int(round(dense_time_sec * sr))
        if sample_idx >= total_dense_samples:
            return float(n_samples) / sr
        return float(time_map[sample_idx]) / sr

    voice_intervals_sec = [
        (float(s) / sr, float(e) / sr) for s, e in merged_intervals
    ]

    logger.info(
        "Dense audio: %.1fs → original %.1fs  (compression ratio %.2fx)",
        len(final_dense) / sr,
        n_samples / sr,
        n_samples / max(len(final_dense), 1),
    )
    return dense_to_original, voice_intervals_sec, dense_voice_intervals_sec

# =====================================================================
# PHASE 2: ACOUSTIC CHUNKING
# =====================================================================
def generate_audio_chunks(audio_path: str,
                          target_sec: float = CHUNK_TARGET_SEC,
                          max_sec: float = CHUNK_MAX_SEC,
                          precomputed_intervals: list | None = None) -> list:
    """Slice the vocal track into chunks bounded by natural acoustic silences.
    
    If ``precomputed_intervals`` (from dense audio processing) are provided, 
    they are used directly. This avoids running librosa.effects.split a second 
    time, which would strip the carefully computed 1.2s phoneme padding and 
    cause edge-word cropping or hallucinations on trailing silence.
    """
    logger.info("Slicing audio into ~%.1fs acoustic windows (max %.1fs)...", target_sec, max_sec)
    y, sr = librosa.load(audio_path, sr=ASR_SR)
    duration = librosa.get_duration(y=y, sr=sr)

    if precomputed_intervals:
        intervals_sec = precomputed_intervals
        logger.info("Using %d precomputed intervals (bypassing second split).", len(intervals_sec))
    else:
        intervals = librosa.effects.split(y, top_db=24.0)
        if len(intervals) == 0:
            return _fixed_windows(duration, target_sec)
        intervals_sec = [(s / sr, e / sr) for s, e in intervals]

    chunks = []
    curr_start = None
    curr_end = None

    for s_sec, e_sec in intervals_sec:
        if curr_start is None:
            curr_start, curr_end = s_sec, e_sec
            continue

        # If extending to include this voiced interval AND the gap before it exceeds max_sec,
        # close the current chunk first.
        if e_sec - curr_start > max_sec:
            chunks.append((curr_start, curr_end))
            curr_start, curr_end = s_sec, e_sec
            continue

        # Otherwise, extend the current chunk to include this interval.
        # Setting curr_end = e_sec naturally includes the compressed silence 
        # that sits between the previous interval's end and this one's start.
        curr_end = e_sec
        
        # Close at the next natural break if we've reached the target.
        if curr_end - curr_start >= target_sec:
            chunks.append((curr_start, curr_end))
            curr_start = curr_end = None

    if curr_start is not None:
        chunks.append((curr_start, curr_end))

    # If a chunk somehow exceeded max_sec (continuous singing with no breaks),
    # split it into equal sub-windows.
    safe = []
    for s, e in chunks:
        if e - s <= max_sec:
            safe.append((s, e))
        else:
            n = int(np.ceil((e - s) / target_sec))
            step = (e - s) / n
            for i in range(n):
                safe.append((s + i * step, s + (i + 1) * step))

    if not safe:
        safe = _fixed_windows(duration, target_sec)

    logger.info("Produced %d chunks (duration %.1fs)", len(safe), duration)
    return safe


def _fixed_windows(duration: float, target_sec: float) -> list:
    """Last-resort fixed-width windowing if librosa.effects.split finds nothing."""
    n = max(1, int(np.ceil(duration / target_sec)))
    step = duration / n
    return [(i * step, min((i + 1) * step, duration)) for i in range(n)]


# =====================================================================
# PHASE 2a: ACOUSTIC LANGUAGE ID
# =====================================================================
def detect_language_acoustic(audio_path: str, model_size: str = "small") -> str | None:
    """Acoustic LID via Whisper detect_language. Falls back to faster-whisper.

    Returns an ISO code or None. The qwen-asr server also returns a language,
    which is used as a downstream fallback if both Whisper paths fail (see
    `run_pipeline`).
    """
    try:
        import whisper
        wmodel = whisper.load_model(model_size)
        audio = whisper.load_audio(audio_path)
        audio = audio[: 30 * ASR_SR]
        audio = whisper.pad_or_trim(audio)
        try:
            mel = whisper.log_mel_spectrogram(audio, n_mels=wmodel.dims.n_mels).to(wmodel.device)
        except TypeError:
            mel = whisper.log_mel_spectrogram(audio).to(wmodel.device)
        _, probs = wmodel.detect_language(mel)
        lang = max(probs, key=probs.get)
        conf = probs[lang]
        del wmodel
        flush_vram("after Whisper LID")
        logger.info("Whisper acoustic LID: %s (p=%.2f)", lang, conf)
        return lang
    except Exception as exc:
        logger.warning("openai-whisper LID unavailable/failed (%s); trying faster-whisper.", exc)

    try:
        from faster_whisper import WhisperModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ctype = "float16" if torch.cuda.is_available() else "int8"
        m = WhisperModel(model_size, device=device, compute_type=ctype)
        _, info = m.transcribe(audio_path, language=None, beam_size=1, vad_filter=True)
        lang = info.language
        del m
        flush_vram("after faster-whisper LID")
        logger.info("faster-whisper LID: %s (p=%.2f)", lang, getattr(info, "language_probability", 0.0))
        return lang
    except Exception as exc:
        logger.warning("faster-whisper LID unavailable/failed (%s).", exc)

    return None

def generate_whisper_hints(audio_path: str, chunk_times: list, model_size: str = "large-v3", language: str | None = None) -> list[str]:
    """Run faster-whisper on the chunked audio to generate cross-model acoustic hints."""
    logger.info("Generating Whisper acoustic hints for %d chunks...", len(chunk_times))
    try:
        from faster_whisper import WhisperModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ctype = "float16" if torch.cuda.is_available() else "int8"
        model = WhisperModel(model_size, device=device, compute_type=ctype)
        
        y, sr = librosa.load(audio_path, sr=ASR_SR)
        hints = []
        
        for i, (start, end) in enumerate(chunk_times):
            clip = y[int(start * sr): int(end * sr)]
            if clip.size == 0:
                hints.append("")
                continue
                
            # Write clip to a temporary buffer
            buf = io.BytesIO()
            sf.write(buf, clip, sr, format="WAV", subtype="PCM_16")
            buf.seek(0)
            
            # Transcribe the chunk
            segments, _ = model.transcribe(buf, language=language, beam_size=1, vad_filter=False)
            
            valid_texts = []
            for seg in segments:
                # The Hallucination/Low-Confidence Guard
                if seg.avg_logprob < -0.8 or seg.compression_ratio > 2.4 or seg.no_speech_prob > 0.6:
                    logger.warning(
                        "Chunk %d Whisper filter tripped: logprob=%.2f, comp=%.2f, nospeech=%.2f. Dropping: '%s'",
                        i + 1, seg.avg_logprob, seg.compression_ratio, seg.no_speech_prob, seg.text.strip()
                    )
                    continue
                    
                valid_texts.append(seg.text)
                
            text = " ".join(valid_texts).strip()
            hints.append(text)
            
        del model
        flush_vram("after Whisper hints")
        logger.info("Whisper hints generated successfully.")
        return hints
        
    except Exception as exc:
        logger.warning("Whisper hint generation failed (%s). Proceeding without hints.", exc)
        flush_vram("after Whisper hint failure")
        return [""] * len(chunk_times)
        
    except Exception as exc:
        logger.warning("Whisper hint generation failed (%s). Proceeding without hints.", exc)
        flush_vram("after Whisper hint failure")
        return [""] * len(chunk_times)
        
        
# =====================================================================
# PHASE 3: ASR (QWEN3-ASR VIA HTTP SERVER)
# =====================================================================
def _strip_qwen_tags(text: str) -> str:
    """Remove stray <asr_text>/</asr_text> markers Qwen may leak into the body."""
    return re.sub(r"</?\s*asr_text\s*>", "", text, flags=re.IGNORECASE).strip()


def _qwen_parse_response(content: str) -> tuple[str | None, str]:
    """Extract (language, transcript) from a Qwen-ASR completion.

    Strips the "language German<asr_text>..." LID preamble/leak, but is careful
    NEVER to return an empty transcript when the body actually contains text:
    each extraction strategy is only accepted if it yields non-empty content,
    otherwise we fall through. An empty return therefore means Qwen genuinely
    transcribed nothing for this clip (handled by the empty-chunk retry).
    """
    content = (content or "").strip()
    lang = None

    # Leading "language German" LID preamble, if present.
    m = re.match(r"language\s+([A-Za-z]+)\b", content, flags=re.IGNORECASE)
    if m:
        lang = m.group(1)

    # Preferred: text after an <asr_text> marker -- but only if non-empty.
    m = re.search(r"<asr_text>\s*(.*)", content, flags=re.IGNORECASE | re.DOTALL)
    if m and _strip_qwen_tags(m.group(1)):
        return lang, _strip_qwen_tags(m.group(1))

    # <language>..</language> / <text>..</text> / "transcription: .."
    lm = re.search(r"<\s*lang(?:uage)?\s*>\s*([^<]+?)\s*<\s*/\s*lang", content, re.IGNORECASE)
    if lm:
        lang = lm.group(1).strip()
    tm = re.search(r"<\s*text\s*>\s*(.*?)\s*<\s*/\s*text\s*>", content, re.IGNORECASE | re.DOTALL)
    if tm and tm.group(1).strip():
        return lang, tm.group(1).strip()
    tm = re.search(r"transcription\s*:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
    if tm and tm.group(1).strip():
        return lang, tm.group(1).strip()

    # Last resort: drop the LID preamble + stray tags, return whatever remains.
    cleaned = re.sub(r"^\s*language\s+[A-Za-z]+\b[:\s]*", "", content, flags=re.IGNORECASE)
    return lang, _strip_qwen_tags(cleaned)

def _qwen_request_once(endpoint: str, clip: np.ndarray, sr: int, model_id: str,
                       *, temperature: float = 0.0) -> tuple[str | None, str, str]:
    """Send one clip to the Qwen-ASR server (AUDIO ONLY) and return
    (lang, text, raw_content).

    Qwen3-ASR is a fixed-template ASR model whose canonical request is the audio
    with NO text or system instruction (see the upstream README examples). Adding
    English instructions does NOT reliably force an output language over
    qwen-asr-serve / vllm (QwenLM/Qwen3-ASR#93) and can actively push the model to
    emit English on hard/ambiguous audio (vllm#33768). So we send audio only and
    let the model run its own LID; the authoritative source language for the rest
    of the pipeline comes from Whisper LID, not from this prompt.
    """
    import requests
    buf = io.BytesIO()
    sf.write(buf, clip, sr, format="WAV", subtype="PCM_16")
    data_uri = f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    messages = [{
        "role": "user",
        "content": [{"type": "audio_url", "audio_url": {"url": data_uri}}],
    }]
    payload = {"model": model_id, "messages": messages, "temperature": temperature}
    r = requests.post(
        endpoint, headers={"Content-Type": "application/json"},
        data=json.dumps(payload), timeout=ASR_REQUEST_TIMEOUT_SEC,
    )
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    lang_here, text = _qwen_parse_response(raw)
    return lang_here, text.strip(), raw


def execute_qwen_asr_server(audio_path: str, chunk_times: list, server_url: str, model_id: str, language: str | None = None) -> tuple[list, str | None]:
    # ``language`` is retained for API compatibility but intentionally NOT sent
    # to the server: instruction prompts don't reliably force Qwen's output
    # language and can corrupt it. See _qwen_request_once.
    endpoint = f"{server_url.rstrip('/')}/v1/chat/completions"
    logger.info("Sending %d chunks to Qwen-ASR server (audio only)...", len(chunk_times))

    y, sr = librosa.load(audio_path, sr=ASR_SR)
    total_samples = len(y)
    results, reported_lang = [], None

    for i, (start, end) in enumerate(chunk_times):
        clip = y[int(start * sr): int(end * sr)]
        lang_here, text, raw = _qwen_request_once(endpoint, clip, sr, model_id, temperature=0.0)

        # RECOVERY: Qwen occasionally greedily collapses a clip to an empty
        # completion (typically when the clip opens on an instrumental lead-in
        # or a long quiet stretch). Re-query ONCE on a context-padded clip with a
        # little sampling before giving up. We do NOT substitute Whisper here --
        # Whisper hallucinates on exactly these non-vocal sections.
        if not text:
            logger.warning("Chunk %d: Qwen returned empty text. Raw response: %r", i + 1, raw)
            pad = int(0.5 * sr)
            ps = max(0, int(start * sr) - pad)
            pe = min(total_samples, int(end * sr) + pad)
            lang_retry, text_retry, raw_retry = _qwen_request_once(
                endpoint, y[ps:pe], sr, model_id, temperature=0.2
            )
            if text_retry:
                logger.info("Chunk %d: recovered via padded retry.", i + 1)
                text = text_retry
                lang_here = lang_here or lang_retry
            else:
                logger.warning(
                    "Chunk %d: still empty after retry (Qwen heard no speech). Raw: %r",
                    i + 1, raw_retry,
                )

        if not reported_lang and lang_here:
            reported_lang = lang_here

        results.append({
            "start": start,
            "end": end,
            "text": text,
            "alternatives": [],  # Whisper hints are attached later (companion only)
        })

    return results, reported_lang

# =====================================================================
# PHASE 4: MULTIMODAL REFINEMENT (GEMMA 4 E4B, AUDIO + TEXT)
# =====================================================================
def _build_refiner_prompt(candidates: list[str], src_lang_name: str) -> str:
    """Refinement prompt presenting the ASR outputs as CO-EQUAL candidates.

    Both recognisers are fallible and their relative quality is
    language-dependent (e.g. Whisper tends to beat Qwen on German; elsewhere the
    reverse), so neither is framed as 'primary'. The model is told to treat them
    symmetrically, use the audio as ground truth, and SYNTHESISE (merge the
    fragments that match what it hears) rather than copy one wholesale.
    """
    cand_block = "\n".join(
        f"  Candidate {chr(ord('A') + i)}: {c}" for i, c in enumerate(candidates)
    )
    n = len(candidates)
    intro = (
        f"{n} independent automatic transcriptions of it"
        if n > 1 else "a rough automatic transcription of it"
    )
    multi_rules = (
        f"The candidate transcriptions are EQUALLY (un)reliable -- each was produced "
        f"by a different speech recogniser and each may contain misrecognised or "
        f"misspelled words. Do NOT assume either one is more correct.\n\n"
        f"Method:\n"
        f"  1. The AUDIO is the ground truth -- listen first.\n"
        f"  2. Go phrase by phrase. Where the candidates disagree, choose the wording "
        f"that matches what you actually hear. You MAY take some words from one candidate "
        f"and some from the other, or use a word that neither got exactly right, if that "
        f"is what the audio says and it is more semantically logical.\n"
        f"  3. Do NOT merely copy one candidate from start to end.\n\n"
        if n > 1 else
        f"The transcription may contain misrecognised or misspelled words; the AUDIO is "
        f"the ground truth -- listen first and correct against it.\n\n"
    )
    return (
        f"You are an expert native-speaker transcriber and editor for {src_lang_name}. "
        f"You are given an audio clip and {intro}. Produce ONE clean, properly punctuated "
        f"paragraph that best captures what is actually said in the audio, written as "
        f"correct {src_lang_name}.\n\n"
        f"{multi_rules}"
        f"Rules:\n"
        f"  - Every word must be a real, correctly spelled {src_lang_name} word (or a genuine proper name).\n"
        f"  - Use full standard written forms, not clipped or slurred spoken forms.\n"
        f"  - Spell out numbers as words; never use digits.\n"
        f"  - Output ONE coherent paragraph of {src_lang_name} only -- no quotes, labels, or translation.\n\n"
        f"Candidate transcriptions:\n{cand_block}\n\n"
        f"Corrected paragraph:"
    )


def _extract_message_text(parsed) -> str:
    """Pull plain text out of whatever parse_response returns: a bare string,
    a chat-message dict, a content-block list, etc. Without this, a returned
    dict gets stringified into the output (e.g. "{'role': 'assistant', ...}")."""
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        if "content" in parsed:
            return _extract_message_text(parsed["content"])
        if parsed.get("type") == "text":
            return parsed.get("text", "")
        return ""
    if isinstance(parsed, (list, tuple)):
        return " ".join(t for t in (_extract_message_text(x) for x in parsed) if t).strip()
    return str(parsed)


def _clean_refiner_output(text: str) -> str:
    """Tidy a refiner completion: drop code fences, a leading label, a single
    pair of wrapping quotes, and collapse whitespace."""
    text = re.sub(r"^```.*\n|```$", "", text.strip(), flags=re.MULTILINE).strip()
    text = re.sub(
        r"^\s*(corrected paragraph|corrected line|corrected|correct|output|antwort|korrigiert)\s*:\s*",
        "", text, flags=re.IGNORECASE,
    )
    quote_chars = "\"'«»\u201c\u201d\u201a\u2018\u2019"
    text = text.strip()
    if len(text) >= 2 and text[0] in quote_chars and text[-1] in quote_chars:
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text).strip()


def execute_omni_refinement(audio_path: str, chunks: list, model_id: str,
                            src_lang_name: str, base_dir: str) -> list:
    """Refine each chunk's text against its audio with Gemma 4 E4B.

    Uses the documented Gemma 4 multimodal path: ``AutoModelForMultimodalLM``
    with the audio referenced INSIDE the message and
    ``apply_chat_template(..., tokenize=True, return_dict=True)``. This
    requires the custom transformers PR you've installed.
    """
    logger.info("Loading omni refiner (%s) via Transformers...", model_id)
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as exc:
        raise RuntimeError(
            "Your transformers does not expose AutoModelForMultimodalLM. "
            "Install the Gemma 4 transformers PR/branch noted in requirements.txt."
        ) from exc

    tmp_dir = os.path.join(base_dir, "_refiner_clips")
    os.makedirs(tmp_dir, exist_ok=True)
    clip_path = os.path.join(tmp_dir, "chunk.wav")

    model, processor = None, None
    audio_inputs_seen = False
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        attn_impl = resolve_attn_implementation()
        load_kwargs = dict(device_map="auto", dtype="auto")
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
            logger.info("Refiner attention backend: %s", attn_impl)
        # Flash-attn 2 requires bf16 or fp16 weights. ``dtype="auto"`` picks the
        # model's native dtype (bf16 for Gemma 4) so this combination is safe.
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)

        y, sr = librosa.load(audio_path, sr=ASR_SR)
        duration = len(y) / sr if sr else 0.0
        corrected = []

        logger.info("Refining %d chunks (audio + draft)...", len(chunks))
        # If Whisper hints are unavailable for the entire run, fall back to a
        # Qwen-only requirement so we don't discard every chunk.
        whisper_available = any(c.get("whisper", "").strip() for c in chunks)
        for i, chunk in enumerate(chunks):
            qwen_text = chunk["text"].strip()
            whisper_text = chunk.get("whisper", "").strip()

            # DISCARD criterion: keep a segment only if BOTH recognisers
            # produced output. If either is silent the segment is marginal
            # (intro / outro / instrumental) and is dropped rather than refined
            # from noise. (Relaxed to Qwen-only when Whisper is unavailable.)
            if not qwen_text or (whisper_available and not whisper_text):
                corrected.append("")
                continue

            # Co-equal candidates: Qwen + Whisper (deduplicated if identical).
            candidates = [qwen_text]
            if whisper_text and whisper_text != qwen_text:
                candidates.append(whisper_text)

            s = max(0.0, chunk["start"] - REFINER_CONTEXT_PAD)
            e = min(duration, chunk["end"] + REFINER_CONTEXT_PAD)
            sf.write(clip_path, y[int(s * sr): int(e * sr)], sr)

            prompt = _build_refiner_prompt(candidates, src_lang_name)
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": clip_path},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            # First-iteration diagnostic: verify audio actually reached the model.
            if i == 0:
                audio_keys = [k for k in inputs
                              if any(t in k.lower() for t in
                                     ("audio", "input_features", "feature", "mel"))]
                if audio_keys:
                    audio_inputs_seen = True
                    logger.info("Refiner audio inputs present: %s", ", ".join(sorted(audio_keys)))
                else:
                    logger.warning(
                        "NO audio tensors in refiner inputs (keys: %s). The model is running "
                        "TEXT-ONLY -- audio-dependent rules will have no effect.",
                        ", ".join(sorted(inputs.keys())),
                    )

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

            raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
            text_out = ""
            if hasattr(processor, "parse_response"):
                try:
                    text_out = _extract_message_text(processor.parse_response(raw))
                except Exception:
                    text_out = ""
            if not text_out:
                text_out = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

            text_out = _clean_refiner_output(text_out)
            corrected.append(text_out or chunk["text"])

            if (i + 1) % 5 == 0 or i == len(chunks) - 1:
                logger.info("  ... refined %d/%d chunks", i + 1, len(chunks))
        return corrected
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if model is not None:
            del model
        if processor is not None:
            del processor
        flush_vram("after omni refiner")


# =====================================================================
# PHASE 5: FORCED ALIGNMENT (STABLE-TS) + SEGMENTATION (SAT)
# =====================================================================
def _clamp_to_dense_voice(
    t_dense: float,
    dense_voice_intervals: list[tuple[float, float]],
) -> float:
    """Pull a DENSE-time timestamp into the nearest dense voice fragment.

    The aligner occasionally places a word inside one of the (<=3 s) silences
    that separate voice fragments in the dense track. Correcting this in DENSE
    space is safe: the gaps are tiny and adjacency is local, so we just move the
    timestamp to the nearer edge of the surrounding fragments.

    This MUST be done before remapping. The previous implementation snapped in
    the ORIGINAL timeline, where the same gaps are the full multi-second
    instrumental breaks -- so a word nudged a fraction past a voice region was
    catapulted to the next region (tens of seconds away) and start/end could
    invert, which the chronological clamp then collapsed into glued phrases.
    """
    if not dense_voice_intervals:
        return t_dense
    for vs, ve in dense_voice_intervals:
        if vs <= t_dense <= ve:
            return t_dense  # already inside a voice fragment — leave it
    best_edge, best_dist = t_dense, float("inf")
    for vs, ve in dense_voice_intervals:
        for edge in (vs, ve):
            d = abs(edge - t_dense)
            if d < best_dist:
                best_dist, best_edge = d, edge
    return best_edge


def execute_stable_ts_alignment(
    dense_audio_path: str,
    chunks: list,
    aligner_model: str,
    language: str | None,
    dense_to_original: Callable[[float], float],
    dense_voice_intervals: list[tuple[float, float]] | None = None,
) -> list:
    """Lock the refined transcript to the audio using per-chunk alignment on dense audio."""
    logger.info("Loading stable-ts (%s) for per-chunk forced alignment…", aligner_model)
    import stable_whisper

    model = stable_whisper.load_model(aligner_model)
    
    # LOAD DENSE AUDIO (This is the timeline the chunk timestamps belong to)
    y, sr = librosa.load(dense_audio_path, sr=ASR_SR) 

    global_words = []
    gap_words = 0

    try:
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            start_sec = chunk["start"]
            end_sec = chunk["end"]

            # Slice the DENSE audio array
            clip = y[int(start_sec * sr) : int(end_sec * sr)]

            if clip.size == 0:
                continue

            logger.info("  Aligning chunk %d [%.2fs - %.2fs]...", i + 1, start_sec, end_sec)

            # Align just this text to this dense audio clip
            result = model.align(
                clip,
                text,
                language=language or None,
                vad=False, 
            )

            # Remap timestamps
            for seg in result.segments:
                for w in seg.words:
                    txt = (getattr(w, "word", None) or "").strip()
                    if not txt:
                        continue

                        # 1. Calculate absolute time in the DENSE timeline
                    abs_dense_start = start_sec + float(w.start)
                    abs_dense_end = start_sec + float(w.end)

                    # 2. Clamp to nearest dense voice (prevents landing in mini-gaps)
                    if dense_voice_intervals:
                        pre_start, pre_end = abs_dense_start, abs_dense_end
                        abs_dense_start = _clamp_to_dense_voice(abs_dense_start, dense_voice_intervals)
                        abs_dense_end = _clamp_to_dense_voice(abs_dense_end, dense_voice_intervals)
                        if pre_start != abs_dense_start or pre_end != abs_dense_end:
                            gap_words += 1

                    # 3. Convert to the ORIGINAL global timeline
                    orig_start = dense_to_original(abs_dense_start)
                    orig_end = dense_to_original(abs_dense_end)

                    # Safety against zero/negative duration
                    if orig_end <= orig_start:
                        mid = (orig_start + orig_end) / 2.0
                        orig_start = mid - 0.005
                        orig_end = mid + 0.005

                    global_words.append({
                        "text": txt,
                        "start": orig_start,
                        "end": orig_end
                    })

        if gap_words:
            logger.info("Clamped %d word(s) inside dense silences.", gap_words)

        # Safety: ensure chronological order across chunk boundaries
        global_words.sort(key=lambda x: x["start"])
        for i in range(1, len(global_words)):
            if global_words[i]["start"] < global_words[i - 1]["end"]:
                global_words[i]["start"] = global_words[i - 1]["end"]

        logger.info("Aligned %d total words.", len(global_words))
        return global_words

    finally:
        del model
        flush_vram("after stable-ts")

def segment_aligned_words(aligned_words: list, max_gap_sec: float = 0.6, linger_sec: float = 2.0, min_words: int = 3) -> list:
    """
    Segment the aligned-word stream into subtitle phrases using acoustic gaps.
    Includes linger time and a min-words safety net to prevent orphaned words.
    """
    if not aligned_words:
        return []
    logger.info("Segmenting aligned transcript using acoustic gaps and punctuation...")

    phrases = []
    current_phrase = []

    for i, word in enumerate(aligned_words):
        current_phrase.append(word)
        should_break = False
        
        text_clean = word["text"].strip()
        has_punctuation = text_clean.endswith((".", "?", "!", ":", ";", ","))
        
        # Check the acoustic gap to the next word
        if i < len(aligned_words) - 1:
            next_word = aligned_words[i + 1]
            gap = next_word["start"] - word["end"]
        else:
            gap = float('inf')  # End of track
            
        # --- THE SMART BREAK LOGIC ---
        # 1. Hard cap reached
        if len(current_phrase) >= MAX_PHRASE_WORDS:
            should_break = True
        # 2. End of track reached
        elif gap == float('inf'):
            should_break = True
        # 3. Punctuation OR Acoustic Gap detected...
        elif has_punctuation or gap >= max_gap_sec:
            # ...BUT only break if we have enough words, OR if it's a huge silence (over 1.5s)
            if len(current_phrase) >= min_words or gap >= 1.5:
                should_break = True
                
        # Commit the phrase
        if should_break:
            phrase_start = current_phrase[0]["start"]
            phrase_end = current_phrase[-1]["end"]
            
            # Linger logic
            if gap != float('inf'):
                available_silence = gap
            else:
                available_silence = linger_sec
                
            extended_display_end = phrase_end + min(linger_sec, max(0, available_silence - 0.1))
            
            phrases.append({
                "text": " ".join(w["text"] for w in current_phrase).strip(),
                "start": phrase_start,
                "end": extended_display_end,
                "words": current_phrase
            })
            current_phrase = []

    logger.info("Produced %d subtitle phrases.", len(phrases))
    return phrases

# =====================================================================
# PHASE 6: TRANSLATION
# =====================================================================
def _parse_numbered_lines(completion: str, expected_count: int) -> list:
    """Parse ``N| text`` numbered lines. The pipe is treated as optional; a
    numeric prefix out of range is treated as content (a continuation of the
    current line) rather than being silently discarded."""
    line_re = re.compile(r"^(\d+)\s*\|?\s*(.*)")
    out = [""] * expected_count
    current_idx = -1

    for line in completion.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = line_re.match(line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < expected_count:
                current_idx = idx
                out[idx] = m.group(2).strip()
                continue
        if current_idx != -1:
            out[current_idx] = (out[current_idx] + " " + line).strip()

    return [r if r else OMITTED for r in out]


def execute_hf_task(model, tokenizer, prompt: str, expected_count: int) -> list:
    """Send a single prompt to a HF causal LM and parse numbered-line output."""
    messages = [{
        "role": "user",
        "content": (
            "You are a precise data formatting AI. Output strictly the requested "
            f"numbered lines. Output NOTHING else.\n\n{prompt}"
        ),
    }]
    text_inputs = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=8192, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    completion = re.sub(r"^```.*\n|```$", "", completion.strip(), flags=re.MULTILINE).strip()
    if "<think>" in completion:
        completion = completion.split("</think>")[-1].strip() if "</think>" in completion else ""
    return _parse_numbered_lines(completion, expected_count)


def execute_task_with_recovery(model, tokenizer, prompt_builder: Callable[[list], str],
                               expected_count: int, recovery_label: str = "task") -> list:
    """Execute a numbered-line task and retry any lines that came back OMITTED."""
    if expected_count == 0:
        return []

    result = execute_hf_task(
        model, tokenizer, prompt_builder(list(range(expected_count))), expected_count
    )
    omitted = [i for i, r in enumerate(result) if r == OMITTED]
    if not omitted:
        return result
    if len(omitted) > expected_count // 2:
        logger.warning("[%s] Skipping recovery (systemic failure).", recovery_label)
        return result

    logger.info("[%s] Recovering %d omitted line(s)...", recovery_label, len(omitted))
    recovered = execute_hf_task(model, tokenizer, prompt_builder(omitted), len(omitted))
    for slot, rec_text in zip(omitted, recovered):
        if rec_text != OMITTED:
            result[slot] = rec_text
    return result


def _build_translation_prompt(corrected_texts: list, global_indices: list,
                              src_name: str, tgt_name: str) -> str:
    source_block = "\n".join(
        f"{slot + 1}| {corrected_texts[orig_idx]}"
        for slot, orig_idx in enumerate(global_indices)
    )
    return (
        f"Translate the following lines from {src_name} into {tgt_name}.\n\n"
        f"RULES:\n"
        f"- There are exactly {len(global_indices)} numbered lines. "
        f"Do not shift content across line boundaries.\n"
        f"- Output the translation only.\n"
        f"- Maintain the exact format: '<number>| <translated text>'.\n"
        f"- Start your response directly with '1| '.\n\n"
        f"--- SOURCE ({src_name}) ---\n{source_block}\n--- TARGET ({tgt_name}) ---\n"
    )


def translate_phrases(corrected_texts: list, translation_model: str,
                      src_name: str, tgt_name: str) -> list:
    """Translate phrases in batches and return aligned translations."""
    logger.info("Loading translator LLM (%s)...", translation_model)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    expected_count = len(corrected_texts)
    translations = [""] * expected_count
    model_trans, tokenizer = None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(translation_model)
        attn_impl = resolve_attn_implementation()
        load_kwargs = dict(
            device_map="auto",
            torch_dtype=best_compute_dtype(),
            trust_remote_code=True,
        )
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
            logger.info("Translator attention backend: %s", attn_impl)
        model_trans = AutoModelForCausalLM.from_pretrained(translation_model, **load_kwargs)

        for batch_start in range(0, expected_count, TRANSLATION_BATCH_SIZE):
            batch_end = min(batch_start + TRANSLATION_BATCH_SIZE, expected_count)
            batch_global = list(range(batch_start, batch_end))
            logger.info("Translating phrases %d-%d...", batch_start + 1, batch_end)

            batch_result = execute_task_with_recovery(
                model_trans, tokenizer,
                prompt_builder=lambda local, _g=batch_global: _build_translation_prompt(
                    corrected_texts, [_g[i] for i in local], src_name, tgt_name
                ),
                expected_count=len(batch_global),
                recovery_label="translation",
            )
            for idx, res in zip(batch_global, batch_result):
                translations[idx] = "" if res == OMITTED else res
        return translations
    finally:
        if model_trans is not None:
            del model_trans
        if tokenizer is not None:
            del tokenizer
        flush_vram("after translator unload")


# =====================================================================
# PHASE 7: COMPILATION
# =====================================================================
def gen_ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.cc (centiseconds)."""
    total_cs = int(round(seconds * 100))
    hours, remainder_cs = divmod(total_cs, 360000)
    minutes, remainder_cs = divmod(remainder_cs, 6000)
    secs, cs = divmod(remainder_cs, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def get_video_resolution(path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream; default 1920x1080."""
    try:
        res = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", path],
            text=True,
        ).strip()
        w, h = res.split("x")
        return int(w), int(h)
    except Exception:
        logger.warning("Could not probe video resolution; defaulting to 1920x1080.")
        return 1920, 1080


def build_ass_document(phrases: list, translations: list, vid_w: int, vid_h: int) -> str:
    """Build the full ASS subtitle document with per-word \\k karaoke timing."""
    main_fs = int(vid_h * 0.06)
    trans_fs = int(vid_h * 0.04)
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "WrapStyle: 0",
        f"PlayResX: {vid_w}", f"PlayResY: {vid_h}", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        f"Style: StudyMain,Arial,{main_fs},&H0000FFFF,&H00FFFFFF,&H00000000,"
        "&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1", "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]

    for idx, phrase in enumerate(phrases):
        words = phrase["words"]
        if not words:
            continue
        line_start = words[0]["start"]
        line_end = words[-1]["end"]

        # Per-word ``\k`` karaoke tags time the in-line highlight to each word.
        k_parts, curr_t = [], line_start
        for w in words:
            gap = int(round((w["start"] - curr_t) * 100))
            if gap > 0:
                k_parts.append(f"{{\\k{gap}}}")
            duration = max(1, int(round((w["end"] - w["start"]) * 100)))
            k_parts.append(f"{{\\k{duration}}}{sanitize_ass_text(w['text'])} ")
            curr_t = w["end"]
        read_along = "".join(k_parts).strip()

        translation = translations[idx] if idx < len(translations) else ""
        if translation:
            combined = (
                f"{read_along}\\N{{\\r\\c&H00A0A0A0&\\fs{trans_fs}}}"
                f"{sanitize_ass_text(translation)}"
            )
        else:
            combined = read_along

        lines.append(
            f"Dialogue: 0,{gen_ass_time(line_start)},{gen_ass_time(line_end)},"
            f"StudyMain,,0,0,0,,{combined}"
        )
    return "\n".join(lines)


def compile_video(clean_video_path: str, ass_file: str, final_vid: str) -> None:
    """Burn the ASS subtitles onto the video, keeping the original audio."""
    escaped_ass = escape_for_ffmpeg_subtitles(ass_file)
    cmd = [
        "ffmpeg", "-y", "-i", clean_video_path,
        "-vf", f"subtitles={escaped_ass}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        final_vid,
    ]
    run_ffmpeg(cmd, action="video compilation")


# =====================================================================
# ORCHESTRATION
# =====================================================================
def run_pipeline(
    video_path: str,
    target_lang: str,
    source_override: str = "auto",
    asr_server: str = DEFAULT_ASR_SERVER,
    asr_model: str = "Qwen/Qwen3-ASR-1.7B",
    correction_model: Optional[str] = "google/gemma-4-E4B-it",
    translation_model: Optional[str] = "tencent/Hy-MT2-7B",
    skip_correction: bool = False,
    output_dir: Optional[str] = None,
    model_cache_dir: Optional[str] = None,
    aligner_model: str = "large-v3",
    whisper_lid_model: str = "large-v3",
    keep_workspace: bool = False,
    qwen_env_path: str = "qwen-env",
) -> str:
    """Run the end-to-end subtitle pipeline and return the output video path."""
    check_external_tools()

    clean_video_path = video_path.strip("'\" ")
    if not os.path.isfile(clean_video_path):
        raise FileNotFoundError(f"Input video not found: {clean_video_path}")

    target_lang = (target_lang or "en").lower().strip()
    source_override = (source_override or "auto").lower().strip()

    out_dir = os.path.abspath(output_dir) if output_dir else os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    cache_dir = os.path.abspath(model_cache_dir or DEFAULT_MODEL_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Using persistent model cache: %s", cache_dir)

    base_dir = os.path.abspath(
        os.path.join(tempfile.gettempdir(), f"langstudy_{uuid.uuid4().hex[:8]}")
    )
    os.makedirs(base_dir, exist_ok=True)

    extracted = os.path.join(base_dir, "extracted_raw.wav")
    v_stft = os.path.join(base_dir, "isolated_vocals_master.wav")
    v_deecho = os.path.join(base_dir, "isolated_vocals_deecho.wav")
    ass_file = os.path.join(base_dir, "compiled_subtitles.ass")

    file_name = os.path.basename(clean_video_path)
    final_vid = os.path.join(
        out_dir, os.path.splitext(file_name)[0] + f"_Study_{target_lang.upper()}.mp4"
    )

    server_process = None
    try:
        # ---- PHASE 1: AUDIO ISOLATION ----
        extract_audio_from_video(clean_video_path, extracted)
        v_bs = run_isolation_inference(
            extracted, MODEL_BS, "bs", base_dir, model_cache_dir=cache_dir
        )
        v_mel = run_isolation_inference(
            extracted, MODEL_MEL, "mel", base_dir, model_cache_dir=cache_dir
        )
        ensemble_vocals_stft(v_bs, v_mel, v_stft)
        ensure_dereverb_model(cache_dir)
        d_stem = run_isolation_inference(
            v_stft, MODEL_DEREVERB, "dereverb", base_dir, "dry", "no dry",
            model_cache_dir=cache_dir,
        )
        os.replace(d_stem, v_deecho)

        # NEU: Erstelle Dense-Track (für LLM) und Gated-Track (für stable-ts)
        v_dense = os.path.join(base_dir, "isolated_vocals_dense.wav")
        v_gated = os.path.join(base_dir, "isolated_vocals_gated.wav")
        remap_dense_time, voice_intervals, dense_voice_intervals = extract_vocals_for_pipeline(v_deecho, v_dense, v_gated)


        # ---- PHASE 2: CHUNKING + LID ----
        # IMPORTANT: Use v_dense for all ASR & LLM reasoning!
        chunk_times = generate_audio_chunks(v_dense, precomputed_intervals=dense_voice_intervals)
        if not chunk_times:
            raise RuntimeError("Chunking produced no output.")

        source_lang: str | None = None
        if source_override == "auto":
            source_lang = detect_language_acoustic(v_dense, model_size=whisper_lid_model)
        else:
            source_lang = source_override
            
        source_lang = normalize_lang_code(source_lang)

        # ---- PHASE 3: DYNAMIC QWEN SERVER ----
        if not is_server_running(asr_server):
            server_process = start_qwen_server_dynamically(qwen_env_path, asr_server)
        else:
            logger.info("Qwen-ASR server is already running on %s.", asr_server)

        chunks, qwen_lang = execute_qwen_asr_server(
            v_dense, chunk_times, asr_server, asr_model, language=source_lang
        )
        
        # Kill the server ASAP to clear VRAM for Gemma!
        if server_process is not None:
            logger.info("Tearing down dynamically started Qwen-ASR server to free VRAM...")
            server_process.terminate()
            server_process.wait(timeout=10)
            # Only kill child processes of the dynamic server, leave manual servers alone!
            subprocess.run(["pkill", "-P", str(server_process.pid)])
            flush_vram("after Qwen Server teardown")
            server_process = None

        if source_lang is None and qwen_lang:
            source_lang = qwen_lang
            logger.info("Using Qwen-reported language as LID fallback: %s", source_lang)

        src_name = lang_name(source_lang)
        tgt_name = OUTPUT_LANGUAGES.get(target_lang, target_lang.upper())
        logger.info(
            "Source language: %s (%s) | Target: %s (%s)",
            source_lang or "unknown", src_name, target_lang, tgt_name,
        )

        # =====================================================================
        # ---- INSERT PHASE 2b HERE ----
        # =====================================================================
        # ---- PHASE 2b: WHISPER ACOUSTIC HINTS ----
        whisper_hints = generate_whisper_hints(v_dense, chunk_times, model_size=whisper_lid_model, language=to_whisper_lang_code(source_lang))
        
        # Store each Whisper hint alongside the Qwen line as a CO-EQUAL second
        # ASR candidate. Whisper beats Qwen on some languages (e.g. German) and
        # vice versa elsewhere, so neither is treated as 'primary'. The refiner
        # (Phase 4) decides per-phrase against the audio; if either recogniser
        # is empty for a chunk, that chunk is discarded there.
        for i, hint in enumerate(whisper_hints):
            if i < len(chunks):
                chunks[i]["whisper"] = hint.strip()
        # =====================================================================

        # ---- PHASE 4: MULTIMODAL REFINEMENT ----
        do_refine = (
            not skip_correction
            and correction_model
            and correction_model.lower() != "none"
        )
        if do_refine:
            logger.info("Phase 4: Multimodal refinement with (%s)...", correction_model)
            refined = execute_omni_refinement(
                v_dense, chunks, correction_model, src_name, base_dir
            )
      
            logger.info("=========================================================")
            logger.info("Phase 4: Chunk-level Correction Results")
            logger.info("=========================================================")
            corrections = 0
            for i, (orig_chunk, corr) in enumerate(zip(chunks, refined)):
                qwen_text = orig_chunk["text"].strip()
                whisper_text = orig_chunk.get("whisper", "").strip()
                # "" here means execute_omni_refinement discarded the chunk
                # (one of the recognisers was empty). Do NOT fall back to the
                # Qwen text, or the discard would be undone.
                corr_text = corr.strip()
                orig_chunk["text"] = corr_text  # propagate to Phase 5 ("" => dropped)

                final_disp = corr_text or "(discarded -- an ASR was empty)"
                if whisper_text and whisper_text != qwen_text:
                    logger.info(
                        "Chunk %d:\n  [Qwen   ]: %s\n  [Whisper]: %s\n  [Final  ]: %s",
                        i + 1, qwen_text or "(empty)", whisper_text, final_disp,
                    )
                else:
                    logger.info(
                        "Chunk %d:\n  [Qwen   ]: %s\n  [Final  ]: %s",
                        i + 1, qwen_text or "(empty)", final_disp,
                    )

                if corr_text and corr_text != qwen_text:
                    corrections += 1
                    
            logger.info("=========================================================")
            logger.info("Total chunk-level corrections: %d", corrections)
        else:
            logger.info("Phase 4 skipped: using Qwen ASR text directly.")

        # ---- PHASE 5: FORCED ALIGNMENT + SEGMENTATION ----
        aligned_words = execute_stable_ts_alignment(
            dense_audio_path=v_dense,
            chunks=chunks,
            aligner_model=aligner_model,
            language=to_whisper_lang_code(source_lang),
            dense_to_original=remap_dense_time,
            dense_voice_intervals=dense_voice_intervals,
        )
        
        phrases = segment_aligned_words(aligned_words)
        if not phrases:
            raise RuntimeError("Segmentation produced no phrases.")
            
        logger.info("=========================================================")
        logger.info("Phase 5: Synchronized Subtitle Phrases (SaT Segments)")
        logger.info("=========================================================")
        for i, ph in enumerate(phrases):
            logger.info("  Phrase %d [%.2fs - %.2fs]: %s", i+1, ph["start"], ph["end"], ph["text"])
        logger.info("=========================================================")
        
        corrected_texts = [p["text"] for p in phrases]

        # ---- PHASE 6: TRANSLATION ----
        needs_translation = bool(target_lang) and target_lang != source_lang
        can_translate = bool(translation_model) and translation_model.lower() != "none"

        if needs_translation and can_translate:
            translations = translate_phrases(
                corrected_texts, translation_model, src_name, tgt_name
            )
        else:
            logger.info(
                "Target matches source or no translator provided; skipping translation."
            )
            translations = list(corrected_texts)

        # ---- PHASE 7: COMPILATION ----
        vid_w, vid_h = get_video_resolution(clean_video_path)
        ass_content = build_ass_document(phrases, translations, vid_w, vid_h)
        with open(ass_file, "w", encoding="utf-8") as f:
            f.write(ass_content)

        compile_video(clean_video_path, ass_file, final_vid)
        logger.info("Success! Master exported: %s", final_vid)
        return final_vid
    except Exception:
        logger.error("Pipeline crashed.", exc_info=True)
        raise
    finally:
        if server_process is not None:
            logger.info("Emergency teardown of Qwen-ASR server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except Exception:
                server_process.kill()
            subprocess.run(["pkill", "-P", str(server_process.pid)])
            
        flush_vram("end of pipeline")
        if keep_workspace:
            logger.info("Workspace preserved at: %s", base_dir)
        else:
            shutil.rmtree(base_dir, ignore_errors=True)


# =====================================================================
# CLI ENTRY POINT
# =====================================================================
def _format_dict_for_help(d: dict, row_length: int = 5) -> str:
    items = [f"{k} ({v})" for k, v in d.items()]
    rows = [", ".join(items[i: i + row_length]) for i in range(0, len(items), row_length)]
    return "\n  ".join(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multilingual Language-Learning Subtitle Builder "
                    "(Decoupled ASR + Forced Alignment).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("video_path", help="Source media file.")
    parser.add_argument(
        "-t", "--target-lang", default="en",
        help="ISO translation target. Supported outputs:\n  "
             + _format_dict_for_help(OUTPUT_LANGUAGES),
    )
    parser.add_argument(
        "-s", "--source-lang", default="auto",
        help="ISO acoustic source. Supported inputs (Qwen3-ASR-1.7B):\n  "
             + _format_dict_for_help(INPUT_LANGUAGES),
    )
    parser.add_argument(
        "--qwen-env-path", default="qwen-env",
        help="Relative or absolute path to the isolated Qwen python environment.",
    )
    parser.add_argument(
        "--asr-server", default=DEFAULT_ASR_SERVER,
        help=f"HTTP base URL of the Qwen3-ASR server (default {DEFAULT_ASR_SERVER}). "
             f"See QWEN_ASR_SERVER_SETUP.md for setup.",
    )
    parser.add_argument(
        "-a", "--asr-model", default="Qwen/Qwen3-ASR-1.7B",
        help="HuggingFace model ID requested from the ASR server.",
    )
    parser.add_argument(
        "-c", "--correction-model", default="google/gemma-4-E4B-it",
        help="HuggingFace model ID for multimodal (audio) refinement. "
             "Must be an omni/audio-capable model. Use 'none' to skip.",
    )
    parser.add_argument(
        "-m", "--translation-model", default="tencent/Hy-MT2-7B",
        help="HuggingFace model ID for translation. Use 'none' to skip.",
    )
    parser.add_argument(
        "--aligner-model", default="large-v3",
        help="Whisper model size used by stable-ts for forced alignment "
             "(e.g. tiny, base, small, medium, large-v3).",
    )
    parser.add_argument(
        "--whisper-lid-model", default="large-v3",
        help="Whisper model size used for language detection only.",
    )
    parser.add_argument(
        "--skip-correction", action="store_true",
        help="Skip Phase 4 (multimodal refinement).",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Directory for the final video (default: current working directory).",
    )
    parser.add_argument(
        "--model-cache-dir", default=None,
        help=f"Persistent directory for downloaded separation models "
             f"(default: {DEFAULT_MODEL_CACHE_DIR}).",
    )
    parser.add_argument(
        "--keep-workspace", action="store_true",
        help="Do not delete the temporary work directory on exit.",
    )
    parser.add_argument(
        "--sat-model", default=SAT_MODEL,
        help=f"SaT (wtpsplit) sentence-segmentation model (default {SAT_MODEL}).",
    )
    parser.add_argument(
        "--sat-threshold", type=float, default=SAT_THRESHOLD,
        help=f"SaT split threshold (default {SAT_THRESHOLD}); lower => more splits.",
    )
    parser.add_argument(
        "--max-phrase-words", type=int, default=MAX_PHRASE_WORDS,
        help=f"Hard cap on words per subtitle phrase (default {MAX_PHRASE_WORDS}).",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Segmentation knobs are module-level; apply CLI overrides before the run.
    global SAT_MODEL, SAT_THRESHOLD, MAX_PHRASE_WORDS
    SAT_MODEL = args.sat_model
    SAT_THRESHOLD = args.sat_threshold
    MAX_PHRASE_WORDS = args.max_phrase_words

    try:
        run_pipeline(
            args.video_path,
            args.target_lang,
            args.source_lang,
            args.asr_server,
            args.asr_model,
            args.correction_model,
            args.translation_model,
            args.skip_correction,
            output_dir=args.output_dir,
            model_cache_dir=args.model_cache_dir,
            aligner_model=args.aligner_model,
            whisper_lid_model=args.whisper_lid_model,
            keep_workspace=args.keep_workspace,
            qwen_env_path=args.qwen_env_path,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception:
        # run_pipeline already logged the traceback.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
