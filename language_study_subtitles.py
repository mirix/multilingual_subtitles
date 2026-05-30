"""
Multilingual Language-Learning Subtitle Builder (candidate-grounded)
====================================================================

End-to-end pipeline that takes a video, isolates the vocals, transcribes them
with word-level timing, refines the transcript against the audio *and the
Parakeet n-best candidates*, optionally translates it, then burns synchronized
dual subtitles (source text plus translation) back onto the video as a
read-along language-learning aid.

This is the "study" variant: lean, fast, and llama.cpp-free. All intelligence
runs through NeMo (Parakeet) and Transformers (a small omni refiner + a small
translator), loaded and unloaded one at a time to fit a modest GPU.

What changed vs. the audio-only study script
--------------------------------------------
The multimodal refinement (Phase 2.5) used to see only the audio and the single
greedy ASR draft. It now also receives the top-N Parakeet beam hypotheses for
each phrase (Phase 2b), borrowed from the "grounded" pipeline. The refiner thus
gets both the acoustic truth (audio) and a menu of acoustically-plausible
spellings (candidates), which sharpens near-homophone and diacritic decisions
without leaving the lightweight Transformers path.

Pipeline stages
---------------
1.   Audio isolation (audio-separator: BS-RoFormer + Mel-RoFormer ensemble,
     followed by a de-reverb/de-echo pass) -- purely to give the ASR a clean
     vocal signal; the original audio is preserved in the output.
2.   ASR with NVIDIA Parakeet-TDT-0.6B-v3 (greedy pass, word-level timestamps).
2a.  Acoustic language ID via Whisper *detect_language only* (no transcription,
     so song-lyric hallucination is irrelevant). Resolves the exact source
     language, which is then named explicitly in every prompt.
2b.  Per-phrase Parakeet beam-search n-best -> the top acoustic hypotheses for
     each phrase, used as phonetic grounding. Candidates detected to be in a
     different language than the source are dropped. Disabled with --nbest 1.
2.5. Multimodal refinement: a small omni LLM (Gemma) reconciles the audio with
     the ASR draft and the n-best candidates into the most coherent reading.
3.   Optional translation of the refined transcript.
4.   ASS subtitle generation (per-word read-along timing) and ffmpeg compilation.

This module is self-contained and driven from the command line; see ``--help``.
"""
from __future__ import annotations

import argparse
import difflib
import gc
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import Callable, Optional

import librosa
import numpy as np
import soundfile as sf
import torch

# =====================================================================
# CONSTANTS
# =====================================================================
MASTER_SR = 44100          # working sample rate for isolation / output stems
ASR_SR = 16000             # sample rate expected by Parakeet / the omni refiner
STFT_NFFT = 2048
STFT_HOP = 512
MAX_GAP_SECONDS = 3.5      # silence gap that forces an acoustic phrase break
TRANSLATION_BATCH_SIZE = 40
PARAKEET_SUBSAMPLING = 8   # Parakeet-TDT time-stride subsampling factor
REFINER_CONTEXT_PAD = 0.2  # seconds of audio padding around each refined phrase
CANDIDATE_LID_MIN_WORDS = 3  # below this, text-LID is too noisy to trust; keep candidate
NBEST_DEFAULT = 3          # Parakeet hypotheses per phrase for grounding (1 disables)
MAX_CANDIDATES_IN_PROMPT = 4  # cap on alternatives shown to the refiner per phrase

MODEL_BS = "bs_roformer_vocals_resurrection_unwa.ckpt"
MODEL_MEL = "melband_roformer_big_beta5e.ckpt"
MODEL_DEREVERB = "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt"
# Workaround for an audio-separator download bug: requesting MODEL_DEREVERB by
# name fails to fetch a usable checkpoint, but the same weights download fine
# under this alias, which we then rename to MODEL_DEREVERB. See
# ensure_dereverb_model().
MODEL_DEREVERB_DOWNLOAD_ALIAS = "dereverb_echo_mbr_fused.ckpt"

# Persistent, shared model cache. This must NOT live inside the per-run work
# directory (which is deleted at the end of every run); otherwise audio-separator
# re-downloads every model on each invocation and overwrites any locally swapped
# checkpoint. Override with --model-cache-dir or the AUDIO_SEPARATOR_MODEL_DIR
# environment variable.
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
INPUT_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "bg": "Bulgarian", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "et": "Estonian", "fi": "Finnish", "el": "Greek",
    "hu": "Hungarian", "it": "Italian", "lv": "Latvian", "lt": "Lithuanian",
    "mt": "Maltese", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish", "ru": "Russian",
    "uk": "Ukrainian",
}

# Broader code->name map for translation targets / Whisper LID output naming.
LANGUAGE_NAMES = dict(INPUT_LANGUAGES)
LANGUAGE_NAMES.update({
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "vi": "Vietnamese", "id": "Indonesian", "tr": "Turkish",
    "th": "Thai", "he": "Hebrew", "fa": "Persian", "ms": "Malay",
    "tl": "Filipino", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "ur": "Urdu", "no": "Norwegian",
    "ca": "Catalan",
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
_lingua_detector = None
_lingua_unavailable = False


def get_sat_model():
    """Lazily load and cache the SaT (Segment any Text) 12-layer model."""
    global _sat_model
    if _sat_model is None:
        logger.info("Initializing SaT (Segment any Text) 12-layer model...")
        from wtpsplit import SaT
        _sat_model = SaT("sat-12l-sm")
    return _sat_model


def get_lingua_detector():
    """Lazily build (and cache) a lingua detector for text language ID."""
    global _lingua_detector, _lingua_unavailable
    if _lingua_detector is not None or _lingua_unavailable:
        return _lingua_detector
    try:
        from lingua import LanguageDetectorBuilder
        logger.info("Initializing lingua language detector (one-time)...")
        _lingua_detector = LanguageDetectorBuilder.from_all_languages().build()
    except Exception as exc:  # noqa: BLE001
        logger.warning("lingua not available (%s); candidate language filtering disabled.", exc)
        _lingua_unavailable = True
    return _lingua_detector


def lang_name(code: str | None) -> str:
    if not code:
        return "its natively detected language"
    return LANGUAGE_NAMES.get(code.lower(), code.upper())


# =====================================================================
# UTILITIES
# =====================================================================
def flush_vram(stage: str = "") -> None:
    """Force garbage collection and release cached CUDA memory."""
    gc.collect()
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if stage:
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            logger.info(
                "VRAM [%s]: %.2f GB allocated, %.2f GB reserved",
                stage, allocated_gb, reserved_gb,
            )


def best_compute_dtype() -> torch.dtype:
    """Pick bf16 where supported, else fp16 (avoids crashes on older GPUs)."""
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def check_external_tools() -> None:
    """Fail fast with a clear message if ffmpeg/ffprobe are missing."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"Required tool(s) not found on PATH: {', '.join(missing)}. "
            "Install ffmpeg (which bundles ffprobe) and retry."
        )


def run_ffmpeg(cmd: list, *, action: str) -> None:
    """Run an ffmpeg command, surfacing stderr on failure.

    The original script captured stderr but discarded it, so ffmpeg failures
    produced opaque tracebacks. Here we log the tail of stderr and raise a
    clear RuntimeError.
    """
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
    """Neutralize characters that would corrupt an ASS dialogue line.

    Braces start override blocks and a backslash followed by N/n/h is treated
    as a line-break/space even outside braces, so lyric content containing
    those characters could silently break rendering.
    """
    return (
        text.replace("\\", "/")
            .replace("{", "(")
            .replace("}", ")")
            .replace("\r", " ")
            .replace("\n", " ")
    )


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
    """Run audio-separator with a model and return the primary-stem path.

    Only the primary stem (the isolated vocals, used downstream for ASR) is
    needed; the secondary stem is left in ``output_dir`` and cleaned up with the
    temp directory.

    ``model_cache_dir`` is the persistent model store and is intentionally kept
    separate from ``output_dir`` (the ephemeral per-run work dir) so models are
    downloaded once and any locally swapped checkpoint survives across runs.
    """
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
    """Ensure the dereverb checkpoint exists in the cache, working around an
    audio-separator download bug.

    If ``MODEL_DEREVERB`` is already cached, do nothing. Otherwise fetch the
    equivalent weights via ``MODEL_DEREVERB_DOWNLOAD_ALIAS`` (which downloads
    correctly) and rename the file to ``MODEL_DEREVERB`` so the rest of the
    pipeline can load it by its expected name.
    """
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
        # load_model() triggers the download into model_cache_dir.
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

    # Normalize both to the same channel layout so a mono/stereo mismatch
    # between models cannot crash the blend.
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

    # soundfile expects (samples, channels); collapse a single channel to 1-D.
    out = final_audio[0] if channels == 1 else final_audio.T
    sf.write(output_path, out, sr1)


# =====================================================================
# PHASE 2: ASR (PARAKEET, GREEDY -> WORD TIMINGS)
# =====================================================================
def execute_parakeet_asr(audio_path: str, base_dir: str, lang_override: str = "auto") -> list:
    """Run NVIDIA Parakeet-TDT-0.6B-v3 ASR and return word-level results."""
    logger.info("Initializing NVIDIA Parakeet-TDT-0.6B-v3 (multilingual)...")
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    lang_override_norm = (lang_override or "auto").lower().strip()
    temp_mono_path = None
    asr_model = None

    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        asr_model.eval()

        y, sr = librosa.load(audio_path, sr=ASR_SR, mono=True)
        temp_mono_path = os.path.join(base_dir, f"temp_asr_mono_{uuid.uuid4().hex[:8]}.wav")
        sf.write(temp_mono_path, y, sr)

        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.preserve_alignments = True
            decoding_cfg.compute_timestamps = True
            decoding_cfg.word_separator = " "
            if lang_override_norm != "auto" and lang_override_norm in INPUT_LANGUAGES:
                decoding_cfg.language = lang_override_norm
        asr_model.change_decoding_strategy(decoding_cfg)

        logger.info("Transcribing audio (greedy, with timestamps)...")
        hypothesis = asr_model.transcribe(
            [temp_mono_path], return_hypotheses=True, timestamps=True
        )[0]

        raw_words = []
        timestamp_dict = getattr(hypothesis, "timestamp", {})
        if isinstance(timestamp_dict, dict) and "word" in timestamp_dict:
            window_stride = getattr(asr_model.cfg.preprocessor, "window_stride", 0.01)
            time_stride = window_stride * PARAKEET_SUBSAMPLING
            for w in timestamp_dict["word"]:
                text = (w.get("word") or w.get("char") or "").strip()
                if not text:
                    continue
                if "start" in w and "end" in w:
                    start, end = float(w["start"]), float(w["end"])
                elif "start_offset" in w and "end_offset" in w:
                    start = w["start_offset"] * time_stride
                    end = w["end_offset"] * time_stride
                else:
                    continue
                if end > start:
                    raw_words.append({"text": text, "start": start, "end": end})
        return raw_words
    finally:
        if temp_mono_path and os.path.exists(temp_mono_path):
            os.remove(temp_mono_path)
        if asr_model is not None:
            del asr_model
        flush_vram("after Parakeet")


# =====================================================================
# PHASE 2a: ACOUSTIC LANGUAGE ID (WHISPER detect_language ONLY)
# =====================================================================
def detect_language_acoustic(audio_path: str, words: list | None = None,
                              model_size: str = "small") -> str | None:
    """Acoustic language ID.

    Uses Whisper's detect_language (no transcription, so the lyric-hallucination
    problem with sung audio never arises). Falls back to faster-whisper, then to
    a text LID over the Parakeet transcript. Returns an ISO code or None.
    """
    start = max(0.0, float(words[0]["start"])) if words else 0.0

    # 1) openai-whisper: pure language detection on a vocal-rich 30 s window.
    try:
        import whisper
        wmodel = whisper.load_model(model_size)
        audio = whisper.load_audio(audio_path)  # 16 kHz mono float32
        i0 = int(start * ASR_SR)
        audio = audio[i0:i0 + 30 * ASR_SR]
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("openai-whisper LID unavailable/failed (%s); trying fallbacks.", exc)

    # 2) faster-whisper.
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("faster-whisper LID unavailable/failed (%s); trying text LID.", exc)

    # 3) Text LID over the Parakeet transcript (last resort).
    if words:
        det = get_lingua_detector()
        if det is not None:
            try:
                text = " ".join(w["text"] for w in words[:200])
                lg = det.detect_language_of(text)
                if lg:
                    code = lg.iso_code_639_1.name.lower()
                    logger.info("Text LID (lingua) fallback: %s", code)
                    return code
            except Exception as exc:  # noqa: BLE001
                logger.warning("lingua text LID failed (%s).", exc)

    logger.warning("Language detection failed; leaving the source generic in prompts.")
    return None


def segment_words_into_phrases(words: list, max_gap_seconds: float = MAX_GAP_SECONDS) -> list:
    """Combine acoustic silence detection with SaT-12L semantic segmentation."""
    logger.info("Applying acoustic + semantic (SaT-12L) segmentation...")

    # 1) Split on long silences.
    acoustic_chunks, current_chunk = [], []
    for i, word in enumerate(words):
        current_chunk.append(word)
        gap = (words[i + 1]["start"] - word["end"]) if i < len(words) - 1 else 0.0
        if gap >= max_gap_seconds:
            acoustic_chunks.append(current_chunk)
            current_chunk = []
    if current_chunk:
        acoustic_chunks.append(current_chunk)

    # 2) Sub-split each acoustic chunk semantically with SaT.
    final_phrases = []
    for chunk in acoustic_chunks:
        words_with_spans, curr_pos = [], 0
        for w in chunk:
            words_with_spans.append(
                {"word": w, "start_pos": curr_pos, "end_pos": curr_pos + len(w["text"])}
            )
            curr_pos += len(w["text"]) + 1  # +1 for the joining space

        raw_text = " ".join(w["text"] for w in chunk)
        if not raw_text.strip():
            continue

        sentences = get_sat_model().split(raw_text)
        word_idx = 0
        curr_search_idx = 0
        for sentence in sentences:
            sentence_start_pos = raw_text.find(sentence, curr_search_idx)
            if sentence_start_pos == -1:
                sentence_start_pos = curr_search_idx
            curr_sentence_end = sentence_start_pos + len(sentence)
            curr_search_idx = curr_sentence_end

            phrase_words = []
            while word_idx < len(words_with_spans):
                w_info = words_with_spans[word_idx]
                midpoint = (w_info["start_pos"] + w_info["end_pos"]) / 2.0
                if midpoint <= curr_sentence_end:
                    phrase_words.append(w_info["word"])
                    word_idx += 1
                else:
                    break
            if phrase_words:
                final_phrases.append(_make_phrase(phrase_words))

        remaining = [w_info["word"] for w_info in words_with_spans[word_idx:]]
        if remaining:
            final_phrases.append(_make_phrase(remaining))
    return final_phrases


def _make_phrase(phrase_words: list) -> dict:
    """Build a phrase record from an ordered list of word dicts."""
    return {
        "text": " ".join(w["text"] for w in phrase_words).strip(),
        "start": phrase_words[0]["start"],
        "end": phrase_words[-1]["end"],
        "words": phrase_words,
    }


# =====================================================================
# PHASE 2b: PARAKEET N-BEST (PHONETIC GROUNDING) + LANGUAGE FILTER
# =====================================================================
def _extract_nbest_texts(res, nbest: int) -> list:
    """Pull text out of NeMo's varied n-best return shapes."""
    cands = None
    if hasattr(res, "n_best_hypotheses") and res.n_best_hypotheses:
        cands = res.n_best_hypotheses
    elif isinstance(res, (list, tuple)):
        cands = res
    elif hasattr(res, "text"):
        cands = [res]
    elif isinstance(res, str):
        return [res.strip()]
    else:
        return []
    out = []
    for h in cands[:nbest]:
        t = getattr(h, "text", h if isinstance(h, str) else None)
        if t:
            out.append(t.strip())
    return out


def _filter_candidates_by_language(candidates: list, source_code: str | None) -> list:
    """Drop alternative candidates whose detected language differs from the
    source. Short candidates (text-LID is unreliable on them) are always kept.

    ``candidates`` should EXCLUDE the primary; the primary is never filtered.
    Beam search on a shared multilingual vocabulary leaks e.g. Ukrainian
    variants into Russian audio, and feeding those wrong-language spellings to
    the refiner is exactly the kind of anchoring we want to avoid.
    """
    if not source_code:
        return candidates
    det = get_lingua_detector()
    if det is None:
        return candidates
    source_code = source_code.lower()
    kept = []
    for c in candidates:
        if len(c.split()) < CANDIDATE_LID_MIN_WORDS:
            kept.append(c)
            continue
        try:
            lg = det.detect_language_of(c)
        except Exception:  # noqa: BLE001
            kept.append(c)
            continue
        if lg is None:
            kept.append(c)
            continue
        code = lg.iso_code_639_1.name.lower()
        if code == source_code:
            kept.append(c)
        else:
            logger.info("    dropped cross-language candidate [%s]: %s", code, c)
    return kept


def augment_phrases_with_nbest(audio_path: str, phrases: list, base_dir: str,
                               source_code: str | None = None, nbest: int = NBEST_DEFAULT,
                               pad: float = REFINER_CONTEXT_PAD) -> None:
    """Decode each phrase span with Parakeet beam search and store the top-N
    acoustic hypotheses on ``phrase['alternatives']`` (primary first, deduped,
    cross-language alternatives removed).

    This is the second Parakeet pass: the greedy pass (Phase 2) gives the
    word-level timings used everywhere downstream, while this beam pass gives
    the phonetic menu the omni refiner reasons over. They are separate loads
    because beam n-best needs the phrase boundaries that only exist after
    segmentation.

    NOTE: TDT/RNN-T beam n-best support varies across NeMo versions. This is
    wrapped so that any failure degrades gracefully to single-hypothesis mode,
    in which case Phase 2.5 behaves like the original audio-only refiner.
    """
    if not nbest or nbest <= 1:
        logger.info("N-best grounding disabled (nbest<=1); refiner will see audio + draft only.")
        for p in phrases:
            p["alternatives"] = [p["text"]]
        return

    logger.info("Phase 2b: generating top-%d Parakeet hypotheses per phrase (beam search)...", nbest)
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    asr_model = None
    tmp_dir = os.path.join(base_dir, "_nbest_clips")
    os.makedirs(tmp_dir, exist_ok=True)
    clip_paths = []
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        asr_model.eval()

        y, sr = librosa.load(audio_path, sr=ASR_SR, mono=True)
        duration = len(y) / sr if sr else 0.0
        for i, p in enumerate(phrases):
            s = max(0.0, p["start"] - pad)
            e = min(duration, p["end"] + pad)
            clip = y[int(s * sr):int(e * sr)]
            cp = os.path.join(tmp_dir, f"nb_{i:05d}.wav")
            sf.write(cp, clip, sr)
            clip_paths.append(cp)

        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "beam"
            if decoding_cfg.get("beam", None) is None:
                decoding_cfg.beam = {}
            decoding_cfg.beam.beam_size = max(2, nbest)
            decoding_cfg.beam.return_best_hypothesis = False
            if source_code and source_code in INPUT_LANGUAGES:
                decoding_cfg.language = source_code
        asr_model.change_decoding_strategy(decoding_cfg)

        results = asr_model.transcribe(clip_paths, return_hypotheses=True, batch_size=8)

        # Normalize the (best, all_nbest) tuple form some NeMo versions return.
        if isinstance(results, tuple) and len(results) == 2 and isinstance(results[0], list):
            results = results[1] if results[1] is not None else results[0]

        for i, res in enumerate(results):
            primary = phrases[i]["text"].strip()
            alts = [a for a in _extract_nbest_texts(res, nbest) if a and a != primary]
            alts = _filter_candidates_by_language(alts, source_code)
            # Dedupe with a whitespace- and case-insensitive key. Beam variants
            # that differ only by a word-internal space (e.g. "Schnupper lang"
            # vs "Schnupperlang") are the SAME hypothesis; showing both would
            # fake a consensus and bias the refiner toward that reading.
            ordered, seen = [], set()
            for a in [primary] + alts:
                if not a:
                    continue
                key = re.sub(r"\s+", "", a.lower())
                if key not in seen:
                    seen.add(key)
                    ordered.append(a)
            phrases[i]["alternatives"] = ordered[:nbest] if ordered else [primary]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Parakeet n-best failed (%s). Continuing with primary ASR text only. "
                       "If your NeMo build needs a different beam config for TDT, try "
                       "strategy='maes' here.", exc)
        for p in phrases:
            p.setdefault("alternatives", [p["text"]])
    finally:
        for cp in clip_paths:
            try:
                if os.path.exists(cp):
                    os.remove(cp)
            except OSError:
                pass
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if asr_model is not None:
            del asr_model
        flush_vram("after Parakeet n-best")


# =====================================================================
# PHASE 2.5: MULTIMODAL AUDIO-TEXT REFINER (CANDIDATE-GROUNDED)
# =====================================================================
def _norm_key(s: str) -> str:
    """Whitespace- and case-insensitive key. Two readings that differ only by a
    word-internal space (e.g. 'Schnupper lang' vs 'Schnupperlang') collapse to
    the same key, so they are treated as one option rather than a fake majority."""
    return re.sub(r"\s+", "", s.lower())


def _word_key(s: str) -> str:
    """Punctuation-, whitespace- and case-insensitive key. Used to decide whether
    an alternative is a *genuine* word difference: 'Horizont.' / 'Horizont!' /
    'Horizont' collapse to the same key, so punctuation-only forks never surface
    as phonetic alternatives, while 'Dösen' vs 'Düsen' stay distinct."""
    return re.sub(r"[^\w]", "", s.lower(), flags=re.UNICODE)


def _clean_candidate(text: str) -> str:
    """Strip beam-search noise from a candidate. Parakeet n-best frequently emits
    a stray trailing single character ('... für mich? H', '... Leute J'); a lone
    1-char token at the end is almost never a real word, so drop it."""
    text = text.strip()
    toks = text.split()
    if len(toks) >= 2 and len(toks[-1].strip(".,!?;:\u2026")) <= 1:
        toks = toks[:-1]
    return " ".join(toks).strip()


def build_confusion_slots(primary: str, candidates: list,
                          max_options: int = MAX_CANDIDATES_IN_PROMPT) -> list:
    """Keep the primary draft as an intact backbone and attach word/phrase-level
    alternatives wherever a candidate disagrees.

    Each candidate is aligned to the PRIMARY independently (pairwise), so the
    granularity never collapses just because one noisy candidate disagrees
    broadly, and an alternative for one position never drags in that candidate's
    mistakes elsewhere -- the model can take the good reading for spot A from one
    candidate and for spot B from another. Disagreement regions from all
    candidates are unioned into groups: a group is a single word for a local
    conflict ('sell' -> 'cell') and a short phrase when the conflict spans a
    merge/split ('prison sell' -> 'prisoncell'), rather than a doomed word-by-word
    alignment inside the divergent region.

    Returns tokens:
        ("text", "<agreed words>")
        ("choice", "<primary span>", ["<alt>", ...], omit: bool)
    The primary words are always preserved in order across the tokens, so the
    backbone sentence reconstructs exactly to ``primary``.
    """
    P = primary.split()
    cands = []
    for c in candidates:
        if not c or not c.strip():
            continue
        c = _clean_candidate(c)
        if c and _word_key(c) != _word_key(primary):
            cands.append(c.split())
    if not P:
        return []
    if not cands:
        return [("text", primary)]

    # 1) Pairwise edits vs the primary: (i1, i2, cand_index, alt or None=omit).
    #    Insertions are attached to a neighbouring primary word so i1 < i2 always.
    edits = []
    for ci, cw in enumerate(cands):
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, P, cw, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace":
                edits.append((i1, i2, ci, " ".join(cw[j1:j2])))
            elif tag == "delete":
                edits.append((i1, i2, ci, None))
            elif tag == "insert":
                if i1 > 0:
                    edits.append((i1 - 1, i1, ci, " ".join(P[i1 - 1:i1] + cw[j1:j2])))
                elif P:
                    edits.append((0, 1, ci, " ".join(cw[j1:j2] + P[0:1])))
    if not edits:
        return [("text", primary)]

    # 2) Union overlapping primary spans into groups.
    edits.sort(key=lambda e: (e[0], e[1]))
    groups = []  # [g1, g2, [edits]]
    for e in edits:
        if groups and e[0] < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], e[1])
            groups[-1][2].append(e)
        else:
            groups.append([e[0], e[1], [e]])

    # 3) Walk the primary, emitting agreed text and one choice per group. Each
    #    candidate's reading of a group is reconstructed from ITS edits only.
    tokens, pending, gi, pos = [], [], 0, 0

    def flush():
        if pending:
            tokens.append(("text", " ".join(pending)))
            pending.clear()

    while pos < len(P):
        if gi < len(groups) and groups[gi][0] == pos:
            g1, g2, gedits = groups[gi]
            prim_span = " ".join(P[g1:g2])

            by_cand = {}
            for (i1, i2, ci, alt) in gedits:
                by_cand.setdefault(ci, []).append((i1, i2, alt))

            alts, omit = [], False
            seen = {_word_key(prim_span)}
            for ci, ce in by_cand.items():
                ce.sort()
                words, p, k = [], g1, 0
                while p < g2:
                    if k < len(ce) and ce[k][0] == p:
                        _, e2, alt = ce[k]
                        if alt:
                            words.append(alt)
                        p, k = e2, k + 1
                    else:
                        words.append(P[p])
                        p += 1
                reading = " ".join(w for w in words if w).strip()
                if reading == "":
                    omit = True
                    continue
                key = _word_key(reading)
                if key not in seen:
                    seen.add(key)
                    alts.append(reading)

            alts = alts[:max_options]
            if alts or omit:
                flush()
                tokens.append(("choice", prim_span, alts, omit))
            else:
                pending.extend(P[g1:g2])
            pos, gi = g2, gi + 1
        else:
            pending.append(P[pos])
            pos += 1
    flush()
    return tokens


def extract_uncertain_spans(tokens: list) -> list:
    """Return [(draft_span, [alternatives]), ...] for the forked slots only.
    A possibly-omitted span gets '(omitted)' appended to its alternatives."""
    spans = []
    for tok in tokens:
        if tok[0] == "choice":
            _, prim, alts, omit = tok
            vals = list(alts) + (["(omitted)"] if omit else [])
            if prim and vals:
                spans.append((prim, vals))
    return spans


def build_uncertain_json(primary: str, candidates: list,
                         max_options: int = MAX_CANDIDATES_IN_PROMPT) -> str:
    """Build the JSON grounding payload: an ordered array of uncertain spans,
    each {"draft": <draft text>, "options": [<draft first>, <alternatives>...]}.

    An array (not a flat object) keeps order and avoids duplicate-key clobbering
    when the same span occurs twice in a line. The draft reading is always first
    in ``options`` so 'keep the draft' is an explicit choice. ``max_options`` caps
    the number of *alternatives* shown per span. Returns '' when nothing is
    uncertain (the caller then uses the audio-only prompt)."""
    spans = extract_uncertain_spans(build_confusion_slots(primary, candidates, max_options))
    if not spans:
        return ""
    payload = [{"draft": key, "options": [key] + vals} for key, vals in spans]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_refiner_prompt(phrase: dict, src_lang_name: str,
                          max_options: int = MAX_CANDIDATES_IN_PROMPT) -> str:
    """Build the per-phrase refinement prompt.

    The audio is the arbiter. When Parakeet proposed competing readings, they are
    supplied as a JSON array of per-span options (draft reading first) -- a format
    the model parses reliably. The model reads the structured grounding but writes
    back ONE plain corrected line, never JSON.
    """
    primary = phrase["text"]
    # Pass ALL stored alternatives through. Diversity is capped per span inside
    # build_confusion_slots (MAX_CANDIDATES_IN_PROMPT), so clipping whole
    # sentences here would needlessly starve spans of distinct readings.
    alts = [a for a in phrase.get("alternatives", []) if a and a.strip() and a.strip() != primary]

    base = (
        f"You are an expert native-speaker editor and transcriptionist for {src_lang_name}. "
        f"The draft below is a strong first hypothesis from an automatic recognizer, but it may contain "
        f"misheard words, wrong word boundaries, casual spoken elisions, or dropped punctuation and "
        f"diacritics. Your job is to produce the single best WRITTEN line: the version a careful native "
        f"writer would consider correct.\n\n"
        f"Your output must satisfy ALL of these together: (a) it is consistent with what the audio actually "
        f"says, (b) it is grammatically and syntactically correct {src_lang_name}, and (c) it is "
        f"semantically coherent -- the sentence makes sense. Use the audio and the listed options as "
        f"EVIDENCE that narrows the possibilities; do not copy a reading that breaks grammar or meaning just "
        f"because it is acoustically close.\n\n"
    )

    # The real-word + coherence constraints are the key guard against adopting an
    # ASR-coined non-word (e.g. 'Schnupperlang') just because it was acoustically near.
    common_rules = (
        f"- Every word MUST be a real, correctly spelled {src_lang_name} word (or a genuine proper name). "
        f"NEVER output an invented word or nonsense compound. Discard any option that is not a real word or "
        f"that makes the sentence incoherent, even if it is the closest acoustic match; if no option works, "
        f"write the correct standard {src_lang_name} word that fits the audio and the meaning.\n"
        f"- Prefer the standard written form over a casual or slurred spoken form (e.g. write the full "
        f"standard word, not a clipped colloquial spelling), even when the singer shortens or slurs it.\n"
        f"- Fix grammar, case, declension, agreement, word boundaries, capitalization, diacritics and "
        f"punctuation so the line reads as correct, idiomatic written {src_lang_name}.\n"
        f"- Write numbers as digits (e.g. 99), never spelled out as words. You may correct a number's value "
        f"if the audio clearly says a different one, but never replace an already-correct number.\n"
        f"- Change ONLY what improves correctness. Do NOT delete or replace words that are already correct, "
        f"and do NOT add words the audio does not support.\n"
        f"- Be conservative: the draft is the default. Make a change only when audio, grammar and meaning "
        f"together give you clear reason to. When the draft and an alternative are about equally plausible, "
        f"keep the draft.\n"
        f"- Do NOT translate, do NOT paraphrase for style, and do NOT add commentary, labels or quotes.\n"
    )

    def audio_only() -> str:
        return base + "RULES:\n" + common_rules + (
            f"\nDraft: {primary}\n\n"
            f"Output STRICTLY the single corrected line in {src_lang_name} and nothing else."
        )

    if not alts:
        return audio_only()

    grounding = build_uncertain_json(primary, alts, max_options)
    if not grounding:
        return audio_only()

    return base + (
        "Most of the draft is correct. The recognizer was unsure about a few spans, given below as JSON: "
        'each entry has "draft" (exactly what the draft says there) and "options" (competing readings for '
        'that span). The FIRST option is always the recognizer\'s main choice (identical to "draft") and is '
        'your safe default; the rest are alternatives it also considered. "(omitted)" as an option means the '
        "span may not be present at all. The options are suggestions, not a closed list.\n\n"
        f"Draft: {primary}\n\n"
        "Uncertain spans (JSON):\n"
        f"```json\n{grounding}\n```\n\n"
        "RULES:\n"
        + common_rules
        + (
            f"- At each span, keep the first option (the draft) unless another option is clearly more "
            f"correct and coherent AND consistent with the audio; if no option works, write the correct "
            f"standard {src_lang_name} word instead.\n"
            f"- Reply with ONE corrected line of natural {src_lang_name} only. Do NOT output JSON, keys, "
            f"brackets, or any explanation.\n\n"
            "Example input -> output:\n"
            '  Draft: There are eight people in the prison sell\n'
            '  Uncertain spans (JSON): [{"draft": "eight", "options": ["eight", "ate"]}, '
            '{"draft": "prison sell", "options": ["prison sell", "prison cell", "prisoncell"]}]\n'
            "  Corrected line: There are eight people in the prison cell\n\n"
            "Now produce the corrected line for the draft above.\n"
            "Corrected line:"
        )
    )


def _recover_if_json_echo(text: str) -> str:
    """Safety net: if the model echoes the JSON grounding instead of writing a
    line, reconstruct a best-effort sentence from the spans' first options. If we
    cannot, return '' so the caller falls back to the draft."""
    stripped = text.strip()
    if not (stripped.startswith("[") or stripped.startswith("{") or '"options"' in stripped):
        return text
    logger.warning("Refiner echoed JSON instead of a line; reconstructing from first options.")
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            data = [data]
        parts = []
        for entry in data:
            opts = entry.get("options") or []
            if opts:
                parts.append(str(opts[0]))
            elif entry.get("draft"):
                parts.append(str(entry["draft"]))
        return " ".join(parts).strip()
    except Exception:  # noqa: BLE001 - unparseable echo; let caller use the draft
        return ""


def _extract_message_text(parsed) -> str:
    """Pull plain text out of whatever parse_response returns: a bare string, a
    chat message dict ({'role','content'}), a list of messages, or content blocks
    like [{'type':'text','text':...}]. Without this, a returned dict gets
    stringified into the output (e.g. "{'role': 'assistant', 'content': '...'}")."""
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
    """Tidy a single-line refiner completion: drop code fences, a leading
    'Corrected line:'/'Correct:' label, a single pair of wrapping quotes, and
    recover gracefully if the model echoed the JSON grounding."""
    text = _strip_code_fences(text)
    text = re.sub(
        r"^\s*(corrected line|corrected|correct|output|antwort|korrigiert)\s*:\s*",
        "", text, flags=re.IGNORECASE,
    )
    text = _recover_if_json_echo(text)
    quote_chars = "\"'«»\u201c\u201d\u201a\u2018\u2019"
    text = text.strip()
    if len(text) >= 2 and text[0] in quote_chars and text[-1] in quote_chars:
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text).strip()


def execute_omni_refinement(audio_path: str, phrases: list, model_id: str,
                            src_lang_name: str = "the audio's language",
                            base_dir: Optional[str] = None,
                            max_options: int = MAX_CANDIDATES_IN_PROMPT) -> list:
    """Use an omni-modal LLM (Gemma 4 E2B/E4B) to refine ASR text against the
    actual audio AND the Parakeet n-best candidates carried on each phrase.

    Each phrase is processed independently with its own audio slice. The Parakeet
    candidates for that phrase are decomposed into per-span options and supplied
    as a small JSON payload in the text prompt -- no batching machinery is needed
    and the pass stays lean.

    Uses the documented Gemma 4 multimodal path: ``AutoModelForMultimodalLM`` with
    the audio referenced INSIDE the message (placed before the text, per the model
    card) and ``apply_chat_template(..., tokenize=True, return_dict=True)``. The
    old separate ``audios=`` processor kwarg is NOT used -- it is silently ignored
    by this processor, which would feed the model text only.
    """
    logger.info("Loading omni refiner (%s) via Transformers...", model_id)
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as exc:  # pragma: no cover - depends on transformers version
        raise RuntimeError(
            "Your transformers is too old for Gemma 4 audio (needs "
            "AutoModelForMultimodalLM). Upgrade: pip install -U transformers"
        ) from exc

    tmp_dir = os.path.join(base_dir or tempfile.gettempdir(), "_refiner_clips")
    os.makedirs(tmp_dir, exist_ok=True)

    model, processor = None, None
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, device_map="auto", dtype="auto",
        )

        y, sr = librosa.load(audio_path, sr=ASR_SR)
        duration = len(y) / sr if sr else 0.0
        corrected_texts = []
        clip_path = os.path.join(tmp_dir, "phrase.wav")

        logger.info("Refining %d phrases via omni engine (audio + candidates)...", len(phrases))
        for i, phrase in enumerate(phrases):
            start_time = max(0.0, phrase["start"] - REFINER_CONTEXT_PAD)
            end_time = min(duration, phrase["end"] + REFINER_CONTEXT_PAD)
            chunk = y[int(start_time * sr): int(end_time * sr)]
            # Write this phrase's slice; the chat template loads audio by path.
            sf.write(clip_path, chunk, sr)

            prompt = _build_refiner_prompt(phrase, src_lang_name, max_options)
            # Audio BEFORE text, per the Gemma 4 model card's modality-order note.
            messages = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": clip_path},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            # Sanity check (first phrase only): confirm the audio actually made it
            # into the model inputs. If apply_chat_template silently returned
            # text-only tensors, every "audio-guided" correction is really just
            # text cleanup -- which would explain prompt changes having no effect.
            if i == 0:
                audio_keys = [k for k in inputs
                              if any(t in k.lower() for t in
                                     ("audio", "input_features", "feature", "mel"))]
                if audio_keys:
                    logger.info("Refiner audio inputs present: %s", ", ".join(sorted(audio_keys)))
                else:
                    logger.warning(
                        "NO audio tensors in refiner inputs (keys: %s). The model is "
                        "running TEXT-ONLY -- audio is being ignored, so audio-dependent "
                        "prompt rules have no effect. Check the processor's audio support "
                        "and that the WAV path in the message is being loaded.",
                        ", ".join(sorted(inputs.keys())),
                    )
                if os.environ.get("REFINER_DEBUG_PROMPT"):
                    logger.info("Refiner prompt (phrase 0):\n%s", prompt)

            with torch.no_grad():
                # Deterministic decoding: the task is constrained correction, not
                # open generation, so we override the model card's creative defaults.
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

            raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
            # parse_response strips Gemma 4's thinking/answer control tokens, but it
            # returns a chat message ({'role': 'assistant', 'content': ...}), NOT a
            # bare string -- so we must extract the text, or every line ends up as a
            # stringified dict.
            text_out = ""
            if hasattr(processor, "parse_response"):
                try:
                    text_out = _extract_message_text(processor.parse_response(raw))
                except Exception:  # noqa: BLE001 - fall back to a plain decode
                    text_out = ""
            if not text_out:
                text_out = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            transcription = _clean_refiner_output(text_out.strip())
            corrected_texts.append(transcription or phrase["text"])

            if (i + 1) % 10 == 0 or i == len(phrases) - 1:
                logger.info("  ... refined %d/%d phrases", i + 1, len(phrases))
        return corrected_texts
    except OSError as exc:
        logger.error(
            "Failed to load model %s. Verify the HF token or that the model ID "
            "exists. Details: %s", model_id, exc,
        )
        raise
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if model is not None:
            del model
        if processor is not None:
            del processor
        flush_vram("after omni refiner")


# =====================================================================
# PHASE 3: TRANSLATION
# =====================================================================
def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from an LLM completion."""
    return re.sub(r"^```.*\n|```$", "", text.strip(), flags=re.MULTILINE).strip()


def _parse_numbered_lines(completion: str, expected_count: int) -> list:
    """Parse ``N| text`` numbered lines from an LLM completion.

    The pipe is treated as optional because models occasionally drop it. A
    numeric prefix that is out of range is treated as ordinary content (a
    continuation of the current line) rather than being silently discarded.
    """
    line_re = re.compile(r"^(\d+)\s*\|?\s*(.*)")
    result_lines = [""] * expected_count
    current_idx = -1

    for line in completion.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = line_re.match(line)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < expected_count:
                current_idx = idx
                result_lines[idx] = match.group(2).strip()
                continue
        # Not a valid numbered line -> continuation of the current line.
        if current_idx != -1:
            result_lines[current_idx] = (result_lines[current_idx] + " " + line).strip()

    return [res if res else OMITTED for res in result_lines]


def execute_hf_task(model, tokenizer, prompt: str, expected_count: int) -> list:
    """Send a single prompt to a HF causal LM and parse numbered-line output.

    Decoding is greedy and deterministic, which suits the strict numbered-line
    format. (The original passed ``temperature``/``top_p`` without
    ``do_sample=True``; transformers silently ignored them, so this was already
    greedy in practice.)
    """
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
    completion = _strip_code_fences(completion)

    if "<think>" in completion:
        if "</think>" in completion:
            completion = completion.split("</think>")[-1].strip()
        else:
            logger.warning("Thinking block was not closed; output discarded.")
            completion = ""
    return _parse_numbered_lines(completion, expected_count)


def execute_task_with_recovery(
    model, tokenizer, prompt_builder: Callable[[list], str],
    expected_count: int, recovery_label: str = "task",
) -> list:
    """Execute an LLM task and retry any lines that came back as OMITTED."""
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


def _build_translation_prompt(
    corrected_texts: list, global_indices: list, src_name: str, tgt_name: str
) -> str:
    """Build the numbered-line translation prompt for a set of line indices."""
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


def translate_phrases(
    corrected_texts: list, translation_model: str, src_name: str, tgt_name: str
) -> list:
    """Translate corrected phrases in batches and return aligned translations."""
    logger.info("Phase 3: Loading translator LLM (%s)...", translation_model)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    expected_count = len(corrected_texts)
    translations = [""] * expected_count
    model_trans, tokenizer = None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(translation_model)
        model_trans = AutoModelForCausalLM.from_pretrained(
            translation_model, device_map="auto", torch_dtype=best_compute_dtype(),
            trust_remote_code=True,
        )

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
    except OSError as exc:
        logger.error(
            "Failed to load translation model %s. Details: %s", translation_model, exc
        )
        raise
    finally:
        if model_trans is not None:
            del model_trans
        if tokenizer is not None:
            del tokenizer
        flush_vram("after translator unload")


# =====================================================================
# PHASE 4: TIMING REDISTRIBUTION + ASS COMPILATION
# =====================================================================
def redistribute_word_timings(orig_words: list, corrected_text: str) -> list:
    """Map corrected words onto the original word timings via a diff alignment."""
    corr_words = corrected_text.split()
    if not corr_words or not orig_words:
        return []

    orig_words_str = [w["text"] for w in orig_words]
    matcher = difflib.SequenceMatcher(None, orig_words_str, corr_words)
    new_words = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for idx in range(i1, i2):
                new_words.append({
                    "text": corr_words[j1 + (idx - i1)],
                    "start": orig_words[idx]["start"],
                    "end": orig_words[idx]["end"],
                })
        else:
            start_time = orig_words[i1 - 1]["end"] if i1 > 0 else orig_words[0]["start"]
            if i1 < i2:
                end_time = orig_words[i2 - 1]["end"]
            elif i1 < len(orig_words):
                end_time = orig_words[i1]["start"]
            else:
                end_time = orig_words[-1]["end"]
            end_time = max(start_time, end_time)

            block = corr_words[j1:j2]
            if block:
                step = (end_time - start_time) / len(block)
                curr = start_time
                for word in block:
                    new_words.append({"text": word, "start": curr, "end": curr + step})
                    curr += step
    return new_words


def gen_ass_time(seconds: float) -> str:
    """Format seconds as an ASS timestamp ``H:MM:SS.cc`` (centiseconds)."""
    total_cs = int(round(seconds * 100))
    hours, remainder_cs = divmod(total_cs, 360000)
    minutes, remainder_cs = divmod(remainder_cs, 6000)
    secs, cs = divmod(remainder_cs, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def get_video_resolution(path: str) -> tuple[int, int]:
    """Return ``(width, height)`` of the first video stream; default 1920x1080."""
    try:
        res = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", path,
            ],
            text=True,
        ).strip()
        w, h = res.split("x")
        return int(w), int(h)
    except Exception:
        logger.warning("Could not probe video resolution; defaulting to 1920x1080.")
        return 1920, 1080


def build_ass_document(phrases: list, translations: list, vid_w: int, vid_h: int) -> str:
    """Build the full ASS subtitle document as a string."""
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

        # The ASS `\k` tag (libass's "karaoke" tag) is used here purely to
        # highlight each word in sync with the audio as a read-along cue.
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
    """Burn the ASS subtitles onto the video, keeping the original audio.

    The original audio is copied unchanged: a language learner needs to hear
    the source vocals while reading the synced source text and translation.
    """
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
    correction_model: Optional[str] = None,
    translation_model: Optional[str] = None,
    skip_correction: bool = False,
    output_dir: Optional[str] = None,
    model_cache_dir: Optional[str] = None,
    whisper_lid_model: str = "small",
    nbest: int = NBEST_DEFAULT,
    keep_workspace: bool = False,
    max_options: int = MAX_CANDIDATES_IN_PROMPT,
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

    try:
        # ---- PHASE 1: AUDIO ISOLATION (preprocessing for clean ASR only) ----
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

        # ---- PHASE 2: ASR (greedy + word timings) ----
        words = execute_parakeet_asr(v_deecho, base_dir, source_override)
        if not words:
            logger.warning("ASR returned no words. Aborting pipeline.")
            raise RuntimeError("ASR produced no transcribable words.")

        # ---- PHASE 2a: LANGUAGE ID ----
        if source_override == "auto":
            source_lang = detect_language_acoustic(v_deecho, words=words, model_size=whisper_lid_model)
        else:
            source_lang = source_override
        src_name = lang_name(source_lang)
        tgt_name = OUTPUT_LANGUAGES.get(target_lang, target_lang.upper())
        logger.info("Source language: %s (%s) | Target: %s (%s)",
                    source_lang or "unknown", src_name, target_lang, tgt_name)

        phrases = segment_words_into_phrases(words, max_gap_seconds=MAX_GAP_SECONDS)
        expected_count = len(phrases)
        if expected_count == 0:
            logger.warning("Segmentation produced no phrases. Aborting pipeline.")
            raise RuntimeError("Segmentation produced no phrases.")

        # ---- PHASE 2b: N-BEST GROUNDING (language-filtered) ----
        # Only worth the second Parakeet pass if we will actually refine.
        do_refine = (
            not skip_correction
            and correction_model
            and correction_model.lower() != "none"
        )
        if do_refine:
            augment_phrases_with_nbest(
                v_deecho, phrases, base_dir, source_code=source_lang, nbest=nbest
            )
        else:
            for p in phrases:
                p["alternatives"] = [p["text"]]

        # ---- PHASE 2.5: MULTIMODAL REFINEMENT (audio + candidates) ----
        if do_refine:
            logger.info("Phase 2.5: Multimodal refinement with (%s)...", correction_model)
            refined = execute_omni_refinement(
                v_deecho, phrases, correction_model, src_name,
                base_dir=base_dir, max_options=max_options,
            )

            logger.info("--- Phase 2.5: audio+candidate correction diff ---")
            corrections_made = 0
            for i in range(expected_count):
                orig = phrases[i]["text"].strip()
                corr = refined[i].strip() if i < len(refined) else orig
                if not corr:
                    corr = orig
                phrases[i]["corrected_text"] = corr
                if orig != corr:
                    logger.info("Line %d (corrected):\n  - %s\n  + %s", i + 1, orig, corr)
                    corrections_made += 1
                # Show the per-span decomposition that actually fed this decision
                # (this is the JSON grounding the refiner saw, not raw candidates).
                grounding = build_uncertain_json(orig, phrases[i].get("alternatives", []), max_options)
                if grounding:
                    # Always tag with the line number so spans are never mis-read as
                    # belonging to a neighbouring line (esp. when a line was unchanged
                    # and therefore prints no diff header of its own).
                    tag = "" if orig != corr else " (unchanged)"
                    logger.info("    Line %d%s uncertain spans (JSON):\n%s", i + 1, tag, grounding)
            logger.info("Total audio-guided corrections: %d", corrections_made)
        else:
            logger.info("Phase 2.5 skipped: using raw ASR text directly.")
            for i in range(expected_count):
                phrases[i]["corrected_text"] = phrases[i]["text"]

        # Single source of truth for downstream phases.
        corrected_texts = [p["corrected_text"] for p in phrases]

        # ---- PHASE 3: TRANSLATION ----
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

        # ---- PHASE 4: TIMING + COMPILATION ----
        vid_w, vid_h = get_video_resolution(clean_video_path)
        for i in range(expected_count):
            corr_text = phrases[i]["corrected_text"]
            phrases[i]["text"] = corr_text
            phrases[i]["words"] = redistribute_word_timings(phrases[i]["words"], corr_text)

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
        description="Standalone Multilingual Language-Learning Subtitle Builder "
                    "(candidate-grounded refinement).",
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
        help="ISO acoustic source. Supported inputs:\n  "
             + _format_dict_for_help(INPUT_LANGUAGES),
    )
    parser.add_argument(
        "-c", "--correction-model", default="google/gemma-4-E4B-it",
        help="HuggingFace model ID for multimodal (audio) ASR refinement. "
             "Must be an omni/audio-capable model. Use 'none' to skip.",
    )
    parser.add_argument(
        "-m", "--translation-model", default="tencent/Hy-MT2-7B",
        help="HuggingFace model ID for translation. Use 'none' to skip.",
    )
    parser.add_argument(
        "--nbest", type=int, default=NBEST_DEFAULT,
        help="Number of Parakeet hypotheses per phrase used as phonetic grounding "
             "for the refiner (1 disables the second beam pass; refiner then sees "
             "audio + greedy draft only).",
    )
    parser.add_argument(
        "--skip-correction", action="store_true",
        help="Skip Phase 2.5 (multimodal refinement) AND Phase 2b (n-best) to save VRAM/time.",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Directory for the final video (default: current working directory).",
    )
    parser.add_argument(
        "--model-cache-dir", default=None,
        help="Persistent directory for downloaded separation models "
             f"(default: {DEFAULT_MODEL_CACHE_DIR}). Keep this stable to avoid "
             "re-downloading models and to preserve any locally swapped checkpoint.",
    )
    parser.add_argument(
        "--whisper-lid-model", default="small",
        help="Whisper model size used ONLY for language detection (e.g. base, small, medium).",
    )
    parser.add_argument(
        "--max-options", type=int, default=MAX_CANDIDATES_IN_PROMPT,
        help=f"Max ALTERNATIVE readings shown per uncertain span in the refiner "
             f"prompt (default {MAX_CANDIDATES_IN_PROMPT}; the draft is always shown "
             f"in addition). Separate from --nbest, which controls how many beam "
             f"hypotheses Parakeet generates. Keep this small (3-5): too many options "
             f"degrade a small model's choice.",
    )
    parser.add_argument(
        "--keep-workspace", action="store_true",
        help="Do not delete the temporary work directory on exit (useful for "
             "inspecting isolated vocals, the n-best clips, or the generated ASS).",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run_pipeline(
            args.video_path,
            args.target_lang,
            args.source_lang,
            args.correction_model,
            args.translation_model,
            args.skip_correction,
            output_dir=args.output_dir,
            model_cache_dir=args.model_cache_dir,
            whisper_lid_model=args.whisper_lid_model,
            nbest=args.nbest,
            keep_workspace=args.keep_workspace,
            max_options=args.max_options,
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
