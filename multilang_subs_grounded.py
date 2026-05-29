"""
Multilingual Language-Learning / Karaoke Subtitle Builder
=========================================================

Pipeline
--------
1.   Audio isolation (BS-RoFormer + Mel-RoFormer ensemble + de-reverb pass) to
     give the ASR a clean vocal signal. Original audio is preserved in output.
2.   ASR with NVIDIA Parakeet-TDT-0.6B-v3 (word-level timestamps, greedy pass).
2a.  Acoustic language ID via Whisper *detect_language only* (no transcription,
     so song-lyric hallucination is irrelevant). Resolves the exact source
     language, which is then named explicitly in every English prompt.
2b.  Per-phrase Parakeet beam-search n-best -> the top acoustic hypotheses for
     each phrase, used as phonetic grounding. Candidates detected to be in a
     different language than the source are dropped (beam search on a shared
     multilingual vocabulary leaks e.g. Ukrainian variants into Russian audio).
2.5. Grounded semantic correction: a text LLM (Qwen3.6 via llama.cpp) reconciles
     the n-best candidates into the most coherent reading -- linguistically
     ambitious, but leashed to what was acoustically plausible.
3.   Translation of the corrected transcript.
4.   ASS karaoke generation (per-word read-along timing) + ffmpeg compile.

Models are loaded and unloaded one at a time to fit a 24 GB GPU.
"""
from __future__ import annotations

import os
import sys
import gc
import re
import logging
import argparse
import subprocess
import difflib

import numpy as np
import librosa
import soundfile as sf
import torch
from wtpsplit import SaT
from audio_separator.separator import Separator

# =====================================================================
# LOGGING
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# =====================================================================
# CONSTANTS
# =====================================================================
MASTER_SR = 44100
ASR_SR = 16000
STFT_NFFT = 2048
STFT_HOP = 512
MAX_GAP_SECONDS = 3.5
PARAKEET_SUBSAMPLING = 8
REFINER_CONTEXT_PAD = 0.2          # seconds of pad around each phrase for n-best decode
CANDIDATE_LID_MIN_WORDS = 3        # below this, text-LID is too noisy to trust; keep candidate

CORRECTION_BATCH_SIZE = 30         # smaller than translation: n-best inflates the prompt
TRANSLATION_BATCH_SIZE = 40

OMITTED = "[OMITTED_BY_LLM]"

# Reasoning / meta markers used to scrub leaked chain-of-thought from output.
_META_MARKERS = (
    "</think>", "<think>", "check against rules", "self-correction",
    "output generation", "output matches", "all lines verified",
    "i'll output", "i will output", "let's verify", "final check", "proceeds.",
)

# Parakeet-TDT-0.6B-v3 is trained on 25 (mostly European) languages.
PARAKEET_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "bg": "Bulgarian", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "et": "Estonian", "fi": "Finnish", "el": "Greek",
    "hu": "Hungarian", "it": "Italian", "lv": "Latvian", "lt": "Lithuanian",
    "mt": "Maltese", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish", "ru": "Russian",
    "uk": "Ukrainian",
}

# Broader code->name map for translation targets / Whisper LID output naming.
LANGUAGE_NAMES = dict(PARAKEET_LANGUAGES)
LANGUAGE_NAMES.update({
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "vi": "Vietnamese", "id": "Indonesian", "tr": "Turkish",
    "th": "Thai", "he": "Hebrew", "fa": "Persian", "ms": "Malay",
    "tl": "Filipino", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "ur": "Urdu", "no": "Norwegian",
    "ca": "Catalan",
})

# =====================================================================
# GLOBALS / LAZY LOADING
# =====================================================================
_sat_model = None
_lingua_detector = None
_lingua_unavailable = False


def get_sat_model():
    global _sat_model
    if _sat_model is None:
        logger.info("Initializing SaT (Segment any Text) semantic segmentation model...")
        _sat_model = SaT("sat-3l-sm")
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
def flush_vram(stage: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if stage:
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            logger.info("VRAM [%s]: %.2f GB allocated, %.2f GB reserved", stage, a, r)


def _load_llm(model_path: str, context_size: int = 16384):
    """Load a GGUF model with llama-cpp-python (full GPU offload)."""
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("llama-cpp-python not found. Install it with CUDA support.")
        raise
    abs_path = os.path.abspath(model_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"GGUF file not found: {abs_path}")
    logger.info("Loading GGUF model: %s", abs_path)
    return Llama(model_path=abs_path, n_gpu_layers=-1, n_ctx=context_size, verbose=False)


def _escape_for_ffmpeg_subtitles(path: str) -> str:
    if os.name == "nt":
        return path.replace("\\", "/").replace(":", "\\:")
    return path.replace(":", "\\:")


def _sanitize_ass_text(text: str) -> str:
    return (text.replace("\\", "/").replace("{", "(").replace("}", ")")
                .replace("\r", " ").replace("\n", " "))


# =====================================================================
# PHASE 1: ISOLATION
# =====================================================================
def extract_audio_from_video(video_path: str, output_audio_path: str):
    logger.info("Extracting raw uncompressed audio from video source...")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
           "-ar", str(MASTER_SR), "-ac", "2", output_audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)


def run_isolation_inference(input_path: str, model_filename: str, output_key: str, output_dir: str,
                            primary_stem: str = "Vocals", secondary_stem: str = "Instrumental") -> str:
    logger.info("Running audio-separator with model: %s", model_filename)
    separator = Separator(
        log_level=logging.WARNING, model_file_dir=os.path.join(output_dir, "models_cache"),
        output_dir=output_dir, output_format="WAV", normalization_threshold=0.9, use_autocast=True,
    )
    separator.load_model(model_filename=model_filename)
    expected_primary = f"{primary_stem.lower()}_{output_key}"
    expected_secondary = f"{secondary_stem.lower()}_{output_key}"
    output_files = separator.separate(
        input_path, {primary_stem: expected_primary, secondary_stem: expected_secondary})

    target_file = None
    if output_files:
        target_file = next((f for f in output_files if expected_primary in f and expected_secondary not in f), None)
        target_file = target_file or next((f for f in output_files if expected_secondary not in f), None)
        target_file = target_file or output_files[0]
    del separator
    flush_vram()
    return os.path.join(output_dir, target_file)


def ensemble_vocals_stft(file1: str, file2: str, output_path: str):
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

    ensembled = []
    for ch in range(channels):
        s1 = librosa.stft(y1[ch], n_fft=STFT_NFFT, hop_length=STFT_HOP)
        s2 = librosa.stft(y2[ch], n_fft=STFT_NFFT, hop_length=STFT_HOP)
        ensembled.append(librosa.istft(0.5 * s1 + 0.5 * s2, hop_length=STFT_HOP))

    final_audio = np.stack(ensembled, axis=0)
    peak = float(np.max(np.abs(final_audio))) if final_audio.size else 0.0
    if peak > 1.0:
        final_audio = final_audio / peak
    out = final_audio[0] if channels == 1 else final_audio.T
    sf.write(output_path, out, sr1)


# =====================================================================
# PHASE 2: ASR (PARAKEET, GREEDY -> WORD TIMINGS)
# =====================================================================
def execute_parakeet_asr(audio_path: str, lang_override: str = "auto") -> list:
    logger.info("Initializing NVIDIA Parakeet-TDT-0.6B-v3 (multilingual)...")
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    lang_override_norm = (lang_override or "auto").lower().strip()
    asr_model = None
    temp_mono_path = os.path.join("workspace", "temp_asr_mono.wav")
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        asr_model.eval()

        y, sr = librosa.load(audio_path, sr=ASR_SR, mono=True)
        sf.write(temp_mono_path, y, sr)

        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.preserve_alignments = True
            decoding_cfg.compute_timestamps = True
            decoding_cfg.word_separator = " "
            # NOTE: Parakeet-TDT-v3 does not truly honor a forced language; this
            # is harmless if ignored. Real language control happens downstream.
            if lang_override_norm != "auto" and lang_override_norm in PARAKEET_LANGUAGES:
                decoding_cfg.language = lang_override_norm
        asr_model.change_decoding_strategy(decoding_cfg)

        logger.info("Transcribing audio (greedy, with timestamps)...")
        hypothesis = asr_model.transcribe([temp_mono_path], return_hypotheses=True, timestamps=True)[0]

        raw_words = []
        timestamp_dict = getattr(hypothesis, "timestamp", {})
        if isinstance(timestamp_dict, dict) and "word" in timestamp_dict:
            time_stride = getattr(asr_model.cfg.preprocessor, "window_stride", 0.01) * PARAKEET_SUBSAMPLING
            for w in timestamp_dict["word"]:
                text = (w.get("word") or w.get("char") or "").strip()
                if not text:
                    continue
                if "start" in w and "end" in w:
                    start, end = float(w["start"]), float(w["end"])
                elif "start_offset" in w and "end_offset" in w:
                    start, end = w["start_offset"] * time_stride, w["end_offset"] * time_stride
                else:
                    continue
                if end > start:
                    raw_words.append({"text": text, "start": start, "end": end})
        return raw_words
    finally:
        if os.path.exists(temp_mono_path):
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


# =====================================================================
# SEGMENTATION
# =====================================================================
def _make_phrase(phrase_words: list) -> dict:
    return {
        "text": " ".join(w["text"] for w in phrase_words).strip(),
        "start": phrase_words[0]["start"],
        "end": phrase_words[-1]["end"],
        "words": phrase_words,
    }


def segment_words_into_phrases(words: list, max_gap_seconds: float = MAX_GAP_SECONDS) -> list:
    logger.info("Applying acoustic + semantic (SaT) segmentation...")
    acoustic_chunks, current = [], []
    for i, word in enumerate(words):
        current.append(word)
        gap = (words[i + 1]["start"] - word["end"]) if i < len(words) - 1 else 0.0
        if gap >= max_gap_seconds:
            acoustic_chunks.append(current)
            current = []
    if current:
        acoustic_chunks.append(current)

    final_phrases = []
    for chunk in acoustic_chunks:
        spans, pos = [], 0
        for w in chunk:
            spans.append({"word": w, "start_pos": pos, "end_pos": pos + len(w["text"])})
            pos += len(w["text"]) + 1
        raw_text = " ".join(w["text"] for w in chunk)
        if not raw_text.strip():
            continue

        sentences = get_sat_model().split(raw_text)
        word_idx, search_idx = 0, 0
        for sentence in sentences:
            s_start = raw_text.find(sentence, search_idx)
            if s_start == -1:
                s_start = search_idx
            s_end = s_start + len(sentence)
            search_idx = s_end
            phrase_words = []
            while word_idx < len(spans):
                info = spans[word_idx]
                midpoint = (info["start_pos"] + info["end_pos"]) / 2.0
                if midpoint <= s_end:
                    phrase_words.append(info["word"])
                    word_idx += 1
                else:
                    break
            if phrase_words:
                final_phrases.append(_make_phrase(phrase_words))
        remaining = [info["word"] for info in spans[word_idx:]]
        if remaining:
            final_phrases.append(_make_phrase(remaining))
    return final_phrases


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

    `candidates` should EXCLUDE the primary; the primary is never filtered.
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


def augment_phrases_with_nbest(audio_path: str, phrases: list, source_code: str | None = None,
                               nbest: int = 4, pad: float = REFINER_CONTEXT_PAD):
    """Decode each phrase span with Parakeet beam search and store the top-N
    acoustic hypotheses on phrase['alternatives'] (primary first, deduped,
    cross-language alternatives removed).

    NOTE: TDT/RNN-T beam n-best support varies across NeMo versions. This is
    wrapped so that any failure degrades gracefully to single-hypothesis mode.
    """
    if not nbest or nbest <= 1:
        for p in phrases:
            p["alternatives"] = [p["text"]]
        return

    logger.info("Generating top-%d Parakeet hypotheses per phrase (beam search)...", nbest)
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    asr_model = None
    tmp_dir = os.path.join("workspace", "_nbest_clips")
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
            if source_code and source_code in PARAKEET_LANGUAGES:
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
            ordered, seen = [], set()
            for a in [primary] + alts:
                if a and a not in seen:
                    seen.add(a)
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
        if asr_model is not None:
            del asr_model
        flush_vram("after Parakeet n-best")


# =====================================================================
# PHASE 2.5 & 3: LLM EXECUTION (llama.cpp)
# =====================================================================
def _clean_llm_output(text: str) -> str:
    """Strip reasoning robustly, including Qwen3.6's case where only a closing
    </think> appears (the opening tag is seeded into the prompt by the template)."""
    text = re.sub(r"<think\b.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)  # full span
    has_num = lambda s: re.search(r"(?m)^\s*\d+\s*\|", s) is not None
    if "</think>" in text:                       # lone closing tag: keep the side with the answer
        head, _, tail = text.rpartition("</think>")
        text = tail if has_num(tail) else (head if has_num(head) else tail)
    if "<think>" in text:                        # lone opening tag (token-limit cutoff)
        head, _, tail = text.partition("<think>")
        text = head if has_num(head) else tail
    return text.strip()


def _parse_numbered_lines(completion: str, expected_count: int) -> list:
    """Parse 'N| text' output, immune to leaked reasoning / meta commentary."""
    completion = _clean_llm_output(completion)
    strict = re.compile(r"^(\d+)\s*\|\s*(.*)")
    result = [""] * expected_count
    current = -1
    for line in completion.split("\n"):
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if any(mk in low for mk in _META_MARKERS):   # reasoning leaked -> stop the block
            current = -1
            continue
        m = strict.match(line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < expected_count:
                current = idx
                captured = m.group(2).strip()
                for mk in _META_MARKERS:              # also truncate inline pollution
                    pos = captured.lower().find(mk)
                    if pos != -1:
                        captured = captured[:pos].strip()
                result[current] = captured
            else:
                current = -1
        elif current != -1:
            result[current] = (result[current] + " " + line).strip()
    return [r if r else OMITTED for r in result]


def _chatml(system: str, user: str, no_think: bool = False) -> str:
    """Build a raw ChatML prompt. With no_think, pre-seed a closed empty think
    block so the model skips reasoning entirely (faster, deterministic)."""
    seed = "<think>\n\n</think>\n\n" if no_think else ""
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{seed}")


def execute_llm_task(model, prompt: str, expected_count: int, temperature: float = 0.3,
                     max_tokens: int = 8192, no_think: bool = False) -> list:
    system_text = ("You are a precise data-formatting AI. Output strictly the requested "
                   "numbered lines and nothing else.")
    if no_think:
        full = _chatml(system_text, prompt, no_think=True)
        out = model.create_completion(full, max_tokens=max_tokens, temperature=temperature,
                                      top_p=0.85, min_p=0.05, stop=["<|im_end|>"])
        completion = out["choices"][0]["text"]
    else:
        resp = model.create_chat_completion(
            messages=[{"role": "system", "content": system_text},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, top_p=0.85, min_p=0.05)
        completion = resp["choices"][0]["message"]["content"]
    return _parse_numbered_lines(completion, expected_count)


def execute_task_with_recovery(model, prompt_builder, expected_count: int,
                               recovery_label: str = "task", temperature: float = 0.3,
                               no_think: bool = False) -> list:
    """Run one prompt over `expected_count` lines; retry lines that came back OMITTED.

    `prompt_builder` takes a list of LOCAL indices (0..expected_count-1).
    """
    if expected_count == 0:
        return []
    result = execute_llm_task(model, prompt_builder(list(range(expected_count))),
                              expected_count, temperature=temperature, no_think=no_think)
    omitted = [i for i, r in enumerate(result) if r == OMITTED]
    if not omitted:
        return result
    if len(omitted) > expected_count // 2:
        logger.warning("[%s] Skipping recovery (systemic failure).", recovery_label)
        return result
    logger.info("[%s] Recovering %d omitted line(s)...", recovery_label, len(omitted))
    recovered = execute_llm_task(model, prompt_builder(omitted), len(omitted),
                                 temperature=temperature, no_think=no_think)
    for slot, text in zip(omitted, recovered):
        if text != OMITTED:
            result[slot] = text
    return result


def run_batched_task(model, global_prompt_builder, expected_count: int, batch_size: int,
                     recovery_label: str, temperature: float = 0.3, no_think: bool = False) -> list:
    """Chunk the work into batches and stitch results back. `global_prompt_builder`
    takes a list of GLOBAL indices."""
    results = [""] * expected_count
    for start in range(0, expected_count, batch_size):
        batch = list(range(start, min(start + batch_size, expected_count)))
        logger.info("[%s] Processing lines %d-%d...", recovery_label, batch[0] + 1, batch[-1] + 1)
        out = execute_task_with_recovery(
            model,
            prompt_builder=lambda local, _b=batch: global_prompt_builder([_b[i] for i in local]),
            expected_count=len(batch),
            recovery_label=recovery_label,
            temperature=temperature,
            no_think=no_think,
        )
        for slot, gi in enumerate(batch):
            results[gi] = out[slot]
    return results


# =====================================================================
# PROMPT BUILDERS
# =====================================================================
def _build_correction_prompt(phrases: list, global_indices: list, src_name: str) -> str:
    blocks = []
    for slot, oi in enumerate(global_indices):
        p = phrases[oi]
        primary = p["text"]
        line = f"{slot + 1}| {primary}"
        alts = [a for a in p.get("alternatives", []) if a and a != primary]
        if alts:
            line += "\n   candidates: " + " / ".join(alts)
        blocks.append(line)
    source_block = "\n".join(blocks)
    return (
        f"You are an expert audio transcription editor and native speaker of {src_name}.\n"
        f"Each numbered line below is the primary output of an acoustic speech-recognition "
        f"(ASR) model, optionally followed by alternative hypotheses the same model also "
        f"considered ('candidates'). These candidates are evidence of what was acoustically "
        f"plausible at that moment.\n\n"
        f"YOUR TASK: produce the single most linguistically correct and coherent reading of "
        f"each line in {src_name}.\n\n"
        f"RULES:\n"
        f"- Be decisive. The primary output is frequently wrong; actively correct it.\n"
        f"- Treat the candidates as the menu of what was likely sung/said. Prefer a reading "
        f"supported by the primary or one of the candidates.\n"
        f"- You MAY split glued words, merge over-split words, or substitute near-homophones "
        f"when it yields a more coherent sentence that stays phonetically consistent with the "
        f"evidence.\n"
        f"- The song is in {src_name}. DISCARD any candidate written in a different language "
        f"or dialect (wrong-language homophones or foreign spellings); they are ASR artifacts, "
        f"not evidence. Always render the line in {src_name}.\n"
        f"- Restore correct diacritics, accents, capitalization and punctuation that the ASR "
        f"dropped, using proper {src_name} orthography.\n"
        f"- Do NOT invent content unsupported by any hypothesis, and do NOT add commentary.\n"
        f"- Do NOT make purely stylistic rewrites, and do NOT translate.\n"
        f"- LINE INTEGRITY: there are exactly {len(global_indices)} lines. Do not merge, split, "
        f"or reorder them. Output line N must correspond to input line N.\n"
        f"- Start your final response directly with '1| '. No conversational text.\n\n"
        f"--- ASR HYPOTHESES ({src_name}) ---\n{source_block}\n--- CORRECTED ({src_name}) ---\n"
    )


def _build_translation_prompt(corrected_texts: list, global_indices: list,
                              src_name: str, tgt_name: str) -> str:
    source_block = "\n".join(
        f"{slot + 1}| {corrected_texts[oi]}" for slot, oi in enumerate(global_indices))
    return (
        f"Translate the following lines from {src_name} into {tgt_name}.\n\n"
        f"RULES:\n"
        f"- There are exactly {len(global_indices)} numbered lines. Do not shift content "
        f"across line boundaries.\n"
        f"- Output the translation only.\n"
        f"- Maintain the exact format: '<number>| <translated text>'.\n"
        f"- Start your response directly with '1| '.\n\n"
        f"--- SOURCE ({src_name}) ---\n{source_block}\n--- TARGET ({tgt_name}) ---\n"
    )


# =====================================================================
# PHASE 4: TIMING + ASS
# =====================================================================
def redistribute_word_timings(orig_words: list, corrected_text: str) -> list:
    """Map corrected words onto original word timings via a diff alignment."""
    corr_words = corrected_text.split()
    if not corr_words or not orig_words:
        return []
    orig_str = [w["text"] for w in orig_words]
    matcher = difflib.SequenceMatcher(None, orig_str, corr_words)
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
                t = start_time
                for w in block:
                    new_words.append({"text": w, "start": t, "end": t + step})
                    t += step
    return new_words


def gen_ass_time(seconds: float) -> str:
    total_cs = int(round(seconds * 100))
    hours, rem = divmod(total_cs, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def get_video_resolution(path: str) -> tuple[int, int]:
    try:
        res = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=s=x:p=0", path], text=True).strip()
        w, h = res.split("x")
        return int(w), int(h)
    except Exception:  # noqa: BLE001
        logger.warning("Could not probe video resolution; defaulting to 1920x1080.")
        return 1920, 1080


def build_ass_document(phrases: list, translations: list, vid_w: int, vid_h: int) -> str:
    main_fs, trans_fs = int(vid_h * 0.06), int(vid_h * 0.04)
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "WrapStyle: 0",
        f"PlayResX: {vid_w}", f"PlayResY: {vid_h}", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: KaraokeMain,Arial,{main_fs},&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1", "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for idx, phrase in enumerate(phrases):
        words = phrase["words"]
        if not words:
            continue
        line_start, line_end = words[0]["start"], words[-1]["end"]
        k_parts, curr_t = [], line_start
        for w in words:
            gap = int(round((w["start"] - curr_t) * 100))
            if gap > 0:
                k_parts.append(f"{{\\k{gap}}}")
            dur = max(1, int(round((w["end"] - w["start"]) * 100)))
            k_parts.append(f"{{\\k{dur}}}{_sanitize_ass_text(w['text'])} ")
            curr_t = w["end"]
        read_along = "".join(k_parts).strip()

        translation = translations[idx] if idx < len(translations) else ""
        if translation:
            combined = (f"{read_along}\\N{{\\r\\c&H00A0A0A0&\\fs{trans_fs}}}"
                        f"{_sanitize_ass_text(translation)}")
        else:
            combined = read_along
        lines.append(
            f"Dialogue: 0,{gen_ass_time(line_start)},{gen_ass_time(line_end)},"
            f"KaraokeMain,,0,0,0,,{combined}")
    return "\n".join(lines)


def compile_video(clean_video_path: str, ass_file: str, final_vid: str):
    cmd = ["ffmpeg", "-y", "-i", clean_video_path,
           "-vf", f"subtitles='{_escape_for_ffmpeg_subtitles(ass_file)}'",
           "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-c:a", "copy", final_vid]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)


# =====================================================================
# ORCHESTRATION
# =====================================================================
def run_pipeline(video_path: str, target_lang: str, source_override: str = "auto",
                 correction_model: str | None = None, translation_model: str | None = None,
                 skip_correction: bool = False, nbest: int = 4, whisper_lid_model: str = "small",
                 no_think: bool = False):
    base_dir = "workspace"
    os.makedirs(base_dir, exist_ok=True)
    clean_video_path = video_path.strip("'\" ")
    if not os.path.isfile(clean_video_path):
        raise FileNotFoundError(f"Input video not found: {clean_video_path}")

    target_lang = (target_lang or "en").lower().strip()
    source_override = (source_override or "auto").lower().strip()

    extracted = os.path.join(base_dir, "extracted_raw.wav")
    v_stft = os.path.join(base_dir, "isolated_vocals_master.wav")
    v_deecho = os.path.join(base_dir, "isolated_vocals_deecho.wav")
    ass_file = os.path.join(base_dir, "compiled_subtitles.ass")
    final_vid = os.path.splitext(clean_video_path)[0] + f"_Karaoke_{target_lang.upper()}.mp4"

    try:
        # ---- PHASE 1: ISOLATION ----
        extract_audio_from_video(clean_video_path, extracted)
        v_bs = run_isolation_inference(extracted, "bs_roformer_vocals_resurrection_unwa.ckpt", "bs", base_dir)
        v_mel = run_isolation_inference(extracted, "melband_roformer_big_beta5e.ckpt", "mel", base_dir)
        ensemble_vocals_stft(v_bs, v_mel, v_stft)
        d_stem = run_isolation_inference(
            v_stft, "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt", "dereverb", base_dir, "dry", "no dry")
        os.replace(d_stem, v_deecho)

        # ---- PHASE 2: ASR (greedy + timings) ----
        words = execute_parakeet_asr(v_deecho, source_override)
        if not words:
            raise RuntimeError("ASR produced no transcribable words.")

        # ---- PHASE 2a: LANGUAGE ID ----
        if source_override == "auto":
            source_lang = detect_language_acoustic(v_deecho, words=words, model_size=whisper_lid_model)
        else:
            source_lang = source_override
        src_name = lang_name(source_lang)
        tgt_name = lang_name(target_lang)
        logger.info("Source language: %s (%s) | Target: %s (%s)",
                    source_lang or "unknown", src_name, target_lang, tgt_name)

        # ---- SEGMENTATION ----
        phrases = segment_words_into_phrases(words, max_gap_seconds=MAX_GAP_SECONDS)
        expected_count = len(phrases)
        if expected_count == 0:
            raise RuntimeError("Segmentation produced no phrases.")

        # ---- PHASE 2b: N-BEST GROUNDING (language-filtered) ----
        augment_phrases_with_nbest(v_deecho, phrases, source_code=source_lang, nbest=nbest)

        corrected_texts = [p["text"] for p in phrases]

        # ---- PHASE 2.5: GROUNDED CORRECTION ----
        if not skip_correction and correction_model and correction_model.lower() != "none":
            logger.info("Phase 2.5: Loading corrector LLM (%s)...", correction_model)
            model_corr = _load_llm(correction_model)
            corrected_texts = run_batched_task(
                model_corr,
                global_prompt_builder=lambda gidx: _build_correction_prompt(phrases, gidx, src_name),
                expected_count=expected_count, batch_size=CORRECTION_BATCH_SIZE,
                recovery_label="correction", temperature=0.2, no_think=no_think,
            )
            if hasattr(model_corr, "close"):
                model_corr.close()
            del model_corr
            flush_vram("after corrector unload")

            logger.info("--- Phase 2.5: semantic correction diff ---")
            made = 0
            for i in range(expected_count):
                orig = phrases[i]["text"].strip()
                corr = corrected_texts[i].strip()
                if corr == OMITTED or not corr:
                    corr = orig
                corrected_texts[i] = corr
                phrases[i]["corrected_text"] = corr
                if orig != corr:
                    logger.info("Line %d:\n  - %s\n  + %s", i + 1, orig, corr)
                    made += 1
                # Show the grounding candidates that fed this decision.
                cands = [c for c in phrases[i].get("alternatives", []) if c.strip() and c.strip() != orig]
                if cands:
                    logger.info("    candidates: %s", "  |  ".join(cands))
            logger.info("Total semantic corrections: %d", made)
        else:
            logger.info("Phase 2.5 skipped: using raw ASR text directly.")
            for i in range(expected_count):
                phrases[i]["corrected_text"] = phrases[i]["text"]
            corrected_texts = [p["corrected_text"] for p in phrases]

        # ---- PHASE 3: TRANSLATION ----
        needs_translation = bool(target_lang) and target_lang != source_lang
        if needs_translation and translation_model and translation_model.lower() != "none":
            logger.info("Phase 3: Loading translator LLM (%s)...", translation_model)
            model_trans = _load_llm(translation_model)
            translations = run_batched_task(
                model_trans,
                global_prompt_builder=lambda gidx: _build_translation_prompt(
                    corrected_texts, gidx, src_name, tgt_name),
                expected_count=expected_count, batch_size=TRANSLATION_BATCH_SIZE,
                recovery_label="translation", temperature=0.3, no_think=no_think,
            )
            if hasattr(model_trans, "close"):
                model_trans.close()
            del model_trans
            flush_vram("after translator unload")
            translations = ["" if t == OMITTED else t for t in translations]
        else:
            logger.info("Target matches source or no translator provided; skipping translation.")
            translations = list(corrected_texts)

        # ---- PHASE 4: TIMING + COMPILE ----
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


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Multilingual Karaoke / Study Subtitle Builder.")
    parser.add_argument("video_path", help="Source media file.")
    parser.add_argument("-t", "--target-lang", default="en", help="ISO translation target.")
    parser.add_argument("-s", "--source-lang", default="auto",
                        help="ISO acoustic source, or 'auto' to detect with Whisper.")
    parser.add_argument("-c", "--correction-model",
                        default="workspace/models_cache/Qwen3.6-27B-UD-Q4_K_XL.gguf",
                        help="Local .gguf path for grounded semantic correction. Use 'none' to skip.")
    parser.add_argument("-m", "--translation-model",
                        default="workspace/models_cache/Qwen3.6-27B-UD-Q4_K_XL.gguf",
                        help="Local .gguf path for translation. Use 'none' to skip.")
    parser.add_argument("--nbest", type=int, default=4,
                        help="Number of Parakeet hypotheses per phrase for grounding (1 disables).")
    parser.add_argument("--whisper-lid-model", default="small",
                        help="Whisper model size used ONLY for language detection (e.g. base, small).")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable LLM thinking (faster, more deterministic; some quality tradeoff).")
    parser.add_argument("--skip-correction", action="store_true",
                        help="Skip Phase 2.5 (grounded correction).")
    args = parser.parse_args()

    try:
        run_pipeline(args.video_path, args.target_lang, args.source_lang,
                     args.correction_model, args.translation_model,
                     args.skip_correction, nbest=args.nbest,
                     whisper_lid_model=args.whisper_lid_model, no_think=args.no_think)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception:
        sys.exit(1)
    sys.exit(0)
