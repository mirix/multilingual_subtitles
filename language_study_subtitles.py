"""
Multilingual Language-Learning Subtitle Builder
===============================================

End-to-end pipeline that takes a video, isolates the vocals, transcribes them
with word-level timing, optionally refines the transcript against the audio and
translates it, then burns synchronized dual subtitles (source text plus
translation) back onto the video as a read-along language-learning aid.

Pipeline stages
----------------
1.   Audio isolation (audio-separator: BS-RoFormer + Mel-RoFormer ensemble,
     followed by a de-reverb/de-echo pass) -- purely to give the ASR a clean
     vocal signal; the original audio is preserved in the output.
2.   ASR with NVIDIA Parakeet-TDT-0.6B-v3 (word-level timestamps).
2.5. Optional multimodal refinement of the transcript against the audio.
3.   Optional translation of the refined transcript.
4.   ASS subtitle generation (per-word read-along timing) and ffmpeg compilation.

This module is self-contained and driven from the command line; see ``--help``.
"""
from __future__ import annotations

import argparse
import difflib
import gc
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

MODEL_BS = "bs_roformer_vocals_resurrection_unwa.ckpt"
MODEL_MEL = "melband_roformer_big_beta5e.ckpt"
MODEL_DEREVERB = "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt"

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
    """Lazily load and cache the SaT (Segment any Text) 12-layer model."""
    global _sat_model
    if _sat_model is None:
        logger.info("Initializing SaT (Segment any Text) 12-layer model...")
        from wtpsplit import SaT
        _sat_model = SaT("sat-12l-sm")
    return _sat_model


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
# PHASE 2: ASR (PARAKEET)
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

        logger.info("Transcribing audio...")
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
# PHASE 2.5: MULTIMODAL AUDIO-TEXT REFINER
# =====================================================================
def execute_omni_refinement(audio_path: str, phrases: list, model_id: str) -> list:
    """Use an omni-modal LLM to refine ASR text against the actual audio."""
    logger.info("Loading omni refiner (%s) via Transformers...", model_id)
    from transformers import AutoModelForCausalLM, AutoProcessor

    model, processor = None, None
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=best_compute_dtype(),
            trust_remote_code=True,
        )

        y, sr = librosa.load(audio_path, sr=ASR_SR)
        duration = len(y) / sr if sr else 0.0
        corrected_texts = []

        logger.info("Refining %d phrases via omni engine...", len(phrases))
        for i, phrase in enumerate(phrases):
            start_time = max(0.0, phrase["start"] - REFINER_CONTEXT_PAD)
            end_time = min(duration, phrase["end"] + REFINER_CONTEXT_PAD)
            chunk = y[int(start_time * sr): int(end_time * sr)]

            prompt = (
                "You are an expert audio editor and linguist. Listen to the provided "
                "audio chunk and read the drafted ASR transcription below. Correct the "
                "transcription, fixing any phonetic hallucinations or grammatical "
                f"errors.\n\nDraft: {phrase['text']}\n\n"
                "Output STRICTLY the corrected text and nothing else."
            )
            conversation = [{
                "role": "user",
                "content": [{"type": "audio"}, {"type": "text", "text": prompt}],
            }]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=text_prompt, audios=[chunk], return_tensors="pt",
                sampling_rate=ASR_SR,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                    pad_token_id=getattr(processor, "tokenizer", processor).pad_token_id
                    or getattr(processor, "tokenizer", processor).eos_token_id,
                )

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            transcription = processor.decode(generated_ids, skip_special_tokens=True).strip()
            transcription = _strip_code_fences(transcription)
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
        d_stem = run_isolation_inference(
            v_stft, MODEL_DEREVERB, "dereverb", base_dir, "dry", "no dry",
            model_cache_dir=cache_dir,
        )
        os.replace(d_stem, v_deecho)

        # ---- PHASE 2: ASR ----
        words = execute_parakeet_asr(v_deecho, base_dir, source_override)
        if not words:
            logger.warning("ASR returned no words. Aborting pipeline.")
            raise RuntimeError("ASR produced no transcribable words.")

        phrases = segment_words_into_phrases(words, max_gap_seconds=MAX_GAP_SECONDS)
        expected_count = len(phrases)
        if expected_count == 0:
            logger.warning("Segmentation produced no phrases. Aborting pipeline.")
            raise RuntimeError("Segmentation produced no phrases.")

        src_name = (
            INPUT_LANGUAGES.get(source_override, source_override.upper())
            if source_override != "auto" else "the source language"
        )
        tgt_name = OUTPUT_LANGUAGES.get(target_lang, target_lang.upper())

        # ---- PHASE 2.5: MULTIMODAL REFINEMENT ----
        do_refine = (
            not skip_correction
            and correction_model
            and correction_model.lower() != "none"
        )
        if do_refine:
            logger.info("Phase 2.5: Multimodal refinement with (%s)...", correction_model)
            refined = execute_omni_refinement(v_deecho, phrases, correction_model)

            logger.info("--- Phase 2.5: audio-text correction diff ---")
            corrections_made = 0
            for i in range(expected_count):
                orig = phrases[i]["text"].strip()
                corr = refined[i].strip()
                phrases[i]["corrected_text"] = corr
                if orig != corr:
                    logger.info("Line %d:\n  - %s\n  + %s", i + 1, orig, corr)
                    corrections_made += 1
            logger.info("Total audio-guided corrections: %d", corrections_made)
        else:
            logger.info("Phase 2.5 skipped: using raw ASR text directly.")
            for i in range(expected_count):
                phrases[i]["corrected_text"] = phrases[i]["text"]

        # Single source of truth for downstream phases.
        corrected_texts = [p["corrected_text"] for p in phrases]

        # ---- PHASE 3: TRANSLATION ----
        needs_translation = bool(target_lang) and target_lang != source_override
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
        description="Standalone Multilingual Language-Learning Subtitle Builder.",
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
        help="HuggingFace model ID for multimodal ASR refinement. Use 'none' to skip.",
    )
    parser.add_argument(
        "-m", "--translation-model", default="tencent/Hy-MT2-7B",
        help="HuggingFace model ID for translation. Use 'none' to skip.",
    )
    parser.add_argument(
        "--skip-correction", action="store_true",
        help="Skip Phase 2.5 (multimodal refinement) to save VRAM.",
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
