"""
Multilingual Karaoke Builder — Refactored, Sanitized & Polished
"""
import os
import sys
import gc
import logging
import subprocess
import argparse
import shutil
import tempfile
from typing import Optional
import numpy as np
import librosa
import soundfile as sf
import torch
import re
import difflib
import uuid

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
# GLOBALS, LAZY LOADING & DICTIONARIES
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

OMITTED = "[OMITTED_BY_LLM]"
_sat_model = None

def get_sat_model():
    """Lazily load and cache the SaT (Segment any Text) 12-Layer model."""
    global _sat_model
    if _sat_model is None:
        logger.info("Initializing SaT (Segment any Text) 12-Layer Model...")
        from wtpsplit import SaT
        _sat_model = SaT("sat-12l-sm")
    return _sat_model

# =====================================================================
# UTILITIES
# =====================================================================
def flush_vram(stage: str = ""):
    """Force garbage collection and release CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if stage and torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        logger.info(f"VRAM [{stage}]: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")

def escape_for_ffmpeg_subtitles(path: str) -> str:
    """Escape a file path for use inside an ffmpeg subtitles filter."""
    path = path.replace("\\", "/")
    for ch in (":", "[", "]", "'"):
        path = path.replace(ch, "\\" + ch)
    return path

# =====================================================================
# PHASE 1 & 2: ISOLATION AND ASR
# =====================================================================
def extract_audio_from_video(video_path: str, output_audio_path: str):
    """Extract raw uncompressed PCM audio from a video source."""
    logger.info("Extracting raw uncompressed audio from video source...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        output_audio_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
    except KeyboardInterrupt:
        logger.warning("User interrupted! Halting ffmpeg audio extraction...")
        raise

def run_isolation_inference(
    input_path: str, model_filename: str, output_key: str, output_dir: str,
    primary_stem: str = "Vocals", secondary_stem: str = "Instrumental",
) -> str:
    """Run audio-separator with a given model and return the primary-stem path."""
    logger.info(f"Running audio-separator with model: {model_filename}")
    from audio_separator.separator import Separator
    
    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=os.path.join(output_dir, "models_cache"),
        output_dir=output_dir, output_format="WAV",
        normalization_threshold=0.9, use_autocast=True,
    )
    separator.load_model(model_filename=model_filename)

    expected_primary = f"{primary_stem.lower()}_{output_key}"
    expected_secondary = f"{secondary_stem.lower()}_{output_key}"
    output_files = separator.separate(
        input_path, {primary_stem: expected_primary, secondary_stem: expected_secondary}
    )

    if not output_files:
        del separator
        flush_vram()
        raise RuntimeError(f"Audio separator returned no output files for model '{model_filename}'")

    target_file = next((f for f in output_files if expected_primary in f and expected_secondary not in f), None)
    target_file = target_file or next((f for f in output_files if expected_secondary not in f), None)
    target_file = target_file or output_files[0]

    del separator
    flush_vram()
    return os.path.join(output_dir, target_file)

def ensemble_vocals_stft(file1: str, file2: str, output_path: str):
    """Blend two isolated vocal tracks via 50/50 STFT averaging."""
    logger.info("Blending isolated vocal components using STFT...")
    y1, sr1 = librosa.load(file1, sr=44100, mono=False)
    y2, sr2 = librosa.load(file2, sr=44100, mono=False)
    min_len = min(y1.shape[-1], y2.shape[-1])
    y1, y2 = y1[..., :min_len], y2[..., :min_len]
    channels = 1 if y1.ndim == 1 else y1.shape[0]
    ensembled_channels = []

    for ch in range(channels):
        arr1 = y1 if channels == 1 else y1[ch]
        arr2 = y2 if channels == 1 else y2[ch]
        stft1 = librosa.stft(arr1, n_fft=2048, hop_length=512)
        stft2 = librosa.stft(arr2, n_fft=2048, hop_length=512)
        ensembled_channels.append(librosa.istft(0.5 * stft1 + 0.5 * stft2, hop_length=512))

    final_audio = np.stack(ensembled_channels, axis=0) if channels > 1 else ensembled_channels[0]
    if (max_val := np.max(np.abs(final_audio))) > 1.0:
        final_audio /= max_val
    sf.write(output_path, final_audio.T, sr1)

def execute_parakeet_asr(audio_path: str, base_dir: str, lang_override: str = "auto") -> list:
    """Run NVIDIA Parakeet-TDT-0.6B-v3 ASR and return word-level results."""
    logger.info("Initializing NVIDIA Parakeet-TDT-0.6B-v3 (Multilingual)...")
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    
    lang_override_norm = (lang_override or "auto").lower().strip()
    temp_mono_path = None

    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        asr_model.eval()

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
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
        hypothesis = asr_model.transcribe([temp_mono_path], return_hypotheses=True, timestamps=True)[0]

        raw_words = []
        timestamp_dict = getattr(hypothesis, "timestamp", {})
        if isinstance(timestamp_dict, dict) and "word" in timestamp_dict:
            time_stride = getattr(asr_model.cfg.preprocessor, "window_stride", 0.01) * 8
            for w in timestamp_dict["word"]:
                text = w.get("word", w.get("char", "")).strip()
                if not text: continue
                if "start" in w and "end" in w:
                    start, end = float(w["start"]), float(w["end"])
                elif "start_offset" in w and "end_offset" in w:
                    start = w["start_offset"] * time_stride
                    end = w["end_offset"] * time_stride
                else: continue
                if end > start:
                    raw_words.append({"text": text, "start": start, "end": end})
        return raw_words

    finally:
        if temp_mono_path and os.path.exists(temp_mono_path):
            os.remove(temp_mono_path)
        if "asr_model" in locals():
            del asr_model
        flush_vram("after Parakeet")

def segment_words_into_phrases(words: list, max_gap_seconds: float = 3.5) -> list:
    """Combine acoustic silence detection with SaT-12L semantic segmentation."""
    logger.info("Applying Acoustic + Semantic (SaT-12L) Segmentation...")
    acoustic_chunks, current_chunk = [], []
    for i, word in enumerate(words):
        current_chunk.append(word)
        gap = (words[i + 1]["start"] - word["end"]) if i < len(words) - 1 else 0
        if gap >= max_gap_seconds:
            acoustic_chunks.append(current_chunk)
            current_chunk = []
    if current_chunk:
        acoustic_chunks.append(current_chunk)

    final_phrases = []
    for chunk in acoustic_chunks:
        words_with_spans, curr_pos, raw_text_parts = [], 0, []
        for w in chunk:
            words_with_spans.append({"word": w, "start_pos": curr_pos, "end_pos": curr_pos + len(w["text"])})
            raw_text_parts.append(w["text"])
            curr_pos += len(w["text"]) + 1

        raw_text = " ".join(raw_text_parts)
        if not raw_text.strip(): continue

        sentences = get_sat_model().split(raw_text)
        word_idx = 0
        curr_search_idx = 0
        for sentence in sentences:
            sentence_start_pos = raw_text.find(sentence, curr_search_idx)
            if sentence_start_pos == -1: sentence_start_pos = curr_search_idx
            curr_sentence_end = sentence_start_pos + len(sentence)
            curr_search_idx = curr_sentence_end

            phrase_words = []
            while word_idx < len(words_with_spans):
                w_info = words_with_spans[word_idx]
                midpoint = (w_info["start_pos"] + w_info["end_pos"]) / 2.0
                if midpoint <= curr_sentence_end:
                    phrase_words.append(w_info["word"])
                    word_idx += 1
                else: break

            if phrase_words:
                final_phrases.append({
                    "text": " ".join(w["text"] for w in phrase_words).strip(),
                    "start": phrase_words[0]["start"], "end": phrase_words[-1]["end"],
                    "words": phrase_words,
                })

        remaining = [w_info["word"] for w_info in words_with_spans[word_idx:]]
        if remaining:
            final_phrases.append({
                "text": " ".join(w["text"] for w in remaining).strip(),
                "start": remaining[0]["start"], "end": remaining[-1]["end"], "words": remaining,
            })
    return final_phrases

# =====================================================================
# PHASE 2.5: MULTIMODAL AUDIO-TEXT REFINER (GEMMA 4 OMNI)
# =====================================================================
def execute_omni_refinement(audio_path: str, base_dir: str, phrases: list, model_id: str) -> list:
    """Use an omni-modal LLM to refine ASR text against actual audio."""
    logger.info(f"Loading Omni Refiner ({model_id}) via Transformers...")
    from transformers import AutoProcessor, AutoModelForCausalLM
    model, processor = None, None

    try:
        # Prevent hardware-specific crashes if running on older GPUs
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=compute_dtype, trust_remote_code=True,
        )

        y, sr = librosa.load(audio_path, sr=16000)
        corrected_texts = []

        logger.info(f"Refining {len(phrases)} phrases via Omni Engine...")
        for i, phrase in enumerate(phrases):
            start_time = max(0.0, phrase["start"] - 0.2)
            end_time = min(len(y) / sr, phrase["end"] + 0.2)
            chunk = y[int(start_time * sr):int(end_time * sr)]

            prompt_instructions = (
                "You are an expert audio editor and linguist. Listen to the provided audio chunk "
                "and read the drafted ASR transcription below. Correct the transcription, fixing any "
                "phonetic hallucinations or grammatical errors.\n\n"
                f"Draft: {phrase['text']}\n\nOutput STRICTLY the corrected text and nothing else."
            )

            conversation = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": prompt_instructions}]}]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            inputs = processor(text=text_prompt, audios=[chunk], return_tensors="pt", sampling_rate=16000).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            transcription = processor.decode(generated_ids, skip_special_tokens=True).strip()
            transcription = re.sub(r"^```.*\n|```$", "", transcription, flags=re.MULTILINE).strip()
            corrected_texts.append(transcription or phrase["text"])
            
            # UX: Keep terminal alive and show progress
            if (i + 1) % 10 == 0 or i == len(phrases) - 1:
                logger.info(f"  ... Refined {i + 1}/{len(phrases)} phrases")

    except OSError as e:
        logger.error(f"Failed to load model {model_id}. Verify HF token or model ID exists. Details: {e}")
        raise
    finally:
        if model is not None: del model
        if processor is not None: del processor
        flush_vram("after Omni Refiner")
    return corrected_texts

# =====================================================================
# PHASE 3: TRANSLATION (TRANSLATEGEMMA VIA TRANSFORMERS)
# =====================================================================
def _parse_numbered_lines(completion: str, expected_count: int) -> list:
    """Parse `N| text` numbered lines from LLM completion."""
    # LLM might occasionally omit the pipe character. Use optional matching for safety.
    strict_pipe = re.compile(r"^(\d+)\s*\|?\s*(.*)")
    result_lines = [""] * expected_count
    current_idx = -1
    for line in completion.strip().split("\n"):
        line = line.strip()
        if not line: continue
        match = strict_pipe.match(line)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < expected_count:
                current_idx = idx
                result_lines[current_idx] = match.group(2).strip()
        else:
            if current_idx != -1:
                result_lines[current_idx] += " " + line
    return [res if res else OMITTED for res in result_lines]

def execute_hf_task(model, tokenizer, prompt: str, expected_count: int) -> list:
    """Send a single prompt to a HF causal LM and parse numbered-line output."""
    messages = [{"role": "user", "content": f"You are a precise data formatting AI. Output strictly the requested numbered lines. Output NOTHING else.\n\n{prompt}"}]
    text_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=8192, temperature=0.3, top_p=0.85)

    completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    completion = re.sub(r"^```.*\n|```$", "", completion.strip(), flags=re.MULTILINE)

    if "<think>" in completion:
        if "</think>" in completion:
            completion = completion.split("</think>")[-1].strip()
        else:
            logger.warning("Thinking block not closed. Output truncated.")
            completion = ""
    return _parse_numbered_lines(completion, expected_count)

def execute_task_with_recovery(model, tokenizer, prompt_builder, expected_count: int, recovery_label: str = "task") -> list:
    """Execute an LLM task and retry any lines that came back as OMITTED."""
    if expected_count == 0: return []
    primary_prompt = prompt_builder(list(range(expected_count)))
    result = execute_hf_task(model, tokenizer, primary_prompt, expected_count)

    omitted = [i for i, r in enumerate(result) if r == OMITTED]
    if not omitted: return result

    if len(omitted) > expected_count // 2:
        logger.warning(f"[{recovery_label}] Skipping recovery (systemic failure).")
        return result

    logger.info(f"[{recovery_label}] Recovering {len(omitted)} omitted lines...")
    recovery_prompt = prompt_builder(omitted)
    recovered = execute_hf_task(model, tokenizer, recovery_prompt, len(omitted))

    for slot, rec_text in zip(omitted, recovered):
        if rec_text != OMITTED: result[slot] = rec_text
    return result

def _build_translation_prompt(phrases: list, corrected_texts: list, global_indices: list, src_name: str, tgt_name: str) -> str:
    """Build the numbered-line translation prompt."""
    corrected_source = "\n".join(f"{slot + 1}| {corrected_texts[orig_idx]}" for slot, orig_idx in enumerate(global_indices))
    return (
        f"Translate the following lines from {src_name} into {tgt_name}.\n\n"
        f"RULES:\n"
        f"- There are exactly {len(global_indices)} numbered lines. Do not shift content across line boundaries.\n"
        f"- Output the translation only.\n"
        f"- Maintain the exact format: '<number>| <translated text>'.\n"
        f"- Start your response directly with '1| '.\n\n"
        f"--- SOURCE ({src_name}) ---\n{corrected_source}\n--- TARGET ({tgt_name}) ---\n"
    )

# =====================================================================
# GLOBAL ORCHESTRATION LOOP
# =====================================================================
def run_pipeline(
    video_path: str, target_lang: str, source_override: str = "auto",
    correction_model: Optional[str] = None, translation_model: Optional[str] = None,
    skip_correction: bool = False,
):
    """End-to-end karaoke subtitle pipeline."""
    base_dir = os.path.abspath(os.path.join(tempfile.gettempdir(), f"karaoke_{uuid.uuid4().hex[:8]}"))
    os.makedirs(base_dir, exist_ok=True)
    clean_video_path = video_path.strip("'\" ")

    target_lang = target_lang.lower().strip()
    source_override = source_override.lower().strip()

    extracted = os.path.join(base_dir, "extracted_raw.wav")
    v_stft = os.path.join(base_dir, "isolated_vocals_master.wav")
    v_deecho = os.path.join(base_dir, "isolated_vocals_deecho.wav")
    ass_file = os.path.join(base_dir, "compiled_subtitles.ass")
    
    # Export to the user's current working directory, avoiding read-only volume crashes
    file_name = os.path.basename(clean_video_path)
    final_vid = os.path.join(os.getcwd(), os.path.splitext(file_name)[0] + f"_Karaoke_{target_lang.upper()}.mp4")

    try:
        # ---- PHASE 1: AUDIO ISOLATION ----
        extract_audio_from_video(clean_video_path, extracted)
        v_bs = run_isolation_inference(extracted, "bs_roformer_vocals_resurrection_unwa.ckpt", "bs", base_dir)
        v_mel = run_isolation_inference(extracted, "melband_roformer_big_beta5e.ckpt", "mel", base_dir)
        ensemble_vocals_stft(v_bs, v_mel, v_stft)
        d_stem = run_isolation_inference(v_stft, "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt", "dereverb", base_dir, "dry", "no dry")
        os.replace(d_stem, v_deecho)

        # ---- PHASE 2: ASR ----
        words = execute_parakeet_asr(v_deecho, base_dir, source_override)
        if not words:
            logger.warning("ASR returned no words. Aborting pipeline.")
            return
        phrases = segment_words_into_phrases(words, max_gap_seconds=3.5)
        expected_count = len(phrases)
        if expected_count == 0:
            logger.warning("Segmentation produced no phrases. Aborting pipeline.")
            return

        src_name = INPUT_LANGUAGES.get(source_override, source_override.upper()) if source_override != "auto" else "the source language"
        tgt_name = OUTPUT_LANGUAGES.get(target_lang, target_lang.upper())
        corrected_texts = [p["text"] for p in phrases]

        # ---- PHASE 2.5: MULTIMODAL AUDIO-TEXT REFINEMENT ----
        if not skip_correction and correction_model and correction_model.lower() != "none":
            logger.info(f"Phase 2.5: Executing Multimodal Refinement with ({correction_model})...")
            corrected_texts = execute_omni_refinement(v_deecho, base_dir, phrases, correction_model)

            logger.info("--- Phase 2.5: Audio-Text Correction Diff ---")
            corrections_made = 0
            for i in range(expected_count):
                orig = phrases[i]["text"].strip()
                corr = corrected_texts[i].strip()
                if orig != corr:
                    logger.info(f"Line {i + 1}:\n  - {orig}\n  + {corr}")
                    corrections_made += 1
                phrases[i]["corrected_text"] = corr
            logger.info(f"Total audio-guided corrections: {corrections_made}")
        else:
            logger.info("Phase 2.5 Skipped: Using raw ASR text directly.")
            for i in range(expected_count):
                phrases[i]["corrected_text"] = phrases[i]["text"]

        # ---- PHASE 3: TRANSLATION ----
        translations = [""] * expected_count
        needs_translation = target_lang and (target_lang != source_override)

        if needs_translation and translation_model and translation_model.lower() != "none":
            logger.info(f"Phase 3: Loading Translator LLM ({translation_model})...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_trans, tokenizer = None, None

            try:
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                tokenizer = AutoTokenizer.from_pretrained(translation_model)
                model_trans = AutoModelForCausalLM.from_pretrained(
                    translation_model, device_map="auto", torch_dtype=compute_dtype, trust_remote_code=True,
                )

                BATCH_SIZE = 40
                for batch_start in range(0, expected_count, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, expected_count)
                    batch_global_indices = list(range(batch_start, batch_end))

                    logger.info(f"Translating phrases {batch_start + 1}–{batch_end}...")
                    batch_result = execute_task_with_recovery(
                        model_trans, tokenizer,
                        prompt_builder=lambda local_indices, _globals=batch_global_indices: _build_translation_prompt(
                            phrases, corrected_texts, [_globals[i] for i in local_indices], src_name, tgt_name
                        ),
                        expected_count=len(batch_global_indices), recovery_label="translation",
                    )

                    for idx, res in zip(batch_global_indices, batch_result):
                        translations[idx] = "" if res == OMITTED else res

            except OSError as e:
                logger.error(f"Failed to load translation model {translation_model}. Details: {e}")
                raise
            finally:
                if model_trans is not None: del model_trans
                if tokenizer is not None: del tokenizer
                flush_vram("after translator unload")
        else:
            logger.info("Target language matches source or no translator provided; skipping translation.")
            translations = list(corrected_texts)

        # ---- PHASE 4: COMPILE ----
        def get_vid_res(p: str):
            try:
                res = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", p], text=True).strip()
                w, h = res.split("x")
                return int(w), int(h)
            except Exception:
                return 1920, 1080

        vid_w, vid_h = get_vid_res(clean_video_path)

        for i in range(expected_count):
            corr_text = phrases[i]["corrected_text"]
            phrases[i]["text"] = corr_text
            corr_words_str = corr_text.split()
            orig_words = phrases[i]["words"]
            if not corr_words_str or not orig_words: continue

            orig_words_str = [w["text"] for w in orig_words]
            matcher = difflib.SequenceMatcher(None, orig_words_str, corr_words_str)
            new_words = []

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    for idx in range(i1, i2):
                        new_words.append({"text": corr_words_str[j1 + (idx - i1)], "start": orig_words[idx]["start"], "end": orig_words[idx]["end"]})
                else:
                    start_time = orig_words[i1 - 1]["end"] if i1 > 0 else (orig_words[0]["start"] if orig_words else 0.0)
                    if i1 < i2: end_time = orig_words[i2 - 1]["end"]
                    elif i1 < len(orig_words): end_time = orig_words[i1]["start"]
                    elif orig_words: end_time = orig_words[-1]["end"]
                    else: end_time = start_time
                    
                    end_time = max(start_time, end_time)
                    block_corr_words = corr_words_str[j1:j2]
                    num_new_words = len(block_corr_words)

                    if num_new_words > 0:
                        time_per_word = (end_time - start_time) / num_new_words
                        curr_time = start_time
                        for w in block_corr_words:
                            new_words.append({"text": w, "start": curr_time, "end": curr_time + time_per_word})
                            curr_time += time_per_word
            phrases[i]["words"] = new_words

        def gen_ass_time(s: float) -> str:
            total_cs = int(round(s * 100))
            hours, remainder_cs = divmod(total_cs, 360000)
            minutes, remainder_cs = divmod(remainder_cs, 6000)
            seconds, cs = divmod(remainder_cs, 100)
            return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"

        ass_content = [
            "[Script Info]", "ScriptType: v4.00+", "WrapStyle: 0", f"PlayResX: {vid_w}", f"PlayResY: {vid_h}", "",
            "[V4+ Styles]", "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: KaraokeMain,Arial,{int(vid_h * 0.06)},&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1", "",
            "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]

        for idx, phrase in enumerate(phrases):
            if not phrase["words"]: continue
            line_start = phrase["words"][0]["start"]
            line_end = phrase["words"][-1]["end"]
            k_str, curr_t = "", line_start

            for w in phrase["words"]:
                gap = int(round((w["start"] - curr_t) * 100))
                if gap > 0: k_str += f"{{\\k{gap}}}"
                duration = max(1, int(round((w["end"] - w["start"]) * 100)))
                k_str += f"{{\\k{duration}}}{w['text']} "
                curr_t = w["end"]

            t_txt = translations[idx] if translations and idx < len(translations) else ""
            if t_txt:
                comb = f"{k_str.strip()}\\N{{\\r\\c&H00A0A0A0&\\fs{int(vid_h * 0.04)}}}{t_txt}"
            else:
                comb = k_str.strip()

            ass_content.append(f"Dialogue: 0,{gen_ass_time(line_start)},{gen_ass_time(line_end)},KaraokeMain,,0,0,0,,{comb}")

        with open(ass_file, "w", encoding="utf-8") as f:
            f.write("\n".join(ass_content))

        escaped_ass = escape_for_ffmpeg_subtitles(ass_file)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", clean_video_path, "-vf", f"subtitles={escaped_ass}", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-c:a", "copy", final_vid],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True,
            )
        except KeyboardInterrupt:
            logger.warning("User interrupted! Halting ffmpeg video compilation...")
            raise
            
        logger.info(f"Success! Master exported: {final_vid}")

    except Exception:
        logger.error("Pipeline crashed.", exc_info=True)
    finally:
        flush_vram("end of pipeline")
        try:
            shutil.rmtree(base_dir, ignore_errors=True)
        except Exception:
            pass

# =====================================================================
# CLI ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    def format_dict_for_help(d: dict, row_length: int = 5) -> str:
        items = [f"{k} ({v})" for k, v in d.items()]
        rows = [", ".join(items[i : i + row_length]) for i in range(0, len(items), row_length)]
        return "\n  ".join(rows)

    parser = argparse.ArgumentParser(
        description="Standalone Multilingual Karaoke Builder.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("video_path", help="Source media file.")
    parser.add_argument("-t", "--target-lang", default="en", help=f"ISO translation target. Supported Outputs:\n  {format_dict_for_help(OUTPUT_LANGUAGES)}")
    parser.add_argument("-s", "--source-lang", default="auto", help=f"ISO acoustic source. Supported Inputs:\n  {format_dict_for_help(INPUT_LANGUAGES)}")
    parser.add_argument("-c", "--correction-model", default="google/gemma-4-E4B-it", help="HuggingFace model ID for Omni multimodal ASR refinement. Use 'none' to skip.")
    parser.add_argument("-m", "--translation-model", default="google/translategemma-12b-it", help="HuggingFace model ID for translation. Use 'none' to skip.")
    parser.add_argument("--skip-correction", action="store_true", help="Skip Phase 2.5 (Multimodal Refinement) to save VRAM.")

    args = parser.parse_args()
    run_pipeline(args.video_path, args.target_lang, args.source_lang, args.correction_model, args.translation_model, args.skip_correction)