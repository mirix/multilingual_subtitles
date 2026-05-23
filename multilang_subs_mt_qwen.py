import os
import sys
import gc
import logging
import subprocess
import argparse
import numpy as np
import librosa
import soundfile as sf
import torch
import re
from wtpsplit import SaT
from audio_separator.separator import Separator

# Configure System Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# Load SaT (Segment any Text) Globally
logger.info("Initializing SaT (Segment any Text) Semantic Segmentation Model...")
sat_model = SaT("sat-3l-sm")

# Parakeet v3 Multilingual Support
PARAKEET_LANGUAGES = {
    "en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French",
    "de": "German", "ru": "Russian", "it": "Italian", "pt": "Portuguese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "nl": "Dutch",
    "sv": "Swedish", "pl": "Polish", "tr": "Turkish", "hi": "Hindi",
    "vi": "Vietnamese", "id": "Indonesian", "cs": "Czech", "uk": "Ukrainian",
    "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "da": "Danish", "fi": "Finnish"
}

# Qwen3 emits its chain-of-thought wrapped in <think>...</think> before the actual answer.
# The closing tag is a special token (ID 151668) which `skip_special_tokens=True` would silently
# remove from decoded text — so we MUST split on the token ID, not on a string match.
QWEN3_THINK_END_TOKEN_ID = 151668

# Substrings in a model repo name that indicate a quantized model. For these we skip the
# explicit torch_dtype override and let the quantization config drive the load.
_QUANT_MARKERS = ("awq", "gptq", "int4", "int8", "-int", "_int", "4bit", "8bit")


# =====================================================================
# UTILITIES
# =====================================================================

def flush_vram(stage: str = ""):
    """Aggressively purges Python garbage and forces PyTorch to release VRAM to the OS.

    Pass a stage label to log allocator state after cleanup — useful for diagnosing
    fragmentation across sequential model loads on a 24 GB card.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if stage and torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        logger.info(f"VRAM [{stage}]: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")


def _is_quantized_repo(model_name: str) -> bool:
    """Heuristically detect whether a HuggingFace repo refers to a pre-quantized model."""
    lower = model_name.lower()
    return any(marker in lower for marker in _QUANT_MARKERS)


def _load_causal_lm(model_name: str):
    """Load a causal LM, automatically picking the right dtype strategy.

    - For quantized repos (AWQ/GPTQ/int4/int8): don't pass torch_dtype; the quantization
      config in the model determines the weight dtype and activation dtype.
    - For full-precision repos: load in bfloat16 (the dtype Qwen3 and most modern LMs were
      trained in). fp16 can overflow on long contexts.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = dict(device_map="auto", trust_remote_code=True)
    if not _is_quantized_repo(model_name):
        load_kwargs["torch_dtype"] = torch.bfloat16

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except ImportError as e:
        if "awq" in str(e).lower() or "autoawq" in str(e).lower():
            raise ImportError(
                "AWQ model requested but `autoawq` is not installed. "
                "Install with: pip install autoawq"
            ) from e
        raise

    return model, tokenizer


def _release_model(*objs):
    """Move models off GPU before deleting references. More reliable than bare `del`
    for releasing the full weight allocation back to the OS."""
    for obj in objs:
        try:
            if hasattr(obj, "cpu"):
                obj.cpu()
        except Exception:
            pass
    for obj in objs:
        del obj


def _escape_for_ffmpeg_subtitles(path: str) -> str:
    """Escape a filesystem path for ffmpeg's -vf subtitles='...' filter.

    The previous version applied the Windows-specific drive-letter escape on all platforms,
    which mangled paths on Linux/macOS. Now we branch on os.name.
    """
    if os.name == 'nt':
        return path.replace('\\', '/').replace(':', '\\:')
    # POSIX: only need to escape the colon (filter-arg separator); slashes are native.
    return path.replace(':', '\\:')


# =====================================================================
# PHASE 1: VOCAL ISOLATION & DE-ECHO
# =====================================================================

def extract_audio_from_video(video_path: str, output_audio_path: str):
    logger.info("Extracting raw uncompressed audio from video source...")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
           "-ar", "44100", "-ac", "2", output_audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)


def run_isolation_inference(input_path: str, model_filename: str, output_key: str, output_dir: str,
                            primary_stem: str = "Vocals", secondary_stem: str = "Instrumental") -> str:
    """Run audio-separator and return the path to the primary stem.

    Output selection is defensive: if exact-name matching fails (e.g. because the checkpoint
    uses unusual stem labels), we fall back to picking the largest output file — for vocal
    isolation the primary stem is almost always significantly larger than the residual.
    """
    logger.info(f"Running audio-separator with model: {model_filename}")
    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=os.path.join(output_dir, "models_cache"),
        output_dir=output_dir, output_format="WAV",
        normalization_threshold=0.9, use_autocast=True
    )
    separator.load_model(model_filename=model_filename)

    expected_primary = f"{primary_stem.lower()}_{output_key}"
    expected_secondary = f"{secondary_stem.lower()}_{output_key}"

    output_names = {primary_stem: expected_primary, secondary_stem: expected_secondary}
    output_files = separator.separate(input_path, output_names)

    target_file = None
    if output_files:
        # Priority 1: exact name match for the primary stem.
        for f in output_files:
            if expected_primary in f and expected_secondary not in f:
                target_file = f
                break

        # Priority 2: anything that isn't the secondary stem.
        if not target_file:
            for f in output_files:
                if expected_secondary not in f:
                    target_file = f
                    break

        # Priority 3 (defensive fallback): pick the largest file in the output set.
        # The residual is usually much smaller than the vocal track, so size-based selection
        # survives unexpected stem-label renames in the checkpoint.
        if not target_file:
            try:
                target_file = max(
                    output_files,
                    key=lambda f: os.path.getsize(os.path.join(output_dir, f))
                )
                logger.warning(
                    f"Could not match stem names for {model_filename}; "
                    f"falling back to largest output file: {target_file}"
                )
            except OSError:
                target_file = output_files[0]

    if not target_file:
        raise FileNotFoundError(
            f"Separation completed, but no output files were found for {model_filename}"
        )

    del separator
    flush_vram()
    return os.path.join(output_dir, target_file)


def ensemble_vocals_stft(file1: str, file2: str, output_path: str):
    logger.info("Blending isolated vocal components using high-fidelity STFT matrix arrays...")
    y1, sr1 = librosa.load(file1, sr=None, mono=False)
    y2, sr2 = librosa.load(file2, sr=None, mono=False)
    min_len = min(y1.shape[-1], y2.shape[-1])
    y1, y2 = y1[..., :min_len], y2[..., :min_len]

    channels = 1 if len(y1.shape) == 1 else y1.shape[0]
    ensembled_channels = []

    for ch in range(channels):
        arr1 = y1 if channels == 1 else y1[ch]
        arr2 = y2 if channels == 1 else y2[ch]
        stft1 = librosa.stft(arr1, n_fft=2048, hop_length=512)
        stft2 = librosa.stft(arr2, n_fft=2048, hop_length=512)
        ensembled_channels.append(librosa.istft((0.5 * stft1) + (0.5 * stft2), hop_length=512))

    final_audio = np.stack(ensembled_channels, axis=0) if channels > 1 else ensembled_channels[0]
    if (max_val := np.max(np.abs(final_audio))) > 1.0:
        final_audio /= max_val
    sf.write(output_path, final_audio.T, sr1)


# =====================================================================
# PHASE 2: ASR ALIGNMENT & SEMANTIC SEGMENTATION
# =====================================================================

def execute_parakeet_asr(audio_path: str, lang_override: str = "auto") -> list:
    """Run Parakeet-TDT-v3 multilingual ASR with word-level timestamps.

    Word-timestamp sanitation: bad outputs (zero/negative duration, empty text) are filtered
    out before returning. A corrupted timestamp would otherwise produce ASS lines with
    negative \\k values that some renderers refuse to display.
    """
    logger.info("Initializing NVIDIA Parakeet-TDT-0.6B-v3 (Multilingual)...")
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    # Normalize the language override once so both checks are consistent. Previously the
    # first guard lowercased and the second didn't, so e.g. "EN" passed the first but failed
    # the second silently for callers using run_pipeline() programmatically.
    lang_override_norm = (lang_override or "auto").lower().strip()

    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        asr_model.eval()

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        temp_mono_path = "workspace/temp_asr_mono.wav"
        sf.write(temp_mono_path, y, sr)

        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.preserve_alignments = True
            decoding_cfg.compute_timestamps = True
            decoding_cfg.word_seperator = " "
            if lang_override_norm != "auto" and lang_override_norm in PARAKEET_LANGUAGES:
                decoding_cfg.language = lang_override_norm
            elif lang_override_norm != "auto":
                logger.warning(
                    f"Source language '{lang_override}' not in Parakeet's supported set; "
                    f"falling back to auto-detection."
                )

        asr_model.change_decoding_strategy(decoding_cfg)

        logger.info("Transcribing audio...")
        results = asr_model.transcribe([temp_mono_path], return_hypotheses=True, timestamps=True)
        hypothesis = results[0]

        raw_words = []
        timestamp_dict = getattr(hypothesis, 'timestamp', {})
        if isinstance(timestamp_dict, dict) and 'word' in timestamp_dict:
            try:
                time_stride = 8 * asr_model.cfg.preprocessor.window_stride
            except Exception:
                time_stride = 0.08

            dropped = 0
            for w in timestamp_dict['word']:
                text = w.get('word', w.get('char', '')).strip()
                if not text:
                    continue
                if 'start' in w and 'end' in w:
                    start, end = w['start'], w['end']
                elif 'start_offset' in w and 'end_offset' in w:
                    start = w['start_offset'] * time_stride
                    end = w['end_offset'] * time_stride
                else:
                    continue
                start, end = float(start), float(end)
                # Drop words with non-positive duration. Some ASR outputs include zero-width
                # words at chunk boundaries or backtracked timestamps.
                if end <= start:
                    dropped += 1
                    continue
                raw_words.append({"text": text, "start": start, "end": end})

            if dropped:
                logger.info(f"Dropped {dropped} word(s) with invalid timestamps.")

        if os.path.exists(temp_mono_path):
            os.remove(temp_mono_path)
        return raw_words
    finally:
        if 'asr_model' in locals():
            _release_model(asr_model)
        flush_vram("after Parakeet")


def segment_words_into_phrases(words: list, max_gap_seconds: float = 3.5) -> list:
    logger.info("Applying Acoustic + Semantic (SaT) Segmentation...")

    acoustic_chunks = []
    current_chunk = []
    for i, word in enumerate(words):
        current_chunk.append(word)
        gap_to_next = (words[i + 1]["start"] - word["end"]) if i < len(words) - 1 else 0
        if gap_to_next >= max_gap_seconds:
            acoustic_chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        acoustic_chunks.append(current_chunk)

    final_phrases = []
    for chunk in acoustic_chunks:
        words_with_spans = []
        curr_pos = 0
        raw_text_parts = []

        for w in chunk:
            text = w["text"]
            text_len = len(text)
            words_with_spans.append({
                "word": w,
                "start_pos": curr_pos,
                "end_pos": curr_pos + text_len
            })
            raw_text_parts.append(text)
            curr_pos += text_len + 1

        raw_text = " ".join(raw_text_parts)
        if not raw_text.strip():
            continue

        sentences = sat_model.split(raw_text)

        word_idx = 0
        curr_sentence_start = 0

        for sentence in sentences:
            curr_sentence_end = curr_sentence_start + len(sentence)
            phrase_words = []

            while word_idx < len(words_with_spans):
                w_info = words_with_spans[word_idx]
                word_center = (w_info["start_pos"] + w_info["end_pos"]) / 2.0

                if word_center <= curr_sentence_end:
                    phrase_words.append(w_info["word"])
                    word_idx += 1
                else:
                    break

            curr_sentence_start = curr_sentence_end

            if phrase_words:
                phrase_text = " ".join([w["text"] for w in phrase_words])
                final_phrases.append({
                    "text": phrase_text.strip(),
                    "start": phrase_words[0]["start"],
                    "end": phrase_words[-1]["end"],
                    "words": phrase_words
                })

        remaining_words = [w_info["word"] for w_info in words_with_spans[word_idx:]]
        if remaining_words:
            phrase_text = " ".join([w["text"] for w in remaining_words])
            final_phrases.append({
                "text": phrase_text.strip(),
                "start": remaining_words[0]["start"],
                "end": remaining_words[-1]["end"],
                "words": remaining_words
            })

    return final_phrases


# =====================================================================
# PHASE 2.5 & 3: FULL CONTEXT LLM EXECUTION
# =====================================================================

# Sentinel for slots the LLM failed to produce. A single constant so callers can detect
# failures without string-matching.
OMITTED = "[OMITTED_BY_LLM]"


def _parse_numbered_lines(completion: str, expected_count: int) -> list:
    """Parse `1| text` style numbered output into a list of length expected_count.

    Strictness vs. permissiveness: the primary pattern REQUIRES the pipe separator (matching
    the prompt's specified format). If a line starts with digits followed by other punctuation
    (e.g. "1. text" or "1: text"), we accept it ONLY if the captured number is within range —
    this prevents a legitimate line like "1999 was a great year" from being misread as a
    nonexistent line index 1999 and then silently dropped, which is what the previous regex
    `r'^(\\d+)[\\s|.:]+(.*)'` did.
    """
    # Unicode-aware merged-verse failsafe: insert a newline before any "<digit>| " sequence
    # that follows a visible character. The previous regex was Latin-script-only, which
    # silently failed on CJK / Cyrillic / Arabic content — exactly the languages where
    # Parakeet v3 most needs post-correction.
    completion = re.sub(r'(\S)\s+(\d+)\s*\|\s+', r'\1\n\2| ', completion)

    strict_pipe = re.compile(r'^(\d+)\s*\|\s+(.*)')
    permissive = re.compile(r'^(\d+)[\s.:]+(.*)')

    result_lines = [""] * expected_count
    current_idx = -1

    for line in completion.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        match = strict_pipe.match(line)
        if not match:
            perm = permissive.match(line)
            if perm and 1 <= int(perm.group(1)) <= expected_count:
                match = perm

        if match:
            line_num = int(match.group(1))
            text = match.group(2).strip()
            idx = line_num - 1
            if 0 <= idx < expected_count:
                current_idx = idx
                result_lines[current_idx] = text
                continue

        # No usable number prefix — treat the line as a continuation of the most recent
        # valid slot (rather than silently dropping it, which is what the old code did
        # when the captured number was out of range).
        if current_idx != -1:
            result_lines[current_idx] += " " + line

    for i in range(expected_count):
        if not result_lines[i]:
            result_lines[i] = OMITTED

    return result_lines


def execute_llm_task(model, tokenizer, prompt: str, expected_count: int,
                     enable_thinking: bool = False, max_new_tokens: int | None = None) -> list:
    """Single-shot LLM execution. Returns one entry per expected line; failures = OMITTED.

    When ``enable_thinking=True`` (Qwen3-style reasoning models), the model first emits a
    ``<think>...</think>`` block. This function locates the closing ``</think>`` token in the
    generated IDs and decodes only the content that follows, guaranteeing that reasoning
    never leaks into the returned lines.

    For reasoning models, the recommended sampling schedule (T=0.6, top_p=0.95, top_k=20,
    min_p=0) is used. For instruction models, we use greedy decoding with a mild repetition
    penalty — more deterministic than the previous T=0.1 + do_sample=True setup, which was
    the worst of both worlds (near-greedy but with run-to-run variance).
    """
    messages = [
        {
            "role": "system",
            "content": "You are a precise data formatting AI. Output strictly the requested "
                       "numbered lines. Do not output any markdown preambles or conversational text."
        },
        {"role": "user", "content": prompt}
    ]

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older tokenizers don't accept enable_thinking — fall back without it.
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # Warn before pushing close to the native context limit. Qwen3-14B has 32k native;
    # leave at least 8k for generation in thinking mode.
    input_len = inputs.input_ids.shape[1]
    if input_len > 24_000:
        logger.warning(
            f"Prompt is {input_len} tokens — close to Qwen3's 32k native context. "
            f"Consider chunking input or enabling YaRN scaling for longer content."
        )

    if enable_thinking:
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or 32768,   # generous budget for reasoning + answer
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or 2048,
            do_sample=False,                          # greedy is fine for non-thinking models
            repetition_penalty=1.05,                  # mild guard against loops
            pad_token_id=tokenizer.eos_token_id,
        )

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs[0][input_len:].tolist()

    if enable_thinking:
        # rindex of </think> — discard everything up to and including it.
        try:
            split_idx = len(generated_ids) - generated_ids[::-1].index(QWEN3_THINK_END_TOKEN_ID)
        except ValueError:
            logger.warning(
                "Thinking mode is enabled but no </think> token was found in the generation. "
                "Reasoning likely did not finish within the token budget. "
                "Falling back to treating the full output as content."
            )
            split_idx = 0
        completion = tokenizer.decode(generated_ids[split_idx:], skip_special_tokens=True)
    else:
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return _parse_numbered_lines(completion, expected_count)


def execute_llm_task_with_recovery(model, tokenizer, prompt_builder, expected_count: int,
                                   enable_thinking: bool = False,
                                   recovery_label: str = "task") -> list:
    """Run an LLM task and recover any omitted lines with a focused re-prompt.

    ``prompt_builder`` is a callable taking a list of 0-based indices it should produce
    output for. It must return a prompt asking for exactly ``len(indices)`` numbered lines
    (renumbered 1..N for the LLM's benefit).

    Returns a list of length ``expected_count`` with OMITTED sentinels for any line that
    survived both the primary pass and the recovery pass.
    """
    if expected_count == 0:
        return []

    primary_prompt = prompt_builder(list(range(expected_count)))
    result = execute_llm_task(model, tokenizer, primary_prompt, expected_count,
                              enable_thinking=enable_thinking)

    omitted = [i for i, r in enumerate(result) if r == OMITTED]
    if not omitted:
        return result

    if len(omitted) > expected_count // 2:
        # More than half the lines failed — the primary pass is broken (wrong format,
        # context overflow, or model refused). A recovery pass is unlikely to help.
        logger.warning(
            f"[{recovery_label}] LLM omitted {len(omitted)}/{expected_count} lines on the "
            f"primary pass. Skipping recovery (likely a systemic prompt or context issue)."
        )
        return result

    logger.info(
        f"[{recovery_label}] Recovering {len(omitted)} omitted lines via targeted re-prompt..."
    )
    recovery_prompt = prompt_builder(omitted)
    recovered = execute_llm_task(model, tokenizer, recovery_prompt, len(omitted),
                                 enable_thinking=enable_thinking)

    for slot, recovered_text in zip(omitted, recovered):
        if recovered_text != OMITTED:
            result[slot] = recovered_text

    still_missing = sum(1 for r in result if r == OMITTED)
    if still_missing:
        logger.warning(f"[{recovery_label}] {still_missing} line(s) remain unfilled after recovery.")
    return result


# =====================================================================
# PHASE 3.5: OPTIONAL EMBEDDING-BASED TRANSLATION VALIDATION
# =====================================================================

def validate_translations(sources: list, translations: list, threshold: float = 0.30) -> list:
    """Flag translations that look semantically distant from their source.

    Uses a small multilingual sentence encoder (paraphrase-multilingual-MiniLM-L12-v2, ~120MB).
    Optional: if `sentence-transformers` isn't installed, we skip validation rather than fail.

    Returns a list of (index, similarity) tuples for suspect lines.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.info(
            "sentence-transformers not installed; skipping translation similarity validation. "
            "Install with: pip install sentence-transformers"
        )
        return []

    eligible = [
        (i, s, t) for i, (s, t) in enumerate(zip(sources, translations))
        if s.strip() and t.strip() and t != OMITTED
    ]
    if not eligible:
        return []

    encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    try:
        idxs = [e[0] for e in eligible]
        src_emb = encoder.encode([e[1] for e in eligible], normalize_embeddings=True)
        trans_emb = encoder.encode([e[2] for e in eligible], normalize_embeddings=True)
        sims = (src_emb * trans_emb).sum(axis=1)
        return [(idx, float(s)) for idx, s in zip(idxs, sims) if s < threshold]
    finally:
        del encoder
        flush_vram("after translation validator")


# =====================================================================
# PHASE 4: ASS FORMATTING & VIDEO COMPILATION
# =====================================================================

def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds % 1) * 100))
    if cs >= 100:
        s += cs // 100
        cs %= 100
        m += s // 60
        s %= 60
        h += m // 60
        m %= 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def get_video_resolution(video_path: str) -> tuple:
    """Probe video dimensions. On failure, log a warning and return 1080p as a fallback,
    rather than silently masking real issues like a missing ffprobe binary."""
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path]
        res = subprocess.check_output(cmd, text=True).strip()
        w, h = map(int, res.split('x'))
        return w, h
    except FileNotFoundError:
        logger.warning("ffprobe not found on PATH; falling back to 1920x1080 for subtitle sizing.")
        return 1920, 1080
    except Exception as e:
        logger.warning(f"Could not probe video resolution ({e}); falling back to 1920x1080.")
        return 1920, 1080


def generate_ass_file(phrases: list, translations: list, output_path: str,
                      video_w: int, video_h: int):
    main_fs = int(video_h * 0.06)
    sub_fs = int(video_h * 0.04)

    ass_content = [
        "[Script Info]", "ScriptType: v4.00+", "WrapStyle: 0",
        f"PlayResX: {video_w}", f"PlayResY: {video_h}", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: KaraokeMain,Arial,{main_fs},&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1", "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]
    for idx, phrase in enumerate(phrases):
        line_start = phrase["words"][0]["start"]
        line_end = phrase["words"][-1]["end"]
        karaoke_str, current_time = "", line_start

        for word_data in phrase["words"]:
            gap_cs = int(round((word_data["start"] - current_time) * 100))
            if gap_cs > 0:
                karaoke_str += f"{{\\k{gap_cs}}}"
            dur_cs = max(1, int(round((word_data['end'] - word_data['start']) * 100)))
            karaoke_str += f"{{\\k{dur_cs}}}{word_data['text']} "
            current_time = word_data["end"]

        trans = translations[idx] if translations and idx < len(translations) else ""
        # Skip the translation line entirely if it's an unfilled sentinel. Previously the
        # script fell back to the source text here, which made the viewer see the source
        # language twice — once karaoke-timed on top, and once as a fake "translation" below.
        if trans == OMITTED:
            trans = ""

        if trans:
            combined_text = (
                f"{karaoke_str.strip()}\\N{{\\r\\c&H00A0A0A0&\\fs{sub_fs}}}{trans}"
            )
        else:
            combined_text = karaoke_str.strip()

        ass_content.append(
            f"Dialogue: 0,{format_ass_time(line_start)},{format_ass_time(line_end)},"
            f"KaraokeMain,,0,0,0,,{combined_text}"
        )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(ass_content))


def burn_subtitles_to_video(input_video: str, ass_path: str, output_video: str):
    escaped = _escape_for_ffmpeg_subtitles(ass_path)
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", f"subtitles='{escaped}'",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy", output_video
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)


# =====================================================================
# PROMPT BUILDERS
# =====================================================================

def _build_correction_prompt(phrases: list, indices: list, src_name: str) -> str:
    """Build the corrector prompt for a given subset of phrase indices.

    The prompt has been generalized away from song-specific framing — earlier versions said
    "in the context of the song", which biased the LLM toward lyrical interpretations on
    podcasts, lectures, and interviews. The new framing is content-agnostic.
    """
    expected_count = len(indices)
    raw_source = "\n".join(
        f"{slot + 1}| {phrases[orig_idx]['text']}"
        for slot, orig_idx in enumerate(indices)
    )

    return (
        f"You are an expert audio transcription editor.\n"
        f"The following numbered lines are raw outputs from an acoustic speech recognition "
        f"(ASR) model in {src_name}. The ASR output is flawed: it contains spelling errors, "
        f"grammatical mistakes, and — most importantly — phonetic hallucinations, where "
        f"a word has been misheard as a near-homophone that doesn't fit the surrounding "
        f"context. You MUST actively fix these errors.\n\n"
        f"CORRECTION RULES:\n"
        f"- Replace illogical words with near-homophones that make sense in the surrounding "
        f"context. The acoustic confusion is usually the giveaway — ask yourself what word "
        f"sounds similar AND fits the flow.\n"
        f"- Enforce consistency: if a name, number, or recurring term is transcribed correctly "
        f"in most places but poorly in another (e.g. '99' vs '90', or 'Anna' vs 'Hannah'), "
        f"correct the outliers to match the majority.\n"
        f"- Fix grammar and punctuation, but stay faithful to the speaker's register and tone.\n"
        f"- Do NOT translate. Keep the output in {src_name}.\n"
        f"- There are exactly {expected_count} numbered lines. Do not merge, split, or "
        f"reorder them. Each output line must correspond to its input line.\n"
        f"- Start your response directly with '1| '.\n\n"
        f"--- RAW ASR ---\n{raw_source}\n"
        f"--- CORRECTED ---\n"
    )


def _build_translation_prompt(phrases: list, corrected_texts: list, indices: list,
                              src_name: str, tgt_name: str) -> str:
    """Build the translator prompt for a given subset of phrase indices.

    Strict 1-to-1 line mapping is preserved (necessary for karaoke alignment with
    the original audio). Each line is presented as plain numbered text without duration
    metadata so the model focuses purely on producing a natural translation.
    """
    expected_count = len(indices)

    corrected_source = "\n".join(
        f"{slot + 1}| {corrected_texts[orig_idx]}"
        for slot, orig_idx in enumerate(indices)
    )

    return (
        f"Translate the following lines from {src_name} into {tgt_name}.\n\n"
        f"CONTEXT: These are subtitle lines for timed audio. Prefer natural, readable "
        f"phrasing over word-for-word literalism.\n\n"
        f"RULES:\n"
        f"- There are exactly {expected_count} numbered lines. Do not merge or split them. "
        f"Each output line must correspond to its input line — do not shift content across "
        f"line boundaries (the lines are independently timed to the audio).\n"
        f"- Output the translation only. Do NOT add any line numbers other than the "
        f"'1| ', '2| ' etc. prefix.\n"
        f"- Maintain the exact format: '<number>| <translated text>', one per line.\n"
        f"- If a source line is gibberish or untranslatable, render it as a best-effort "
        f"phonetic transliteration rather than omitting it.\n"
        f"- Start your response directly with '1| '.\n\n"
        f"--- SOURCE ({src_name}) ---\n{corrected_source}\n"
        f"--- TARGET ({tgt_name}) ---\n"
    )


# =====================================================================
# GLOBAL ORCHESTRATION LOOP
# =====================================================================

def run_pipeline(video_path: str, target_lang: str, source_override: str = "auto",
                 correction_model: str = "Qwen/Qwen3-14B-AWQ",
                 translation_model: str = "tencent/Hy-MT2-7B",
                 validate_translations_flag: bool = True):
    base_dir = "workspace"
    os.makedirs(base_dir, exist_ok=True)
    clean_video_path = video_path.strip("'\"")

    # Normalize language codes once for the whole pipeline.
    target_lang = target_lang.lower().strip()
    source_override = source_override.lower().strip()

    extracted = os.path.join(base_dir, "extracted_raw.wav")
    v_stft = os.path.join(base_dir, "isolated_vocals_master.wav")
    v_deecho = os.path.join(base_dir, "isolated_vocals_deecho.wav")
    ass_file = os.path.join(base_dir, "compiled_subtitles.ass")
    final_vid = os.path.splitext(clean_video_path)[0] + f"_Karaoke_{target_lang.upper()}.mp4"

    try:
        # ---- PHASE 1: AUDIO ISOLATION ----
        extract_audio_from_video(clean_video_path, extracted)
        v_bs = run_isolation_inference(extracted, "bs_roformer_vocals_resurrection_unwa.ckpt", "bs", base_dir)
        v_mel = run_isolation_inference(extracted, "melband_roformer_big_beta5e.ckpt", "mel", base_dir)
        ensemble_vocals_stft(v_bs, v_mel, v_stft)

        logger.info("Initiating Phase 1.5: Deep Acoustic De-Echo Pass...")
        d_stem = run_isolation_inference(
            v_stft, "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt", "dereverb", base_dir,
            "dry", "no dry"
        )
        os.rename(d_stem, v_deecho)

        # ---- PHASE 2: ASR + SEGMENTATION ----
        words = execute_parakeet_asr(v_deecho, source_override)
        if not words:
            logger.error("No words aligned. Aborting.")
            return

        phrases = segment_words_into_phrases(words, max_gap_seconds=3.5)
        expected_count = len(phrases)
        if expected_count == 0:
            logger.error("Segmentation produced no phrases. Aborting.")
            return
        logger.info(f"Segmented into {expected_count} phrases.")

        if source_override == "auto":
            src_name = "its natively detected language"
        else:
            src_name = PARAKEET_LANGUAGES.get(source_override, source_override.upper())
        tgt_name = PARAKEET_LANGUAGES.get(target_lang, target_lang.upper())

        # ---- PHASE 2.5: SEMANTIC CORRECTION ----
        logger.info(f"Phase 2.5: Loading Corrector LLM ({correction_model})...")
        model_corr, tokenizer_corr = _load_causal_lm(correction_model)
        flush_vram("after corrector load")

        # Qwen3 thinking mode — the reasoning chain helps catch homophones and contextual
        # inconsistencies that a non-thinking model would miss. Reasoning is stripped from
        # the output by token-ID (see _parse_numbered_lines / execute_llm_task).
        corrected_texts = execute_llm_task_with_recovery(
            model_corr, tokenizer_corr,
            prompt_builder=lambda indices: _build_correction_prompt(phrases, indices, src_name),
            expected_count=expected_count,
            enable_thinking=True,
            recovery_label="correction",
        )

        _release_model(model_corr, tokenizer_corr)
        flush_vram("after corrector unload")

        # Diff log
        logger.info("--- Phase 2.5: Semantic Correction Diff ---")
        corrections_made = 0
        for i in range(expected_count):
            orig = phrases[i]["text"].strip()
            corr = corrected_texts[i].strip()
            if corr == OMITTED:
                # Recovery still failed for this line — keep the original ASR text.
                corr = orig
                corrected_texts[i] = orig
            if orig != corr:
                logger.info(f"Line {i + 1}:")
                logger.info(f"  - {orig}")
                logger.info(f"  + {corr}")
                corrections_made += 1
            phrases[i]["corrected_text"] = corr

        if corrections_made == 0:
            logger.info("No semantic changes were made by the LLM.")
        else:
            logger.info(f"Total semantic corrections: {corrections_made}")

        # ---- PHASE 3: FULL CONTEXT TRANSLATION ----
        translations = []
        needs_translation = target_lang and target_lang != source_override
        if needs_translation:
            logger.info(f"Phase 3: Loading Translator LLM ({translation_model})...")
            model_trans, tokenizer_trans = _load_causal_lm(translation_model)
            flush_vram("after translator load")

            translations = execute_llm_task_with_recovery(
                model_trans, tokenizer_trans,
                prompt_builder=lambda indices: _build_translation_prompt(
                    phrases, corrected_texts, indices, src_name, tgt_name
                ),
                expected_count=expected_count,
                enable_thinking=False,
                recovery_label="translation",
            )

            _release_model(model_trans, tokenizer_trans)
            flush_vram("after translator unload")

            # Replace any leftover OMITTED sentinels with empty strings so the ASS writer
            # produces a karaoke line WITHOUT a translation row, rather than falling back
            # to the source text (which would show the viewer the source language twice).
            translations = ["" if t == OMITTED else t for t in translations]

            # ---- PHASE 3.5: OPTIONAL VALIDATION ----
            if validate_translations_flag:
                suspects = validate_translations(corrected_texts, translations, threshold=0.30)
                if suspects:
                    logger.warning(
                        f"--- Phase 3.5: {len(suspects)} translation(s) flagged as "
                        f"semantically distant from source (cosine < 0.30) ---"
                    )
                    for idx, sim in suspects:
                        logger.warning(
                            f"  Line {idx + 1} [sim={sim:.2f}]:"
                            f"\n    src: {corrected_texts[idx]}"
                            f"\n    tgt: {translations[idx]}"
                        )
                    logger.warning(
                        "These may be hallucinations — inspect manually before publishing."
                    )
        else:
            logger.info("Target language matches source; skipping translation phase.")
            translations = list(corrected_texts)

        # ---- PHASE 4: COMPILE ----
        vid_w, vid_h = get_video_resolution(clean_video_path)

        # Promote corrected text into the phrase records so the karaoke-timed line reflects
        # the cleaned-up version.
        for i in range(expected_count):
            phrases[i]["text"] = phrases[i]["corrected_text"]

        generate_ass_file(phrases, translations, ass_file, vid_w, vid_h)
        burn_subtitles_to_video(clean_video_path, ass_file, final_vid)

        logger.info(f"Success! Master exported: {final_vid}")

    except Exception:
        logger.error("Pipeline crashed.", exc_info=True)
    finally:
        flush_vram("end of pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Multilingual Karaoke Builder.")
    parser.add_argument("video_path", help="Source media file.")
    parser.add_argument("-t", "--target-lang", default="en", help="ISO translation target.")
    parser.add_argument("-s", "--source-lang", default="auto",
                        help="ISO acoustic source (e.g. 'ru', 'de', or 'auto').")
    parser.add_argument("-c", "--correction-model", default="Qwen/Qwen3-14B-AWQ",
                        help="LLM for semantic editing. Default fits a 24 GB GPU with 32k "
                             "thinking budget. Alternative: google/gemma-4-26B-A4B-it (needs "
                             "4-bit quant).")
    parser.add_argument("-m", "--translation-model", default="tencent/Hy-MT2-7B",
                        help="LLM for translation.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the embedding-based translation similarity check.")

    args = parser.parse_args()
    run_pipeline(
        video_path=args.video_path,
        target_lang=args.target_lang,
        source_override=args.source_lang,
        correction_model=args.correction_model,
        translation_model=args.translation_model,
        validate_translations_flag=not args.no_validate,
    )
