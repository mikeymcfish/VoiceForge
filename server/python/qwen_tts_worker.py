#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def emit(event: str, **payload) -> None:
    message = {"event": event}
    message.update(payload)
    print(json.dumps(message), flush=True)


def is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _module_name_from_spec(spec: str) -> str:
    base = spec
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in base:
            base = base.split(sep, 1)[0]
            break
    if "[" in base:
        base = base.split("[", 1)[0]
    return base


def _install_package(
    package_spec: str,
    module_name: str,
    *,
    index_url: Optional[str] = None,
    extra_index_urls: Optional[Sequence[str]] = None,
) -> None:
    emit("log", level="info", message=f"Installing dependency: {package_spec}")
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", package_spec]
    if index_url:
        command.extend(["--index-url", index_url])
    if extra_index_urls:
        for url in extra_index_urls:
            if url:
                command.extend(["--extra-index-url", url])
    subprocess.check_call(command)
    to_delete = [name for name in sys.modules if name == module_name or name.startswith(f"{module_name}.")]
    for name in to_delete:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


def ensure_package(
    package_spec: str,
    import_name: Optional[str] = None,
    *,
    index_url: Optional[str] = None,
    extra_index_urls: Optional[Sequence[str]] = None,
) -> None:
    module_name = import_name or _module_name_from_spec(package_spec)
    try:
        importlib.import_module(module_name)
        return
    except Exception:
        _install_package(
            package_spec,
            module_name,
            index_url=index_url,
            extra_index_urls=extra_index_urls,
        )


def ensure_dependencies() -> None:
    use_cuda = is_truthy(os.environ.get("QWEN_TTS_ENABLE_CUDA"))
    torch_spec = os.environ.get("QWEN_TTS_TORCH_SPEC")
    torch_index_url = os.environ.get("QWEN_TTS_TORCH_INDEX_URL")
    torch_extra_indexes: Optional[Tuple[str, ...]] = None

    if use_cuda:
        torch_spec = torch_spec or "torch==2.3.1"
        extra_url = os.environ.get("QWEN_TTS_TORCH_EXTRA_INDEX_URL")
        if extra_url:
            torch_extra_indexes = (extra_url,)
        emit("log", level="info", message=f"Using CUDA torch dependency {torch_spec}")
    else:
        torch_spec = torch_spec or "torch==2.3.1+cpu"
        torch_index_url = torch_index_url or "https://download.pytorch.org/whl/cpu"
        extra_url = os.environ.get("QWEN_TTS_TORCH_EXTRA_INDEX_URL", "https://pypi.org/simple")
        torch_extra_indexes = tuple(filter(None, (extra_url,)))
        emit("log", level="info", message="Using CPU torch dependency for Qwen3 TTS")

    torchaudio_spec = os.environ.get("QWEN_TTS_TORCHAUDIO_SPEC")
    if not torchaudio_spec:
        torchaudio_spec = "torchaudio==2.3.1"
        if not use_cuda:
            torchaudio_spec += "+cpu"
    if not use_cuda and "+cpu" not in torchaudio_spec and "==" in torchaudio_spec:
        name, version = torchaudio_spec.split("==", 1)
        if "+" not in version:
            torchaudio_spec = f"{name}=={version}+cpu"

    ensure_package(torch_spec, "torch", index_url=torch_index_url, extra_index_urls=torch_extra_indexes)
    ensure_package(
        torchaudio_spec,
        "torchaudio",
        index_url=torch_index_url if not use_cuda else None,
        extra_index_urls=torch_extra_indexes,
    )
    ensure_package("transformers>=4.52.1", "transformers")
    ensure_package("accelerate>=0.33.0", "accelerate")
    ensure_package("sentencepiece>=0.2.0", "sentencepiece")
    ensure_package("safetensors>=0.4.3", "safetensors")
    ensure_package("numpy>=1.24.3", "numpy")
    ensure_package("soundfile>=0.12.1", "soundfile")


def _split_by_words(text: str, max_chars: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            for start in range(0, len(word), max_chars):
                part = word[start : start + max_chars]
                if part:
                    chunks.append(part)
            continue
        word_len = len(word) + (1 if current else 0)
        if current and current_len + word_len > max_chars:
            chunks.append(" ".join(current).strip())
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += word_len
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def split_text(text: str, max_chars: int) -> List[str]:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []
    max_chars = max(50, max_chars)
    sentence_splits = re.split(r"(?<=[.!?。！？])\\s+", cleaned)
    chunks: List[str] = []
    buffer: List[str] = []
    length = 0
    for sentence in sentence_splits:
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if buffer:
                chunks.append(" ".join(buffer).strip())
                buffer = []
                length = 0
            chunks.extend(_split_by_words(sentence, max_chars))
            continue
        if length + len(sentence) + (1 if buffer else 0) > max_chars:
            chunks.append(" ".join(buffer).strip())
            buffer = [sentence]
            length = len(sentence)
        else:
            buffer.append(sentence)
            length += len(sentence) + (1 if len(buffer) > 1 else 0)
    if buffer:
        chunks.append(" ".join(buffer).strip())
    return chunks


def _load_voice_path(voice_path: str) -> str:
    voice = Path(voice_path)
    if not voice.exists():
        raise FileNotFoundError(f"Voice sample not found: {voice}")
    return str(voice)


def _build_forward_params(voice_path: Optional[str]) -> dict:
    if not voice_path:
        return {}
    return {
        "speaker_wav": voice_path,
        "prompt_wav": voice_path,
        "voice": voice_path,
        "voice_path": voice_path,
        "reference_audio": voice_path,
    }


def synthesize(
    model_id: str,
    voice_path: str,
    text: str,
    output_path: str,
    max_chars: int,
    gap_ms: int,
) -> None:
    ensure_dependencies()
    from transformers import pipeline  # type: ignore
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore

    device = os.environ.get("QWEN_TTS_DEVICE")
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    tts = pipeline(
        "text-to-speech",
        model=model_id,
        device=device,
        trust_remote_code=True,
    )

    voice_ref = _load_voice_path(voice_path) if voice_path else None
    forward_params = _build_forward_params(voice_ref)

    segments = split_text(text, max_chars=max_chars)
    if not segments:
        raise ValueError("No text segments produced for synthesis")

    audio_segments: List[np.ndarray] = []
    sample_rate: Optional[int] = None
    total = len(segments)
    for idx, segment in enumerate(segments, start=1):
        emit("progress", progress=idx / total, message=f"Synthesizing segment {idx}/{total}")
        try:
            result = tts(segment, forward_params=forward_params)
        except TypeError as exc:
            emit("log", level="warn", message=f"Voice cloning parameters rejected: {exc}. Retrying without reference.")
            result = tts(segment)
        audio = result.get("audio") if isinstance(result, dict) else None
        rate = result.get("sampling_rate") if isinstance(result, dict) else None
        if audio is None or rate is None:
            raise RuntimeError("TTS pipeline did not return audio output")
        if sample_rate is None:
            sample_rate = int(rate)
        elif int(rate) != sample_rate:
            raise RuntimeError("Inconsistent sample rate across segments")
        audio_segments.append(np.asarray(audio))

    if sample_rate is None:
        raise RuntimeError("No audio generated")

    if gap_ms > 0 and len(audio_segments) > 1:
        gap_samples = int(sample_rate * gap_ms / 1000)
        silence = np.zeros(gap_samples, dtype=audio_segments[0].dtype)
        merged: List[np.ndarray] = []
        for seg in audio_segments:
            if merged:
                merged.append(silence)
            merged.append(seg)
        combined = np.concatenate(merged)
    else:
        combined = np.concatenate(audio_segments)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), combined, sample_rate)
    emit("log", level="info", message=f"Saved audio to {output}")
    emit("result", output_path=str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 TTS worker")
    parser.add_argument("--root-dir", type=str, default=str(Path.home() / ".voiceforge" / "qwen3_tts"))
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--jobs-dir", type=str, default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)
    synth_parser = subparsers.add_parser("synthesize", help="Run Qwen3 TTS synthesis")
    synth_parser.add_argument("--model-id", type=str, default=os.environ.get("QWEN_TTS_MODEL_ID", "Qwen/Qwen3-TTS"))
    synth_parser.add_argument("--voice", type=str, required=True)
    synth_parser.add_argument("--text", type=str, required=True)
    synth_parser.add_argument("--output", type=str, required=True)
    synth_parser.add_argument("--max-chars", type=int, default=int(os.environ.get("QWEN_TTS_MAX_CHARS", "320")))
    synth_parser.add_argument("--gap-ms", type=int, default=int(os.environ.get("QWEN_TTS_GAP_MS", "120")))

    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    models_dir = Path(args.models_dir) if args.models_dir else root_dir / "models"
    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else root_dir / "jobs"
    models_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(models_dir))
    os.environ.setdefault("TORCH_HOME", str(root_dir / "torch_cache"))

    if args.command == "synthesize":
        text_path = Path(args.text)
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_path}")
        text = text_path.read_text(encoding="utf-8")
        synthesize(
            args.model_id,
            args.voice,
            text,
            args.output,
            args.max_chars,
            args.gap_ms,
        )


if __name__ == "__main__":
    main()
