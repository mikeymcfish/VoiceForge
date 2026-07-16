#!/usr/bin/env python3
from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
PINNED_MODEL_REVISIONS = {
    DEFAULT_MODEL_ID: "5d83992436eae1d760afd27aff78a71d676296fc",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "fd4b254389122332181a7c3db7f27e918eec64e3",
}
SUPPORTED_MODEL_IDS = frozenset(PINNED_MODEL_REVISIONS)
QWEN_TTS_PACKAGE = "qwen-tts==0.1.1"
MODEL_MANIFEST_NAME = ".voiceforge-qwen-models.json"
MODEL_MANIFEST_VERSION = 1
MINIMUM_MODEL_BYTES = 100 * 1024 * 1024
MODEL_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}


def configure_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


configure_utf8_streams()


class WorkerError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def emit(event: str, **payload: Any) -> None:
    message = {"event": event}
    message.update(payload)
    print(json.dumps(message, ensure_ascii=False), flush=True)


def is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _torch_major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)", version)
    if not match:
        raise WorkerError(
            "UNSUPPORTED_RUNTIME",
            f"Unable to parse the installed PyTorch version: {version!r}",
        )
    return int(match.group(1)), int(match.group(2))


def ensure_dependencies() -> None:
    """Validate the isolated Qwen runtime without changing it at request time."""
    expected_version = QWEN_TTS_PACKAGE.partition("==")[2]
    try:
        current_version = importlib_metadata.version("qwen-tts")
    except importlib_metadata.PackageNotFoundError as exc:
        raise WorkerError(
            "RUNTIME_NOT_CONFIGURED",
            "Qwen3-TTS is not installed in this interpreter. Create a dedicated "
            f"environment, install {QWEN_TTS_PACKAGE}, and set QWEN_TTS_PYTHON "
            "to its Python executable.",
        ) from exc

    if current_version != expected_version:
        raise WorkerError(
            "UNSUPPORTED_RUNTIME",
            f"Expected {QWEN_TTS_PACKAGE}, found qwen-tts=={current_version}.",
            details={"expected": expected_version, "installed": current_version},
        )

    try:
        import numpy  # noqa: F401
        import qwen_tts  # noqa: F401
        import soundfile  # noqa: F401
        import torch
    except (ImportError, OSError) as exc:
        raise WorkerError(
            "RUNTIME_NOT_CONFIGURED",
            "The isolated Qwen3-TTS runtime is incomplete. Reinstall "
            f"{QWEN_TTS_PACKAGE} in the interpreter selected by QWEN_TTS_PYTHON "
            f"({exc}).",
        ) from exc

    if _torch_major_minor(str(torch.__version__)) < (2, 6):
        raise WorkerError(
            "UNSUPPORTED_RUNTIME",
            "Qwen3-TTS requires PyTorch 2.6 or newer; "
            f"found {torch.__version__}.",
        )


def prepare_environment(root_dir: Path, models_dir: Path) -> tuple[Path, Path]:
    root = root_dir.expanduser().resolve()
    models = models_dir.expanduser().resolve()
    cache = models / "hf_cache"
    root.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(models / "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
    os.environ.setdefault("TORCH_HOME", str(root / "torch_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return models, cache


def enable_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def require_supported_model(model_id: str) -> str:
    normalized = (model_id or "").strip()
    if normalized not in PINNED_MODEL_REVISIONS:
        raise WorkerError(
            "UNSUPPORTED_MODEL",
            f"Unsupported Qwen3-TTS model {normalized!r}. Allowed models: "
            + ", ".join(sorted(PINNED_MODEL_REVISIONS)),
            details={"model_id": normalized, "allowed": sorted(PINNED_MODEL_REVISIONS)},
        )
    return normalized


def _manifest_path(models_root: Path) -> Path:
    return models_root / MODEL_MANIFEST_NAME


def _expected_snapshot_path(models_root: Path, model_id: str) -> Path:
    revision = PINNED_MODEL_REVISIONS[model_id]
    cache_name = f"models--{model_id.replace('/', '--')}"
    return (models_root / "hf_cache" / cache_name / "snapshots" / revision).resolve()


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _snapshot_inventory(models_root: Path, snapshot_path: Path) -> List[Dict[str, Any]]:
    root = models_root.resolve()
    snapshot = snapshot_path.resolve()
    if not snapshot.is_dir():
        raise WorkerError("MODEL_NOT_READY", f"Model snapshot is missing: {snapshot}")
    if not _is_within(root, snapshot):
        raise WorkerError(
            "UNSAFE_MODEL_SNAPSHOT",
            f"Model snapshot resolves outside the managed directory: {snapshot}",
        )

    inventory: List[Dict[str, Any]] = []
    incomplete: List[str] = []
    for candidate in sorted(snapshot.rglob("*")):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if not _is_within(root, resolved):
            raise WorkerError(
                "UNSAFE_MODEL_SNAPSHOT",
                f"Model file resolves outside the managed directory: {candidate}",
            )
        relative = candidate.relative_to(snapshot).as_posix()
        if relative.endswith((".incomplete", ".lock", ".tmp")):
            incomplete.append(relative)
        inventory.append({"path": relative, "size": candidate.stat().st_size})

    if incomplete:
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            "The model snapshot contains interrupted download files.",
            details={"files": incomplete[:20]},
        )
    if not inventory:
        raise WorkerError("INCOMPLETE_MODEL_SNAPSHOT", "The model snapshot contains no files.")
    if not any(item["path"] == "config.json" and item["size"] > 0 for item in inventory):
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            "The Qwen3-TTS snapshot is missing a non-empty config.json.",
        )
    if not any(Path(item["path"]).suffix.lower() in MODEL_WEIGHT_SUFFIXES for item in inventory):
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            "The Qwen3-TTS snapshot contains no recognized model weights.",
        )
    total_size = sum(int(item["size"]) for item in inventory)
    if total_size < MINIMUM_MODEL_BYTES:
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            "The Qwen3-TTS snapshot is incomplete (model files are too small).",
            details={"bytes": total_size, "minimum_bytes": MINIMUM_MODEL_BYTES},
        )
    return inventory


def _read_manifest(models_root: Path, *, required: bool) -> Dict[str, Any]:
    path = _manifest_path(models_root)
    if not path.is_file():
        if required:
            raise WorkerError(
                "MODEL_NOT_READY",
                "Pinned Qwen3-TTS models have not been downloaded. Run setup first.",
            )
        return {"manifest_version": MODEL_MANIFEST_VERSION, "models": {}}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if not required:
            return {"manifest_version": MODEL_MANIFEST_VERSION, "models": {}}
        raise WorkerError(
            "INVALID_MODEL_MANIFEST",
            "The Qwen3-TTS model manifest is unreadable; run setup again.",
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_version") != MODEL_MANIFEST_VERSION
        or not isinstance(manifest.get("models"), dict)
    ):
        if not required:
            return {"manifest_version": MODEL_MANIFEST_VERSION, "models": {}}
        raise WorkerError(
            "INVALID_MODEL_MANIFEST",
            "The Qwen3-TTS model manifest version or structure is unsupported; run setup again.",
        )
    return manifest


def _write_manifest_entry(
    models_root: Path,
    model_id: str,
    snapshot_path: Path,
) -> None:
    expected = _expected_snapshot_path(models_root, model_id)
    snapshot = snapshot_path.resolve()
    if snapshot != expected:
        raise WorkerError(
            "UNEXPECTED_MODEL_REVISION",
            "The downloaded Qwen3-TTS snapshot did not resolve to the pinned revision.",
            details={"expected": str(expected), "actual": str(snapshot)},
        )
    inventory = _snapshot_inventory(models_root, snapshot)
    manifest = _read_manifest(models_root, required=False)
    models = manifest.setdefault("models", {})
    models[model_id] = {
        "repo_id": model_id,
        "revision": PINNED_MODEL_REVISIONS[model_id],
        "snapshot_path": snapshot.relative_to(models_root).as_posix(),
        "files": inventory,
    }
    path = _manifest_path(models_root)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_manifest_inventory(
    models_root: Path,
    snapshot_path: Path,
    raw_inventory: Any,
) -> None:
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise WorkerError(
            "INVALID_MODEL_MANIFEST",
            "The Qwen3-TTS model manifest has no completeness inventory.",
        )
    expected: Dict[str, int] = {}
    for item in raw_inventory:
        if not isinstance(item, dict):
            raise WorkerError("INVALID_MODEL_MANIFEST", "Invalid model manifest file entry.")
        relative = item.get("path")
        size = item.get("size")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in expected
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise WorkerError("INVALID_MODEL_MANIFEST", "Invalid model manifest file entry.")
        logical = Path(os.path.abspath(snapshot_path / relative))
        if not _is_within(snapshot_path, logical):
            raise WorkerError(
                "INVALID_MODEL_MANIFEST",
                "The Qwen3-TTS model manifest contains an unsafe file path.",
            )
        expected[relative] = size

    actual = {item["path"]: item["size"] for item in _snapshot_inventory(models_root, snapshot_path)}
    if actual != expected:
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            "The pinned Qwen3-TTS snapshot does not match its completeness inventory; run setup again.",
        )


def load_pinned_snapshot(models_root: Path, model_id: str) -> Path:
    model_id = require_supported_model(model_id)
    manifest = _read_manifest(models_root, required=True)
    raw_entry = manifest["models"].get(model_id)
    if not isinstance(raw_entry, dict):
        raise WorkerError(
            "MODEL_NOT_READY",
            f"Pinned model {model_id} has not been downloaded. Run setup for this model first.",
            details={"model_id": model_id},
        )
    revision = PINNED_MODEL_REVISIONS[model_id]
    if raw_entry.get("repo_id") != model_id or raw_entry.get("revision") != revision:
        raise WorkerError(
            "UNEXPECTED_MODEL_REVISION",
            f"The installed {model_id} snapshot is not the pinned revision; run setup again.",
        )
    relative_snapshot = raw_entry.get("snapshot_path")
    if not isinstance(relative_snapshot, str) or not relative_snapshot or Path(relative_snapshot).is_absolute():
        raise WorkerError(
            "INVALID_MODEL_MANIFEST",
            "The Qwen3-TTS model manifest contains an invalid snapshot path.",
        )
    logical_snapshot = Path(os.path.abspath(models_root / relative_snapshot))
    if not _is_within(models_root, logical_snapshot):
        raise WorkerError(
            "INVALID_MODEL_MANIFEST",
            "The Qwen3-TTS model manifest points outside the managed directory.",
        )
    snapshot = logical_snapshot.resolve()
    if snapshot != _expected_snapshot_path(models_root, model_id):
        raise WorkerError(
            "UNEXPECTED_MODEL_REVISION",
            "The Qwen3-TTS model manifest does not point to the pinned snapshot path.",
        )
    _validate_manifest_inventory(models_root, snapshot, raw_entry.get("files"))
    return snapshot


def setup_model(models_root: Path, model_id: str) -> Path:
    model_id = require_supported_model(model_id)
    try:
        from huggingface_hub import snapshot_download
    except (ImportError, OSError) as exc:
        raise WorkerError(
            "RUNTIME_NOT_CONFIGURED",
            "huggingface_hub is required to download Qwen3-TTS models. Install the "
            f"isolated {QWEN_TTS_PACKAGE} runtime first.",
        ) from exc

    revision = PINNED_MODEL_REVISIONS[model_id]
    emit(
        "progress",
        progress=0.05,
        message=f"Downloading {model_id} at pinned revision {revision[:12]}...",
    )
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=str(models_root / "hf_cache"),
            )
        )
    except Exception as exc:
        raise WorkerError(
            "MODEL_DOWNLOAD_FAILED",
            f"Failed to download pinned Qwen3-TTS model {model_id}: {exc}",
            details={"model_id": model_id, "revision": revision},
        ) from exc
    emit("progress", progress=0.9, message="Verifying downloaded model snapshot...")
    _write_manifest_entry(models_root, model_id, snapshot)
    verified = load_pinned_snapshot(models_root, model_id)
    emit(
        "complete",
        progress=1.0,
        message="Qwen3-TTS model setup complete",
        model_id=model_id,
        revision=revision,
        snapshot_path=str(verified),
    )
    return verified


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


def _sentence_separator(previous: str) -> str:
    return "" if previous.endswith(("\u3002", "\uff01", "\uff1f")) else " "


def _join_sentences(sentences: Sequence[str]) -> str:
    if not sentences:
        return ""
    joined = sentences[0]
    for sentence in sentences[1:]:
        joined += _sentence_separator(joined) + sentence
    return joined.strip()


def split_text(text: str, max_chars: int) -> List[str]:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []
    max_chars = max(50, max_chars)
    # CJK text commonly has no whitespace after sentence punctuation.  A
    # zero-width split keeps those boundaries useful without requiring spaces.
    sentence_splits = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s*", cleaned)
    chunks: List[str] = []
    buffer: List[str] = []
    length = 0
    for sentence in sentence_splits:
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if buffer:
                chunks.append(_join_sentences(buffer))
                buffer = []
                length = 0
            chunks.extend(_split_by_words(sentence, max_chars))
            continue
        separator_len = len(_sentence_separator(buffer[-1])) if buffer else 0
        if buffer and length + separator_len + len(sentence) > max_chars:
            chunks.append(_join_sentences(buffer))
            buffer = [sentence]
            length = len(sentence)
        else:
            buffer.append(sentence)
            length += separator_len + len(sentence)
    if buffer:
        chunks.append(_join_sentences(buffer))
    return chunks


def _load_voice_path(voice_path: str) -> str:
    voice = Path(voice_path).expanduser().resolve()
    if not voice.is_file():
        raise WorkerError("FILE_NOT_FOUND", f"Voice sample not found: {voice}")
    return str(voice)


def synthesize(
    models_root: Path,
    model_id: str,
    voice_path: str,
    reference_text: str,
    language: str,
    text: str,
    output_path: str,
    max_chars: int,
    gap_ms: int,
) -> None:
    model_id = require_supported_model(model_id)
    ensure_dependencies()
    snapshot = load_pinned_snapshot(models_root, model_id)
    enable_offline_mode()

    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore
    from qwen_tts import Qwen3TTSModel  # type: ignore

    device = (os.environ.get("QWEN_TTS_DEVICE") or "").strip()
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise WorkerError(
            "DEVICE_UNAVAILABLE",
            f"QWEN_TTS_DEVICE={device!r} requests CUDA, but CUDA is unavailable.",
        )
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    load_options: Dict[str, Any] = {
        "device_map": device,
        "dtype": dtype,
        "local_files_only": True,
    }
    if is_truthy(os.environ.get("QWEN_TTS_USE_FLASH_ATTENTION")):
        load_options["attn_implementation"] = "flash_attention_2"

    emit(
        "progress",
        progress=0.05,
        message=f"Loading pinned {model_id} snapshot on {device}",
    )
    try:
        model = Qwen3TTSModel.from_pretrained(str(snapshot), **load_options)
    except Exception as exc:
        raise WorkerError(
            "MODEL_LOAD_FAILED",
            f"Failed to load the pinned local Qwen3-TTS snapshot: {exc}",
            details={"model_id": model_id, "snapshot_path": str(snapshot)},
        ) from exc

    voice_ref = _load_voice_path(voice_path)
    reference_text = (reference_text or "").strip()
    if not reference_text:
        emit(
            "log",
            level="warn",
            message="No reference transcript supplied; using speaker-embedding-only cloning "
            "with lower expected fidelity.",
        )
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=voice_ref,
        ref_text=reference_text or None,
        x_vector_only_mode=not bool(reference_text),
    )

    segments = split_text(text, max_chars=max_chars)
    if not segments:
        raise WorkerError("INVALID_ARGUMENT", "Text input is empty.")
    if gap_ms < 0:
        raise WorkerError("INVALID_ARGUMENT", "gap_ms must be zero or greater.")

    audio_segments: List[Any] = []
    sample_rate: Optional[int] = None
    total = len(segments)
    for index, segment in enumerate(segments, start=1):
        emit(
            "progress",
            progress=0.15 + (0.8 * index / total),
            message=f"Synthesizing segment {index}/{total}",
        )
        try:
            wavs, rate = model.generate_voice_clone(
                text=segment,
                language=(language or "Auto").strip() or "Auto",
                voice_clone_prompt=voice_prompt,
            )
        except Exception as exc:
            raise WorkerError(
                "SYNTHESIS_FAILED",
                f"Qwen3-TTS synthesis failed on segment {index}/{total}: {exc}",
                details={"segment": index, "segment_count": total},
            ) from exc
        if not wavs:
            raise WorkerError(
                "INVALID_MODEL_OUTPUT",
                f"Qwen3-TTS returned no audio for segment {index}.",
            )
        if sample_rate is None:
            sample_rate = int(rate)
        elif int(rate) != sample_rate:
            raise WorkerError(
                "INVALID_MODEL_OUTPUT",
                "Qwen3-TTS returned inconsistent sample rates across segments.",
            )
        audio = np.asarray(wavs[0]).squeeze()
        if audio.ndim != 1 or audio.size == 0:
            raise WorkerError(
                "INVALID_MODEL_OUTPUT",
                f"Qwen3-TTS returned an invalid waveform for segment {index}.",
            )
        audio_segments.append(audio)

    if sample_rate is None:
        raise WorkerError("INVALID_MODEL_OUTPUT", "Qwen3-TTS generated no audio.")
    if gap_ms > 0 and len(audio_segments) > 1:
        gap_samples = int(sample_rate * gap_ms / 1000)
        silence = np.zeros(gap_samples, dtype=audio_segments[0].dtype)
        merged: List[Any] = []
        for audio in audio_segments:
            if merged:
                merged.append(silence)
            merged.append(audio)
        combined = np.concatenate(merged)
    else:
        combined = np.concatenate(audio_segments)

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), combined, sample_rate)
    emit(
        "complete",
        progress=1.0,
        message="Qwen3-TTS synthesis complete",
        output_path=str(output),
        model_id=model_id,
        revision=PINNED_MODEL_REVISIONS[model_id],
        segments=total,
        sample_rate=sample_rate,
    )


def _add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-id",
        default=os.environ.get("QWEN_TTS_MODEL_ID", DEFAULT_MODEL_ID),
        choices=sorted(PINNED_MODEL_REVISIONS),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pinned, offline Qwen3-TTS worker")
    parser.add_argument(
        "--root-dir",
        default=str(Path.home() / ".voiceforge" / "qwen3_tts"),
    )
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--jobs-dir", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "setup", "download"):
        command_parser = subparsers.add_parser(command)
        _add_model_argument(command_parser)

    synth_parser = subparsers.add_parser("synthesize", help="Run local Qwen3-TTS synthesis")
    _add_model_argument(synth_parser)
    synth_parser.add_argument("--voice", required=True)
    synth_parser.add_argument("--reference-text", default="")
    synth_parser.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
    )
    synth_parser.add_argument("--text", required=True, help="UTF-8 text file")
    synth_parser.add_argument("--output", required=True)
    synth_parser.add_argument(
        "--max-chars",
        type=int,
        default=int(os.environ.get("QWEN_TTS_MAX_CHARS", "320")),
    )
    synth_parser.add_argument(
        "--gap-ms",
        type=int,
        default=int(os.environ.get("QWEN_TTS_GAP_MS", "120")),
    )
    return parser


def _error_payload(exc: BaseException) -> Dict[str, Any]:
    if isinstance(exc, WorkerError):
        return {
            "code": exc.code,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "details": exc.details,
        }
    if isinstance(exc, FileNotFoundError):
        code = "FILE_NOT_FOUND"
    elif isinstance(exc, (ValueError, argparse.ArgumentError)):
        code = "INVALID_ARGUMENT"
    else:
        code = "UNEXPECTED_ERROR"
    return {
        "code": code,
        "error": str(exc) or type(exc).__name__,
        "error_type": type(exc).__name__,
        "details": {},
    }


def run(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    root_dir = Path(args.root_dir)
    models_dir = Path(args.models_dir) if args.models_dir else root_dir / "models"
    models_root, _ = prepare_environment(root_dir, models_dir)
    if args.jobs_dir:
        Path(args.jobs_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    if args.command in {"setup", "download"}:
        setup_model(models_root, args.model_id)
        return
    if args.command == "check":
        ensure_dependencies()
        snapshot = load_pinned_snapshot(models_root, args.model_id)
        emit(
            "complete",
            progress=1.0,
            message="Qwen3-TTS runtime and pinned model are ready",
            ready=True,
            model_id=args.model_id,
            revision=PINNED_MODEL_REVISIONS[args.model_id],
            snapshot_path=str(snapshot),
        )
        return
    if args.command == "synthesize":
        text_path = Path(args.text).expanduser().resolve()
        if not text_path.is_file():
            raise WorkerError("FILE_NOT_FOUND", f"Text file not found: {text_path}")
        text = text_path.read_text(encoding="utf-8")
        synthesize(
            models_root,
            args.model_id,
            args.voice,
            args.reference_text,
            args.language,
            text,
            args.output,
            args.max_chars,
            args.gap_ms,
        )
        return
    raise WorkerError("INVALID_ARGUMENT", f"Unknown command: {args.command}")


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        emit(
            "error",
            code="CANCELLED",
            error="Qwen3-TTS operation cancelled",
            error_type="KeyboardInterrupt",
            details={},
        )
        raise SystemExit(130)
    except Exception as exc:
        emit("error", message="Qwen3-TTS operation failed", **_error_payload(exc))
        if is_truthy(os.environ.get("VOICEFORGE_WORKER_DEBUG")):
            traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
