#!/usr/bin/env python3
from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import importlib.util
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


MODEL_REPO_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"
MODEL_REVISION = "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"
CODEC_REPO_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
CODEC_REVISION = "3cd226ba2947efa357ef453bcad111b6eafba782"
PINNED_ARTIFACTS = {
    "model": {
        "repo_id": MODEL_REPO_ID,
        "revision": MODEL_REVISION,
        "minimum_bytes": 1024 * 1024 * 1024,
    },
    "codec": {
        "repo_id": CODEC_REPO_ID,
        "revision": CODEC_REVISION,
        "minimum_bytes": 1024 * 1024 * 1024,
    },
}
TRANSFORMERS_PACKAGE = "transformers==5.0.0"
MODEL_MANIFEST_NAME = ".voiceforge-moss-models.json"
MODEL_MANIFEST_VERSION = 1
MODEL_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TEMPERATURE = 1.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 25
DEFAULT_REPETITION_PENALTY = 1.0


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


_FFMPEG_DLL_DIRECTORY_HANDLES: List[Any] = []


def _is_compatible_shared_ffmpeg_bin(candidate: Path) -> bool:
    if not candidate.is_dir() or not (candidate / "ffmpeg.exe").is_file():
        return False
    compatible_sets = (
        ("avcodec-58.dll", "avfilter-7.dll", "avformat-58.dll", "avutil-56.dll", "swresample-3.dll", "swscale-5.dll"),
        ("avcodec-59.dll", "avfilter-8.dll", "avformat-59.dll", "avutil-57.dll", "swresample-4.dll", "swscale-6.dll"),
        ("avcodec-60.dll", "avfilter-9.dll", "avformat-60.dll", "avutil-58.dll", "swresample-4.dll", "swscale-7.dll"),
        ("avcodec-61.dll", "avfilter-10.dll", "avformat-61.dll", "avutil-59.dll", "swresample-5.dll", "swscale-8.dll"),
    )
    names = {entry.name.lower() for entry in candidate.iterdir() if entry.is_file()}
    return any(all(required.lower() in names for required in required_set) for required_set in compatible_sets)


def configure_windows_ffmpeg_dll_search() -> None:
    if os.name != "nt" or _FFMPEG_DLL_DIRECTORY_HANDLES:
        return
    configured = (os.environ.get("MOSS_TTS_FFMPEG_BIN") or "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
    else:
        ffmpeg = shutil.which("ffmpeg")
        candidate = Path(ffmpeg).resolve().parent if ffmpeg else Path()
    if not _is_compatible_shared_ffmpeg_bin(candidate):
        raise WorkerError(
            "RUNTIME_NOT_CONFIGURED",
            "MOSS-TTS on Windows requires FFmpeg 4-7 with shared DLLs. "
            "Rerun VoiceForge.cmd setup-moss or set MOSS_TTS_FFMPEG_BIN to "
            "the compatible shared build's bin directory.",
        )
    try:
        _FFMPEG_DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(candidate)))
    except (OSError, AttributeError) as exc:
        raise WorkerError(
            "RUNTIME_NOT_CONFIGURED",
            f"Could not register the shared FFmpeg DLL directory {candidate}: {exc}",
        ) from exc


def emit(event: str, **payload: Any) -> None:
    message = {"event": event}
    message.update(payload)
    print(json.dumps(message, ensure_ascii=False), flush=True)


def is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _major_minor(version: str, package_name: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)", version)
    if not match:
        raise WorkerError(
            "UNSUPPORTED_RUNTIME",
            f"Unable to parse the installed {package_name} version: {version!r}",
        )
    return int(match.group(1)), int(match.group(2))


def ensure_dependencies() -> None:
    configure_windows_ffmpeg_dll_search()
    expected_transformers = TRANSFORMERS_PACKAGE.partition("==")[2]
    try:
        installed_transformers = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError as exc:
        raise WorkerError(
            "RUNTIME_NOT_CONFIGURED",
            "MOSS-TTS is not installed in this interpreter. Create a dedicated "
            f"environment with {TRANSFORMERS_PACKAGE}, PyTorch, torchaudio, numpy, "
            "and soundfile, then set MOSS_TTS_PYTHON to its Python executable.",
        ) from exc
    if installed_transformers != expected_transformers:
        raise WorkerError(
            "UNSUPPORTED_RUNTIME",
            f"MOSS-TTS v1.5 requires {TRANSFORMERS_PACKAGE}; found "
            f"transformers=={installed_transformers}.",
            details={
                "expected": expected_transformers,
                "installed": installed_transformers,
            },
        )

    try:
        import numpy  # noqa: F401
        import soundfile  # noqa: F401
        import torch
        import torchaudio  # noqa: F401
        import torchcodec  # noqa: F401
        from transformers import AutoModel, AutoProcessor  # noqa: F401
    except (ImportError, OSError) as exc:
        raise WorkerError(
            "RUNTIME_NOT_CONFIGURED",
            "The isolated MOSS-TTS runtime is incomplete. Install the official "
            f"runtime dependencies in MOSS_TTS_PYTHON ({exc}).",
        ) from exc
    if _major_minor(str(torch.__version__), "PyTorch") < (2, 6):
        raise WorkerError(
            "UNSUPPORTED_RUNTIME",
            f"MOSS-TTS requires PyTorch 2.6 or newer; found {torch.__version__}.",
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


def _manifest_path(models_root: Path) -> Path:
    return models_root / MODEL_MANIFEST_NAME


def _expected_snapshot_path(models_root: Path, artifact_name: str) -> Path:
    artifact = PINNED_ARTIFACTS[artifact_name]
    cache_name = f"models--{artifact['repo_id'].replace('/', '--')}"
    return (
        models_root
        / "hf_cache"
        / cache_name
        / "snapshots"
        / str(artifact["revision"])
    ).resolve()


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _snapshot_inventory(
    models_root: Path,
    snapshot_path: Path,
    artifact_name: str,
) -> List[Dict[str, Any]]:
    root = models_root.resolve()
    snapshot = snapshot_path.resolve()
    if artifact_name not in PINNED_ARTIFACTS:
        raise WorkerError("INVALID_ARGUMENT", f"Unknown MOSS artifact: {artifact_name}")
    if not snapshot.is_dir():
        raise WorkerError(
            "MODEL_NOT_READY",
            f"Pinned MOSS {artifact_name} snapshot is missing: {snapshot}",
        )
    if not _is_within(root, snapshot):
        raise WorkerError(
            "UNSAFE_MODEL_SNAPSHOT",
            f"MOSS {artifact_name} snapshot resolves outside the managed directory: {snapshot}",
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
                f"MOSS model file resolves outside the managed directory: {candidate}",
            )
        relative = candidate.relative_to(snapshot).as_posix()
        if relative.endswith((".incomplete", ".lock", ".tmp")):
            incomplete.append(relative)
        inventory.append({"path": relative, "size": candidate.stat().st_size})

    if incomplete:
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            f"The pinned MOSS {artifact_name} snapshot contains interrupted downloads.",
            details={"files": incomplete[:20]},
        )
    if not inventory:
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            f"The pinned MOSS {artifact_name} snapshot contains no files.",
        )
    if not any(item["path"] == "config.json" and item["size"] > 0 for item in inventory):
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            f"The pinned MOSS {artifact_name} snapshot is missing config.json.",
        )
    if not any(Path(item["path"]).suffix.lower() in MODEL_WEIGHT_SUFFIXES for item in inventory):
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            f"The pinned MOSS {artifact_name} snapshot contains no recognized model weights.",
        )
    total_size = sum(int(item["size"]) for item in inventory)
    minimum_bytes = int(PINNED_ARTIFACTS[artifact_name]["minimum_bytes"])
    if total_size < minimum_bytes:
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            f"The pinned MOSS {artifact_name} snapshot is incomplete (files are too small).",
            details={"bytes": total_size, "minimum_bytes": minimum_bytes},
        )
    return inventory


def _write_manifest(models_root: Path, snapshots: Dict[str, Path]) -> None:
    entries: Dict[str, Any] = {}
    for artifact_name, artifact in PINNED_ARTIFACTS.items():
        if artifact_name not in snapshots:
            raise WorkerError(
                "INCOMPLETE_MODEL_SNAPSHOT",
                f"Missing downloaded MOSS artifact: {artifact_name}",
            )
        snapshot = snapshots[artifact_name].resolve()
        expected = _expected_snapshot_path(models_root, artifact_name)
        if snapshot != expected:
            raise WorkerError(
                "UNEXPECTED_MODEL_REVISION",
                f"Downloaded MOSS {artifact_name} did not resolve to the pinned revision.",
                details={"expected": str(expected), "actual": str(snapshot)},
            )
        entries[artifact_name] = {
            "repo_id": artifact["repo_id"],
            "revision": artifact["revision"],
            "snapshot_path": snapshot.relative_to(models_root).as_posix(),
            "files": _snapshot_inventory(models_root, snapshot, artifact_name),
        }

    manifest = {
        "manifest_version": MODEL_MANIFEST_VERSION,
        "artifacts": entries,
    }
    path = _manifest_path(models_root)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_manifest(models_root: Path) -> Dict[str, Any]:
    path = _manifest_path(models_root)
    if not path.is_file():
        raise WorkerError(
            "MODEL_NOT_READY",
            "Pinned MOSS-TTS model data has not been downloaded. Run setup first.",
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerError(
            "INVALID_MODEL_MANIFEST",
            "The MOSS-TTS model manifest is unreadable; run setup again.",
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_version") != MODEL_MANIFEST_VERSION
        or not isinstance(manifest.get("artifacts"), dict)
    ):
        raise WorkerError(
            "INVALID_MODEL_MANIFEST",
            "The MOSS-TTS model manifest version or structure is unsupported; run setup again.",
        )
    return manifest


def _validate_manifest_inventory(
    models_root: Path,
    snapshot_path: Path,
    artifact_name: str,
    raw_inventory: Any,
) -> None:
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise WorkerError(
            "INVALID_MODEL_MANIFEST",
            f"The MOSS {artifact_name} manifest has no completeness inventory.",
        )
    expected: Dict[str, int] = {}
    for item in raw_inventory:
        if not isinstance(item, dict):
            raise WorkerError("INVALID_MODEL_MANIFEST", "Invalid MOSS manifest file entry.")
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
            raise WorkerError("INVALID_MODEL_MANIFEST", "Invalid MOSS manifest file entry.")
        logical = Path(os.path.abspath(snapshot_path / relative))
        if not _is_within(snapshot_path, logical):
            raise WorkerError(
                "INVALID_MODEL_MANIFEST",
                "The MOSS model manifest contains an unsafe file path.",
            )
        expected[relative] = size
    actual = {
        item["path"]: item["size"]
        for item in _snapshot_inventory(models_root, snapshot_path, artifact_name)
    }
    if actual != expected:
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            f"The pinned MOSS {artifact_name} snapshot does not match its manifest; run setup again.",
        )


def load_pinned_snapshots(models_root: Path) -> Tuple[Path, Path]:
    manifest = _read_manifest(models_root)
    artifacts = manifest["artifacts"]
    resolved: Dict[str, Path] = {}
    for artifact_name, expected_artifact in PINNED_ARTIFACTS.items():
        raw_entry = artifacts.get(artifact_name)
        if not isinstance(raw_entry, dict):
            raise WorkerError(
                "MODEL_NOT_READY",
                f"Pinned MOSS {artifact_name} data is missing; run setup again.",
            )
        if (
            raw_entry.get("repo_id") != expected_artifact["repo_id"]
            or raw_entry.get("revision") != expected_artifact["revision"]
        ):
            raise WorkerError(
                "UNEXPECTED_MODEL_REVISION",
                f"Installed MOSS {artifact_name} data is not the pinned revision; run setup again.",
            )
        relative_snapshot = raw_entry.get("snapshot_path")
        if (
            not isinstance(relative_snapshot, str)
            or not relative_snapshot
            or Path(relative_snapshot).is_absolute()
        ):
            raise WorkerError(
                "INVALID_MODEL_MANIFEST",
                f"The MOSS {artifact_name} manifest contains an invalid snapshot path.",
            )
        logical = Path(os.path.abspath(models_root / relative_snapshot))
        if not _is_within(models_root, logical):
            raise WorkerError(
                "INVALID_MODEL_MANIFEST",
                "The MOSS model manifest points outside the managed directory.",
            )
        snapshot = logical.resolve()
        if snapshot != _expected_snapshot_path(models_root, artifact_name):
            raise WorkerError(
                "UNEXPECTED_MODEL_REVISION",
                f"The MOSS {artifact_name} manifest does not point to the pinned snapshot path.",
            )
        _validate_manifest_inventory(
            models_root,
            snapshot,
            artifact_name,
            raw_entry.get("files"),
        )
        resolved[artifact_name] = snapshot
    return resolved["model"], resolved["codec"]


def setup_models(models_root: Path) -> Tuple[Path, Path]:
    try:
        from huggingface_hub import snapshot_download
    except (ImportError, OSError) as exc:
        raise WorkerError(
            "RUNTIME_NOT_CONFIGURED",
            "huggingface_hub is required to download pinned MOSS-TTS model data.",
        ) from exc

    snapshots: Dict[str, Path] = {}
    progress_ranges = {"model": (0.05, 0.48), "codec": (0.52, 0.88)}
    for artifact_name, artifact in PINNED_ARTIFACTS.items():
        start_progress, end_progress = progress_ranges[artifact_name]
        emit(
            "progress",
            progress=start_progress,
            message=f"Downloading {artifact['repo_id']} at pinned revision "
            f"{str(artifact['revision'])[:12]}...",
        )
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=str(artifact["repo_id"]),
                    revision=str(artifact["revision"]),
                    cache_dir=str(models_root / "hf_cache"),
                )
            )
        except Exception as exc:
            raise WorkerError(
                "MODEL_DOWNLOAD_FAILED",
                f"Failed to download pinned MOSS {artifact_name} data: {exc}",
                details={
                    "artifact": artifact_name,
                    "repo_id": artifact["repo_id"],
                    "revision": artifact["revision"],
                },
            ) from exc
        snapshots[artifact_name] = snapshot
        emit(
            "progress",
            progress=end_progress,
            message=f"Downloaded pinned MOSS {artifact_name} snapshot",
        )

    emit("progress", progress=0.92, message="Verifying MOSS model and codec snapshots...")
    _write_manifest(models_root, snapshots)
    model_snapshot, codec_snapshot = load_pinned_snapshots(models_root)
    emit(
        "complete",
        progress=1.0,
        message="MOSS-TTS v1.5 model setup complete",
        model_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        model_snapshot_path=str(model_snapshot),
        codec_id=CODEC_REPO_ID,
        codec_revision=CODEC_REVISION,
        codec_snapshot_path=str(codec_snapshot),
    )
    return model_snapshot, codec_snapshot


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
            chunks.extend(
                word[start : start + max_chars]
                for start in range(0, len(word), max_chars)
                if word[start : start + max_chars]
            )
            continue
        addition = len(word) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            chunks.append(" ".join(current).strip())
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += addition
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
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return []
    max_chars = max(50, int(max_chars))
    # CJK text commonly has no whitespace after sentence punctuation.  A
    # zero-width split keeps those boundaries useful without requiring spaces.
    sentences = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s*", cleaned)
    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0
    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if buffer:
                chunks.append(_join_sentences(buffer))
                buffer = []
                buffer_len = 0
            chunks.extend(_split_by_words(sentence, max_chars))
            continue
        separator_len = len(_sentence_separator(buffer[-1])) if buffer else 0
        if buffer and buffer_len + separator_len + len(sentence) > max_chars:
            chunks.append(_join_sentences(buffer))
            buffer = [sentence]
            buffer_len = len(sentence)
        else:
            buffer.append(sentence)
            buffer_len += separator_len + len(sentence)
    if buffer:
        chunks.append(_join_sentences(buffer))
    return chunks


def normalize_language(language: Optional[str]) -> Optional[str]:
    normalized = (language or "").strip()
    if normalized.lower() in {"", "auto", "auto (omit)", "none"}:
        return None
    return normalized


def validate_sampling(
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    max_chars: int,
    gap_ms: int,
) -> None:
    if not 0.0 <= temperature <= 5.0:
        raise WorkerError("INVALID_ARGUMENT", "temperature must be between 0 and 5.")
    if not 0.0 < top_p <= 1.0:
        raise WorkerError("INVALID_ARGUMENT", "top_p must be greater than 0 and at most 1.")
    if top_k < 1:
        raise WorkerError("INVALID_ARGUMENT", "top_k must be at least 1.")
    if repetition_penalty <= 0:
        raise WorkerError("INVALID_ARGUMENT", "repetition_penalty must be greater than 0.")
    if max_new_tokens < 1:
        raise WorkerError("INVALID_ARGUMENT", "max_new_tokens must be at least 1.")
    if max_chars < 50:
        raise WorkerError("INVALID_ARGUMENT", "max_chars must be at least 50.")
    if gap_ms < 0:
        raise WorkerError("INVALID_ARGUMENT", "gap_ms must be zero or greater.")


def resolve_device(torch: Any, requested: Optional[str]) -> Any:
    requested_device = (requested or "").strip()
    if not requested_device:
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise WorkerError(
            "DEVICE_UNAVAILABLE",
            f"MOSS_TTS_DEVICE={requested_device!r} requests CUDA, but CUDA is unavailable.",
        )
    try:
        return torch.device(requested_device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise WorkerError(
            "INVALID_ARGUMENT",
            f"Invalid MOSS-TTS device {requested_device!r}.",
        ) from exc


def resolve_attn_implementation(torch: Any, requested: str, device: Any, dtype: Any) -> Optional[str]:
    normalized = (requested or "auto").strip().lower()
    if normalized == "none":
        return None
    if normalized not in {"", "auto"}:
        return requested
    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "flash_attention_2"
    if device.type == "cuda":
        return "sdpa"
    return "eager"


def _configure_torch_attention(torch: Any) -> None:
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        return
    for method_name, enabled in (
        ("enable_cudnn_sdp", False),
        ("enable_flash_sdp", True),
        ("enable_mem_efficient_sdp", True),
        ("enable_math_sdp", True),
    ):
        method = getattr(cuda_backend, method_name, None)
        if callable(method):
            method(enabled)


def build_user_message(
    processor: Any,
    text: str,
    reference_audio: Optional[str],
    language: Optional[str],
) -> Any:
    kwargs: Dict[str, Any] = {"text": text}
    if reference_audio:
        kwargs["reference"] = [reference_audio]
    normalized_language = normalize_language(language)
    if normalized_language:
        kwargs["language"] = normalized_language
    return processor.build_user_message(**kwargs)


def load_backend(
    model_snapshot: Path,
    codec_snapshot: Path,
    requested_device: Optional[str],
    requested_attention: str,
) -> tuple[Any, Any, Any, int]:
    ensure_dependencies()
    enable_offline_mode()
    import torch  # type: ignore
    from transformers import AutoModel, AutoProcessor  # type: ignore

    _configure_torch_attention(torch)
    device = resolve_device(torch, requested_device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    attention = resolve_attn_implementation(torch, requested_attention, device, dtype)
    emit(
        "progress",
        progress=0.05,
        message=f"Loading pinned MOSS-TTS v1.5 snapshots on {device} "
        f"(attention={attention or 'default'})",
    )
    try:
        # The pinned custom processor forwards unconsumed loader kwargs into
        # ProcessorMixin, which rejects local_files_only in Transformers 5.
        # Both paths are local and offline mode above provides the same hard
        # network boundary for its nested tokenizer/codec loads.
        processor = AutoProcessor.from_pretrained(
            str(model_snapshot),
            codec_path=str(codec_snapshot),
            trust_remote_code=True,
        )
        if not hasattr(processor, "audio_tokenizer"):
            raise RuntimeError("MOSS processor did not expose its audio tokenizer")
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        processor.audio_tokenizer.eval()

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
            "torch_dtype": dtype,
        }
        if attention:
            model_kwargs["attn_implementation"] = attention
        model = AutoModel.from_pretrained(str(model_snapshot), **model_kwargs).to(device)
        model.eval()
    except Exception as exc:
        raise WorkerError(
            "MODEL_LOAD_FAILED",
            f"Failed to load the pinned local MOSS-TTS model and codec: {exc}",
            details={
                "model_snapshot_path": str(model_snapshot),
                "codec_snapshot_path": str(codec_snapshot),
            },
        ) from exc

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    if sample_rate <= 0:
        raise WorkerError(
            "INVALID_MODEL_OUTPUT",
            f"MOSS-TTS reported an invalid sample rate: {sample_rate}",
        )
    return model, processor, device, sample_rate


def synthesize(
    models_root: Path,
    text: str,
    output_path: str,
    voice_path: Optional[str],
    language: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    max_chars: int,
    gap_ms: int,
    requested_attention: str,
) -> None:
    validate_sampling(
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_new_tokens,
        max_chars,
        gap_ms,
    )
    segments = split_text(text, max_chars)
    if not segments:
        raise WorkerError("INVALID_ARGUMENT", "Text input is empty.")

    reference_audio: Optional[str] = None
    if voice_path:
        reference = Path(voice_path).expanduser().resolve()
        if not reference.is_file():
            raise WorkerError("FILE_NOT_FOUND", f"Voice reference not found: {reference}")
        reference_audio = str(reference)

    model_snapshot, codec_snapshot = load_pinned_snapshots(models_root)
    model, processor, device, sample_rate = load_backend(
        model_snapshot,
        codec_snapshot,
        os.environ.get("MOSS_TTS_DEVICE"),
        requested_attention,
    )
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore

    audio_segments: List[Any] = []
    total = len(segments)
    for index, segment in enumerate(segments, start=1):
        emit(
            "progress",
            progress=0.12 + (0.82 * index / total),
            message=f"Synthesizing MOSS segment {index}/{total}",
        )
        try:
            conversation = [[build_user_message(
                processor,
                segment,
                reference_audio,
                language,
            )]]
            batch = processor(conversation, mode="generation")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(max_new_tokens),
                    audio_temperature=float(temperature),
                    audio_top_p=float(top_p),
                    audio_top_k=int(top_k),
                    audio_repetition_penalty=float(repetition_penalty),
                )
            messages = processor.decode(outputs)
        except Exception as exc:
            raise WorkerError(
                "SYNTHESIS_FAILED",
                f"MOSS-TTS synthesis failed on segment {index}/{total}: {exc}",
                details={"segment": index, "segment_count": total},
            ) from exc

        if not messages or messages[0] is None:
            raise WorkerError(
                "INVALID_MODEL_OUTPUT",
                f"MOSS-TTS returned no decodable message for segment {index}.",
            )
        audio_codes = getattr(messages[0], "audio_codes_list", None)
        if not audio_codes:
            raise WorkerError(
                "INVALID_MODEL_OUTPUT",
                f"MOSS-TTS returned no audio for segment {index}.",
            )
        audio = audio_codes[0]
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().float().cpu().numpy()
        else:
            audio_np = np.asarray(audio, dtype=np.float32)
        audio_np = np.asarray(audio_np, dtype=np.float32).squeeze()
        if audio_np.ndim != 1 or audio_np.size == 0:
            raise WorkerError(
                "INVALID_MODEL_OUTPUT",
                f"MOSS-TTS returned an invalid waveform for segment {index}.",
            )
        audio_segments.append(audio_np)

    if gap_ms > 0 and len(audio_segments) > 1:
        silence = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
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
        message="MOSS-TTS v1.5 synthesis complete",
        output_path=str(output),
        model_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        segments=total,
        sample_rate=sample_rate,
        used_voice_reference=bool(reference_audio),
        language=normalize_language(language) or "auto",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pinned, offline MOSS-TTS v1.5 worker")
    parser.add_argument(
        "--root-dir",
        default=str(Path.home() / ".voiceforge" / "moss_tts"),
    )
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--jobs-dir", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    subparsers.add_parser("setup")
    subparsers.add_parser("download")

    synth_parser = subparsers.add_parser("synthesize", help="Run local MOSS-TTS synthesis")
    synth_parser.add_argument("--text", required=True, help="UTF-8 text file")
    synth_parser.add_argument("--output", required=True)
    synth_parser.add_argument("--voice", "--reference-audio", dest="voice", default=None)
    synth_parser.add_argument(
        "--language",
        default=os.environ.get("MOSS_TTS_LANGUAGE", "Auto"),
    )
    synth_parser.add_argument(
        "--temperature",
        "--audio-temperature",
        dest="temperature",
        type=float,
        default=float(os.environ.get("MOSS_TTS_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
    )
    synth_parser.add_argument(
        "--top-p",
        "--audio-top-p",
        dest="top_p",
        type=float,
        default=float(os.environ.get("MOSS_TTS_TOP_P", str(DEFAULT_TOP_P))),
    )
    synth_parser.add_argument(
        "--top-k",
        "--audio-top-k",
        dest="top_k",
        type=int,
        default=int(os.environ.get("MOSS_TTS_TOP_K", str(DEFAULT_TOP_K))),
    )
    synth_parser.add_argument(
        "--repetition-penalty",
        "--audio-repetition-penalty",
        dest="repetition_penalty",
        type=float,
        default=float(
            os.environ.get(
                "MOSS_TTS_REPETITION_PENALTY",
                str(DEFAULT_REPETITION_PENALTY),
            )
        ),
    )
    synth_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(
            os.environ.get("MOSS_TTS_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS))
        ),
    )
    synth_parser.add_argument(
        "--max-chars",
        type=int,
        default=int(os.environ.get("MOSS_TTS_MAX_CHARS", "800")),
    )
    synth_parser.add_argument(
        "--gap-ms",
        type=int,
        default=int(os.environ.get("MOSS_TTS_GAP_MS", "120")),
    )
    synth_parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("MOSS_TTS_ATTN_IMPLEMENTATION", "auto"),
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
        setup_models(models_root)
        return
    if args.command == "check":
        ensure_dependencies()
        model_snapshot, codec_snapshot = load_pinned_snapshots(models_root)
        emit(
            "complete",
            progress=1.0,
            message="MOSS-TTS v1.5 runtime and pinned model data are ready",
            ready=True,
            model_id=MODEL_REPO_ID,
            revision=MODEL_REVISION,
            model_snapshot_path=str(model_snapshot),
            codec_id=CODEC_REPO_ID,
            codec_revision=CODEC_REVISION,
            codec_snapshot_path=str(codec_snapshot),
        )
        return
    if args.command == "synthesize":
        text_path = Path(args.text).expanduser().resolve()
        if not text_path.is_file():
            raise WorkerError("FILE_NOT_FOUND", f"Text file not found: {text_path}")
        text = text_path.read_text(encoding="utf-8")
        synthesize(
            models_root,
            text,
            args.output,
            args.voice,
            args.language,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repetition_penalty,
            args.max_new_tokens,
            args.max_chars,
            args.gap_ms,
            args.attn_implementation,
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
            error="MOSS-TTS operation cancelled",
            error_type="KeyboardInterrupt",
            details={},
        )
        raise SystemExit(130)
    except Exception as exc:
        emit("error", message="MOSS-TTS operation failed", **_error_payload(exc))
        if is_truthy(os.environ.get("VOICEFORGE_WORKER_DEBUG")):
            traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
