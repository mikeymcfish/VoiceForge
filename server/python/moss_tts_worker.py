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
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

try:
    from .chapter_markers import (
        AudioMergeTiming,
        ChapterSegment,
        build_audio_merge_timing,
        build_chapter_manifest,
        split_marked_text,
        write_chapter_manifest,
    )
except ImportError:
    from chapter_markers import (
        AudioMergeTiming,
        ChapterSegment,
        build_audio_merge_timing,
        build_chapter_manifest,
        split_marked_text,
        write_chapter_manifest,
    )


MODEL_REPO_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"
MODEL_REVISION = "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"
CODEC_REPO_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
CODEC_REVISION = "3cd226ba2947efa357ef453bcad111b6eafba782"
LOCAL_MODEL_REPO_ID = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
LOCAL_MODEL_REVISION = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
LOCAL_CODEC_REPO_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
LOCAL_CODEC_REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
DEFAULT_MODEL_ID = MODEL_REPO_ID
PINNED_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    MODEL_REPO_ID: {
        "label": "MOSS-TTS v1.5 8B",
        "manifest_name": ".voiceforge-moss-models.json",
        "local_transformer": False,
        "artifacts": {
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
        },
    },
    LOCAL_MODEL_REPO_ID: {
        "label": "MOSS-TTS Local-Transformer v1.5",
        "manifest_name": ".voiceforge-moss-local-v1.5-models.json",
        "local_transformer": True,
        "artifacts": {
            "model": {
                "repo_id": LOCAL_MODEL_REPO_ID,
                "revision": LOCAL_MODEL_REVISION,
                "minimum_bytes": 1024 * 1024 * 1024,
            },
            "codec": {
                "repo_id": LOCAL_CODEC_REPO_ID,
                "revision": LOCAL_CODEC_REVISION,
                "minimum_bytes": 1024 * 1024 * 1024,
            },
        },
    },
}
# Keep the original public mapping as an alias for tests and integrations that
# inspect or temporarily lower the default checkpoint's minimum-size guards.
PINNED_ARTIFACTS = PINNED_MODEL_SPECS[DEFAULT_MODEL_ID]["artifacts"]
TRANSFORMERS_PACKAGE = "transformers==5.0.0"
MODEL_MANIFEST_NAME = str(PINNED_MODEL_SPECS[DEFAULT_MODEL_ID]["manifest_name"])
MODEL_MANIFEST_VERSION = 1
MODEL_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TEMPERATURE = 1.3
DEFAULT_TOP_P = 0.75
DEFAULT_TOP_K = 25
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_DURATION_OUTLIER_RETRIES = 0
DEFAULT_DURATION_OUTLIER_RATIO = 1.35
REVIEW_MANIFEST_VERSION = 2
DURATION_RATE_HISTORY_SIZE = 5
MIN_DURATION_RATE_HISTORY = 2
MIN_DURATION_RATE_SPEECH_UNITS = 80
MIN_DURATION_RATE_SECONDS = 2.0
DEFAULT_REANCHOR_EVERY_SEGMENTS = 6
REFERENCE_PROFILE_MAX_SECONDS = 90
REFERENCE_PROFILE_MAX_FRAMES = 180
REFERENCE_PROFILE_FRAME_MS = 60
REFERENCE_PROFILE_BAND_COUNT = 24
REFERENCE_CLOSENESS_WARNING_SCORE = 60.0
REFERENCE_CLOSENESS_WARNING_DROP = 12.0
REFERENCE_CLOSENESS_HISTORY_SIZE = 8


class RollingPrefix(NamedTuple):
    transcript: str
    audio_path: str


class DurationOutlier(NamedTuple):
    baseline_rate: float
    rate_ratio: float


class ReferenceCloseness(NamedTuple):
    score: float
    status: str
    baseline: Optional[float]


class StreamingWaveWriter:
    """Append MOSS waveforms to a temporary WAV without retaining prior audio.

    Long-form synthesis previously held every segment and then allocated a
    second, combined NumPy array at the end.  Keeping only the current segment
    makes the final assembly scale with disk space instead of RAM.  The caller
    must call :meth:`finalize` only after all validation has succeeded; failures
    leave the destination untouched.
    """

    def __init__(self, output_path: Path, sample_rate: int) -> None:
        self.output_path = output_path
        self.sample_rate = sample_rate
        self._temporary_path: Optional[Path] = None
        self._writer: Any = None
        self.channels: Optional[int] = None
        self.total_samples = 0

    def write(self, audio: Any, gap_samples: int = 0) -> None:
        """Write one channel-first waveform and any silence before it."""

        if gap_samples < 0:
            raise WorkerError("INVALID_ARGUMENT", "Audio gap samples cannot be negative.")
        channels = int(audio.shape[0])
        if channels not in {1, 2}:
            raise WorkerError(
                "INVALID_MODEL_OUTPUT",
                "MOSS-TTS returned an unsupported channel count.",
            )
        if self.channels is None:
            import soundfile as sf  # type: ignore

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=f".{self.output_path.name}.",
                suffix=".wav",
                dir=self.output_path.parent,
                delete=False,
            ) as handle:
                self._temporary_path = Path(handle.name)
            try:
                self._writer = sf.SoundFile(
                    str(self._temporary_path),
                    mode="w",
                    samplerate=self.sample_rate,
                    channels=channels,
                    format="WAV",
                    # Match soundfile.write(..., format="WAV") used by the
                    # previous in-memory merge, preserving the final file's
                    # PCM-16 encoding and size.
                    subtype="PCM_16",
                )
            except Exception:
                self.discard()
                raise
            self.channels = channels
        elif channels != self.channels:
            raise WorkerError(
                "INVALID_MODEL_OUTPUT",
                "MOSS-TTS changed channel count between generated segments.",
            )

        if self._writer is None:
            raise WorkerError("INVALID_MODEL_OUTPUT", "MOSS-TTS output writer did not initialize.")
        if gap_samples:
            import numpy as np  # type: ignore

            self._writer.write(
                np.zeros((gap_samples, self.channels), dtype=np.float32)
            )
        self._writer.write(soundfile_waveform(audio))
        self.total_samples += gap_samples + int(audio.shape[1])

    def finalize(self) -> None:
        """Close and atomically publish the completed WAV."""

        if self._writer is None or self._temporary_path is None:
            raise WorkerError("INVALID_MODEL_OUTPUT", "MOSS-TTS returned no audio segments.")
        self._writer.close()
        self._writer = None
        temporary_path = self._temporary_path
        try:
            temporary_path.replace(self.output_path)
        except Exception:
            # Keep the path tracked so the caller's discard path can clean up
            # a closed but unpublished temporary file.
            raise
        self._temporary_path = None

    def discard(self) -> None:
        """Close and remove an incomplete temporary WAV, if present."""

        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None
        if self._temporary_path is not None:
            self._temporary_path.unlink(missing_ok=True)
            self._temporary_path = None


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


def _model_spec(model_id: str) -> Dict[str, Any]:
    normalized = (model_id or "").strip()
    spec = PINNED_MODEL_SPECS.get(normalized)
    if spec is None:
        raise WorkerError(
            "UNSUPPORTED_MODEL",
            f"Unsupported MOSS-TTS checkpoint: {model_id!r}.",
            details={"allowed_models": sorted(PINNED_MODEL_SPECS)},
        )
    return spec


def _pinned_artifacts(model_id: str = DEFAULT_MODEL_ID) -> Dict[str, Dict[str, Any]]:
    return _model_spec(model_id)["artifacts"]


def _manifest_path(models_root: Path, model_id: str = DEFAULT_MODEL_ID) -> Path:
    return models_root / str(_model_spec(model_id)["manifest_name"])


def _expected_snapshot_path(
    models_root: Path,
    artifact_name: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> Path:
    artifacts = _pinned_artifacts(model_id)
    if artifact_name not in artifacts:
        raise WorkerError("INVALID_ARGUMENT", f"Unknown MOSS artifact: {artifact_name}")
    artifact = artifacts[artifact_name]
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
    model_id: str = DEFAULT_MODEL_ID,
) -> List[Dict[str, Any]]:
    root = models_root.resolve()
    snapshot = snapshot_path.resolve()
    artifacts = _pinned_artifacts(model_id)
    if artifact_name not in artifacts:
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
    minimum_bytes = int(artifacts[artifact_name]["minimum_bytes"])
    if total_size < minimum_bytes:
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            f"The pinned MOSS {artifact_name} snapshot is incomplete (files are too small).",
            details={"bytes": total_size, "minimum_bytes": minimum_bytes},
        )
    return inventory


def _write_manifest(
    models_root: Path,
    snapshots: Dict[str, Path],
    model_id: str = DEFAULT_MODEL_ID,
) -> None:
    artifacts = _pinned_artifacts(model_id)
    entries: Dict[str, Any] = {}
    for artifact_name, artifact in artifacts.items():
        if artifact_name not in snapshots:
            raise WorkerError(
                "INCOMPLETE_MODEL_SNAPSHOT",
                f"Missing downloaded MOSS artifact: {artifact_name}",
            )
        snapshot = snapshots[artifact_name].resolve()
        expected = _expected_snapshot_path(models_root, artifact_name, model_id)
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
            "files": _snapshot_inventory(
                models_root,
                snapshot,
                artifact_name,
                model_id,
            ),
        }

    manifest = {
        "manifest_version": MODEL_MANIFEST_VERSION,
        "model_id": model_id,
        "artifacts": entries,
    }
    path = _manifest_path(models_root, model_id)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_manifest(
    models_root: Path,
    model_id: str = DEFAULT_MODEL_ID,
) -> Dict[str, Any]:
    path = _manifest_path(models_root, model_id)
    if not path.is_file():
        raise WorkerError(
            "MODEL_NOT_READY",
            f"Pinned {_model_spec(model_id)['label']} data has not been downloaded. "
            "Run setup for this checkpoint first.",
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
    model_id: str = DEFAULT_MODEL_ID,
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
        for item in _snapshot_inventory(
            models_root,
            snapshot_path,
            artifact_name,
            model_id,
        )
    }
    if actual != expected:
        raise WorkerError(
            "INCOMPLETE_MODEL_SNAPSHOT",
            f"The pinned MOSS {artifact_name} snapshot does not match its manifest; run setup again.",
        )


def load_pinned_snapshots(
    models_root: Path,
    model_id: str = DEFAULT_MODEL_ID,
) -> Tuple[Path, Path]:
    artifacts_spec = _pinned_artifacts(model_id)
    manifest = _read_manifest(models_root, model_id)
    artifacts = manifest["artifacts"]
    resolved: Dict[str, Path] = {}
    for artifact_name, expected_artifact in artifacts_spec.items():
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
        if snapshot != _expected_snapshot_path(models_root, artifact_name, model_id):
            raise WorkerError(
                "UNEXPECTED_MODEL_REVISION",
                f"The MOSS {artifact_name} manifest does not point to the pinned snapshot path.",
            )
        _validate_manifest_inventory(
            models_root,
            snapshot,
            artifact_name,
            raw_entry.get("files"),
            model_id,
        )
        resolved[artifact_name] = snapshot
    return resolved["model"], resolved["codec"]


def setup_models(
    models_root: Path,
    model_id: str = DEFAULT_MODEL_ID,
) -> Tuple[Path, Path]:
    spec = _model_spec(model_id)
    artifacts = _pinned_artifacts(model_id)
    try:
        from huggingface_hub import snapshot_download
    except (ImportError, OSError) as exc:
        raise WorkerError(
            "RUNTIME_NOT_CONFIGURED",
            "huggingface_hub is required to download pinned MOSS-TTS model data.",
        ) from exc

    snapshots: Dict[str, Path] = {}
    progress_ranges = {"model": (0.05, 0.48), "codec": (0.52, 0.88)}
    for artifact_name, artifact in artifacts.items():
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
    _write_manifest(models_root, snapshots, model_id)
    model_snapshot, codec_snapshot = load_pinned_snapshots(models_root, model_id)
    model_artifact = artifacts["model"]
    codec_artifact = artifacts["codec"]
    emit(
        "complete",
        progress=1.0,
        message=f"{spec['label']} setup complete",
        model_id=model_id,
        revision=model_artifact["revision"],
        model_snapshot_path=str(model_snapshot),
        codec_id=codec_artifact["repo_id"],
        codec_revision=codec_artifact["revision"],
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


def split_text_with_chapters(text: str, max_chars: int) -> List[ChapterSegment]:
    """Split marked text while preserving the original ``split_text`` API."""

    return split_marked_text(text, lambda section: split_text(section, max_chars))


def split_synthesis_text(
    text: str,
    max_chars: int,
    use_chapter_markers: bool = False,
) -> List[ChapterSegment]:
    """Keep chapter syntax opt-in so legacy text is spoken unchanged."""

    if use_chapter_markers:
        return split_text_with_chapters(text, max_chars)
    return [ChapterSegment(segment) for segment in split_text(text, max_chars)]


def normalize_language(language: Optional[str]) -> Optional[str]:
    normalized = (language or "").strip()
    if normalized.lower() in {"", "auto", "auto (omit)", "none"}:
        return None
    return normalized


def speech_unit_count(text: str) -> int:
    """Count language-neutral visible speech units for relative pace checks."""
    return sum(1 for character in text if character.isalnum())


def segment_speaking_rate(
    text: str,
    audio: Any,
    sample_rate: int,
) -> Optional[float]:
    """Return comparable speech units/second when a segment is suitable."""
    if sample_rate <= 0 or re.search(r"\[\s*pause\b", text, flags=re.IGNORECASE):
        return None
    speech_units = speech_unit_count(text)
    if speech_units < MIN_DURATION_RATE_SPEECH_UNITS:
        return None
    duration_seconds = int(audio.shape[1]) / float(sample_rate)
    if duration_seconds < MIN_DURATION_RATE_SECONDS:
        return None
    return speech_units / duration_seconds


def recent_rate_baseline(rate_history: Sequence[float]) -> Optional[float]:
    recent = sorted(rate_history[-DURATION_RATE_HISTORY_SIZE:])
    if len(recent) < MIN_DURATION_RATE_HISTORY:
        return None
    midpoint = len(recent) // 2
    if len(recent) % 2:
        return recent[midpoint]
    return (recent[midpoint - 1] + recent[midpoint]) / 2.0


def duration_outlier_analysis(
    rate: Optional[float],
    rate_history: Sequence[float],
    maximum_ratio: float,
) -> Optional[DurationOutlier]:
    if rate is None or rate <= 0:
        return None
    baseline = recent_rate_baseline(rate_history)
    if baseline is None or baseline <= 0:
        return None
    ratio = rate / baseline
    if 1.0 / maximum_ratio <= ratio <= maximum_ratio:
        return None
    return DurationOutlier(baseline, ratio)


def duration_candidate_is_closer(
    candidate_rate: Optional[float],
    current_rate: Optional[float],
    baseline_rate: float,
) -> bool:
    if candidate_rate is None or candidate_rate <= 0 or baseline_rate <= 0:
        return False
    if current_rate is None or current_rate <= 0:
        return True
    candidate_deviation = max(
        candidate_rate / baseline_rate,
        baseline_rate / candidate_rate,
    )
    current_deviation = max(
        current_rate / baseline_rate,
        baseline_rate / current_rate,
    )
    return candidate_deviation < current_deviation


def validate_duration_outlier_controls(retries: int, maximum_ratio: float) -> None:
    if not 0 <= retries <= 3:
        raise WorkerError(
            "INVALID_ARGUMENT",
            "duration_outlier_retries must be between 0 and 3.",
        )
    if not 1.05 <= maximum_ratio <= 3.0:
        raise WorkerError(
            "INVALID_ARGUMENT",
            "duration_outlier_ratio must be between 1.05 and 3.",
        )


def should_reanchor_to_reference(
    segment_index: int,
    starts_chapter: bool,
    has_reference_audio: bool,
) -> bool:
    """Whether this segment starts fresh from the supplied voice reference.

    A rolling MOSS continuation keeps prosody smooth, but repeatedly conditioning
    on generated audio can amplify artifacts over a long run.  Clone jobs retain
    that continuity within a block and re-anchor on every marked chapter plus
    after six continuation segments.  Direct-generation jobs have no clean
    reference to return to, so their legacy continuation behavior is preserved.
    """

    if not has_reference_audio or segment_index <= 0:
        return False
    return starts_chapter or segment_index % DEFAULT_REANCHOR_EVERY_SEGMENTS == 0


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def reference_acoustic_profile(
    audio: Any,
    sample_rate: int,
    np: Any,
) -> Optional[Any]:
    """Build a compact, text-independent-ish acoustic profile for drift hints.

    This intentionally does *not* claim biometric speaker verification.  It
    compares broad timbre and pitch-distribution features from the clean reference
    against generated speech, which makes it useful for flagging a changing voice
    without an additional speaker-recognition model or a network dependency.
    """

    if sample_rate <= 0:
        return None
    waveform = normalize_audio_channels(audio)
    mono = np.asarray(waveform, dtype=np.float32).mean(axis=0)
    if mono.size < max(512, sample_rate // 4):
        return None
    mono = mono[np.isfinite(mono)]
    if mono.size < max(512, sample_rate // 4):
        return None
    mono = mono - float(np.mean(mono))
    peak = float(np.max(np.abs(mono)))
    if peak < 1e-5:
        return None
    mono /= peak

    frame_size = max(512, int(sample_rate * REFERENCE_PROFILE_FRAME_MS / 1000))
    if mono.size < frame_size:
        return None
    start_limit = mono.size - frame_size
    frame_count = min(
        REFERENCE_PROFILE_MAX_FRAMES,
        max(1, int(mono.size / max(frame_size // 2, 1))),
    )
    starts = np.linspace(0, start_limit, num=frame_count, dtype=np.int64)
    frames = np.stack([mono[start : start + frame_size] for start in starts])
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    active_floor = max(1e-4, float(np.median(rms)) * 0.35)
    active_frames = frames[rms >= active_floor]
    if active_frames.shape[0] < 4:
        active_frames = frames
    if active_frames.shape[0] < 2:
        return None

    window = np.hanning(frame_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(active_frames * window, axis=1)) + 1e-7
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    highest_frequency = min(8_000.0, float(sample_rate) * 0.45)
    if highest_frequency <= 100.0:
        return None
    band_edges = np.geomspace(80.0, highest_frequency, REFERENCE_PROFILE_BAND_COUNT + 1)
    log_band_energy: List[Any] = []
    for lower, upper in zip(band_edges[:-1], band_edges[1:]):
        band = spectrum[:, (frequencies >= lower) & (frequencies < upper)]
        if band.shape[1] == 0:
            log_band_energy.append(np.zeros(active_frames.shape[0], dtype=np.float32))
        else:
            log_band_energy.append(np.log(np.mean(np.square(band), axis=1) + 1e-8))
    bands = np.stack(log_band_energy, axis=1)
    mean_bands = np.mean(bands, axis=0)
    std_bands = np.std(bands, axis=0)
    mean_bands = mean_bands - float(np.mean(mean_bands))
    std_bands = std_bands - float(np.mean(std_bands))

    voiced = spectrum[:, (frequencies >= 70.0) & (frequencies <= 350.0)]
    voiced_frequencies = frequencies[(frequencies >= 70.0) & (frequencies <= 350.0)]
    if voiced.shape[1]:
        pitch = voiced_frequencies[np.argmax(voiced, axis=1)]
        log_pitch = np.log(np.maximum(pitch, 1.0))
        pitch_features = np.asarray(
            [float(np.median(log_pitch)), float(np.std(log_pitch))], dtype=np.float32
        )
    else:
        pitch_features = np.zeros(2, dtype=np.float32)

    # Pitch has less weight than the broad spectral envelope: different prose
    # naturally changes intonation, while a changing timbre is the useful signal.
    features = np.concatenate(
        [
            mean_bands.astype(np.float32),
            std_bands.astype(np.float32),
            pitch_features * 0.35,
        ]
    )
    magnitude = float(np.linalg.norm(features))
    if magnitude <= 1e-8:
        return None
    return features / magnitude


def load_reference_acoustic_profile(
    reference_audio: str,
    sf: Any,
    np: Any,
) -> Optional[Any]:
    """Read at most a short clean-reference window for bounded diagnostics."""

    try:
        with sf.SoundFile(reference_audio) as handle:
            maximum_frames = max(1, int(handle.samplerate * REFERENCE_PROFILE_MAX_SECONDS))
            audio = handle.read(
                frames=min(len(handle), maximum_frames),
                dtype="float32",
                always_2d=True,
            )
            return reference_acoustic_profile(audio, int(handle.samplerate), np)
    except Exception:
        # MOSS itself may support a reference format that SoundFile cannot read.
        # Diagnostics must never make an otherwise valid clone job fail.
        return None


def reference_closeness(
    reference_profile: Optional[Any],
    audio: Any,
    sample_rate: int,
    history: Sequence[float],
    np: Any,
) -> Optional[ReferenceCloseness]:
    """Score broad acoustic similarity to the clean reference on a 0-100 scale."""

    if reference_profile is None:
        return None
    generated_profile = reference_acoustic_profile(audio, sample_rate, np)
    if generated_profile is None or generated_profile.shape != reference_profile.shape:
        return None
    cosine = float(np.dot(reference_profile, generated_profile))
    score = max(0.0, min(100.0, (cosine + 1.0) * 50.0))
    baseline = _median(history[-REFERENCE_CLOSENESS_HISTORY_SIZE:])
    drifted = score < REFERENCE_CLOSENESS_WARNING_SCORE or (
        baseline is not None and score < baseline - REFERENCE_CLOSENESS_WARNING_DROP
    )
    return ReferenceCloseness(
        score=score,
        status="drift-warning" if drifted else "typical",
        baseline=baseline,
    )


def _review_audio_path(manifest_path: Path, audio_file: object) -> Path:
    if not isinstance(audio_file, str) or not audio_file.strip():
        raise WorkerError(
            "INVALID_REVIEW_MANIFEST",
            "A review segment has no audio file.",
        )
    relative = Path(audio_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise WorkerError(
            "INVALID_REVIEW_MANIFEST",
            "A review segment contains an unsafe audio path.",
        )
    root = manifest_path.parent.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WorkerError(
            "INVALID_REVIEW_MANIFEST",
            "A review segment points outside the managed job directory.",
        ) from exc
    return resolved


def validate_review_manifest(
    manifest: object,
    manifest_path: Path | str,
    *,
    require_audio: bool = True,
) -> Dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    if not isinstance(manifest, dict):
        raise WorkerError(
            "INVALID_REVIEW_MANIFEST",
            "The MOSS review manifest must be a JSON object.",
        )
    required_top_level = {
        "version",
        "sample_rate",
        "gap_ms",
        "chapter_pause_ms",
        "segments",
    }
    if set(manifest) != required_top_level:
        raise WorkerError(
            "INVALID_REVIEW_MANIFEST",
            "The MOSS review manifest has unexpected fields.",
        )
    version = manifest.get("version")
    if version not in {1, REVIEW_MANIFEST_VERSION}:
        raise WorkerError(
            "INVALID_REVIEW_MANIFEST",
            "The MOSS review manifest version is unsupported.",
        )
    for field in ("sample_rate", "gap_ms", "chapter_pause_ms"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise WorkerError(
                "INVALID_REVIEW_MANIFEST",
                f"The MOSS review manifest {field} value is invalid.",
            )
    if int(manifest["sample_rate"]) <= 0:
        raise WorkerError(
            "INVALID_REVIEW_MANIFEST",
            "The MOSS review sample rate must be positive.",
        )
    if int(manifest["gap_ms"]) < 0 or int(manifest["chapter_pause_ms"]) < 0:
        raise WorkerError(
            "INVALID_REVIEW_MANIFEST",
            "MOSS review pauses cannot be negative.",
        )

    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise WorkerError(
            "INVALID_REVIEW_MANIFEST",
            "The MOSS review manifest contains no segments.",
        )
    required_segment_fields = {
        "index",
        "text",
        "audio_file",
        "sample_count",
        "duration_seconds",
        "attempt",
        "starts_chapter",
        "chapter_title",
        "speaking_rate",
        "pace_ratio",
        "pace_status",
        "updated_at",
    }
    if version == REVIEW_MANIFEST_VERSION:
        required_segment_fields.update({
            "reference_closeness",
            "reference_drift_status",
        })
    for expected_index, segment in enumerate(segments):
        if not isinstance(segment, dict) or set(segment) != required_segment_fields:
            raise WorkerError(
                "INVALID_REVIEW_MANIFEST",
                "A MOSS review segment has unexpected fields.",
            )
        if segment.get("index") != expected_index:
            raise WorkerError(
                "INVALID_REVIEW_MANIFEST",
                "MOSS review segment indices are not contiguous.",
            )
        if not isinstance(segment.get("text"), str) or not segment["text"].strip():
            raise WorkerError(
                "INVALID_REVIEW_MANIFEST",
                "A MOSS review segment has no text.",
            )
        for field in ("sample_count", "attempt", "updated_at"):
            value = segment.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise WorkerError(
                    "INVALID_REVIEW_MANIFEST",
                    f"A MOSS review segment has an invalid {field} value.",
                )
        duration = segment.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            raise WorkerError(
                "INVALID_REVIEW_MANIFEST",
                "A MOSS review segment has an invalid duration.",
            )
        if not isinstance(segment.get("starts_chapter"), bool):
            raise WorkerError(
                "INVALID_REVIEW_MANIFEST",
                "A MOSS review segment has invalid chapter state.",
            )
        chapter_title = segment.get("chapter_title")
        if chapter_title is not None and (
            not isinstance(chapter_title, str) or not chapter_title.strip()
        ):
            raise WorkerError(
                "INVALID_REVIEW_MANIFEST",
                "A MOSS review segment has an invalid chapter title.",
            )
        for field in ("speaking_rate", "pace_ratio"):
            value = segment.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise WorkerError(
                    "INVALID_REVIEW_MANIFEST",
                    f"A MOSS review segment has an invalid {field} value.",
                )
        if segment.get("pace_status") not in {
            "typical",
            "unusually-fast",
            "unusually-slow",
            "not-compared",
        }:
            raise WorkerError(
                "INVALID_REVIEW_MANIFEST",
                "A MOSS review segment has an invalid pace status.",
            )
        if version == REVIEW_MANIFEST_VERSION:
            closeness = segment.get("reference_closeness")
            if closeness is not None and (
                isinstance(closeness, bool)
                or not isinstance(closeness, (int, float))
                or not 0.0 <= float(closeness) <= 100.0
            ):
                raise WorkerError(
                    "INVALID_REVIEW_MANIFEST",
                    "A MOSS review segment has an invalid reference closeness score.",
                )
            if segment.get("reference_drift_status") not in {
                "typical",
                "drift-warning",
                "not-compared",
            }:
                raise WorkerError(
                    "INVALID_REVIEW_MANIFEST",
                    "A MOSS review segment has an invalid reference drift status.",
                )
        audio_path = _review_audio_path(path, segment.get("audio_file"))
        if require_audio and not audio_path.is_file():
            raise WorkerError(
                "FILE_NOT_FOUND",
                f"MOSS review audio is missing for segment {expected_index + 1}.",
            )
    return manifest


def read_review_manifest(path: Path | str) -> Dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise WorkerError(
            "FILE_NOT_FOUND",
            f"MOSS review manifest not found: {manifest_path}",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerError(
            "INVALID_REVIEW_MANIFEST",
            "The MOSS review manifest is unreadable.",
        ) from exc
    return validate_review_manifest(manifest, manifest_path)


def write_review_manifest(
    path: Path | str,
    manifest: Dict[str, Any],
) -> Path:
    destination = Path(path).expanduser().resolve()
    validate_review_manifest(manifest, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
        temporary_path.replace(destination)
        return destination
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def upgrade_review_manifest(manifest: Dict[str, Any]) -> None:
    """Keep existing review jobs usable after adding reference diagnostics."""

    if manifest.get("version") != 1:
        return
    manifest["version"] = REVIEW_MANIFEST_VERSION
    for segment in manifest["segments"]:
        segment["reference_closeness"] = None
        segment["reference_drift_status"] = "not-compared"


def pace_metadata(
    rate: Optional[float],
    rate_history: Sequence[float],
    maximum_ratio: float,
) -> tuple[Optional[float], str]:
    if rate is None:
        return None, "not-compared"
    baseline = recent_rate_baseline(rate_history)
    if baseline is None or baseline <= 0:
        return None, "not-compared"
    ratio = rate / baseline
    if ratio > maximum_ratio:
        return ratio, "unusually-fast"
    if ratio < 1.0 / maximum_ratio:
        return ratio, "unusually-slow"
    return ratio, "typical"


def refresh_review_pace_metadata(
    segments: Sequence[Dict[str, Any]],
    maximum_ratio: float,
) -> None:
    history: List[float] = []
    for segment in segments:
        raw_rate = segment.get("speaking_rate")
        rate = float(raw_rate) if raw_rate is not None else None
        pace_ratio, pace_status = pace_metadata(
            rate,
            history,
            maximum_ratio,
        )
        segment["pace_ratio"] = pace_ratio
        segment["pace_status"] = pace_status
        if rate is not None and pace_status not in {
            "unusually-fast",
            "unusually-slow",
        }:
            history.append(rate)


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


def build_segment_conversation(
    processor: Any,
    segment: str,
    reference_audio: Optional[str],
    language: Optional[str],
    rolling_prefix: Optional[RollingPrefix] = None,
) -> tuple[List[List[Any]], str]:
    if rolling_prefix is None:
        return [[build_user_message(
            processor,
            segment,
            reference_audio,
            language,
        )]], "generation"

    if not rolling_prefix.transcript or not rolling_prefix.audio_path:
        raise ValueError("A rolling MOSS prefix requires transcript and audio.")
    continuation_text = (
        rolling_prefix.transcript
        + _sentence_separator(rolling_prefix.transcript)
        + segment
    )
    conversation = [[
        build_user_message(
            processor,
            continuation_text,
            reference_audio,
            language,
        ),
        processor.build_assistant_message(
            audio_codes_list=[rolling_prefix.audio_path],
        ),
    ]]
    return conversation, "continuation"


def load_backend(
    model_snapshot: Path,
    codec_snapshot: Path,
    requested_device: Optional[str],
    requested_attention: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> tuple[Any, Any, Any, int]:
    spec = _model_spec(model_id)
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
        message=f"Loading pinned {spec['label']} snapshots on {device} "
        f"(attention={attention or 'default'})",
    )
    try:
        # The pinned custom processor forwards unconsumed loader kwargs into
        # ProcessorMixin, which rejects local_files_only in Transformers 5.
        # Both paths are local and offline mode above provides the same hard
        # network boundary for its nested tokenizer/codec loads.
        processor_kwargs: Dict[str, Any] = {
            "codec_path": str(codec_snapshot),
            "trust_remote_code": True,
        }
        if bool(spec["local_transformer"]):
            processor_kwargs.update(
                {
                    "codec_weight_dtype": (
                        "bf16" if device.type == "cuda" else "fp32"
                    ),
                    "codec_compute_dtype": dtype,
                    "codec_attention_implementation": (
                        "flash_attention_2"
                        if attention == "flash_attention_2"
                        else "sdpa"
                    ),
                }
            )
        processor = AutoProcessor.from_pretrained(
            str(model_snapshot),
            **processor_kwargs,
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
    if bool(spec["local_transformer"]) and sample_rate != 48_000:
        raise WorkerError(
            "INVALID_MODEL_OUTPUT",
            "MOSS-TTS Local-Transformer v1.5 did not report its pinned 48 kHz "
            f"sample rate (reported {sample_rate}).",
        )
    return model, processor, device, sample_rate


def normalize_audio_channels(audio: Any) -> Any:
    """Return float32 audio in channel-first ``[channels, samples]`` form."""
    import numpy as np  # type: ignore

    waveform = np.asarray(audio, dtype=np.float32)
    while waveform.ndim > 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.reshape(1, -1)
    elif waveform.ndim == 2:
        if waveform.shape[0] not in {1, 2}:
            if waveform.shape[1] in {1, 2}:
                waveform = waveform.T
            else:
                raise WorkerError(
                    "INVALID_MODEL_OUTPUT",
                    "MOSS-TTS returned audio with no recognizable channel axis.",
                )
    else:
        raise WorkerError(
            "INVALID_MODEL_OUTPUT",
            "MOSS-TTS returned audio with an unsupported waveform rank.",
        )
    if waveform.shape[0] not in {1, 2} or waveform.shape[1] == 0:
        raise WorkerError(
            "INVALID_MODEL_OUTPUT",
            "MOSS-TTS returned an empty or unsupported-channel waveform.",
        )
    if not np.isfinite(waveform).all():
        raise WorkerError(
            "INVALID_MODEL_OUTPUT",
            "MOSS-TTS returned non-finite waveform samples.",
        )
    return np.ascontiguousarray(waveform, dtype=np.float32)


def generate_segment_audio(
    model: Any,
    processor: Any,
    device: Any,
    torch: Any,
    np: Any,
    segment_text: str,
    reference_audio: Optional[str],
    language: Optional[str],
    rolling_prefix: Optional[RollingPrefix],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    segment_index: int,
    segment_count: int,
) -> Any:
    conversation, processor_mode = build_segment_conversation(
        processor,
        segment_text,
        reference_audio,
        language,
        rolling_prefix,
    )
    batch = processor(conversation, mode=processor_mode)
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
    if not messages or messages[0] is None:
        raise WorkerError(
            "INVALID_MODEL_OUTPUT",
            f"MOSS-TTS returned no decodable message for segment {segment_index}.",
        )
    audio_codes = getattr(messages[0], "audio_codes_list", None)
    if not audio_codes:
        raise WorkerError(
            "INVALID_MODEL_OUTPUT",
            f"MOSS-TTS returned no audio for segment {segment_index}.",
        )
    audio = audio_codes[0]
    if isinstance(audio, torch.Tensor):
        candidate = audio.detach().float().cpu().numpy()
    else:
        candidate = np.asarray(audio, dtype=np.float32)
    try:
        return normalize_audio_channels(candidate)
    except WorkerError as exc:
        raise WorkerError(
            exc.code,
            (
                "MOSS-TTS returned an invalid waveform for "
                f"segment {segment_index}: {exc}"
            ),
            details={
                "segment": segment_index,
                "segment_count": segment_count,
            },
        ) from exc


def combine_audio_segments(
    audio_segments: Sequence[Any],
    sample_rate: int,
    gap_ms: int,
) -> Any:
    """Join channel-first segments without collapsing Local's stereo output."""

    combined, _timing = combine_audio_segments_with_timing(
        audio_segments,
        sample_rate,
        gap_ms,
    )
    return combined


def combine_audio_segments_with_timing(
    audio_segments: Sequence[Any],
    sample_rate: int,
    gap_ms: int,
    chapter_segment_indices: Sequence[int] = (),
    chapter_pause_ms: int = 0,
) -> Tuple[Any, AudioMergeTiming]:
    """Join channel-first segments and return their exact merged positions."""

    import numpy as np  # type: ignore

    if not audio_segments:
        raise WorkerError("INVALID_MODEL_OUTPUT", "MOSS-TTS returned no audio segments.")
    channel_count = int(audio_segments[0].shape[0])
    if any(int(audio.shape[0]) != channel_count for audio in audio_segments):
        raise WorkerError(
            "INVALID_MODEL_OUTPUT",
            "MOSS-TTS changed channel count between generated segments.",
        )
    timing = build_audio_merge_timing(
        [int(audio.shape[1]) for audio in audio_segments],
        sample_rate,
        gap_ms,
        chapter_segment_indices,
        chapter_pause_ms,
    )
    merged: List[Any] = []
    for index, audio in enumerate(audio_segments):
        gap_samples = timing.gap_samples_before[index]
        if gap_samples:
            merged.append(
                np.zeros((channel_count, gap_samples), dtype=np.float32)
            )
        merged.append(audio)
    combined = np.concatenate(merged, axis=1)
    if int(combined.shape[1]) != timing.total_samples:
        raise WorkerError(
            "INVALID_MODEL_OUTPUT",
            "MOSS-TTS merged audio length does not match its timing plan.",
        )
    return combined, timing


def soundfile_waveform(audio: Any) -> Any:
    """Convert channel-first audio to SoundFile's sample-first convention."""
    return audio[0] if int(audio.shape[0]) == 1 else audio.T


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
    model_id: str = DEFAULT_MODEL_ID,
    chapter_pause_ms: int = 0,
    chapter_manifest_path: Optional[str] = None,
    duration_outlier_retries: int = DEFAULT_DURATION_OUTLIER_RETRIES,
    duration_outlier_ratio: float = DEFAULT_DURATION_OUTLIER_RATIO,
    review_manifest_path: Optional[str] = None,
    enable_chapter_markers: bool = False,
) -> None:
    spec = _model_spec(model_id)
    validate_sampling(
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_new_tokens,
        max_chars,
        gap_ms,
    )
    validate_duration_outlier_controls(
        duration_outlier_retries,
        duration_outlier_ratio,
    )
    if chapter_pause_ms < 0:
        raise WorkerError(
            "INVALID_ARGUMENT",
            "chapter_pause_ms must be zero or greater.",
        )
    chapter_markers_enabled = bool(
        chapter_manifest_path and chapter_manifest_path.strip()
    ) or enable_chapter_markers
    segments = split_synthesis_text(
        text,
        max_chars,
        use_chapter_markers=chapter_markers_enabled,
    )
    if not segments:
        raise WorkerError("INVALID_ARGUMENT", "Text input is empty.")
    output = Path(output_path).expanduser().resolve()
    manifest_path = (
        Path(chapter_manifest_path).expanduser().resolve()
        if chapter_manifest_path and chapter_manifest_path.strip()
        else None
    )
    review_path = (
        Path(review_manifest_path).expanduser().resolve()
        if review_manifest_path and review_manifest_path.strip()
        else None
    )
    if manifest_path == output or review_path == output or (
        manifest_path is not None and manifest_path == review_path
    ):
        raise WorkerError(
            "INVALID_ARGUMENT",
            "Audio, chapter, and review output paths must be different.",
        )

    reference_audio: Optional[str] = None
    if voice_path:
        reference = Path(voice_path).expanduser().resolve()
        if not reference.is_file():
            raise WorkerError("FILE_NOT_FOUND", f"Voice reference not found: {reference}")
        reference_audio = str(reference)

    model_snapshot, codec_snapshot = load_pinned_snapshots(models_root, model_id)
    model, processor, device, sample_rate = load_backend(
        model_snapshot,
        codec_snapshot,
        os.environ.get("MOSS_TTS_DEVICE"),
        requested_attention,
        model_id,
    )
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore

    reference_profile = (
        load_reference_acoustic_profile(reference_audio, sf, np)
        if reference_audio
        else None
    )
    if reference_audio and reference_profile is None:
        emit(
            "log",
            level="warn",
            message=(
                "MOSS reference-closeness checks are unavailable for this "
                "reference format; synthesis will continue normally."
            ),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    streamed_output = StreamingWaveWriter(output, sample_rate)
    segment_sample_counts: List[int] = []
    total = len(segments)
    rolling_prefix: Optional[RollingPrefix] = None
    speaking_rate_history: List[float] = []
    reference_closeness_history: List[float] = []
    duration_retries_attempted = 0
    duration_retried_segments: List[int] = []
    review_segments: List[Dict[str, Any]] = []
    review_audio_dir: Optional[Path] = None
    if review_path is not None:
        review_audio_dir = review_path.parent / "segments"
        review_audio_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".voiceforge-moss-prefix-",
            dir=str(output.parent),
        ) as prefix_dir:
            rolling_audio_path = Path(prefix_dir) / "rolling-prefix.wav"
            for index, segment in enumerate(segments, start=1):
                emit(
                    "progress",
                    progress=0.12 + (0.82 * index / total),
                    message=f"Synthesizing MOSS segment {index}/{total}",
                )
                use_rolling_prefix = rolling_prefix
                reanchored = should_reanchor_to_reference(
                    index - 1,
                    bool(segment.starts_chapter),
                    reference_audio is not None,
                )
                if reanchored:
                    use_rolling_prefix = None
                    reanchor_reason = (
                        "chapter boundary"
                        if segment.starts_chapter
                        else f"every {DEFAULT_REANCHOR_EVERY_SEGMENTS} continuation segments"
                    )
                    emit(
                        "log",
                        level="info",
                        message=(
                            f"MOSS segment {index}/{total}: re-anchoring to the "
                            f"original reference ({reanchor_reason})."
                        ),
                    )
                try:
                    def generate_audio_candidate() -> Any:
                        return generate_segment_audio(
                            model,
                            processor,
                            device,
                            torch,
                            np,
                            segment.text,
                            reference_audio,
                            language,
                            use_rolling_prefix,
                            max_new_tokens,
                            temperature,
                            top_p,
                            top_k,
                            repetition_penalty,
                            index,
                            total,
                        )

                    audio_np = generate_audio_candidate()
                except Exception as exc:
                    raise WorkerError(
                        "SYNTHESIS_FAILED",
                        f"MOSS-TTS synthesis failed on segment {index}/{total}: {exc}",
                        details={"segment": index, "segment_count": total},
                    ) from exc

                selected_rate = segment_speaking_rate(
                    segment.text,
                    audio_np,
                    sample_rate,
                )
                outlier = duration_outlier_analysis(
                    selected_rate,
                    speaking_rate_history,
                    duration_outlier_ratio,
                )
                if outlier is not None and duration_outlier_retries > 0:
                    duration_retried_segments.append(index)
                    pace_label = "faster" if outlier.rate_ratio > 1.0 else "slower"
                    pace_deviation = max(
                        outlier.rate_ratio,
                        1.0 / outlier.rate_ratio,
                    )
                    for retry_number in range(1, duration_outlier_retries + 1):
                        duration_retries_attempted += 1
                        emit(
                            "progress",
                            progress=0.12 + (0.82 * index / total),
                            message=(
                                f"MOSS segment {index}/{total} was a "
                                f"{pace_deviation:.2f}x pace outlier "
                                f"({pace_label}); retrying "
                                f"({retry_number}/{duration_outlier_retries})"
                            ),
                        )
                        try:
                            retry_audio = generate_audio_candidate()
                        except Exception as exc:
                            emit(
                                "log",
                                level="warn",
                                message=(
                                    f"MOSS pace retry failed for segment {index}; "
                                    f"keeping the best completed attempt: {exc}"
                                ),
                            )
                            break
                        retry_rate = segment_speaking_rate(
                            segment.text,
                            retry_audio,
                            sample_rate,
                        )
                        if duration_candidate_is_closer(
                            retry_rate,
                            selected_rate,
                            outlier.baseline_rate,
                        ):
                            audio_np = retry_audio
                            selected_rate = retry_rate
                        outlier = duration_outlier_analysis(
                            selected_rate,
                            speaking_rate_history,
                            duration_outlier_ratio,
                        )
                        if outlier is None:
                            break

                remaining_outlier = duration_outlier_analysis(
                    selected_rate,
                    speaking_rate_history,
                    duration_outlier_ratio,
                )
                pace_ratio, pace_status = pace_metadata(
                    selected_rate,
                    speaking_rate_history,
                    duration_outlier_ratio,
                )
                closeness = reference_closeness(
                    reference_profile,
                    audio_np,
                    sample_rate,
                    reference_closeness_history,
                    np,
                )
                reference_score: Optional[float] = None
                reference_drift_status = "not-compared"
                if closeness is not None:
                    reference_score = closeness.score
                    reference_drift_status = closeness.status
                    reference_closeness_history.append(closeness.score)
                    baseline_text = (
                        f" (recent baseline {closeness.baseline:.0f}/100)"
                        if closeness.baseline is not None
                        else ""
                    )
                    message = (
                        f"MOSS segment {index}/{total}: reference acoustic closeness "
                        f"{closeness.score:.0f}/100{baseline_text}"
                    )
                    if closeness.status == "drift-warning":
                        message += " — possible voice drift; audition this segment."
                    emit(
                        "reference-closeness",
                        level="warn" if closeness.status == "drift-warning" else "info",
                        message=message,
                        segment_index=index - 1,
                        reference_closeness=closeness.score,
                        reference_drift_status=closeness.status,
                        reference_baseline=closeness.baseline,
                    )
                if selected_rate is not None and remaining_outlier is None:
                    speaking_rate_history.append(selected_rate)
                gap_samples = 0
                if index > 1:
                    gap_samples = int(sample_rate * gap_ms / 1000)
                    if segment.starts_chapter:
                        gap_samples += int(sample_rate * chapter_pause_ms / 1000)
                streamed_output.write(audio_np, gap_samples)
                segment_sample_counts.append(int(audio_np.shape[1]))
                if review_path is not None and review_audio_dir is not None:
                    segment_audio_path = review_audio_dir / f"segment-{index:04d}.wav"
                    sf.write(
                        str(segment_audio_path),
                        soundfile_waveform(audio_np),
                        sample_rate,
                        subtype="FLOAT",
                    )
                    review_segments.append({
                        "index": index - 1,
                        "text": segment.text,
                        "audio_file": segment_audio_path.relative_to(
                            review_path.parent
                        ).as_posix(),
                        "sample_count": int(audio_np.shape[1]),
                        "duration_seconds": int(audio_np.shape[1]) / float(sample_rate),
                        "attempt": 1,
                        "starts_chapter": bool(segment.starts_chapter),
                        "chapter_title": segment.chapter_title,
                        "speaking_rate": selected_rate,
                        "pace_ratio": pace_ratio,
                        "pace_status": pace_status,
                        "reference_closeness": reference_score,
                        "reference_drift_status": reference_drift_status,
                        "updated_at": int(time.time() * 1000),
                    })

                if index < total:
                    # Reuse one file and replace one state object: only the most
                    # recent segment is ever retained as continuation context.
                    sf.write(
                        str(rolling_audio_path),
                        soundfile_waveform(audio_np),
                        sample_rate,
                        subtype="FLOAT",
                    )
                    rolling_prefix = RollingPrefix(segment.text, str(rolling_audio_path))

        if review_path is not None:
            write_review_manifest(
                review_path,
                {
                    "version": REVIEW_MANIFEST_VERSION,
                    "sample_rate": sample_rate,
                    "gap_ms": gap_ms,
                    "chapter_pause_ms": chapter_pause_ms,
                    "segments": review_segments,
                },
            )

        chapter_indices = [
            index for index, segment in enumerate(segments) if segment.starts_chapter
        ]
        timing = build_audio_merge_timing(
            segment_sample_counts,
            sample_rate,
            gap_ms,
            chapter_indices,
            chapter_pause_ms,
        )
        if streamed_output.total_samples != timing.total_samples:
            raise WorkerError(
                "INVALID_MODEL_OUTPUT",
                "MOSS-TTS streamed audio length does not match its timing plan.",
            )
        streamed_output.finalize()
    except Exception:
        streamed_output.discard()
        raise

    if manifest_path is not None:
        manifest = build_chapter_manifest(
            segments,
            timing.start_samples,
            sample_rate,
            timing.total_samples,
        )
        write_chapter_manifest(manifest_path, manifest)
    model_artifact = _pinned_artifacts(model_id)["model"]
    emit(
        "complete",
        progress=1.0,
        message=f"{spec['label']} synthesis complete",
        output_path=str(output),
        model_id=model_id,
        revision=model_artifact["revision"],
        segments=total,
        chapters=len(chapter_indices),
        continuation_segments=max(0, total - 1),
        rolling_continuation=total > 1,
        duration_outlier_retries=duration_retries_attempted,
        duration_retried_segments=duration_retried_segments,
        duration_outlier_ratio=duration_outlier_ratio,
        sample_rate=sample_rate,
        total_samples=timing.total_samples,
        channels=streamed_output.channels,
        used_voice_reference=bool(reference_audio),
        language=normalize_language(language) or "auto",
        chapter_manifest_path=str(manifest_path) if manifest_path else None,
        review_manifest_path=str(review_path) if review_path else None,
    )


def rerun_review_segment(
    models_root: Path,
    review_manifest_path: str,
    segment_index: int,
    voice_path: Optional[str],
    language: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    requested_attention: str,
    model_id: str = DEFAULT_MODEL_ID,
    duration_outlier_ratio: float = DEFAULT_DURATION_OUTLIER_RATIO,
) -> None:
    validate_sampling(
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_new_tokens,
        50,
        0,
    )
    validate_duration_outlier_controls(0, duration_outlier_ratio)
    manifest_path = Path(review_manifest_path).expanduser().resolve()
    manifest = read_review_manifest(manifest_path)
    upgrade_review_manifest(manifest)
    segments = manifest["segments"]
    if (
        isinstance(segment_index, bool)
        or not isinstance(segment_index, int)
        or segment_index < 0
        or segment_index >= len(segments)
    ):
        raise WorkerError(
            "INVALID_ARGUMENT",
            "The requested MOSS review segment does not exist.",
        )

    reference_audio: Optional[str] = None
    if voice_path:
        reference = Path(voice_path).expanduser().resolve()
        if not reference.is_file():
            raise WorkerError("FILE_NOT_FOUND", f"Voice reference not found: {reference}")
        reference_audio = str(reference)

    model_snapshot, codec_snapshot = load_pinned_snapshots(models_root, model_id)
    model, processor, device, sample_rate = load_backend(
        model_snapshot,
        codec_snapshot,
        os.environ.get("MOSS_TTS_DEVICE"),
        requested_attention,
        model_id,
    )
    if sample_rate != int(manifest["sample_rate"]):
        raise WorkerError(
            "INVALID_MODEL_OUTPUT",
            "The MOSS model sample rate changed since the review was created.",
        )
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore

    reference_profile = (
        load_reference_acoustic_profile(reference_audio, sf, np)
        if reference_audio
        else None
    )
    reference_closeness_history = [
        float(item["reference_closeness"])
        for index, item in enumerate(segments)
        if index != segment_index and item.get("reference_closeness") is not None
    ]

    segment = segments[segment_index]
    rolling_prefix: Optional[RollingPrefix] = None
    reanchored = should_reanchor_to_reference(
        segment_index,
        bool(segment["starts_chapter"]),
        reference_audio is not None,
    )
    if segment_index > 0 and not reanchored:
        previous = segments[segment_index - 1]
        previous_audio = _review_audio_path(
            manifest_path,
            previous["audio_file"],
        )
        rolling_prefix = RollingPrefix(previous["text"], str(previous_audio))
    elif reanchored:
        reanchor_reason = (
            "chapter boundary"
            if segment["starts_chapter"]
            else f"every {DEFAULT_REANCHOR_EVERY_SEGMENTS} continuation segments"
        )
        emit(
            "log",
            level="info",
            message=(
                f"MOSS segment {segment_index + 1}/{len(segments)}: re-anchoring "
                f"to the original reference ({reanchor_reason})."
            ),
        )
    emit(
        "progress",
        progress=0.12,
        message=(
            f"Re-running MOSS segment {segment_index + 1}/{len(segments)}"
        ),
    )
    try:
        audio = generate_segment_audio(
            model,
            processor,
            device,
            torch,
            np,
            segment["text"],
            reference_audio,
            language,
            rolling_prefix,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            segment_index + 1,
            len(segments),
        )
    except Exception as exc:
        raise WorkerError(
            "SYNTHESIS_FAILED",
            (
                "MOSS-TTS failed while re-running segment "
                f"{segment_index + 1}/{len(segments)}: {exc}"
            ),
            details={
                "segment": segment_index + 1,
                "segment_count": len(segments),
            },
        ) from exc

    rate = segment_speaking_rate(segment["text"], audio, sample_rate)
    closeness = reference_closeness(
        reference_profile,
        audio,
        sample_rate,
        reference_closeness_history,
        np,
    )
    reference_score: Optional[float] = None
    reference_drift_status = "not-compared"
    if closeness is not None:
        reference_score = closeness.score
        reference_drift_status = closeness.status
        baseline_text = (
            f" (recent baseline {closeness.baseline:.0f}/100)"
            if closeness.baseline is not None
            else ""
        )
        message = (
            f"MOSS segment {segment_index + 1}/{len(segments)}: reference acoustic "
            f"closeness {closeness.score:.0f}/100{baseline_text}"
        )
        if closeness.status == "drift-warning":
            message += " — possible voice drift; audition this segment."
        emit(
            "reference-closeness",
            level="warn" if closeness.status == "drift-warning" else "info",
            message=message,
            segment_index=segment_index,
            reference_closeness=closeness.score,
            reference_drift_status=closeness.status,
            reference_baseline=closeness.baseline,
        )
    destination = _review_audio_path(manifest_path, segment["audio_file"])
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}.",
            suffix=".wav",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        sf.write(
            str(temporary_path),
            soundfile_waveform(audio),
            sample_rate,
            subtype="FLOAT",
        )
        temporary_path.replace(destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    segment.update({
        "sample_count": int(audio.shape[1]),
        "duration_seconds": int(audio.shape[1]) / float(sample_rate),
        "attempt": int(segment["attempt"]) + 1,
        "speaking_rate": rate,
        "reference_closeness": reference_score,
        "reference_drift_status": reference_drift_status,
        "updated_at": int(time.time() * 1000),
    })
    refresh_review_pace_metadata(segments, duration_outlier_ratio)
    write_review_manifest(manifest_path, manifest)
    emit(
        "complete",
        progress=1.0,
        message=f"MOSS segment {segment_index + 1} is ready for review",
        segment_index=segment_index,
        attempt=segment["attempt"],
        duration_seconds=segment["duration_seconds"],
        pace_ratio=segment["pace_ratio"],
        pace_status=segment["pace_status"],
        reference_closeness=segment["reference_closeness"],
        reference_drift_status=segment["reference_drift_status"],
    )


def compile_review(
    review_manifest_path: str,
    output_path: str,
    chapter_manifest_path: Optional[str] = None,
) -> None:
    manifest_path = Path(review_manifest_path).expanduser().resolve()
    manifest = read_review_manifest(manifest_path)
    output = Path(output_path).expanduser().resolve()
    chapter_path = (
        Path(chapter_manifest_path).expanduser().resolve()
        if chapter_manifest_path and chapter_manifest_path.strip()
        else None
    )
    if output == manifest_path or chapter_path in {output, manifest_path}:
        raise WorkerError(
            "INVALID_ARGUMENT",
            "Compiled audio, chapter, and review paths must be different.",
        )
    import soundfile as sf  # type: ignore

    sample_rate = int(manifest["sample_rate"])
    streamed_output = StreamingWaveWriter(output, sample_rate)
    segment_sample_counts: List[int] = []
    chapter_segments: List[ChapterSegment] = []
    try:
        for index, segment in enumerate(manifest["segments"]):
            emit(
                "progress",
                progress=0.05 + (0.75 * (index + 1) / len(manifest["segments"])),
                message=(
                    f"Reading reviewed MOSS segment "
                    f"{index + 1}/{len(manifest['segments'])}"
                ),
            )
            audio_path = _review_audio_path(manifest_path, segment["audio_file"])
            samples, actual_rate = sf.read(
                str(audio_path),
                dtype="float32",
                always_2d=True,
            )
            if int(actual_rate) != sample_rate:
                raise WorkerError(
                    "INVALID_MODEL_OUTPUT",
                    f"MOSS review segment {index + 1} has an inconsistent sample rate.",
                )
            audio = normalize_audio_channels(samples.T)
            if int(audio.shape[1]) != int(segment["sample_count"]):
                raise WorkerError(
                    "INVALID_MODEL_OUTPUT",
                    f"MOSS review segment {index + 1} does not match its manifest.",
                )
            gap_samples = 0
            if index > 0:
                gap_samples = int(sample_rate * int(manifest["gap_ms"]) / 1000)
                if bool(segment["starts_chapter"]):
                    gap_samples += int(
                        sample_rate * int(manifest["chapter_pause_ms"]) / 1000
                    )
            streamed_output.write(audio, gap_samples)
            segment_sample_counts.append(int(audio.shape[1]))
            chapter_segments.append(
                ChapterSegment(
                    segment["text"],
                    bool(segment["starts_chapter"]),
                    segment["chapter_title"],
                )
            )

        chapter_indices = [
            index
            for index, segment in enumerate(chapter_segments)
            if segment.starts_chapter
        ]
        timing = build_audio_merge_timing(
            segment_sample_counts,
            sample_rate,
            int(manifest["gap_ms"]),
            chapter_indices,
            int(manifest["chapter_pause_ms"]),
        )
        if streamed_output.total_samples != timing.total_samples:
            raise WorkerError(
                "INVALID_MODEL_OUTPUT",
                "MOSS-TTS streamed review audio length does not match its timing plan.",
            )
        streamed_output.finalize()
    except Exception:
        streamed_output.discard()
        raise

    if chapter_path is not None:
        write_chapter_manifest(
            chapter_path,
            build_chapter_manifest(
                chapter_segments,
                timing.start_samples,
                sample_rate,
                timing.total_samples,
            ),
        )
    emit(
        "complete",
        progress=1.0,
        message="Reviewed MOSS segments compiled",
        output_path=str(output),
        segments=len(segment_sample_counts),
        chapters=len(chapter_indices),
        sample_rate=sample_rate,
        total_samples=timing.total_samples,
        channels=streamed_output.channels,
        chapter_manifest_path=str(chapter_path) if chapter_path else None,
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
    for command in ("check", "setup", "download"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--model-id",
            choices=sorted(PINNED_MODEL_SPECS),
            default=DEFAULT_MODEL_ID,
        )

    synth_parser = subparsers.add_parser("synthesize", help="Run local MOSS-TTS synthesis")
    synth_parser.add_argument(
        "--model-id",
        choices=sorted(PINNED_MODEL_SPECS),
        default=DEFAULT_MODEL_ID,
    )
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
        "--duration-outlier-retries",
        type=int,
        default=int(
            os.environ.get(
                "MOSS_TTS_DURATION_OUTLIER_RETRIES",
                str(DEFAULT_DURATION_OUTLIER_RETRIES),
            )
        ),
        help="Automatic pace-outlier retries per eligible local segment",
    )
    synth_parser.add_argument(
        "--duration-outlier-ratio",
        type=float,
        default=float(
            os.environ.get(
                "MOSS_TTS_DURATION_OUTLIER_RATIO",
                str(DEFAULT_DURATION_OUTLIER_RATIO),
            )
        ),
        help="Maximum speaking-rate ratio versus the recent segment median",
    )
    synth_parser.add_argument(
        "--chapter-pause-ms",
        type=int,
        default=int(os.environ.get("MOSS_TTS_CHAPTER_PAUSE_MS", "0")),
        help="Additional silence before chapter-start segments",
    )
    synth_parser.add_argument(
        "--chapter-manifest",
        default=None,
        help="Optional JSON path for exact chapter start timings",
    )
    synth_parser.add_argument(
        "--chapter-markers",
        action="store_true",
        help="Treat [CHAPTER] markers as non-spoken chapter/re-anchor boundaries",
    )
    synth_parser.add_argument(
        "--review-manifest",
        default=None,
        help="Optional JSON path for per-segment review artifacts",
    )
    synth_parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("MOSS_TTS_ATTN_IMPLEMENTATION", "auto"),
    )

    rerun_parser = subparsers.add_parser(
        "rerun-segment",
        help="Replace one segment in an existing MOSS review",
    )
    rerun_parser.add_argument(
        "--model-id",
        choices=sorted(PINNED_MODEL_SPECS),
        default=DEFAULT_MODEL_ID,
    )
    rerun_parser.add_argument("--review-manifest", required=True)
    rerun_parser.add_argument("--segment-index", type=int, required=True)
    rerun_parser.add_argument(
        "--voice",
        "--reference-audio",
        dest="voice",
        default=None,
    )
    rerun_parser.add_argument(
        "--language",
        default=os.environ.get("MOSS_TTS_LANGUAGE", "Auto"),
    )
    rerun_parser.add_argument(
        "--temperature",
        type=float,
        default=float(
            os.environ.get("MOSS_TTS_TEMPERATURE", str(DEFAULT_TEMPERATURE))
        ),
    )
    rerun_parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.environ.get("MOSS_TTS_TOP_P", str(DEFAULT_TOP_P))),
    )
    rerun_parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("MOSS_TTS_TOP_K", str(DEFAULT_TOP_K))),
    )
    rerun_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(
            os.environ.get(
                "MOSS_TTS_REPETITION_PENALTY",
                str(DEFAULT_REPETITION_PENALTY),
            )
        ),
    )
    rerun_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(
            os.environ.get(
                "MOSS_TTS_MAX_NEW_TOKENS",
                str(DEFAULT_MAX_NEW_TOKENS),
            )
        ),
    )
    rerun_parser.add_argument(
        "--duration-outlier-ratio",
        type=float,
        default=float(
            os.environ.get(
                "MOSS_TTS_DURATION_OUTLIER_RATIO",
                str(DEFAULT_DURATION_OUTLIER_RATIO),
            )
        ),
    )
    rerun_parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("MOSS_TTS_ATTN_IMPLEMENTATION", "auto"),
    )

    compile_parser = subparsers.add_parser(
        "compile-review",
        help="Compile reviewed MOSS segments without loading the model",
    )
    compile_parser.add_argument("--review-manifest", required=True)
    compile_parser.add_argument("--output", required=True)
    compile_parser.add_argument("--chapter-manifest", default=None)
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
        setup_models(models_root, args.model_id)
        return
    if args.command == "check":
        ensure_dependencies()
        spec = _model_spec(args.model_id)
        artifacts = _pinned_artifacts(args.model_id)
        model_snapshot, codec_snapshot = load_pinned_snapshots(
            models_root,
            args.model_id,
        )
        emit(
            "complete",
            progress=1.0,
            message=f"{spec['label']} runtime and pinned model data are ready",
            ready=True,
            model_id=args.model_id,
            revision=artifacts["model"]["revision"],
            model_snapshot_path=str(model_snapshot),
            codec_id=artifacts["codec"]["repo_id"],
            codec_revision=artifacts["codec"]["revision"],
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
            args.model_id,
            args.chapter_pause_ms,
            args.chapter_manifest,
            args.duration_outlier_retries,
            args.duration_outlier_ratio,
            args.review_manifest,
            args.chapter_markers,
        )
        return
    if args.command == "rerun-segment":
        rerun_review_segment(
            models_root,
            args.review_manifest,
            args.segment_index,
            args.voice,
            args.language,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repetition_penalty,
            args.max_new_tokens,
            args.attn_implementation,
            args.model_id,
            args.duration_outlier_ratio,
        )
        return
    if args.command == "compile-review":
        compile_review(
            args.review_manifest,
            args.output,
            args.chapter_manifest,
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
