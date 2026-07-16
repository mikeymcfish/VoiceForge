#!/usr/bin/env python3
"""IndexTTS worker for an operator-managed, isolated official runtime.

This process intentionally never installs Python packages or downloads executable
source code. VoiceForge downloads only the pinned official model snapshot. The
operator supplies the official IndexTTS runtime through INDEX_TTS_PYTHON and,
when the source tree is not installed in that environment,
INDEX_TTS_SOURCE_DIR.
"""

import argparse
import importlib
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple


OFFICIAL_MODEL_REPO_ID = "IndexTeam/IndexTTS-2"
OFFICIAL_MODEL_REVISION = "740dcaff396282ffb241903d150ac011cd4b1ede"
MODEL_MANIFEST_NAME = ".voiceforge-index-model.json"
MODEL_MANIFEST_VERSION = 2
MINIMUM_TORCH_VERSION = (2, 6)
MINIMUM_MODEL_BYTES = 100 * 1024 * 1024
MODEL_WEIGHT_SUFFIXES = {".bin", ".onnx", ".pt", ".pth", ".safetensors"}


def configure_utf8_streams() -> None:
    """Keep official IndexTTS diagnostic output Unicode-safe on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


configure_utf8_streams()


def emit(event: str, **payload: Any) -> None:
    message = {"event": event}
    message.update(payload)
    print(json.dumps(message), flush=True)


def is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _setup_hint() -> str:
    return (
        "Create an isolated environment from the official index-tts/index-tts source, "
        "install its dependencies with PyTorch 2.6 or newer, then set INDEX_TTS_PYTHON "
        "to that environment's Python executable. If the indextts package is not "
        "installed, also set INDEX_TTS_SOURCE_DIR to the official repository root."
    )


def _import_required(module_name: str, purpose: str):
    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"IndexTTS requires {purpose} in its preconfigured Python environment; "
            f"could not import '{module_name}' ({exc}). {_setup_hint()}"
        ) from exc


def _torch_major_minor(version: str) -> Tuple[int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)", version)
    if not match:
        raise RuntimeError(f"Unable to parse the installed PyTorch version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def require_supported_torch():
    torch = _import_required("torch", "PyTorch 2.6 or newer")
    version = str(getattr(torch, "__version__", ""))
    if _torch_major_minor(version) < MINIMUM_TORCH_VERSION:
        raise RuntimeError(
            f"IndexTTS is configured with PyTorch {version or 'unknown'}, but VoiceForge "
            "requires PyTorch 2.6 or newer. Upgrade Torch in the isolated environment "
            "selected by INDEX_TTS_PYTHON before loading model code."
        )
    emit("log", level="info", message=f"Validated PyTorch {version}")
    return torch


def _configure_official_source() -> Optional[Path]:
    configured = os.environ.get("INDEX_TTS_SOURCE_DIR", "").strip()
    if not configured:
        return None

    source = Path(configured).expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(
            f"INDEX_TTS_SOURCE_DIR does not exist or is not a directory: {source}. "
            f"{_setup_hint()}"
        )

    if (source / "indextts" / "infer_v2.py").is_file():
        import_root = source
        repo_root = source
    elif source.name == "indextts" and (source / "infer_v2.py").is_file():
        import_root = source.parent
        repo_root = source.parent
    else:
        raise RuntimeError(
            "INDEX_TTS_SOURCE_DIR must point to the official IndexTTS repository root "
            "containing indextts/infer_v2.py (or to that indextts package directory)."
        )

    import_root_text = str(import_root)
    if import_root_text not in sys.path:
        sys.path.insert(0, import_root_text)
    os.chdir(repo_root)
    emit("log", level="info", message=f"Using configured IndexTTS source: {repo_root}")
    return import_root


def import_index_inference():
    import_root = _configure_official_source()
    try:
        infer_v2 = importlib.import_module("indextts.infer_v2")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The official IndexTTS Python source is not available in the environment "
            f"selected by INDEX_TTS_PYTHON ({exc}). {_setup_hint()}"
        ) from exc

    if import_root is not None:
        module_file = Path(getattr(infer_v2, "__file__", "")).resolve()
        try:
            module_file.relative_to(import_root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "Imported indextts from outside INDEX_TTS_SOURCE_DIR; refusing to run "
                f"unexpected module path: {module_file}"
            ) from exc

    if not hasattr(infer_v2, "IndexTTS2"):
        raise RuntimeError(
            "The configured indextts.infer_v2 module does not expose IndexTTS2. "
            "Verify that INDEX_TTS_SOURCE_DIR points to the official IndexTTS2 source."
        )
    return infer_v2


def prepare_environment(models_dir: str) -> Path:
    models_root = Path(models_dir).expanduser().resolve()
    cache_dir = models_root / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("HF_HOME", str(models_root / "hf_home"))
    return models_root


def _manifest_path(models_root: Path) -> Path:
    return models_root / MODEL_MANIFEST_NAME


def _expected_snapshot_path(models_root: Path) -> Path:
    repository_cache_name = f"models--{OFFICIAL_MODEL_REPO_ID.replace('/', '--')}"
    return (
        models_root
        / "hf_cache"
        / repository_cache_name
        / "snapshots"
        / OFFICIAL_MODEL_REVISION
    ).resolve()


def _snapshot_inventory(models_root: Path, snapshot_path: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    resolved_models_root = models_root.resolve()

    for candidate in sorted(snapshot_path.rglob("*")):
        if not candidate.is_file():
            continue
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(resolved_models_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Pinned model file resolves outside the managed model directory: {candidate}"
            ) from exc
        inventory.append(
            {
                "path": candidate.relative_to(snapshot_path).as_posix(),
                "size": candidate.stat().st_size,
            }
        )

    if not inventory:
        raise RuntimeError("The pinned IndexTTS model snapshot contains no files.")
    if not any(item["path"].endswith("config.yaml") and item["size"] > 0 for item in inventory):
        raise RuntimeError("The pinned IndexTTS model snapshot is missing a non-empty config.yaml.")
    if not any(Path(item["path"]).suffix.lower() in MODEL_WEIGHT_SUFFIXES for item in inventory):
        raise RuntimeError("The pinned IndexTTS model snapshot contains no recognized model weights.")
    if sum(item["size"] for item in inventory) < MINIMUM_MODEL_BYTES:
        raise RuntimeError("The pinned IndexTTS model snapshot is incomplete (model files are too small).")
    return inventory


def _validate_manifest_inventory(
    models_root: Path, snapshot_path: Path, raw_inventory: Any
) -> None:
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise RuntimeError("The IndexTTS model manifest has no completeness inventory.")

    seen_paths: set[str] = set()
    resolved_models_root = models_root.resolve()
    for item in raw_inventory:
        if not isinstance(item, dict):
            raise RuntimeError("The IndexTTS model manifest contains an invalid file entry.")
        relative_path = item.get("path")
        expected_size = item.get("size")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path in seen_paths
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise RuntimeError("The IndexTTS model manifest contains an invalid file entry.")
        seen_paths.add(relative_path)

        logical_candidate = Path(os.path.abspath(snapshot_path / relative_path))
        try:
            logical_candidate.relative_to(snapshot_path)
            logical_candidate.resolve().relative_to(resolved_models_root)
        except ValueError as exc:
            raise RuntimeError("The IndexTTS model manifest contains an unsafe file path.") from exc
        if not logical_candidate.is_file() or logical_candidate.stat().st_size != expected_size:
            raise RuntimeError(
                f"Pinned IndexTTS model file is missing or incomplete: {relative_path}"
            )

    actual_inventory = _snapshot_inventory(models_root, snapshot_path)
    actual_files = {item["path"]: item["size"] for item in actual_inventory}
    expected_files = {item["path"]: item["size"] for item in raw_inventory}
    if actual_files != expected_files:
        raise RuntimeError("The pinned IndexTTS model snapshot does not match its manifest.")


def _write_model_manifest(models_root: Path, snapshot_path: Path) -> None:
    resolved_snapshot = snapshot_path.resolve()
    try:
        relative_snapshot = resolved_snapshot.relative_to(models_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Pinned model snapshot resolved outside the managed model directory: {resolved_snapshot}"
        ) from exc

    expected_snapshot = _expected_snapshot_path(models_root)
    if resolved_snapshot != expected_snapshot:
        raise RuntimeError(
            "Pinned model download did not resolve to the expected Hugging Face revision path."
        )

    manifest = {
        "manifest_version": MODEL_MANIFEST_VERSION,
        "repo_id": OFFICIAL_MODEL_REPO_ID,
        "revision": OFFICIAL_MODEL_REVISION,
        "snapshot_path": relative_snapshot.as_posix(),
        "files": _snapshot_inventory(models_root, resolved_snapshot),
    }
    manifest_path = _manifest_path(models_root)
    temporary_path = manifest_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)


def _load_pinned_snapshot(models_dir: str) -> Path:
    models_root = prepare_environment(models_dir)
    manifest_path = _manifest_path(models_root)
    if not manifest_path.is_file():
        raise RuntimeError(
            "Pinned IndexTTS models have not been downloaded by this VoiceForge version. "
            "Use Download models before loading or synthesizing."
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("The IndexTTS model manifest is unreadable; download models again.") from exc

    if (
        manifest.get("repo_id") != OFFICIAL_MODEL_REPO_ID
        or manifest.get("revision") != OFFICIAL_MODEL_REVISION
    ):
        raise RuntimeError(
            "The installed IndexTTS model snapshot is not the pinned official revision; "
            "download models again."
        )

    relative_snapshot = manifest.get("snapshot_path")
    if not isinstance(relative_snapshot, str) or not relative_snapshot:
        raise RuntimeError("The IndexTTS model manifest does not contain a snapshot path.")

    snapshot_path = (models_root / relative_snapshot).resolve()
    try:
        snapshot_path.relative_to(models_root)
    except ValueError as exc:
        raise RuntimeError("The IndexTTS model manifest points outside the managed directory.") from exc
    if not snapshot_path.is_dir():
        raise RuntimeError("The pinned IndexTTS model snapshot is missing; download models again.")

    if snapshot_path != _expected_snapshot_path(models_root):
        raise RuntimeError(
            "The IndexTTS model manifest does not point to the pinned Hugging Face revision path."
        )

    manifest_version = manifest.get("manifest_version")
    if manifest_version == MODEL_MANIFEST_VERSION:
        _validate_manifest_inventory(models_root, snapshot_path, manifest.get("files"))
    elif manifest_version is None and "files" not in manifest:
        # Version 1 manifests were written only after snapshot_download completed.
        # Rebuild the deterministic inventory so existing trusted downloads survive
        # a VoiceForge restart/upgrade without another multi-gigabyte download.
        _write_model_manifest(models_root, snapshot_path)
    else:
        raise RuntimeError("The IndexTTS model manifest version is unsupported; download models again.")
    return snapshot_path


def find_config(snapshot_path: Path) -> Tuple[str, str]:
    config_paths = sorted(snapshot_path.rglob("config.yaml"))
    if not config_paths:
        raise FileNotFoundError(
            f"Unable to locate config.yaml in pinned model snapshot {snapshot_path}"
        )
    config_path = config_paths[0]
    return str(config_path), str(config_path.parent)


def handle_download(args) -> None:
    models_root = prepare_environment(args.models_dir)
    huggingface_hub = _import_required(
        "huggingface_hub",
        "huggingface_hub for downloading the pinned official model snapshot",
    )
    snapshot_download = getattr(huggingface_hub, "snapshot_download", None)
    if not callable(snapshot_download):
        raise RuntimeError("The installed huggingface_hub does not provide snapshot_download.")

    emit(
        "progress",
        progress=0.05,
        message=f"Downloading {OFFICIAL_MODEL_REPO_ID} at {OFFICIAL_MODEL_REVISION[:12]}…",
    )
    snapshot = Path(
        snapshot_download(
            repo_id=OFFICIAL_MODEL_REPO_ID,
            revision=OFFICIAL_MODEL_REVISION,
            cache_dir=str(models_root / "hf_cache"),
        )
    )
    find_config(snapshot)
    _write_model_manifest(models_root, snapshot)
    emit("progress", progress=0.95, message="Pinned model snapshot verified")
    emit("complete", progress=1.0, message="Download complete", output_path=str(snapshot))


def init_model(models_dir: str):
    require_supported_torch()
    infer_v2 = import_index_inference()
    snapshot_path = _load_pinned_snapshot(models_dir)
    cfg_path, checkpoint_dir = find_config(snapshot_path)
    emit("progress", progress=0.15, message="Initializing IndexTTS2")

    use_fp16 = is_truthy(os.environ.get("INDEX_TTS_USE_FP16"))
    # The optional fused BigVGAN kernel needs a local MSVC/CUDA build toolchain
    # on Windows. Keep it opt-in so a normal prebuilt PyTorch installation does
    # not attempt a JIT compile (and emit a misleading missing-cl warning).
    use_cuda_kernel = is_truthy(os.environ.get("INDEX_TTS_USE_CUDA_KERNEL"))
    use_deepspeed = is_truthy(os.environ.get("INDEX_TTS_ENABLE_DEEPSPEED"))
    device_override = os.environ.get("INDEX_TTS_DEVICE")

    return infer_v2.IndexTTS2(
        cfg_path=cfg_path,
        model_dir=checkpoint_dir,
        use_fp16=use_fp16,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=use_deepspeed,
        device=device_override,
    )


def handle_load(args) -> None:
    tts = init_model(args.models_dir)
    try:
        # The pinned official IndexTTS2 constructor loads and validates all
        # model components. It does not expose a separate load_models method.
        emit("progress", progress=0.9, message="Runtime and model initialization completed")
    finally:
        del tts
    emit("complete", progress=1.0, message="Models loaded")


def handle_synthesize(args) -> None:
    if not os.path.exists(args.voice):
        raise FileNotFoundError(f"Voice prompt not found: {args.voice}")
    if not os.path.exists(args.text):
        raise FileNotFoundError(f"Text file not found: {args.text}")

    with open(args.text, "r", encoding="utf-8") as text_file:
        text = text_file.read().strip()
    if not text:
        raise ValueError("Text input is empty")

    tts = init_model(args.models_dir)
    def gr_progress(value, desc=None):
        try:
            emit("progress", progress=float(value), message=desc)
        except Exception:
            pass

    tts.gr_progress = gr_progress
    emit("progress", progress=0.25, message="Running synthesis")
    tts.infer(
        spk_audio_prompt=args.voice,
        text=text,
        output_path=args.output,
        verbose=True,
    )
    emit("complete", progress=1.0, message="Synthesis complete", output_path=args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="IndexTTS worker")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--models-dir", required=True)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download")
    subparsers.add_parser("load")

    synth_parser = subparsers.add_parser("synthesize")
    synth_parser.add_argument("--voice", required=True)
    synth_parser.add_argument("--text", required=True)
    synth_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    try:
        if args.command == "download":
            handle_download(args)
        elif args.command == "load":
            handle_load(args)
        elif args.command == "synthesize":
            handle_synthesize(args)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        emit("error", error=str(exc), message="IndexTTS operation failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
