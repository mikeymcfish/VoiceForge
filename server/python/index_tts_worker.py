#!/usr/bin/env python3
import argparse
import importlib
import importlib.metadata as importlib_metadata
import json
import os
import shutil
import subprocess
import sys
import traceback
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

# Upstream Premium IndexTTS2 sources used to provide the `indextts` package.
DEFAULT_INDEXTTS_REPO_ZIP = (
    "https://github.com/FurkanGozukara/Premium_IndexTTS2_SECourses/archive/refs/heads/main.zip"
)
INDEXTTS_MODULE_NAME = "indextts"
_TORCH_PLACEHOLDER = {"spec": "__TORCH__", "module": "torch"}
_TORCHAUDIO_PLACEHOLDER = {"spec": "__TORCHAUDIO__", "module": "torchaudio"}
_DEEPSPEED_PLACEHOLDER = {"spec": "__DEEPSPEED__", "module": "deepspeed"}

_BASE_DEPENDENCIES: Tuple[dict[str, Any], ...] = (
    {"spec": "numpy==1.24.3", "module": "numpy"},
    _TORCH_PLACEHOLDER,
    _TORCHAUDIO_PLACEHOLDER,
    {"spec": "accelerate==1.8.1", "module": "accelerate"},
    {"spec": "descript-audiotools==0.7.2", "module": "audiotools"},
    {"spec": "transformers==4.52.1", "module": "transformers"},
    {"spec": "tokenizers==0.21.0", "module": "tokenizers"},
    {"spec": "cn2an==0.5.22", "module": "cn2an"},
    {"spec": "ffmpeg-python==0.2.0", "module": "ffmpeg"},
    {"spec": "Cython==3.0.7", "module": "Cython"},
    {"spec": "g2p-en==2.1.0", "module": "g2p_en"},
    {"spec": "jieba==0.42.1", "module": "jieba"},
    {"spec": "json5==0.10.0", "module": "json5"},
    {"spec": "keras==2.13.1", "module": "keras"},
    {"spec": "tensorflow==2.13.1", "module": "tensorflow"},
    {"spec": "numba==0.58.1", "module": "numba"},
    {"spec": "pandas==2.1.3", "module": "pandas"},
    {"spec": "matplotlib==3.8.2", "module": "matplotlib"},
    {"spec": "munch==4.0.0", "module": "munch"},
    {"spec": "opencv-python==4.9.0.80", "module": "cv2"},
    {"spec": "tensorboard==2.13.0", "module": "tensorboard"},
    {"spec": "librosa==0.10.2.post1", "module": "librosa"},
    {"spec": "safetensors==0.5.2", "module": "safetensors"},
    {"spec": "scipy==1.11.4", "module": "scipy"},
    {"spec": "einops==0.7.0", "module": "einops"},
    {"spec": "soundfile==0.12.1", "module": "soundfile"},
    {"spec": "pyyaml==6.0.2", "module": "yaml"},
    _DEEPSPEED_PLACEHOLDER,
    {"spec": "modelscope==1.27.0", "module": "modelscope"},
    {"spec": "omegaconf>=2.3.0", "module": "omegaconf"},
    {"spec": "sentencepiece>=0.2.1", "module": "sentencepiece"},
    {"spec": "gradio>=5.0.0", "module": "gradio"},
    {"spec": "tqdm>=4.67.1", "module": "tqdm"},
    {"spec": "textstat>=0.7.10", "module": "textstat"},
    {"spec": "huggingface_hub>=0.25.0", "module": "huggingface_hub"},
    {"spec": "spaces>=0.31.0", "module": "spaces"},
)

_RESOLVED_DEPENDENCIES: Optional[Tuple[dict[str, Any], ...]] = None

dependencies_ready = False


def emit(event: str, **payload):
    message = {"event": event}
    message.update(payload)
    print(json.dumps(message), flush=True)


def _module_name_from_spec(spec: str) -> str:
    base = spec
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in base:
            base = base.split(sep, 1)[0]
            break
    if "[" in base:
        base = base.split("[", 1)[0]
    return base


def _parse_version(value: str) -> Tuple[int, ...]:
    parts: list[int] = []
    for token in value.replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                parts.append(int(digits))
    return tuple(parts)


def _validate_transformers(module: Any) -> bool:
    try:
        version_str = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        return False
    if _parse_version(version_str) < _parse_version("4.52.1"):
        return False
    try:
        cache_utils = importlib.import_module("transformers.cache_utils")
    except ImportError:
        return False
    return hasattr(cache_utils, "QuantizedCacheConfig")


PACKAGE_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "transformers": _validate_transformers,
}


def is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_additional_dependencies() -> Tuple[dict[str, Any], ...]:
    global _RESOLVED_DEPENDENCIES
    if _RESOLVED_DEPENDENCIES is not None:
        return _RESOLVED_DEPENDENCIES

    use_cuda = is_truthy(os.environ.get("INDEX_TTS_ENABLE_CUDA"))
    torch_spec = os.environ.get("INDEX_TTS_TORCH_SPEC")
    torch_index_url = os.environ.get("INDEX_TTS_TORCH_INDEX_URL")
    torch_extra_indexes: Optional[Tuple[str, ...]] = None
    torchaudio_spec = os.environ.get("INDEX_TTS_TORCHAUDIO_SPEC")
    torchaudio_index_url = os.environ.get("INDEX_TTS_TORCHAUDIO_INDEX_URL")

    if use_cuda:
        torch_spec = torch_spec or "torch==2.3.1"
        extra_url = os.environ.get("INDEX_TTS_TORCH_EXTRA_INDEX_URL")
        if extra_url:
            torch_extra_indexes = (extra_url,)
        emit(
            "log",
            level="info",
            message=f"Using PyTorch dependency {torch_spec} (CUDA-enabled configuration)",
        )
    else:
        torch_spec = torch_spec or "torch==2.3.1+cpu"
        torch_index_url = torch_index_url or "https://download.pytorch.org/whl/cpu"
        extra_url = os.environ.get("INDEX_TTS_TORCH_EXTRA_INDEX_URL", "https://pypi.org/simple")
        torch_extra_indexes = tuple(filter(None, (extra_url,)))
        emit(
            "log",
            level="info",
            message=(
                "Using CPU-only PyTorch build for IndexTTS dependencies; set "
                "INDEX_TTS_ENABLE_CUDA=1 to install GPU wheels"
            ),
        )

    skip_deepspeed = is_truthy(os.environ.get("INDEX_TTS_SKIP_DEEPSPEED", "1"))
    deepspeed_spec = os.environ.get("INDEX_TTS_DEEPSPEED_SPEC", "deepspeed==0.17.1")

    resolved: list[dict[str, Any]] = []
    # Derive torchaudio spec from torch if not explicitly provided
    default_torchaudio_version = "2.3.1"
    if "==" in torch_spec:
        _, version_part = torch_spec.split("==", 1)
        default_torchaudio_version = version_part
    if not torchaudio_spec:
        torchaudio_spec = f"torchaudio=={default_torchaudio_version}"
    if not use_cuda:
        if "+cpu" not in torchaudio_spec:
            if "==" in torchaudio_spec:
                name, version = torchaudio_spec.split("==", 1)
                if "+" not in version:
                    torchaudio_spec = f"{name}=={version}+cpu"
            elif "@" not in torchaudio_spec:
                torchaudio_spec = f"{torchaudio_spec}+cpu"
    if not torchaudio_index_url and not use_cuda:
        torchaudio_index_url = torch_index_url

    for dep in _BASE_DEPENDENCIES:
        if dep is _TORCH_PLACEHOLDER:
            entry = {
                "spec": torch_spec,
                "module": dep["module"],
            }
            if torch_index_url:
                entry["index_url"] = torch_index_url
            if torch_extra_indexes:
                entry["extra_index_urls"] = torch_extra_indexes
            resolved.append(entry)
        elif dep is _TORCHAUDIO_PLACEHOLDER:
            entry = {
                "spec": torchaudio_spec,
                "module": dep["module"],
            }
            if torchaudio_index_url:
                entry["index_url"] = torchaudio_index_url
            if torch_extra_indexes:
                entry["extra_index_urls"] = torch_extra_indexes
            resolved.append(entry)
        elif dep is _DEEPSPEED_PLACEHOLDER:
            if skip_deepspeed:
                emit(
                    "log",
                    level="info",
                    message=(
                        "Skipping optional dependency deepspeed; set INDEX_TTS_SKIP_DEEPSPEED=0 "
                        "to attempt installation"
                    ),
                )
            else:
                resolved.append({"spec": deepspeed_spec, "module": dep["module"], "optional": True})
        else:
            resolved.append(dict(dep))

    _RESOLVED_DEPENDENCIES = tuple(resolved)
    return _RESOLVED_DEPENDENCIES


def _install_package(
    package_spec: str,
    module_name: str,
    *,
    index_url: Optional[str] = None,
    extra_index_urls: Optional[Sequence[str]] = None,
):
    emit("log", level="info", message=f"Installing dependency: {package_spec}")
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", package_spec]
    if index_url:
        command.extend(["--index-url", index_url])
    if extra_index_urls:
        for url in extra_index_urls:
            if url:
                command.extend(["--extra-index-url", url])
    subprocess.check_call(command)
    # Purge old module cache so the new version is imported.
    to_delete = [name for name in sys.modules if name == module_name or name.startswith(f"{module_name}.")]
    for name in to_delete:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


def ensure_package(
    package_spec: str,
    import_name: Optional[str] = None,
    validator: Optional[Callable[[Any], bool]] = None,
    *,
    index_url: Optional[str] = None,
    extra_index_urls: Optional[Sequence[str]] = None,
    optional: bool = False,
):
    module_name = import_name or _module_name_from_spec(package_spec)
    validator = validator or PACKAGE_VALIDATORS.get(module_name)
    try:
        module = importlib.import_module(module_name)
        if validator and validator(module):
            return
        if validator and not validator(module):
            raise ImportError(f"Validator failed for {module_name}")
    except ImportError:
        try:
            if package_spec == INDEXTTS_MODULE_NAME:
                install_indextts_module()
            else:
                _install_package(
                    package_spec,
                    module_name,
                    index_url=index_url,
                    extra_index_urls=extra_index_urls,
                )
        except subprocess.CalledProcessError as exc:
            if optional:
                emit(
                    "log",
                    level="warning",
                    message=f"Optional dependency {package_spec} failed to install: {exc}",
                )
                return
            raise
        module = importlib.import_module(module_name)
        if validator and not validator(module):
            raise RuntimeError(f"Dependency {module_name} failed validation after installation")


def install_indextts_module():
    """
    Install the IndexTTS python module by downloading a published source tree.
    The upstream project does not provide a pip-distributable package, so we fetch
    the repo archive and place the `indextts` package inside a cache directory.
    """

    cache_root = Path(os.environ.get("INDEX_TTS_ROOT", Path.home() / ".cache" / "index_tts"))
    module_parent = cache_root / "python"
    module_dir = module_parent / INDEXTTS_MODULE_NAME

    if module_dir.exists():
        if (module_dir / "__init__.py").exists() and (module_dir / "infer_v2.py").exists():
            if str(module_parent) not in sys.path:
                sys.path.insert(0, str(module_parent))
            return
        # stale/incomplete install; refresh
        shutil.rmtree(module_dir)

    module_parent.mkdir(parents=True, exist_ok=True)

    repo_zip_url = os.environ.get("INDEX_TTS_PY_MODULE_ZIP_URL", DEFAULT_INDEXTTS_REPO_ZIP)
    emit(
        "log",
        level="info",
        message="Fetching IndexTTS python sources (first-time setup, may take a minute)…",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        zip_path = tmp_path / "indextts.zip"

        with urllib.request.urlopen(repo_zip_url) as response:
            zip_path.write_bytes(response.read())

        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_path)

        repo_root: Optional[Path] = None
        for candidate in tmp_path.iterdir():
            if candidate.is_dir() and (candidate / INDEXTTS_MODULE_NAME).exists():
                repo_root = candidate
                break
        if not repo_root:
            raise RuntimeError("Failed to locate IndexTTS repository root in downloaded archive")

        src_module = repo_root / INDEXTTS_MODULE_NAME
        if not src_module.exists():
            raise RuntimeError(
                f"Archive does not contain '{INDEXTTS_MODULE_NAME}' module; "
                "check INDEX_TTS_PY_MODULE_ZIP_URL"
            )

        if module_dir.exists():
            shutil.rmtree(module_dir)

        shutil.copytree(src_module, module_dir, dirs_exist_ok=True)

    # Ensure the package is importable on subsequent runs.
    if str(module_parent) not in sys.path:
        sys.path.insert(0, str(module_parent))

    emit("log", level="info", message="IndexTTS python module installed")


def ensure_runtime_dependencies():
    global dependencies_ready
    if dependencies_ready:
        return
    for dep in _get_additional_dependencies():
        ensure_package(
            dep["spec"],
            dep.get("module"),
            dep.get("validator"),
            index_url=dep.get("index_url"),
            extra_index_urls=dep.get("extra_index_urls"),
            optional=dep.get("optional", False),
        )
    dependencies_ready = True


def find_config(models_dir: str) -> Tuple[str, str]:
    for root, _dirs, files in os.walk(models_dir):
        if "config.yaml" in files:
            cfg_path = os.path.join(root, "config.yaml")
            return cfg_path, root
    raise FileNotFoundError("Unable to locate config.yaml under models directory")


def prepare_environment(models_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    cache_dir = os.path.join(models_dir, "hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def handle_download(args):
    ensure_package("indextts")
    ensure_package("huggingface_hub")
    ensure_package("modelscope")
    ensure_package("soundfile")
    ensure_package("torch")
    prepare_environment(args.models_dir)

    emit("progress", progress=0.05, message="Downloading IndexTTS models…")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.models_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    emit("progress", progress=0.95, message="Download verified")
    emit("complete", progress=1.0, message="Download complete", output_path=args.models_dir)


def init_model(models_dir: str):
    ensure_package("indextts")
    ensure_runtime_dependencies()
    import indextts.infer_v2 as infer_v2

    prepare_environment(models_dir)
    cfg_path, checkpoint_dir = find_config(models_dir)
    emit("progress", progress=0.15, message="Initializing IndexTTS2")

    use_fp16 = is_truthy(os.environ.get("INDEX_TTS_USE_FP16"))
    use_cuda_kernel = os.environ.get("INDEX_TTS_USE_CUDA_KERNEL")
    use_deepspeed = is_truthy(os.environ.get("INDEX_TTS_ENABLE_DEEPSPEED"))
    hybrid_mode = is_truthy(os.environ.get("INDEX_TTS_HYBRID_MODE"))
    device_override = os.environ.get("INDEX_TTS_DEVICE")

    if use_cuda_kernel is not None:
        use_cuda_kernel = is_truthy(use_cuda_kernel)

    tts = infer_v2.IndexTTS2(
        cfg_path=cfg_path,
        model_dir=checkpoint_dir,
        use_fp16=use_fp16,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=use_deepspeed,
        hybrid_model_device=hybrid_mode,
        device=device_override,
    )
    return tts


def handle_load(args):
    tts = init_model(args.models_dir)
    try:
        tts.load_models()
        emit("progress", progress=0.6, message="Warm loading completed")
    finally:
        # free CUDA / GPU memory if used
        if hasattr(tts, "gpt"):
            del tts
    emit("complete", progress=1.0, message="Models loaded")


def handle_synthesize(args):
    if not os.path.exists(args.voice):
        raise FileNotFoundError(f"Voice prompt not found: {args.voice}")
    if not os.path.exists(args.text):
        raise FileNotFoundError(f"Text file not found: {args.text}")

    with open(args.text, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError("Text input is empty")

    tts = init_model(args.models_dir)
    diffusion_steps = max(1, int(args.steps))
    inference_cfg_rate = float(os.environ.get("INDEX_TTS_INFERENCE_CFG_RATE", "0.7"))
    interval_silence = os.environ.get("INDEX_TTS_INTERVAL_SILENCE")
    if interval_silence is not None:
        try:
            interval_silence = int(interval_silence)
        except ValueError:
            interval_silence = None

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
        diffusion_steps=diffusion_steps,
        inference_cfg_rate=inference_cfg_rate,
        interval_silence=interval_silence if interval_silence is not None else 200,
    )
    emit("complete", progress=1.0, message="Synthesis complete", output_path=args.output)


def main():
    parser = argparse.ArgumentParser(description="IndexTTS worker")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--models-dir", required=True)

    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--repo-id", required=True)

    subparsers.add_parser("load")

    synth_parser = subparsers.add_parser("synthesize")
    synth_parser.add_argument("--voice", required=True)
    synth_parser.add_argument("--text", required=True)
    synth_parser.add_argument("--output", required=True)
    synth_parser.add_argument("--steps", type=int, default=25)

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
