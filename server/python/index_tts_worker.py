#!/usr/bin/env python3
import argparse
import importlib
import importlib.metadata as importlib_metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import textwrap
import traceback
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

DEFAULT_INDEXTTS_REPO_ZIP = "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
INDEXTTS_MODULE_NAME = "indextts"
ADDITIONAL_DEPENDENCIES = (
    {"spec": "librosa==0.10.2.post1", "module": "librosa"},
    {"spec": "omegaconf>=2.3.0", "module": "omegaconf"},
    {"spec": "transformers==4.52.1", "module": "transformers"},
    {"spec": "accelerate==1.8.1", "module": "accelerate"},
    {"spec": "sentencepiece>=0.2.1", "module": "sentencepiece"},
    {"spec": "tokenizers==0.21.0", "module": "tokenizers"},
    {"spec": "textstat>=0.7.10", "module": "textstat"},
    {"spec": "cn2an==0.5.22", "module": "cn2an"},
    {"spec": "g2p_en==2.1.0", "module": "g2p_en"},
    {"spec": "jieba==0.42.1", "module": "jieba"},
    {"spec": "json5==0.10.0", "module": "json5"},
    {"spec": "safetensors==0.5.2", "module": "safetensors"},
)

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


def _install_package(package_spec: str, module_name: str):
    emit("log", level="info", message=f"Installing dependency: {package_spec}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", package_spec]
    )
    # Purge old module cache so the new version is imported.
    to_delete = [name for name in sys.modules if name == module_name or name.startswith(f"{module_name}.")]
    for name in to_delete:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


def ensure_package(
    package_spec: str, import_name: Optional[str] = None, validator: Optional[Callable[[Any], bool]] = None
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
        if package_spec == INDEXTTS_MODULE_NAME:
            install_indextts_module()
        else:
            _install_package(package_spec, module_name)
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
    for dep in ADDITIONAL_DEPENDENCIES:
        ensure_package(dep["spec"], dep.get("module"))
    dependencies_ready = True


def find_config(models_dir: str) -> Tuple[str, str]:
    for root, _dirs, files in os.walk(models_dir):
        if "config.yaml" in files:
            cfg_path = os.path.join(root, "config.yaml")
            return cfg_path, root
    raise FileNotFoundError("Unable to locate config.yaml under models directory")


def patch_diffusion_steps():
    import indextts.infer_v2 as infer_v2_module

    source = inspect.getsource(infer_v2_module.IndexTTS2.infer_generator)
    if "getattr(self, '_diffusion_steps', 25)" in source:
        return

    patched_source = source.replace(
        "diffusion_steps = 25",
        "diffusion_steps = getattr(self, '_diffusion_steps', 25)",
    )
    exec_namespace = {}
    exec(textwrap.dedent(patched_source), infer_v2_module.__dict__, exec_namespace)
    infer_v2_module.IndexTTS2.infer_generator = exec_namespace["infer_generator"]


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
    patch_diffusion_steps()
    cfg_path, checkpoint_dir = find_config(models_dir)
    emit("progress", progress=0.15, message="Initializing IndexTTS2")
    tts = infer_v2.IndexTTS2(cfg_path=cfg_path, model_dir=checkpoint_dir, use_fp16=False)
    return tts


def handle_load(args):
    tts = init_model(args.models_dir)
    emit("progress", progress=0.6, message="Warm loading completed")
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
    tts._diffusion_steps = int(args.steps)

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
