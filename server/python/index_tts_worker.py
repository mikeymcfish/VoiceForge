#!/usr/bin/env python3
import argparse
import importlib
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
from typing import Optional, Tuple

DEFAULT_INDEXTTS_REPO_ZIP = "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
INDEXTTS_MODULE_NAME = "indextts"
ADDITIONAL_DEPENDENCIES = (
    ("librosa", None),
    ("omegaconf", None),
    ("transformers", None),
    ("accelerate", None),
    ("sentencepiece", None),
    ("tokenizers", None),
    ("textstat", None),
    ("cn2an", None),
    ("g2p_en", None),
    ("jieba", None),
    ("json5", None),
    ("safetensors", None),
)

dependencies_ready = False


def emit(event: str, **payload):
    message = {"event": event}
    message.update(payload)
    print(json.dumps(message), flush=True)


def ensure_package(package_name: str, import_name: Optional[str] = None):
    module_name = import_name or package_name
    try:
        importlib.import_module(module_name)
    except ImportError:
        if package_name == INDEXTTS_MODULE_NAME:
            install_indextts_module()
        else:
            emit("log", level="info", message=f"Installing missing package: {package_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        importlib.invalidate_caches()
        importlib.import_module(module_name)


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
    for package_name, import_name in ADDITIONAL_DEPENDENCIES:
        ensure_package(package_name, import_name)
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
