#!/usr/bin/env python3
import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
import traceback
from typing import Optional, Tuple


def emit(event: str, **payload):
    message = {"event": event}
    message.update(payload)
    print(json.dumps(message), flush=True)


def ensure_package(package_name: str, import_name: Optional[str] = None):
    module_name = import_name or package_name
    try:
        importlib.import_module(module_name)
    except ImportError:
        emit("log", level="info", message=f"Installing missing package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


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
