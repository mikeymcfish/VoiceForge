import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def emit(event: str, **payload):
    message = {"event": event}
    message.update(payload)
    print(json.dumps(message), flush=True)


def prepare_environment(root_dir: str) -> Dict[str, str]:
    os.makedirs(root_dir, exist_ok=True)
    models_dir = os.path.join(root_dir, "models")
    cache_dir = os.path.join(root_dir, "cache")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    env = {
        "HF_HOME": cache_dir,
        "HUGGINGFACE_HUB_CACHE": cache_dir,
        "TORCH_HOME": os.path.join(root_dir, "torch_cache"),
        "VIBEVOICE_ROOT": root_dir,
        "VIBEVOICE_MODELS_DIR": models_dir,
    }
    for key, value in env.items():
        os.environ.setdefault(key, value)
    return env


def run_process(
    cmd: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    stream: bool = True,
    log_level: str = "info",
) -> None:
    emit("log", level="info", message=f"Running: {' '.join(cmd)}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=merged_env,
        stdout=subprocess.PIPE if stream else None,
        stderr=subprocess.STDOUT if stream else None,
        text=True,
    )

    if stream and process.stdout:
        for line in process.stdout:
            if line.strip():
                emit("log", level=log_level, message=line.rstrip())

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def ensure_repo(repo_url: str, branch: str, repo_dir: str) -> None:
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        emit("progress", progress=0.15, message="Updating existing VibeVoice repository…")
        run_process(["git", "fetch", "origin"], cwd=repo_dir)
        run_process(["git", "checkout", branch], cwd=repo_dir)
        run_process(["git", "pull", "origin", branch], cwd=repo_dir)
    else:
        emit("progress", progress=0.1, message="Cloning VibeVoice repository…")
        parent = os.path.dirname(repo_dir)
        os.makedirs(parent, exist_ok=True)
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        run_process(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                repo_url,
                repo_dir,
            ],
            cwd=parent,
        )


def install_requirements(repo_dir: str) -> None:
    requirement_files = [
        "requirements.txt",
        "requirements.in",
        "requirements-dev.txt",
    ]
    for file_name in requirement_files:
        req_path = os.path.join(repo_dir, file_name)
        if os.path.isfile(req_path):
            emit("progress", progress=0.35, message=f"Installing dependencies from {file_name}…")
            run_process([sys.executable, "-m", "pip", "install", "-r", req_path], cwd=repo_dir)
            return

    pyproject_path = Path(repo_dir) / "pyproject.toml"
    if pyproject_path.is_file():
        emit("progress", progress=0.32, message="Installing VibeVoice package (editable)…")
        run_process([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(repo_dir)], cwd=repo_dir)

        try:
            import tomllib  # type: ignore[attr-defined]
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore

        with pyproject_path.open("rb") as fh:
            data = tomllib.load(fh)

        raw_dependencies: Iterable[str] = data.get("project", {}).get("dependencies", [])
        dependencies: List[str] = []
        for spec in raw_dependencies:
            clean = spec.strip()
            if not clean:
                continue
            # Avoid overriding torch installations supplied by the environment.
            if clean.split("[")[0].strip().startswith("torch"):
                continue
            dependencies.append(clean)

        if dependencies:
            emit(
                "progress",
                progress=0.36,
                message="Installing runtime dependencies from pyproject…",
            )
            run_process([sys.executable, "-m", "pip", "install", *dependencies], cwd=repo_dir)
        else:
            emit(
                "log",
                level="info",
                message="No additional dependencies declared in pyproject.toml",
            )
        return

    emit("log", level="warn", message="No requirements or pyproject file found; skipping dependency installation")


def try_download_assets(repo_dir: str) -> None:
    download_candidates: List[Tuple[List[str], str]] = [
        ([sys.executable, "scripts/download_models.py"], "scripts/download_models.py"),
        ([sys.executable, "download_models.py"], "download_models.py"),
        ([sys.executable, "tools/download_models.py"], "tools/download_models.py"),
        ([sys.executable, "scripts/download_assets.py"], "scripts/download_assets.py"),
        ([sys.executable, "demo", "download_models.py"], "demo/download_models.py"),
    ]

    for cmd, relative in download_candidates:
        script_path = os.path.join(repo_dir, relative)
        if os.path.isfile(script_path):
            emit("progress", progress=0.55, message=f"Running {relative} to fetch assets…")
            run_process(cmd, cwd=repo_dir)
            return

    emit("log", level="info", message="No dedicated download script detected; assuming manual model management")


def ensure_hf_hub_installed() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        emit("log", level="info", message="Installing huggingface_hub for model downloads…")
        run_process([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub>=0.23.0"], stream=True)


def parse_model_ids_from_env() -> List[str]:
    raw = os.environ.get("VIBEVOICE_MODELS") or os.environ.get("VIBEVOICE_MODEL_IDS")
    if raw and raw.strip():
        # allow comma, semicolon or whitespace separators
        parts = [p.strip() for p in raw.replace(";", ",").replace("\n", ",").split(",")]
        return [p for p in parts if p]
    # sensible defaults per user request
    return [
        "microsoft/VibeVoice-1.5B",
        "aoi-ot/VibeVoice-Large",
    ]


def download_hf_models(models_dir: str) -> None:
    models = parse_model_ids_from_env()
    if not models:
        emit("log", level="info", message="No VibeVoice models specified; skipping HF download")
        return

    ensure_hf_hub_installed()

    # Import after potential install
    from huggingface_hub import snapshot_download  # type: ignore

    token = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_TOKEN")

    total = len(models)
    for idx, repo_id in enumerate(models, start=1):
        safe_name = repo_id.replace("/", "__")
        local_dir = os.path.join(models_dir, safe_name)
        os.makedirs(local_dir, exist_ok=True)

        base_prog = 0.6  # start progress after repo setup stage
        end_prog = 0.95
        prog = base_prog + (idx / max(total, 1)) * (end_prog - base_prog)
        emit("progress", progress=prog, message=f"Downloading {repo_id}…")

        try:
            snapshot_download(
                repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=token,
                resume_download=True,
                max_workers=8,
            )
            try:
                with open(os.path.join(local_dir, "repo_id.txt"), "w", encoding="utf-8") as f:
                    f.write(repo_id)
            except Exception:
                pass
            emit("log", level="info", message=f"Downloaded {repo_id} -> {local_dir}")
            emit("log", level="info", message=f"Downloaded {repo_id} → {local_dir}")
        except Exception as exc:  # noqa: BLE001
            emit(
                "log",
                level="warn",
                message=f"Failed to download {repo_id}: {exc}. You can pre-download manually or adjust VIBEVOICE_MODELS.",
            )


def _normalize_speaker_id(raw_id: int) -> int:
    return raw_id - 1 if raw_id > 0 else raw_id


def _extract_speaker_sequence(text_path: str) -> List[int]:
    sequence: List[int] = []
    seen: set[int] = set()
    pattern = re.compile(r"^Speaker\s+(\d+)\s*:")

    try:
        with open(text_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                match = pattern.match(line)
                if match:
                    speaker = _normalize_speaker_id(int(match.group(1)))
                else:
                    speaker = 0
                if speaker not in seen:
                    seen.add(speaker)
                    sequence.append(speaker)
    except FileNotFoundError:
        return []

    return sequence


def _prepare_voice_prompts(text_path: str, voices: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not voices:
        return None

    unique_speakers = _extract_speaker_sequence(text_path)
    if not unique_speakers:
        unique_speakers = [0]

    planned: List[str] = []
    voice_list = [os.path.abspath(v) for v in voices if v]
    if not voice_list:
        return None

    for index in range(len(unique_speakers)):
        if index < len(voice_list):
            planned.append(voice_list[index])
        else:
            planned.append(voice_list[-1])

    return planned


def _candidate_model_dirs(root_dir: str, model_id: str) -> List[str]:
    safe_name = model_id.replace("/", "__")
    models_root = os.path.join(root_dir, "models")
    return [
        os.path.join(models_root, safe_name),
        os.path.join(models_root, model_id),
    ]


def _resolve_local_model(root_dir: str, model_id: str) -> Optional[str]:
    if not model_id:
        return None

    for candidate in _candidate_model_dirs(root_dir, model_id):
        if os.path.isdir(candidate):
            return candidate

    models_root = os.path.join(root_dir, "models")
    try:
        for name in os.listdir(models_root):
            sub = os.path.join(models_root, name)
            if not os.path.isdir(sub):
                continue
            marker = os.path.join(sub, "repo_id.txt")
            if os.path.isfile(marker):
                try:
                    with open(marker, "r", encoding="utf-8") as handle:
                        repo_id = handle.read().strip()
                except Exception:
                    continue
                if repo_id == model_id:
                    return sub
    except Exception:
        pass
    return None


def resolve_model_path(root_dir: str, requested: Optional[str]) -> Tuple[str, Optional[str]]:
    env_override = os.environ.get("VIBEVOICE_MODEL_PATH")
    if env_override:
        return env_override, os.environ.get("VIBEVOICE_MODEL_ID")

    if requested:
        local = _resolve_local_model(root_dir, requested)
        if local:
            return local, requested
        if os.path.isdir(requested):
            return requested, requested
        return requested, requested

    defaults = parse_model_ids_from_env()
    for model_id in defaults:
        local = _resolve_local_model(root_dir, model_id)
        if local:
            return local, model_id

    if defaults:
        return defaults[0], defaults[0]

    raise RuntimeError(
        "No VibeVoice model configured. Set VIBEVOICE_MODEL_PATH or provide --model-id."
    )


def choose_device() -> str:
    preferred = os.environ.get("VIBEVOICE_DEVICE")

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for VibeVoice inference but is not installed") from exc

    def available(name: str) -> bool:
        if name == "cuda":
            return torch.cuda.is_available()
        if name == "mps":
            return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        return True

    if preferred and available(preferred):
        return preferred

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_vibevoice_modules(repo_dir: str):
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from vibevoice.modular.modeling_vibevoice_inference import (  # noqa: PLC0415
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor  # noqa: PLC0415

    return VibeVoiceProcessor, VibeVoiceForConditionalGenerationInference


def run_inprocess_inference(
    args: argparse.Namespace,
    *,
    model_path: str,
    model_id: Optional[str],
    voice_samples: Optional[List[str]],
) -> None:
    emit("progress", progress=0.18, message="Preparing VibeVoice runtime…")

    VibeVoiceProcessor, VibeVoiceForConditionalGenerationInference = _load_vibevoice_modules(args.repo_dir)

    import torch

    device = choose_device()
    emit("log", level="info", message=f"Using VibeVoice device: {device}")

    emit("progress", progress=0.28, message="Loading processor…")
    processor = VibeVoiceProcessor.from_pretrained(model_path)

    emit("progress", progress=0.38, message="Loading model weights…")
    if device == "mps":
        torch_dtype = torch.float32
        attn_impl = "sdpa"
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            device_map=None,
        )
        model.to("mps")
    elif device == "cuda":
        torch_dtype = torch.bfloat16
        attn_impl = "flash_attention_2"
        try:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="cuda",
                attn_implementation=attn_impl,
            )
        except Exception:
            emit(
                "log",
                level="warn",
                message="Falling back to SDPA attention implementation for VibeVoice",
            )
            attn_impl = "sdpa"
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="cuda",
                attn_implementation=attn_impl,
            )
    else:
        torch_dtype = torch.float32
        attn_impl = "sdpa"
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
            attn_implementation=attn_impl,
        )

    model.eval()

    steps_env = os.environ.get("VIBEVOICE_INFERENCE_STEPS") or os.environ.get("VIBEVOICE_DIFFUSION_STEPS")
    try:
        diffusion_steps = int(steps_env) if steps_env else 10
    except ValueError:
        diffusion_steps = 10
    model.set_ddpm_inference_steps(num_steps=diffusion_steps)

    cfg_scale = os.environ.get("VIBEVOICE_CFG_SCALE")
    cfg_value: Optional[float] = None
    if cfg_scale:
        try:
            cfg_value = float(cfg_scale)
        except ValueError:
            cfg_value = None
    if cfg_value is None and getattr(args, "temperature", None) is not None:
        cfg_value = float(args.temperature)
    if cfg_value is None:
        cfg_value = 1.3

    emit("progress", progress=0.48, message="Preparing synthesis inputs…")

    voice_batch: Optional[List[List[str]]] = None
    if voice_samples:
        voice_batch = [voice_samples]

    inputs = processor(
        text=[args.text],
        voice_samples=voice_batch,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    target_device = device if device in {"cuda", "mps"} else "cpu"
    for key, value in list(inputs.items()):
        if hasattr(value, "to") and callable(value.to):
            inputs[key] = value.to(target_device)

    emit("progress", progress=0.68, message="Running VibeVoice generation…")

    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_value,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=True,
        )
    end = time.time()

    speech_outputs = getattr(outputs, "speech_outputs", None)
    if not speech_outputs or speech_outputs[0] is None:
        raise RuntimeError("VibeVoice did not return any audio output")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    saved_paths = processor.save_audio(
        speech_outputs[0],
        output_path=args.output,
        normalize=True,
    )

    final_path = saved_paths[0] if saved_paths else args.output
    duration = None
    try:
        waveform = speech_outputs[0]
        if hasattr(waveform, "detach"):
            waveform = waveform.detach()
        array = waveform.cpu().numpy() if hasattr(waveform, "cpu") else waveform
        if array.ndim > 1:
            array = array.squeeze(0)
        sample_rate = 24000
        duration = array.shape[-1] / float(sample_rate)
    except Exception:
        duration = None

    elapsed = end - start
    if duration and duration > 0:
        emit(
            "log",
            level="info",
            message=f"Generation completed in {elapsed:.2f}s (RTF {elapsed / duration:.2f}×)",
        )
    else:
        emit("log", level="info", message=f"Generation completed in {elapsed:.2f}s")

    emit(
        "complete",
        progress=1.0,
        message="Synthesis complete",
        output_path=final_path,
        model_id=model_id or model_path,
    )


def setup(args: argparse.Namespace) -> None:
    prepare_environment(args.root_dir)
    repo_url = args.repo_url or "https://github.com/FurkanGozukara/VibeVoice.git"
    branch = args.repo_branch or "main"

    ensure_repo(repo_url, branch, args.repo_dir)
    install_requirements(args.repo_dir)
    try_download_assets(args.repo_dir)
    # Attempt to fetch baseline models for VibeVoice
    download_hf_models(os.path.join(args.root_dir, "models"))

    emit("complete", progress=1.0, message="VibeVoice setup complete", output_path=args.repo_dir)


class CommandCandidate(Tuple[str, List[Tuple[str, str, bool]]]):
    pass


def candidate_commands() -> List[Tuple[str, List[Tuple[str, str, bool]]]]:
    return [
        (
            "inference.py",
            [
                ("--text", "text_path", False),
                ("--output", "output_path", False),
                ("--ref", "voice_path", True),
                ("--style", "style", True),
                ("--temperature", "temperature", True),
            ],
        ),
        (
            "inference.py",
            [
                ("--text-file", "text_path", False),
                ("--output-path", "output_path", False),
                ("--reference-audio", "voice_path", True),
                ("--style", "style", True),
                ("--temperature", "temperature", True),
            ],
        ),
        (
            "cli/inference.py",
            [
                ("--text", "text_path", False),
                ("--output", "output_path", False),
                ("--ref_audio", "voice_path", True),
                ("--style", "style", True),
                ("--temperature", "temperature", True),
            ],
        ),
        (
            "scripts/inference.py",
            [
                ("--text", "text_path", False),
                ("--output", "output_path", False),
                ("--voice", "voice_path", True),
                ("--style", "style", True),
                ("--temperature", "temperature", True),
            ],
        ),
    ]


def build_default_command(
    repo_dir: str,
    replacements: Dict[str, Optional[object]],
) -> List[str]:
    python_bin = replacements.get("python") or sys.executable
    for script, args in candidate_commands():
        script_path = os.path.join(repo_dir, script)
        if not os.path.isfile(script_path):
            continue

        command: List[str] = [python_bin, script_path]
        missing_required = False
        for flag, key, optional in args:
            value = replacements.get(key)
            if value:
                if isinstance(value, (list, tuple)):
                    command.append(flag)
                    command.extend(str(item) for item in value if item is not None)
                else:
                    command.extend([flag, str(value)])
            elif not optional:
                missing_required = True
                break
        if not missing_required:
            return command

    raise RuntimeError(
        "Unable to locate a VibeVoice inference entry point. Set VIBEVOICE_COMMAND_TEMPLATE to override the command."
    )


def build_command(
    repo_dir: str,
    replacements: Dict[str, Optional[object]],
) -> List[str]:
    template = os.environ.get("VIBEVOICE_COMMAND_TEMPLATE")
    if template:
        cleaned_replacements = {
            key: value or ""
            for key, value in replacements.items()
        }
        command_string = template.format(**cleaned_replacements)
        command_tokens = shlex.split(command_string)
        if not command_tokens:
            raise ValueError("VIBEVOICE_COMMAND_TEMPLATE expanded to an empty command")
        return command_tokens

    return build_default_command(repo_dir, replacements)




def synthesize(args: argparse.Namespace) -> None:
    prepare_environment(args.root_dir)
    if not os.path.isdir(args.repo_dir):
        raise FileNotFoundError(
            "VibeVoice repository not found. Run setup before starting synthesis."
        )

    if not os.path.isfile(args.text):
        raise FileNotFoundError(f"Text file not found: {args.text}")

    with open(args.text, "r", encoding="utf-8") as fh:
        text_content = fh.read().strip()
    if not text_content:
        raise ValueError("Text input is empty")

    voices: List[str] = []
    if args.voice:
        for v in args.voice:
            if not os.path.isfile(v):
                raise FileNotFoundError(f"Voice reference not found: {v}")
            voices.append(v)

    job_dir = os.path.dirname(os.path.abspath(args.text))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    voice_samples = _prepare_voice_prompts(args.text, voices)
    if voice_samples and len(set(voice_samples)) < len(voice_samples):
        emit(
            "log",
            level="info",
            message="Reusing voice prompts for speakers without dedicated references",
        )

    model_path, resolved_model_id = resolve_model_path(args.root_dir, getattr(args, "model_id", None))

    try:
        run_inprocess_inference(
            args,
            model_path=model_path,
            model_id=resolved_model_id,
            voice_samples=voice_samples,
        )
        return
    except Exception as exc:  # noqa: BLE001
        emit(
            "log",
            level="warn",
            message=f"In-process VibeVoice inference failed ({exc}); attempting CLI fallback",
        )

    selected_model_dir = _resolve_local_model(args.root_dir, getattr(args, "model_id", "") or "")

    replacements: Dict[str, Optional[object]] = {
        "python": sys.executable,
        "text_path": args.text,
        "output_path": args.output,
        "output_dir": os.path.dirname(args.output),
        "voice_path": voices[0] if voices else None,
        "style": getattr(args, "style", None),
        "temperature": str(args.temperature) if getattr(args, "temperature", None) is not None else None,
        "job_id": args.job_id,
        "job_dir": job_dir,
        "repo_dir": args.repo_dir,
        "root_dir": args.root_dir,
        "models_dir": os.path.join(args.root_dir, "models"),
        "model_id": getattr(args, "model_id", None),
        "model_dir": selected_model_dir,
        "model_path": model_path,
        "voice1": voices[0] if len(voices) > 0 else None,
        "voice2": voices[1] if len(voices) > 1 else None,
        "voice3": voices[2] if len(voices) > 2 else None,
        "voice4": voices[3] if len(voices) > 3 else None,
        "voice_arg": f"--voice {voices[0]}" if voices else None,
        "voices_arg": " ".join([f"--voice {p}" for p in voices]) if voices else None,
        "ref_args": " ".join([f"--ref {p}" for p in voices]) if voices else None,
        "ref_audio_args": " ".join([f"--ref_audio {p}" for p in voices]) if voices else None,
        "reference_audio_args": " ".join([f"--reference-audio {p}" for p in voices]) if voices else None,
        "style_arg": f"--style {getattr(args, 'style', '')}" if getattr(args, "style", None) else None,
        "temperature_arg": f"--temperature {getattr(args, 'temperature', '')}" if getattr(args, "temperature", None) is not None else None,
        "speaker_names": None,
        "cfg_scale": str(getattr(args, "temperature", "")) if getattr(args, "temperature", None) is not None else None,
    }

    command = build_command(args.repo_dir, replacements)

    emit("progress", progress=0.2, message="Running VibeVoice synthesis via CLI…")
    run_process(command, cwd=args.repo_dir)

    if not os.path.isfile(args.output):
        raise FileNotFoundError(
            f"Expected output audio not found at {args.output}. Check your VibeVoice command configuration."
        )

    emit("complete", progress=1.0, message="Synthesis complete", output_path=args.output)


def main():
    parser = argparse.ArgumentParser(description="VibeVoice worker")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--repo-url")
    parser.add_argument("--repo-branch")

    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument("--repo-url")
    setup_parser.add_argument("--repo-branch")

    synth_parser = subparsers.add_parser("synthesize")
    synth_parser.add_argument("--job-id", required=True)
    synth_parser.add_argument("--text", required=True)
    synth_parser.add_argument("--output", required=True)
    synth_parser.add_argument("--voice", action="append")
    synth_parser.add_argument("--style")
    synth_parser.add_argument("--temperature", type=float)
    synth_parser.add_argument("--model-id")

    args = parser.parse_args()

    try:
        if args.command == "setup":
            setup(args)
        elif args.command == "synthesize":
            synthesize(args)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:  # noqa: BLE001
        emit("error", error=str(exc), message="VibeVoice operation failed")
        raise


if __name__ == "__main__":
    main()
