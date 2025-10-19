import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple


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

    emit("log", level="warn", message="No requirements file found; skipping dependency installation")


def try_download_assets(repo_dir: str) -> None:
    download_candidates: List[Tuple[List[str], str]] = [
        ([sys.executable, "scripts/download_models.py"], "scripts/download_models.py"),
        ([sys.executable, "download_models.py"], "download_models.py"),
        ([sys.executable, "tools/download_models.py"], "tools/download_models.py"),
        ([sys.executable, "scripts/download_assets.py"], "scripts/download_assets.py"),
    ]

    for cmd, relative in download_candidates:
        script_path = os.path.join(repo_dir, relative)
        if os.path.isfile(script_path):
            emit("progress", progress=0.55, message=f"Running {relative} to fetch assets…")
            run_process(cmd, cwd=repo_dir)
            return

    emit("log", level="info", message="No dedicated download script detected; assuming manual model management")


def setup(args: argparse.Namespace) -> None:
    prepare_environment(args.root_dir)
    repo_url = args.repo_url or "https://github.com/vibevoice-community/VibeVoice.git"
    branch = args.repo_branch or "main"

    ensure_repo(repo_url, branch, args.repo_dir)
    install_requirements(args.repo_dir)
    try_download_assets(args.repo_dir)

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
    replacements: Dict[str, Optional[str]],
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
                command.extend([flag, value])
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
    replacements: Dict[str, Optional[str]],
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

    if args.voice and not os.path.isfile(args.voice):
        raise FileNotFoundError(f"Voice reference not found: {args.voice}")

    job_dir = os.path.dirname(os.path.abspath(args.text))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    replacements: Dict[str, Optional[str]] = {
        "python": sys.executable,
        "text_path": args.text,
        "output_path": args.output,
        "voice_path": args.voice,
        "style": args.style,
        "temperature": str(args.temperature) if args.temperature is not None else None,
        "job_id": args.job_id,
        "job_dir": job_dir,
        "repo_dir": args.repo_dir,
        "root_dir": args.root_dir,
        "models_dir": os.path.join(args.root_dir, "models"),
        "voice_arg": f"--voice {args.voice}" if args.voice else None,
        "style_arg": f"--style {args.style}" if args.style else None,
        "temperature_arg": f"--temperature {args.temperature}" if args.temperature is not None else None,
    }

    command = build_command(args.repo_dir, replacements)

    emit("progress", progress=0.2, message="Running VibeVoice synthesis…")
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
    synth_parser.add_argument("--voice")
    synth_parser.add_argument("--style")
    synth_parser.add_argument("--temperature", type=float)

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
