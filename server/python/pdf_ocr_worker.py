import argparse
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional


def emit(event: str, **data):
  payload = {"event": event}
  for key, value in data.items():
    if value is not None:
      payload[key] = value
  sys.stdout.write(json.dumps(payload) + "\n")
  sys.stdout.flush()


def read_stream(stream, level: str = "info"):
  for raw_line in iter(stream.readline, ""):
    line = raw_line.strip()
    if not line:
      continue
    emit("log", level=level, message=line)
  stream.close()


def monitor_output(output_dir: Path, total_pages: int, stop_event: threading.Event):
  processed_pages = 0
  last_emit = 0.0
  while not stop_event.is_set():
    txt_files = sorted(p for p in output_dir.rglob("*.txt") if p.is_file())
    count = len(txt_files)
    if count != processed_pages:
      processed_pages = count
      progress = (processed_pages / total_pages) if total_pages > 0 else 0.0
      message = f"Processed {processed_pages}/{total_pages} pages" if total_pages else f"Processed {processed_pages} pages"
      emit(
        "progress",
        processed_pages=processed_pages,
        total_pages=total_pages,
        progress=progress,
        message=message,
      )
      last_emit = time.time()
    else:
      now = time.time()
      if processed_pages and now - last_emit > 10:
        progress = (processed_pages / total_pages) if total_pages > 0 else 0.0
        message = f"Waiting for remaining pages… ({processed_pages}/{total_pages})" if total_pages else "Waiting for OCR output…"
        emit(
          "progress",
          processed_pages=processed_pages,
          total_pages=total_pages,
          progress=progress,
          message=message,
        )
        last_emit = now
    stop_event.wait(0.75)


def combine_text(output_dir: Path) -> str:
  parts: List[str] = []
  for txt_file in sorted(output_dir.rglob("*.txt")):
    try:
      content = txt_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
      content = txt_file.read_text(encoding="latin-1")
    cleaned = content.strip()
    if cleaned:
      parts.append(cleaned)
  return "\n\n".join(parts)


def build_commands(args) -> List[List[str]]:
  python = sys.executable
  pdf_path = str(args.pdf_path)
  output_dir = str(args.output_dir)
  models_dir = str(args.models_dir)

  command_strings: List[str] = []

  env_command = os.environ.get("DEEPSEEK_OCR_PDF_COMMAND")
  if env_command:
    command_strings.append(env_command)

  # CLI variants (deepseek-ocr)
  cli_executable = shutil.which("deepseek-ocr")
  if cli_executable:
    command_strings.extend([
      f"{cli_executable} pdf --input {shlex.quote(pdf_path)} --output {shlex.quote(output_dir)} --models-dir {shlex.quote(models_dir)} --auto-download",
      f"{cli_executable} pdf --input {shlex.quote(pdf_path)} --output-dir {shlex.quote(output_dir)} --models-dir {shlex.quote(models_dir)} --auto-download",
    ])

  # Python module variants
  command_strings.extend([
    f"{python} -m deepseek_ocr.bin.pdf --pdf {shlex.quote(pdf_path)} --output {shlex.quote(output_dir)} --model-dir {shlex.quote(models_dir)} --auto-download",
    f"{python} -m deepseek_ocr.bin.pdf --pdf {shlex.quote(pdf_path)} --output-dir {shlex.quote(output_dir)} --model-dir {shlex.quote(models_dir)} --auto-download",
    f"{python} -m deepseek_ocr.bin.pdf_infer --pdf {shlex.quote(pdf_path)} --output {shlex.quote(output_dir)} --model-dir {shlex.quote(models_dir)} --auto-download",
    f"{python} -m deepseek_ocr.bin.pdf_infer --pdf {shlex.quote(pdf_path)} --output-dir {shlex.quote(output_dir)} --model-dir {shlex.quote(models_dir)} --auto-download",
    f"{python} -m deepseek_ocr pdf --input {shlex.quote(pdf_path)} --output {shlex.quote(output_dir)} --models-dir {shlex.quote(models_dir)} --auto-download",
    f"{python} -m deepseek_ocr pdf --input {shlex.quote(pdf_path)} --output-dir {shlex.quote(output_dir)} --models-dir {shlex.quote(models_dir)} --auto-download",
  ])

  commands: List[List[str]] = []
  for command_string in command_strings:
    try:
      formatted = command_string.format(
        pdf_path=pdf_path,
        output_dir=output_dir,
        models_dir=models_dir,
        python=python,
      )
      commands.append(shlex.split(formatted))
    except Exception:
      try:
        commands.append(shlex.split(command_string))
      except Exception:
        continue
  return commands


def run_command(command: List[str], env: dict, output_dir: Path, total_pages: int) -> Optional[str]:
  try:
    process = subprocess.Popen(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      bufsize=1,
      env=env,
    )
  except FileNotFoundError as exc:
    emit("log", level="error", message=f"Command not found: {command[0]}")
    return f"Command not found: {command[0]}"
  except Exception as exc:
    emit("log", level="error", message=f"Failed to start command: {exc}")
    return str(exc)

  stop_event = threading.Event()
  monitor_thread = threading.Thread(
    target=monitor_output, args=(output_dir, total_pages, stop_event), daemon=True
  )
  monitor_thread.start()

  stdout_thread = threading.Thread(
    target=read_stream, args=(process.stdout, "info"), daemon=True
  )
  stderr_thread = threading.Thread(
    target=read_stream, args=(process.stderr, "warn"), daemon=True
  )
  stdout_thread.start()
  stderr_thread.start()

  exit_code = process.wait()
  stop_event.set()
  monitor_thread.join(timeout=1.0)
  stdout_thread.join(timeout=1.0)
  stderr_thread.join(timeout=1.0)

  if exit_code != 0:
    return f"DeepSeek OCR exited with code {exit_code}"
  return None


def _models_present(models_dir: Path) -> bool:
  if not models_dir.exists():
    return False
  for path in models_dir.rglob("*"):
    if not path.is_file():
      continue
    suffix = path.suffix.lower()
    if suffix in {".bin", ".pt", ".pth", ".onnx", ".safetensors", ".json", ".yaml", ".yml"}:
      try:
        if path.stat().st_size > 0:
          return True
      except OSError:
        continue
  return False


def _run_downloader(command: Iterable[str], env: dict) -> bool:
  command_list = list(command)
  emit("log", level="info", message=f"Attempting DeepSeek OCR model download: {' '.join(command_list)}")
  try:
    result = subprocess.run(
      command_list,
      check=False,
      env=env,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
  except FileNotFoundError:
    emit("log", level="warn", message=f"Downloader command not found: {command_list[0]}")
    return False
  except Exception as exc:
    emit("log", level="warn", message=f"Failed to run downloader command: {exc}")
    return False

  if result.stdout:
    emit("log", level="info", message=result.stdout.strip())
  if result.returncode != 0:
    message = result.stderr.strip() or f"Downloader exited with code {result.returncode}"
    emit("log", level="warn", message=message)
    return False
  if result.stderr:
    emit("log", level="info", message=result.stderr.strip())
  return True


def ensure_models(models_dir: Path) -> None:
  if _models_present(models_dir):
    return

  env = os.environ.copy()
  env.setdefault("DEEPSEEK_OCR_HOME", str(models_dir))

  download_commands: List[Iterable[str]] = [
    [sys.executable, "-m", "deepseek_ocr.bin.download", "--output", str(models_dir)],
    [sys.executable, "-m", "deepseek_ocr.bin.download", "--output-dir", str(models_dir)],
    [sys.executable, "-m", "deepseek_ocr.download", "--output", str(models_dir)],
    [sys.executable, "-m", "deepseek_ocr.download", "--output-dir", str(models_dir)],
  ]

  for command in download_commands:
    if _run_downloader(command, env) and _models_present(models_dir):
      emit("log", level="info", message=f"DeepSeek OCR models downloaded to {models_dir}")
      return

  repo_id = os.environ.get("DEEPSEEK_OCR_MODEL_REPO", "deepseek-ai/deepseek-ocr")
  revision = os.environ.get("DEEPSEEK_OCR_MODEL_REVISION")
  emit(
    "log",
    level="info",
    message=f"Falling back to Hugging Face snapshot download for {repo_id}",
  )
  try:
    from huggingface_hub import snapshot_download  # type: ignore
  except ImportError:
    emit(
      "error",
      error=(
        "huggingface_hub is not installed. Install it in the DeepSeek OCR environment with "
        "'pip install huggingface_hub' so VoiceForge can download the models from Hugging Face."
      ),
    )
    raise SystemExit(1)

  try:
    snapshot_download(
      repo_id=repo_id,
      revision=revision,
      local_dir=str(models_dir),
      local_dir_use_symlinks=False,
    )
  except Exception as exc:
    emit(
      "error",
      error=(
        f"Failed to download DeepSeek OCR models from Hugging Face ({repo_id}). "
        f"Error: {exc}. You can download them manually and place them in {models_dir}."
      ),
    )
    raise SystemExit(1)

  if not _models_present(models_dir):
    emit(
      "error",
      error=(
        f"No DeepSeek OCR models found in {models_dir} after download. Verify the repository contains the inference weights "
        "or provide a populated directory via DEEPSEEK_OCR_HOME."
      ),
    )
    raise SystemExit(1)

  emit("log", level="info", message=f"DeepSeek OCR models synced from Hugging Face to {models_dir}")


def _iter_candidate_roots() -> List[Path]:
  candidates: List[Path] = []

  env_vars = [
    os.environ.get("DEEPSEEK_OCR_REPO"),
    os.environ.get("DEEPSEEK_OCR_ROOT"),
    os.environ.get("DEEPSEEK_OCR_HOME"),
  ]
  for value in env_vars:
    if value:
      candidates.append(Path(value).expanduser())

  script_path = Path(__file__).resolve()
  project_root = script_path.parents[3]
  system_drive = os.environ.get("SystemDrive")
  user_profile = os.environ.get("USERPROFILE")
  common_roots = [
    Path.cwd(),
    project_root,
    project_root.parent,
    Path.home(),
  ]

  if system_drive:
    common_roots.append(Path(system_drive + "\\"))
  if user_profile:
    common_roots.append(Path(user_profile))

  for root in common_roots:
    for name in ("deepseek-ocr", "DeepSeek-OCR"):
      candidates.append(root / name)

  candidates.extend([Path("/deepseek-ocr"), Path("/DeepSeek-OCR")])

  unique_candidates: List[Path] = []
  seen: set = set()
  for candidate in candidates:
    resolved = candidate.resolve() if candidate.exists() else candidate
    key = str(resolved).lower()
    if key not in seen:
      seen.add(key)
      unique_candidates.append(candidate)
  return unique_candidates


def _import_from_candidates() -> Optional[object]:
  for candidate in _iter_candidate_roots():
    try_paths: List[Path] = []
    for path in (candidate, candidate / "src", candidate / "python"):
      if path.exists() and path.is_dir():
        try_paths.append(path)
    for path in try_paths:
      if str(path) not in sys.path:
        sys.path.insert(0, str(path))
      try:
        module = importlib.import_module("deepseek_ocr")
        emit("log", level="info", message=f"Resolved DeepSeek OCR from {path}")
        return module
      except ImportError:
        continue
  return None


def ensure_dependencies():
  module = None
  try:
    import deepseek_ocr as module  # type: ignore
  except ImportError:
    module = _import_from_candidates()
    if module is None:
      emit(
        "error",
        error=(
          "DeepSeek OCR is unavailable. Clone the repository with "
          "'git clone https://github.com/deepseek-ai/deepseek-ocr.git', create and activate "
          "a virtual environment (e.g. 'python -m venv .venv' and 'source .venv/bin/activate'), "
          "install its requirements with 'pip install -r requirements.txt', and either run "
          "VoiceForge from that checkout or set DEEPSEEK_OCR_REPO to the clone directory so "
          "the bundled inference scripts are on the Python path. Run VoiceForge from that "
          "environment so DeepSeek OCR can download the required models."
        ),
      )
      raise SystemExit(1)

  if module is not None:
    module_path = Path(module.__file__).resolve().parent
    emit("log", level="info", message=f"Using DeepSeek OCR from {module_path}")
    emit("config", deepseek_module_path=str(module_path))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="DeepSeek PDF OCR worker")
  parser.add_argument("--job-id")
  parser.add_argument("--pdf-path")
  parser.add_argument("--output-path")
  parser.add_argument("--models-dir", required=True)
  parser.add_argument("--total-pages", type=int, default=1)
  parser.add_argument("--download-models", action="store_true", help="Download models without running OCR")
  args = parser.parse_args()

  models_dir = Path(args.models_dir)
  models_dir.mkdir(parents=True, exist_ok=True)

  ensure_dependencies()

  if args.download_models:
    ensure_models(models_dir)
    emit(
      "complete",
      message="DeepSeek OCR models are ready",
      models_dir=str(models_dir),
    )
    raise SystemExit(0)

  if not args.job_id or not args.pdf_path or not args.output_path:
    emit(
      "error",
      error=(
        "Missing required arguments for OCR job. Provide --job-id, --pdf-path, and --output-path "
        "or run with --download-models."
      ),
    )
    raise SystemExit(1)

  pdf_path = Path(args.pdf_path)
  output_path = Path(args.output_path)
  output_dir = output_path.parent / "deepseek_output"
  output_dir.mkdir(parents=True, exist_ok=True)

  ensure_models(models_dir)

  emit(
    "progress",
    processed_pages=0,
    total_pages=max(1, args.total_pages),
    progress=0.0,
    message="Starting DeepSeek PDF OCR…",
  )

  env = os.environ.copy()
  env.setdefault("DEEPSEEK_OCR_HOME", str(models_dir))

  commands = build_commands(
    SimpleNamespace(pdf_path=pdf_path, output_dir=output_dir, models_dir=models_dir)
  )

  if not commands:
    emit(
      "error",
      error="Unable to determine DeepSeek OCR command. Set DEEPSEEK_OCR_PDF_COMMAND to the correct invocation.",
    )
    raise SystemExit(1)

  last_error = None
  for command in commands:
    emit("log", level="info", message=f"Running command: {' '.join(command)}")
    error_message = run_command(command, env, output_dir, max(1, args.total_pages))
    if error_message is None:
      break
    last_error = error_message
  else:
    emit("error", error=last_error or "DeepSeek OCR command failed")
    raise SystemExit(1)

  combined_text = combine_text(output_dir)
  output_path.write_text(combined_text, encoding="utf-8")

  processed_count = sum(1 for _ in output_dir.rglob("*.txt"))
  final_processed = processed_count if processed_count > 0 else max(1, args.total_pages)

  emit(
    "complete",
    message="DeepSeek OCR completed",
    processed_pages=final_processed,
    total_pages=max(1, args.total_pages),
    output_path=str(output_path),
    text=combined_text,
  )
