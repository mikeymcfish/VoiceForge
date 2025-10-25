import argparse
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
from typing import List, Optional


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


def ensure_dependencies():
  try:
    import deepseek_ocr  # noqa: F401
  except ImportError as exc:
    emit(
      "error",
      error=(
        "The 'deepseek-ocr' package is not installed. Install it with 'pip install deepseek-ocr' "
        "inside the Python environment used for VoiceForge."
      ),
    )
    raise SystemExit(1) from exc


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="DeepSeek PDF OCR worker")
  parser.add_argument("--job-id", required=True)
  parser.add_argument("--pdf-path", required=True)
  parser.add_argument("--output-path", required=True)
  parser.add_argument("--models-dir", required=True)
  parser.add_argument("--total-pages", type=int, default=1)
  args = parser.parse_args()

  ensure_dependencies()

  pdf_path = Path(args.pdf_path)
  output_path = Path(args.output_path)
  output_dir = output_path.parent / "deepseek_output"
  output_dir.mkdir(parents=True, exist_ok=True)
  models_dir = Path(args.models_dir)
  models_dir.mkdir(parents=True, exist_ok=True)

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
