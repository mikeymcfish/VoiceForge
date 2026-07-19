#!/usr/bin/env python3
"""Offline, same-machine benchmark harness for the two MOSS-TTS v1.5 backends.

The harness deliberately does not know how to install or download either model.
It runs two explicit argv templates with Hugging Face/Transformers offline mode
forced on, alternates backend order, and records cold-process latency, WAV
duration, end-to-end RTF, output hashes, worker event timing, and optional GPU
memory samples.

Command templates are JSON arrays. They must contain ``{text}`` and ``{output}``
placeholders and may contain ``{voice}``, ``{seed}``, ``{iteration}``, and
``{phase}``. For example (PowerShell quoting):

  python scripts/moss-backend-benchmark.py `
    --text .\\benchmark.txt `
    --voice .\\reference.wav `
    --output-dir .\\benchmark-results `
    --delay-command-json '["C:\\path\\delay\\python.exe","server/python/moss_tts_worker.py","--root-dir","C:\\path\\delay-root","synthesize","--text","{text}","--output","{output}","--voice","{voice}"]' `
    --local-command-json '["C:\\path\\local\\python.exe","path/to/local_adapter.py","--text","{text}","--output","{output}","--voice","{voice}"]'

The current VoiceForge MOSS worker writes its WAV only after generation, so
``first_output_file_seconds`` is not model time-to-first-audio for that backend.
An adapter can emit ``{"event":"first_audio"}`` when it has a real streaming
sample; that timestamp is recorded separately as ``reported_ttfa_seconds``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


BACKEND_DELAY = "delay-8b"
BACKEND_LOCAL = "local-transformer-v1.5"
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
ALLOWED_PLACEHOLDERS = {"text", "output", "voice", "seed", "iteration", "phase"}
FORBIDDEN_COMMAND_WORDS = {"download", "setup", "snapshot-download"}
PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class Backend:
    name: str
    command_template: tuple[str, ...]


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    uuid: str
    driver_version: str
    memory_total_mib: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_command_json(raw: str, option_name: str) -> tuple[str, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must be a valid JSON array: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"{option_name} must be a non-empty JSON array")
    if not all(isinstance(token, str) and token for token in value):
        raise ValueError(f"{option_name} must contain only non-empty strings")
    return tuple(value)


def validate_command_template(command: Sequence[str], *, voice_supplied: bool) -> None:
    placeholders = {
        match.group(1)
        for token in command
        for match in PLACEHOLDER_PATTERN.finditer(token)
    }
    unknown = placeholders - ALLOWED_PLACEHOLDERS
    if unknown:
        raise ValueError(f"Unsupported command placeholders: {sorted(unknown)}")
    missing = {"text", "output"} - placeholders
    if missing:
        raise ValueError(f"Command template is missing placeholders: {sorted(missing)}")
    if "voice" in placeholders and not voice_supplied:
        raise ValueError("Command template uses {voice}, but --voice was not supplied")

    normalized_words = {
        word.lower().lstrip("-/")
        for token in command
        for word in re.split(r"[\s=:]+", token)
        if word
    }
    forbidden = normalized_words & FORBIDDEN_COMMAND_WORDS
    if forbidden:
        raise ValueError(
            "Benchmark commands may synthesize only; setup/download words are "
            f"rejected: {sorted(forbidden)}"
        )


def expand_command(
    backend: Backend,
    *,
    text: Path,
    output: Path,
    voice: Path | None,
    seed: int,
    iteration: int,
    phase: str,
) -> list[str]:
    replacements = {
        "text": str(text),
        "output": str(output),
        "voice": str(voice) if voice else "",
        "seed": str(seed),
        "iteration": str(iteration),
        "phase": phase,
    }
    return [
        PLACEHOLDER_PATTERN.sub(
            lambda match: replacements.get(match.group(1), match.group(0)),
            token,
        )
        for token in backend.command_template
    ]


def read_wav_metadata(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as audio:
            frame_count = audio.getnframes()
            sample_rate = audio.getframerate()
            channels = audio.getnchannels()
            sample_width_bytes = audio.getsampwidth()
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"Output is not a readable PCM WAV file: {path}: {exc}") from exc
    if frame_count <= 0 or sample_rate <= 0:
        raise ValueError(f"Output WAV is empty or has an invalid sample rate: {path}")
    return {
        "audio_duration_seconds": frame_count / sample_rate,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width_bytes,
        "frame_count": frame_count,
    }


def query_gpu_info(gpu_index: int) -> GpuInfo | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    command = [
        executable,
        f"--id={gpu_index}",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        fields = [field.strip() for field in completed.stdout.strip().split(",", 4)]
        if len(fields) != 5:
            return None
        return GpuInfo(
            index=int(fields[0]),
            name=fields[1],
            uuid=fields[2],
            driver_version=fields[3],
            memory_total_mib=float(fields[4]),
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def query_gpu_memory_mib(gpu_index: int) -> float | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    command = [
        executable,
        f"--id={gpu_index}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(completed.stdout.strip().splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _read_stream(
    stream: Any,
    start_time: float,
    destination: list[tuple[float, str]],
) -> None:
    try:
        for line in iter(stream.readline, ""):
            destination.append((time.perf_counter() - start_time, line.rstrip("\r\n")))
    finally:
        stream.close()


def _event_timings(lines: Sequence[tuple[float, str]]) -> dict[str, float | None]:
    first_event: float | None = None
    generation_start: float | None = None
    reported_ttfa: float | None = None
    complete_event: float | None = None
    for timestamp, line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        event = payload.get("event")
        if isinstance(event, str) and first_event is None:
            first_event = timestamp
        message = payload.get("message")
        if (
            generation_start is None
            and event == "progress"
            and isinstance(message, str)
            and "synthesiz" in message.lower()
        ):
            generation_start = timestamp
        if reported_ttfa is None and event in {"first_audio", "audio_first_chunk"}:
            reported_ttfa = timestamp
        if complete_event is None and event == "complete":
            complete_event = timestamp
    return {
        "first_event_seconds": first_event,
        "generation_start_event_seconds": generation_start,
        "reported_ttfa_seconds": reported_ttfa,
        "complete_event_seconds": complete_event,
    }


def run_once(
    backend: Backend,
    *,
    command: Sequence[str],
    output: Path,
    log_prefix: Path,
    phase: str,
    iteration: int,
    seed: int,
    timeout_seconds: float,
    gpu_index: int,
    sample_gpu: bool,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    if output.exists():
        output.unlink()

    environment = os.environ.copy()
    environment.update(OFFLINE_ENVIRONMENT)
    environment["PYTHONHASHSEED"] = str(seed)
    environment["VOICEFORGE_BENCHMARK"] = "1"
    environment["VOICEFORGE_BENCHMARK_SEED"] = str(seed)

    baseline_gpu_mib = query_gpu_memory_mib(gpu_index) if sample_gpu else None
    stdout_lines: list[tuple[float, str]] = []
    stderr_lines: list[tuple[float, str]] = []
    start_time = time.perf_counter()
    first_output_file_seconds: float | None = None
    peak_gpu_mib = baseline_gpu_mib
    timed_out = False
    launch_error: str | None = None

    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            env=environment,
        )
    except OSError as exc:
        process = None
        launch_error = str(exc)

    if process is not None:
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=_read_stream,
            args=(process.stdout, start_time, stdout_lines),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_read_stream,
            args=(process.stderr, start_time, stderr_lines),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        deadline = start_time + timeout_seconds
        next_gpu_sample = start_time
        while process.poll() is None:
            now = time.perf_counter()
            if first_output_file_seconds is None:
                try:
                    if output.is_file() and output.stat().st_size > 44:
                        first_output_file_seconds = now - start_time
                except OSError:
                    pass
            if sample_gpu and now >= next_gpu_sample:
                used_mib = query_gpu_memory_mib(gpu_index)
                if used_mib is not None:
                    peak_gpu_mib = max(peak_gpu_mib or used_mib, used_mib)
                next_gpu_sample = now + max(0.1, poll_interval_seconds)
            if now >= deadline:
                timed_out = True
                process.kill()
                break
            time.sleep(min(0.05, poll_interval_seconds))
        return_code = process.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
    else:
        return_code = None

    elapsed_seconds = time.perf_counter() - start_time
    if first_output_file_seconds is None:
        try:
            if output.is_file() and output.stat().st_size > 44:
                first_output_file_seconds = elapsed_seconds
        except OSError:
            pass

    stdout_text = "\n".join(line for _, line in stdout_lines)
    stderr_text = "\n".join(line for _, line in stderr_lines)
    log_prefix.with_suffix(".stdout.log").write_text(
        stdout_text + ("\n" if stdout_text else ""),
        encoding="utf-8",
    )
    log_prefix.with_suffix(".stderr.log").write_text(
        stderr_text + ("\n" if stderr_text else ""),
        encoding="utf-8",
    )

    record: dict[str, Any] = {
        "backend": backend.name,
        "phase": phase,
        "iteration": iteration,
        "command": list(command),
        "output_path": str(output),
        "return_code": return_code,
        "timed_out": timed_out,
        "launch_error": launch_error,
        "elapsed_seconds": elapsed_seconds,
        "first_stdout_seconds": stdout_lines[0][0] if stdout_lines else None,
        "first_output_file_seconds": first_output_file_seconds,
        "gpu_baseline_memory_mib": baseline_gpu_mib,
        "gpu_peak_memory_mib": peak_gpu_mib,
        "gpu_peak_delta_memory_mib": (
            max(0.0, peak_gpu_mib - baseline_gpu_mib)
            if peak_gpu_mib is not None and baseline_gpu_mib is not None
            else None
        ),
        "stdout_log": str(log_prefix.with_suffix(".stdout.log")),
        "stderr_log": str(log_prefix.with_suffix(".stderr.log")),
    }
    record.update(_event_timings(stdout_lines))

    successful_process = return_code == 0 and not timed_out and launch_error is None
    if successful_process and output.is_file():
        try:
            audio = read_wav_metadata(output)
            record.update(audio)
            duration = float(audio["audio_duration_seconds"])
            record["end_to_end_rtf"] = elapsed_seconds / duration
            generation_start = record["generation_start_event_seconds"]
            record["post_generation_start_rtf"] = (
                max(0.0, elapsed_seconds - float(generation_start)) / duration
                if generation_start is not None
                else None
            )
            record["output_sha256"] = sha256_file(output)
            record["success"] = True
            record["error"] = None
        except ValueError as exc:
            record["success"] = False
            record["error"] = str(exc)
    else:
        record["success"] = False
        if timed_out:
            record["error"] = f"Timed out after {timeout_seconds:.1f} seconds"
        elif launch_error:
            record["error"] = f"Could not launch command: {launch_error}"
        elif return_code == 0:
            record["error"] = f"Command did not create output WAV: {output}"
        else:
            tail = next(
                (line for _, line in reversed(stderr_lines) if line.strip()),
                "no stderr output",
            )
            record["error"] = f"Command exited {return_code}: {tail}"
    return record


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metrics = (
        "elapsed_seconds",
        "audio_duration_seconds",
        "end_to_end_rtf",
        "post_generation_start_rtf",
        "reported_ttfa_seconds",
        "first_output_file_seconds",
        "gpu_peak_memory_mib",
        "gpu_peak_delta_memory_mib",
    )
    backend_names = sorted({str(record["backend"]) for record in records})
    for backend_name in backend_names:
        selected = [
            record
            for record in records
            if record["backend"] == backend_name
            and record["phase"] == "measured"
            and record["success"]
        ]
        backend_summary: dict[str, Any] = {
            "successful_runs": len(selected),
            "output_sha256s": sorted(
                {
                    str(record["output_sha256"])
                    for record in selected
                    if record.get("output_sha256")
                }
            ),
        }
        for metric in metrics:
            values = [
                float(record[metric])
                for record in selected
                if record.get(metric) is not None
            ]
            if not values:
                backend_summary[metric] = None
                continue
            backend_summary[metric] = {
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
                "minimum": min(values),
                "maximum": max(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        summary[backend_name] = backend_summary
    return summary


def write_csv(records: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "backend",
        "phase",
        "iteration",
        "success",
        "return_code",
        "elapsed_seconds",
        "audio_duration_seconds",
        "end_to_end_rtf",
        "post_generation_start_rtf",
        "reported_ttfa_seconds",
        "first_output_file_seconds",
        "gpu_baseline_memory_mib",
        "gpu_peak_memory_mib",
        "gpu_peak_delta_memory_mib",
        "sample_rate",
        "channels",
        "output_sha256",
        "output_path",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark preinstalled MOSS delay-pattern 8B and Local-Transformer "
            "v1.5 backends without downloading models."
        )
    )
    parser.add_argument("--text", type=Path, help="UTF-8 benchmark text file")
    parser.add_argument("--voice", type=Path, default=None, help="Optional clone reference")
    parser.add_argument("--output-dir", type=Path, help="Directory for WAVs, logs, JSON, and CSV")
    parser.add_argument("--delay-command-json", help="Delay-pattern backend argv JSON array")
    parser.add_argument("--local-command-json", help="Local-Transformer backend argv JSON array")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per backend")
    parser.add_argument("--warmups", type=int, default=1, help="Excluded warm-up runs per backend")
    parser.add_argument("--seed", type=int, default=1234, help="Base value for {seed}")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--no-gpu-sampling", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=0.2)
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    required = {
        "--text": args.text,
        "--output-dir": args.output_dir,
        "--delay-command-json": args.delay_command_json,
        "--local-command-json": args.local_command_json,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not 0 <= args.cooldown_seconds <= 60:
        parser.error("--cooldown-seconds must be between 0 and 60")
    if not 0.05 <= args.poll_interval_seconds <= 10:
        parser.error("--poll-interval-seconds must be between 0.05 and 10")


def run_benchmark(args: argparse.Namespace) -> int:
    text = args.text.expanduser().resolve()
    voice = args.voice.expanduser().resolve() if args.voice else None
    output_dir = args.output_dir.expanduser().resolve()
    if not text.is_file():
        raise ValueError(f"Benchmark text file not found: {text}")
    if voice is not None and not voice.is_file():
        raise ValueError(f"Voice reference not found: {voice}")

    delay_command = parse_command_json(args.delay_command_json, "--delay-command-json")
    local_command = parse_command_json(args.local_command_json, "--local-command-json")
    for command in (delay_command, local_command):
        validate_command_template(command, voice_supplied=voice is not None)

    backends = (
        Backend(BACKEND_DELAY, delay_command),
        Backend(BACKEND_LOCAL, local_command),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_info = None if args.no_gpu_sampling else query_gpu_info(args.gpu_index)
    sample_gpu = gpu_info is not None
    records: list[dict[str, Any]] = []
    started_at = utc_now()
    failed = False

    phases = (("warmup", args.warmups), ("measured", args.runs))
    for phase, count in phases:
        for zero_based_iteration in range(count):
            iteration = zero_based_iteration + 1
            ordered = backends if zero_based_iteration % 2 == 0 else tuple(reversed(backends))
            for backend in ordered:
                slug = backend.name.replace(".", "-")
                prefix = output_dir / f"{phase}-{iteration:02d}-{slug}"
                output = prefix.with_suffix(".wav")
                command = expand_command(
                    backend,
                    text=text,
                    output=output,
                    voice=voice,
                    seed=args.seed,
                    iteration=iteration,
                    phase=phase,
                )
                print(
                    f"[{phase} {iteration}/{count}] {backend.name}",
                    flush=True,
                )
                record = run_once(
                    backend,
                    command=command,
                    output=output,
                    log_prefix=prefix,
                    phase=phase,
                    iteration=iteration,
                    seed=args.seed,
                    timeout_seconds=args.timeout_seconds,
                    gpu_index=args.gpu_index,
                    sample_gpu=sample_gpu,
                    poll_interval_seconds=args.poll_interval_seconds,
                )
                records.append(record)
                if record["success"]:
                    print(
                        f"  {record['elapsed_seconds']:.3f}s elapsed, "
                        f"{record['audio_duration_seconds']:.3f}s audio, "
                        f"RTF {record['end_to_end_rtf']:.3f}",
                        flush=True,
                    )
                else:
                    print(f"  FAILED: {record['error']}", file=sys.stderr, flush=True)
                    failed = True
                    break
                if args.cooldown_seconds:
                    time.sleep(args.cooldown_seconds)
            if failed:
                break
        if failed:
            break

    report = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": utc_now(),
        "benchmark_kind": "cold-process-same-machine",
        "offline_environment": OFFLINE_ENVIRONMENT,
        "inputs": {
            "text_path": str(text),
            "text_sha256": sha256_file(text),
            "text_characters": len(text.read_text(encoding="utf-8")),
            "voice_path": str(voice) if voice else None,
            "voice_sha256": sha256_file(voice) if voice else None,
            "runs": args.runs,
            "warmups": args.warmups,
            "base_seed": args.seed,
        },
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": sys.version,
            "gpu": vars(gpu_info) if gpu_info else None,
        },
        "notes": [
            "The harness never invokes setup/download and forces Hugging Face and Transformers offline mode.",
            "Runs launch a fresh backend process, so elapsed time and end-to-end RTF include model loading.",
            "first_output_file_seconds is not TTFA when a backend writes its WAV only at completion.",
            "reported_ttfa_seconds is populated only when a backend emits an explicit first_audio event.",
            "GPU memory is whole-device nvidia-smi memory; peak delta can include unrelated processes.",
            "Seed values are backend hints; the current VoiceForge worker does not guarantee reproducible hashes.",
        ],
        "records": records,
        "summary": summarize(records),
    }
    json_path = output_dir / "moss-backend-benchmark.json"
    csv_path = output_dir / "moss-backend-benchmark.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(records, csv_path)
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")
    return 1 if failed else 0


def run_self_test() -> int:
    synth_code = (
        "import json,os,sys,wave;"
        "assert os.environ['HF_HUB_OFFLINE']=='1';"
        "assert os.environ['TRANSFORMERS_OFFLINE']=='1';"
        "text=open(sys.argv[2],encoding='utf-8').read();"
        "print(json.dumps({'event':'progress','message':'Synthesizing fixture'}),flush=True);"
        "w=wave.open(sys.argv[1],'wb');"
        "w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000);"
        "w.writeframes(b'\\x00\\x00'*2000);w.close();"
        "print(json.dumps({'event':'complete','characters':len(text)}),flush=True)"
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        text = root / "input.txt"
        text.write_text("Offline benchmark self-test.", encoding="utf-8")
        command = (
            sys.executable,
            "-c",
            synth_code,
            "{output}",
            "{text}",
        )
        validate_command_template(command, voice_supplied=False)
        backend = Backend("fixture", command)
        output = root / "fixture.wav"
        expanded = expand_command(
            backend,
            text=text,
            output=output,
            voice=None,
            seed=1234,
            iteration=1,
            phase="measured",
        )
        record = run_once(
            backend,
            command=expanded,
            output=output,
            log_prefix=root / "fixture",
            phase="measured",
            iteration=1,
            seed=1234,
            timeout_seconds=30,
            gpu_index=0,
            sample_gpu=False,
            poll_interval_seconds=0.05,
        )
        assert record["success"], record["error"]
        assert abs(record["audio_duration_seconds"] - 0.25) < 1e-9
        assert record["generation_start_event_seconds"] is not None
        assert record["end_to_end_rtf"] > 0
        assert len(record["output_sha256"]) == 64
        benchmark_args = argparse.Namespace(
            text=text,
            voice=None,
            output_dir=root / "results",
            delay_command_json=json.dumps(command),
            local_command_json=json.dumps(command),
            runs=1,
            warmups=0,
            seed=1234,
            timeout_seconds=30.0,
            cooldown_seconds=0.0,
            gpu_index=0,
            no_gpu_sampling=True,
            poll_interval_seconds=0.05,
        )
        assert run_benchmark(benchmark_args) == 0
        report_path = benchmark_args.output_dir / "moss-backend-benchmark.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["summary"][BACKEND_DELAY]["successful_runs"] == 1
        assert report["summary"][BACKEND_LOCAL]["successful_runs"] == 1
        assert (benchmark_args.output_dir / "moss-backend-benchmark.csv").is_file()
        try:
            validate_command_template(
                ("python", "worker.py", "setup", "{text}", "{output}"),
                voice_supplied=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("setup command was not rejected")
    print("MOSS backend benchmark self-test passed.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    validate_args(args, parser)
    try:
        return run_benchmark(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
