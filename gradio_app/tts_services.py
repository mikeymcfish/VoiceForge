from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gradio as gr


@dataclass
class WorkerEvent:
    event: str
    message: Optional[str] = None
    progress: Optional[float] = None
    level: Optional[str] = None
    payload: Dict[str, object] = None  # type: ignore[assignment]


def _read_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_event(line: str) -> WorkerEvent:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return WorkerEvent(event="log", message=line.strip(), payload={})
    event = str(payload.get("event", "log"))
    return WorkerEvent(
        event=event,
        message=payload.get("message") or payload.get("error"),
        progress=payload.get("progress"),
        level=payload.get("level"),
        payload=payload,
    )


def _run_worker(command: Sequence[str]) -> Tuple[List[WorkerEvent], str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    events: List[WorkerEvent] = []
    logs: List[str] = []

    assert process.stdout is not None
    for line in process.stdout:
        if not line:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        logs.append(stripped)
        events.append(_parse_event(stripped))

    code = process.wait()
    if code != 0:
        preview = "\n".join(logs[-20:])
        raise RuntimeError(f"Worker exited with status {code}. Logs:\n{preview}")

    return events, "\n".join(logs)


def _summarize(events: Iterable[WorkerEvent], fallback: str) -> Tuple[str, Optional[str]]:
    last_message: Optional[str] = None
    output_path: Optional[str] = None
    for event in events:
        if event.payload:
            candidate = event.payload.get("output_path") or event.payload.get("outputFile")
            if isinstance(candidate, str):
                output_path = candidate
        if event.message:
            last_message = event.message
    return last_message or fallback, output_path


class IndexTTSService:
    def __init__(self) -> None:
        repo_root = _read_repo_root()
        self.worker_path = repo_root / "server" / "python" / "index_tts_worker.py"
        if not self.worker_path.exists():
            raise FileNotFoundError(
                "IndexTTS worker script not found. Expected at server/python/index_tts_worker.py"
            )
        base_dir = Path(os.getenv("INDEXTTS_ROOT", Path.home() / ".voiceforge" / "indextts"))
        self.root_dir = _ensure_directory(base_dir)
        self.models_dir = _ensure_directory(Path(os.getenv("INDEXTTS_MODELS", self.root_dir / "models")))
        self.jobs_dir = _ensure_directory(self.root_dir / "jobs")

    def _command(self) -> List[str]:
        return [
            sys.executable,
            str(self.worker_path),
            "--root-dir",
            str(self.root_dir),
            "--models-dir",
            str(self.models_dir),
        ]

    def download_models(self, repo_id: str) -> Tuple[str, str]:
        if not repo_id.strip():
            raise gr.Error("Provide a HuggingFace repo id for IndexTTS")
        events, log = _run_worker(self._command() + ["download", "--repo-id", repo_id.strip()])
        summary, _ = _summarize(events, "Download finished")
        return summary, log

    def load_models(self) -> Tuple[str, str]:
        events, log = _run_worker(self._command() + ["load"])
        summary, _ = _summarize(events, "Models loaded")
        return summary, log

    def synthesize(self, voice_path: Path, text: str, steps: int = 25) -> Tuple[str, str, Optional[Path]]:
        if not voice_path.exists():
            raise gr.Error("Voice prompt file missing")
        if not text.strip():
            raise gr.Error("Provide text to synthesize")
        job_id = uuid.uuid4().hex
        text_file = self.jobs_dir / f"{job_id}.txt"
        text_file.write_text(text, encoding="utf-8")
        output_path = self.jobs_dir / f"{job_id}.wav"
        command = self._command() + [
            "synthesize",
            "--voice",
            str(voice_path),
            "--text",
            str(text_file),
            "--output",
            str(output_path),
            "--steps",
            str(int(max(1, steps))),
        ]
        events, log = _run_worker(command)
        summary, reported_output = _summarize(events, "Synthesis complete")
        if reported_output and Path(reported_output).exists():
            output_path = Path(reported_output)
        return summary, log, output_path if output_path.exists() else None


class VibeVoiceService:
    def __init__(self) -> None:
        repo_root = _read_repo_root()
        self.worker_path = repo_root / "server" / "python" / "vibevoice_worker.py"
        if not self.worker_path.exists():
            raise FileNotFoundError(
                "VibeVoice worker script not found. Expected at server/python/vibevoice_worker.py"
            )
        base_dir = Path(os.getenv("VIBEVOICE_ROOT", Path.home() / ".voiceforge" / "vibevoice"))
        self.root_dir = _ensure_directory(base_dir)
        self.repo_dir = _ensure_directory(Path(os.getenv("VIBEVOICE_REPO_DIR", self.root_dir / "repo")))
        self.jobs_dir = _ensure_directory(Path(os.getenv("VIBEVOICE_JOBS_DIR", self.root_dir / "jobs")))

    def _command(self) -> List[str]:
        return [
            sys.executable,
            str(self.worker_path),
            "--root-dir",
            str(self.root_dir),
            "--repo-dir",
            str(self.repo_dir),
            "--jobs-dir",
            str(self.jobs_dir),
        ]

    def setup(self, repo_url: str, branch: str) -> Tuple[str, str]:
        args = self._command() + [
            "setup",
        ]
        if repo_url.strip():
            args += ["--repo-url", repo_url.strip()]
        if branch.strip():
            args += ["--repo-branch", branch.strip()]
        events, log = _run_worker(args)
        summary, _ = _summarize(events, "Setup complete")
        return summary, log

    def synthesize(
        self,
        text: str,
        voices: Sequence[Path],
        style: Optional[str],
        temperature: Optional[float],
        model_id: Optional[str],
    ) -> Tuple[str, str, Optional[Path]]:
        if not text.strip():
            raise gr.Error("Provide text for VibeVoice synthesis")
        if not voices:
            raise gr.Error("Upload at least one voice reference file")
        for voice in voices:
            if not voice.exists():
                raise gr.Error(f"Voice reference missing: {voice}")
        job_id = uuid.uuid4().hex
        text_file = self.jobs_dir / f"{job_id}.txt"
        text_file.write_text(text, encoding="utf-8")
        output_path = self.jobs_dir / f"{job_id}.wav"
        args: List[str] = list(self._command())
        args += [
            "synthesize",
            "--job-id",
            job_id,
            "--text",
            str(text_file),
            "--output",
            str(output_path),
        ]
        for voice in voices:
            args += ["--voice", str(voice)]
        if style and style.strip():
            args += ["--style", style.strip()]
        if temperature is not None:
            args += ["--temperature", str(temperature)]
        if model_id and model_id.strip():
            args += ["--model-id", model_id.strip()]
        events, log = _run_worker(args)
        summary, reported = _summarize(events, "Synthesis complete")
        if reported and Path(reported).exists():
            output_path = Path(reported)
        return summary, log, output_path if output_path.exists() else None
