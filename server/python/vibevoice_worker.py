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


VIBEVOICE_REPO_URL = "https://github.com/vibevoice-community/VibeVoice.git"
VIBEVOICE_REPO_REVISION = "07cb79feadd2d3fd7f47530d4c964a12857936a0"
PINNED_MODEL_REVISIONS = {
    "microsoft/VibeVoice-1.5B": "c00898d257e6b46004e3e2866a47534085fb685a",
    "aoi-ot/VibeVoice-Large": "8229be00d7c036aa32321e4dae8a81d433f6413a",
}
PINNED_TOKENIZERS = {
    "microsoft/VibeVoice-1.5B": (
        "Qwen/Qwen2.5-1.5B",
        "8faed761d45a263340a0528343f099c05c9a4323",
    ),
    "aoi-ot/VibeVoice-Large": (
        "Qwen/Qwen2.5-7B",
        "d149729398750b98c0af14eb82c78cfe92750796",
    ),
}


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


def validate_torch_runtime() -> None:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the VibeVoice environment. Configure "
            "VIBEVOICE_PYTHON to a dedicated environment with PyTorch 2.6 or newer."
        ) from exc

    version_match = re.match(r"^(\d+)\.(\d+)", str(torch.__version__))
    if not version_match or tuple(map(int, version_match.groups())) < (2, 6):
        raise RuntimeError(
            f"VibeVoice requires a maintained PyTorch runtime (>=2.6); found {torch.__version__}."
        )


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


def _read_git_head(repo_dir: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stdout.strip()
        raise RuntimeError(f"Unable to verify VibeVoice checkout: {details}")
    return completed.stdout.strip().lower()


def ensure_repo(repo_dir: str) -> None:
    """Create and verify a fresh checkout of the one reviewed VibeVoice revision."""
    parent = os.path.dirname(os.path.abspath(repo_dir))
    checkout_dir = f"{os.path.abspath(repo_dir)}.checkout"
    os.makedirs(parent, exist_ok=True)

    if os.path.exists(checkout_dir):
        shutil.rmtree(checkout_dir)
    os.makedirs(checkout_dir)

    emit(
        "progress",
        progress=0.1,
        message=f"Fetching pinned VibeVoice revision {VIBEVOICE_REPO_REVISION[:12]}…",
    )
    try:
        run_process(["git", "init"], cwd=checkout_dir)
        run_process(
            ["git", "remote", "add", "origin", VIBEVOICE_REPO_URL],
            cwd=checkout_dir,
        )
        run_process(
            [
                "git",
                "fetch",
                "--depth",
                "1",
                "origin",
                VIBEVOICE_REPO_REVISION,
            ],
            cwd=checkout_dir,
        )
        run_process(
            ["git", "checkout", "--detach", "--force", "FETCH_HEAD"],
            cwd=checkout_dir,
        )

        actual_revision = _read_git_head(checkout_dir)
        if actual_revision != VIBEVOICE_REPO_REVISION:
            raise RuntimeError(
                "VibeVoice checkout verification failed: "
                f"expected {VIBEVOICE_REPO_REVISION}, got {actual_revision or 'unknown'}"
            )

        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        os.replace(checkout_dir, repo_dir)
    except Exception:
        if os.path.exists(checkout_dir):
            shutil.rmtree(checkout_dir)
        raise

    emit(
        "log",
        level="info",
        message=f"Using pinned VibeVoice revision {VIBEVOICE_REPO_REVISION}",
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


def _model_manifest_path(model_dir: str) -> str:
    return os.path.join(model_dir, "voiceforge-model.json")


def _validate_model_snapshot(
    model_dir: str,
    *,
    expected_repo_id: Optional[str] = None,
    expected_revision: Optional[str] = None,
    require_manifest: bool = False,
) -> Dict[str, object]:
    """Reject interrupted or structurally incomplete local HF snapshots."""
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Model directory does not exist: {model_dir}")

    incomplete_files: List[str] = []
    for root, dirs, files in os.walk(model_dir):
        dirs[:] = [name for name in dirs if name != ".cache"]
        for name in files:
            if name.endswith(".incomplete"):
                incomplete_files.append(os.path.relpath(os.path.join(root, name), model_dir))
    if incomplete_files:
        raise RuntimeError(
            "Model snapshot contains unfinished downloads: " + ", ".join(incomplete_files[:5])
        )

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path) or os.path.getsize(config_path) <= 0:
        raise RuntimeError("Model snapshot is missing a non-empty config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Model config.json is invalid: {exc}") from exc

    required_weights: set[str] = set()
    index_files = [
        name
        for name in os.listdir(model_dir)
        if name.endswith((".safetensors.index.json", ".bin.index.json"))
    ]
    for index_name in index_files:
        index_path = os.path.join(model_dir, index_name)
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                index_data = json.load(handle)
            weight_map = index_data.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weight_map is empty")
            required_weights.update(str(value) for value in weight_map.values())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Model weight index {index_name} is invalid: {exc}") from exc

    if not required_weights:
        required_weights.update(
            name
            for name in os.listdir(model_dir)
            if name.endswith((".safetensors", ".bin", ".pt", ".pth"))
            and os.path.isfile(os.path.join(model_dir, name))
        )
    if not required_weights:
        raise RuntimeError("Model snapshot does not contain model weights")

    missing_weights = [
        relative
        for relative in sorted(required_weights)
        if not os.path.isfile(os.path.join(model_dir, relative))
        or os.path.getsize(os.path.join(model_dir, relative)) <= 0
    ]
    if missing_weights:
        raise RuntimeError(
            "Model snapshot is missing weight shards: " + ", ".join(missing_weights[:5])
        )

    manifest_path = _model_manifest_path(model_dir)
    existing_manifest: Optional[Dict[str, object]] = None
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                parsed = json.load(handle)
            if isinstance(parsed, dict):
                existing_manifest = parsed
        except Exception as exc:  # noqa: BLE001
            if require_manifest:
                raise RuntimeError(f"Model install manifest is invalid: {exc}") from exc
    if require_manifest:
        if not existing_manifest or existing_manifest.get("complete") is not True:
            raise RuntimeError("Model snapshot has no completed VoiceForge install manifest")
        if expected_repo_id and existing_manifest.get("repo_id") != expected_repo_id:
            raise RuntimeError("Model install manifest repository does not match the requested model")
        if expected_revision and existing_manifest.get("revision") != expected_revision:
            raise RuntimeError("Model install manifest revision does not match the pinned revision")
        artifacts = existing_manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RuntimeError("Model install manifest has no artifacts")
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise RuntimeError("Model install manifest contains an invalid artifact")
            relative = artifact.get("path")
            expected_size = artifact.get("size")
            if not isinstance(relative, str) or not isinstance(expected_size, int):
                raise RuntimeError("Model install manifest contains an invalid artifact entry")
            artifact_path = os.path.abspath(os.path.join(model_dir, relative))
            if os.path.commonpath([model_dir, artifact_path]) != model_dir:
                raise RuntimeError("Model install manifest contains a path outside the snapshot")
            if not os.path.isfile(artifact_path) or os.path.getsize(artifact_path) != expected_size:
                raise RuntimeError(f"Model artifact is missing or changed: {relative}")

    return {
        "required_weights": sorted(required_weights),
        "manifest": existing_manifest or {},
    }


def _write_model_manifest(model_dir: str, repo_id: str, revision: Optional[str]) -> None:
    artifacts: List[Dict[str, object]] = []
    for root, dirs, files in os.walk(model_dir):
        dirs[:] = [name for name in dirs if name != ".cache"]
        for name in files:
            if name in {"voiceforge-model.json", "repo_id.txt"} or name.endswith(".incomplete"):
                continue
            file_path = os.path.join(root, name)
            if os.path.isfile(file_path):
                artifacts.append(
                    {
                        "path": os.path.relpath(file_path, model_dir).replace(os.sep, "/"),
                        "size": os.path.getsize(file_path),
                    }
                )
    artifacts.sort(key=lambda item: str(item["path"]))
    if not artifacts:
        raise RuntimeError("Cannot mark an empty model snapshot as complete")

    manifest = {
        "schema_version": 1,
        "complete": True,
        "repo_id": repo_id,
        "revision": revision,
        "artifacts": artifacts,
    }
    manifest_path = _model_manifest_path(model_dir)
    temporary_path = f"{manifest_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, manifest_path)


def _install_pinned_tokenizer(
    model_dir: str,
    repo_id: str,
    token: Optional[str],
    snapshot_download,
) -> None:
    tokenizer_pin = PINNED_TOKENIZERS.get(repo_id)
    if not tokenizer_pin:
        raise RuntimeError(
            f"No reviewed tokenizer snapshot is pinned for operator-configured model {repo_id}"
        )
    tokenizer_id, tokenizer_revision = tokenizer_pin
    tokenizer_dir = os.path.join(model_dir, "voiceforge-tokenizer")
    snapshot_download(
        tokenizer_id,
        revision=tokenizer_revision,
        local_dir=tokenizer_dir,
        token=token,
        allow_patterns=[
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
        ],
    )
    tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
    vocab_json = os.path.join(tokenizer_dir, "vocab.json")
    merges_txt = os.path.join(tokenizer_dir, "merges.txt")
    if not os.path.isfile(tokenizer_json) and not (
        os.path.isfile(vocab_json) and os.path.isfile(merges_txt)
    ):
        raise RuntimeError(f"Pinned tokenizer snapshot is incomplete for {repo_id}")

    processor_config_path = os.path.join(model_dir, "preprocessor_config.json")
    if not os.path.isfile(processor_config_path):
        raise RuntimeError(f"Model snapshot {repo_id} is missing preprocessor_config.json")
    with open(processor_config_path, "r", encoding="utf-8") as handle:
        processor_config = json.load(handle)
    processor_config["language_model_pretrained_name"] = os.path.abspath(tokenizer_dir)
    temporary_path = f"{processor_config_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(processor_config, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, processor_config_path)


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
    failures: List[str] = []
    for idx, repo_id in enumerate(models, start=1):
        revision = PINNED_MODEL_REVISIONS.get(repo_id)
        safe_name = repo_id.replace("/", "__")
        local_dir = os.path.join(models_dir, safe_name)
        os.makedirs(local_dir, exist_ok=True)
        for stale_marker in (_model_manifest_path(local_dir), os.path.join(local_dir, "repo_id.txt")):
            try:
                os.remove(stale_marker)
            except FileNotFoundError:
                pass

        base_prog = 0.6  # start progress after repo setup stage
        end_prog = 0.95
        prog = base_prog + (idx / max(total, 1)) * (end_prog - base_prog)
        emit("progress", progress=prog, message=f"Downloading {repo_id}…")

        try:
            if revision is None:
                emit(
                    "log",
                    level="warn",
                    message=f"Operator-configured model {repo_id} has no pinned revision.",
                )
            snapshot_download(
                repo_id,
                revision=revision,
                local_dir=local_dir,
                token=token,
                max_workers=8,
            )
            _install_pinned_tokenizer(local_dir, repo_id, token, snapshot_download)
            _validate_model_snapshot(local_dir)
            with open(os.path.join(local_dir, "repo_id.txt"), "w", encoding="utf-8") as f:
                f.write(repo_id)
            _write_model_manifest(local_dir, repo_id, revision)
            _validate_model_snapshot(
                local_dir,
                expected_repo_id=repo_id,
                expected_revision=revision,
                require_manifest=True,
            )
            emit("log", level="info", message=f"Downloaded and verified {repo_id} -> {local_dir}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{repo_id}: {exc}")
            emit(
                "log",
                level="error",
                message=f"Failed to download or verify {repo_id}: {exc}",
            )

    if failures:
        raise RuntimeError("VibeVoice model setup incomplete: " + "; ".join(failures))


def _chapter_token_limit() -> int:
    env_keys = [
        "VIBEVOICE_MAX_CHAPTER_TOKENS",
        "VIBEVOICE_CHAPTER_TOKEN_LIMIT",
        "VIBEVOICE_SEGMENT_TOKEN_LIMIT",
        "VIBEVOICE_TOKEN_LIMIT",
    ]
    for key in env_keys:
        raw_value = os.environ.get(key)
        if not raw_value:
            continue
        try:
            parsed = int(raw_value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    # Empirically VibeVoice handles ~2k tokens comfortably per run.
    # Keep some headroom for long sentences and narration tags.
    return 1400


def _coerce_offsets(encoding) -> List[Tuple[int, int]]:
    offsets = encoding.get("offset_mapping") if hasattr(encoding, "get") else None
    if offsets is None:
        encodings = getattr(encoding, "encodings", None)
        if encodings:
            offsets = encodings[0].offsets  # type: ignore[index]
    if offsets is None:
        return []
    if len(offsets) > 0 and isinstance(offsets[0], list):
        return [tuple(item) for item in offsets]  # type: ignore[arg-type]
    return list(offsets)


def _coerce_input_ids(encoding) -> List[int]:
    ids = encoding.get("input_ids") if hasattr(encoding, "get") else None
    if ids is None:
        return []
    if len(ids) > 0 and isinstance(ids[0], list):
        return ids[0]  # type: ignore[index]
    return list(ids)


def _find_segment_boundary(text: str, preferred_index: int, lookahead: int = 400) -> int:
    if preferred_index >= len(text):
        return len(text)

    window_end = min(len(text), preferred_index + max(lookahead, 1))
    search_window = text[preferred_index:window_end]

    # Prefer double newlines (chapter/paragraph boundaries)
    newline_match = search_window.find("\n\n")
    if newline_match != -1:
        return preferred_index + newline_match + 2

    # Prefer sentence endings.
    sentence_match = re.search(r"[\.\?!][\)\]\"']?\s", search_window)
    if sentence_match:
        return preferred_index + sentence_match.end()

    # Fall back to the next whitespace boundary.
    whitespace_match = re.search(r"\s", search_window)
    if whitespace_match:
        return preferred_index + whitespace_match.end()

    return window_end


def _segment_text_by_tokens(text: str, tokenizer, max_tokens: int) -> List[str]:
    remaining = text.strip()
    segments: List[str] = []
    safety_counter = 0

    while remaining:
        safety_counter += 1
        if safety_counter > 10000:
            raise RuntimeError("Segmentation safety limit exceeded while splitting text")

        encoding = tokenizer(
            remaining,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        input_ids = _coerce_input_ids(encoding)
        offsets = _coerce_offsets(encoding)

        if not input_ids:
            if remaining.strip():
                segments.append(remaining.strip())
            break

        if len(input_ids) <= max_tokens:
            trimmed = remaining.strip()
            if trimmed:
                segments.append(trimmed)
            break

        cut_index = min(max_tokens, len(offsets))
        if cut_index <= 0:
            trimmed = remaining.strip()
            if trimmed:
                segments.append(trimmed)
            break

        cutoff = offsets[cut_index - 1][1] if offsets else len(remaining)
        boundary = _find_segment_boundary(remaining, cutoff)
        boundary = max(boundary, cutoff)

        if boundary >= len(remaining):
            trimmed = remaining.strip()
            if trimmed:
                segments.append(trimmed)
            break

        segment = remaining[:boundary].strip()
        if not segment:
            decoded = tokenizer.decode(
                input_ids[:cut_index],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if not decoded:
                decoded = remaining[:boundary].strip()
            if decoded:
                segments.append(decoded)
            remaining = remaining[boundary:].lstrip()
            continue

        segments.append(segment)
        remaining = remaining[boundary:].lstrip()

    return segments


def _progress_between(start: float, end: float, index: int, total: int) -> float:
    if total <= 0:
        return end
    fraction = max(0.0, min(1.0, index / total))
    return start + (end - start) * fraction


_ROLE_LINE_RE = re.compile(
    r"^\s*(?:(Narrator)|Speaker\s+(\d+)|\[(\d+)\])\s*:\s*(.*)$",
    re.IGNORECASE,
)


def _normalize_script_for_vibe(text: str) -> Tuple[str, List[str]]:
    """Convert VoiceForge labels to VibeVoice's sequential Speaker 1..4 format."""
    role_indexes: Dict[str, int] = {}
    role_names: List[str] = []
    normalized_lines: List[str] = []

    def index_for(role_key: str, role_name: str) -> int:
        if role_key not in role_indexes:
            if len(role_indexes) >= 4:
                raise ValueError(
                    "VibeVoice supports at most four unique speaking roles per script. "
                    "Reduce or merge speaker labels before synthesis."
                )
            role_indexes[role_key] = len(role_indexes)
            role_names.append(role_name)
        return role_indexes[role_key] + 1

    for line in text.splitlines():
        if not line.strip():
            normalized_lines.append("")
            continue

        match = _ROLE_LINE_RE.match(line)
        if match:
            if match.group(1):
                role_key, role_name = "narrator", "Narrator"
            else:
                raw_number = match.group(2) or match.group(3) or "1"
                role_key, role_name = f"speaker:{raw_number}", f"Speaker {raw_number}"
            content = match.group(4).strip()
        else:
            role_key, role_name = "narrator", "Narrator"
            content = line.strip()

        speaker_index = index_for(role_key, role_name)
        normalized_lines.append(f"Speaker {speaker_index}: {content}")

    return "\n".join(normalized_lines).strip(), role_names


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
                    speaker = int(match.group(1))
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
            try:
                _validate_model_snapshot(
                    candidate,
                    expected_repo_id=model_id,
                    expected_revision=PINNED_MODEL_REVISIONS.get(model_id),
                    require_manifest=True,
                )
                return candidate
            except RuntimeError:
                continue

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
                    try:
                        _validate_model_snapshot(
                            sub,
                            expected_repo_id=model_id,
                            expected_revision=PINNED_MODEL_REVISIONS.get(model_id),
                            require_manifest=True,
                        )
                        return sub
                    except RuntimeError:
                        continue
    except Exception:
        pass
    return None


def resolve_model_path(root_dir: str, requested: Optional[str]) -> Tuple[str, Optional[str]]:
    env_override = os.environ.get("VIBEVOICE_MODEL_PATH")
    if env_override:
        _validate_model_snapshot(env_override)
        return env_override, os.environ.get("VIBEVOICE_MODEL_ID")

    if requested:
        local = _resolve_local_model(root_dir, requested)
        if local:
            return local, requested
        if os.path.isdir(requested):
            _validate_model_snapshot(requested)
            return requested, requested
        raise RuntimeError(
            f"VibeVoice model {requested!r} is not a verified local snapshot. Run setup again."
        )

    defaults = parse_model_ids_from_env()
    for model_id in defaults:
        local = _resolve_local_model(root_dir, model_id)
        if local:
            return local, model_id

    raise RuntimeError(
        "No verified VibeVoice model is installed. Run setup again or set VIBEVOICE_MODEL_PATH."
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
    text: str,
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

    cfg_value: Optional[float] = None
    if getattr(args, "guidance_scale", None) is not None:
        cfg_value = float(args.guidance_scale)
    cfg_scale = os.environ.get("VIBEVOICE_CFG_SCALE")
    if cfg_value is None and cfg_scale:
        try:
            cfg_value = float(cfg_scale)
        except ValueError:
            cfg_value = None
    if cfg_value is None:
        cfg_value = 1.3
    if not 0.5 <= cfg_value <= 3.0:
        raise ValueError("VibeVoice guidance scale must be between 0.5 and 3.0")

    emit("progress", progress=0.48, message="Preparing synthesis inputs…")

    voice_batch: Optional[List[List[str]]] = None
    if voice_samples:
        voice_batch = [voice_samples]

    max_tokens = max(128, _chapter_token_limit())
    segments = _segment_text_by_tokens(text, processor.tokenizer, max_tokens)
    if not segments:
        segments = [text.strip()]

    total_segments = len(segments)
    emit(
        "log",
        level="info",
        message=f"Segmented script into {total_segments} chapter(s) (≤{max_tokens} tokens each)",
    )
    emit(
        "progress",
        progress=0.52,
        message=f"Queued {total_segments} chapter{'s' if total_segments != 1 else ''} for synthesis…",
    )

    target_device = device if device in {"cuda", "mps"} else "cpu"
    combined_waveform = None
    generation_start = time.time()

    for index, segment_text in enumerate(segments, start=1):
        seg_encoding = processor.tokenizer(
            segment_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        token_count = len(_coerce_input_ids(seg_encoding))
        emit(
            "log",
            level="info",
            message=f"Chapter {index}/{total_segments}: {len(segment_text)} chars, ~{token_count} tokens",
        )

        emit(
            "progress",
            progress=_progress_between(0.52, 0.9, index - 1, total_segments),
            message=f"Generating chapter {index}/{total_segments}…",
        )

        inputs = processor(
            text=[segment_text],
            voice_samples=voice_batch,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for key, value in list(inputs.items()):
            if hasattr(value, "to") and callable(value.to):
                inputs[key] = value.to(target_device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_value,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=True,
            )

        speech_outputs = getattr(outputs, "speech_outputs", None)
        if not speech_outputs or speech_outputs[0] is None:
            raise RuntimeError(f"VibeVoice did not return audio for chapter {index}")

        waveform = speech_outputs[0]
        if hasattr(waveform, "detach"):
            waveform = waveform.detach()
        waveform = waveform.to("cpu") if hasattr(waveform, "to") else waveform
        if hasattr(waveform, "ndim") and waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if combined_waveform is None:
            combined_waveform = waveform
        else:
            combined_waveform = torch.cat([combined_waveform, waveform], dim=-1)

        emit(
            "progress",
            progress=_progress_between(0.52, 0.9, index, total_segments),
            message=f"Chapter {index}/{total_segments} ready",
        )

    generation_end = time.time()

    if combined_waveform is None:
        raise RuntimeError("VibeVoice did not produce any audio segments")

    emit("progress", progress=0.92, message="Normalizing and saving audio…")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    saved_paths = processor.save_audio(
        combined_waveform,
        output_path=args.output,
        normalize=True,
    )

    final_path = saved_paths[0] if saved_paths else args.output
    duration = None
    try:
        waveform_any = combined_waveform
        if hasattr(waveform_any, "detach"):
            waveform_any = waveform_any.detach()
        array = waveform_any.cpu().numpy() if hasattr(waveform_any, "cpu") else waveform_any
        if hasattr(array, "ndim") and array.ndim > 1:
            array = array.squeeze(0)
        sample_rate = 24000
        duration = array.shape[-1] / float(sample_rate)
    except Exception:
        duration = None

    elapsed = generation_end - generation_start
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

    ensure_repo(args.repo_dir)
    install_requirements(args.repo_dir)
    validate_torch_runtime()
    try_download_assets(args.repo_dir)
    # Attempt to fetch baseline models for VibeVoice
    download_hf_models(os.path.join(args.root_dir, "models"))

    emit("complete", progress=1.0, message="VibeVoice setup complete", output_path=args.repo_dir)


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

    raise RuntimeError(
        "The pinned VibeVoice CLI cannot consume arbitrary uploaded voice paths. "
        "Set VIBEVOICE_COMMAND_TEMPLATE only for an operator-reviewed compatible CLI."
    )




def synthesize(args: argparse.Namespace) -> None:
    prepare_environment(args.root_dir)
    # Synthesis must use the setup-time pinned snapshots; never resolve mutable Hub state.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    validate_torch_runtime()
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

    text_content, role_names = _normalize_script_for_vibe(text_content)
    normalized_path = os.path.join(
        os.path.dirname(os.path.abspath(args.text)),
        f"{args.job_id}-vibe-script.txt",
    )
    with open(normalized_path, "w", encoding="utf-8") as handle:
        handle.write(text_content)
    args.text = normalized_path
    emit(
        "log",
        level="info",
        message="Voice order: " + ", ".join(
            f"Voice {index + 1} = {name}" for index, name in enumerate(role_names)
        ),
    )

    voices: List[str] = []
    if args.voice:
        for v in args.voice:
            if not os.path.isfile(v):
                raise FileNotFoundError(f"Voice reference not found: {v}")
            voices.append(v)

    job_dir = os.path.dirname(os.path.abspath(args.text))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    voice_samples = _prepare_voice_prompts(args.text, voices)
    if len(voices) < len(role_names):
        emit(
            "log",
            level="warn",
            message=(
                f"The script has {len(role_names)} speaking roles but only {len(voices)} voice references; "
                "the last voice will be reused."
            ),
        )
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
            text=text_content,
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
        "guidance_scale": str(args.guidance_scale) if getattr(args, "guidance_scale", None) is not None else None,
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
        "guidance_scale_arg": f"--cfg_scale {getattr(args, 'guidance_scale', '')}" if getattr(args, "guidance_scale", None) is not None else None,
        "speaker_names": None,
        "cfg_scale": str(getattr(args, "guidance_scale", "")) if getattr(args, "guidance_scale", None) is not None else None,
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

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("setup")

    synth_parser = subparsers.add_parser("synthesize")
    synth_parser.add_argument("--job-id", required=True)
    synth_parser.add_argument("--text", required=True)
    synth_parser.add_argument("--output", required=True)
    synth_parser.add_argument("--voice", action="append")
    synth_parser.add_argument("--style")
    synth_parser.add_argument("--guidance-scale", type=float, dest="guidance_scale")
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
