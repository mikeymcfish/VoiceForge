#!/usr/bin/env bash
# One-shot VoiceForge setup for a fresh RunPod (PyTorch) image.
# Env vars you can set before running:
#   REPO_URL=https://github.com/mikeymcfish/VoiceForge.git
#   APP_DIR=$HOME/VoiceForge          # defaults to this script's directory
#   PORT=5000
#   NODE_MAJOR=22
#   SESSION_SECRET=...           # auto-generated if absent
#   HUGGINGFACE_API_TOKEN=...
#   OLLAMA_BASE_URL=...
#   INDEX_TTS_PYTHON=/path/to/indextts-env/bin/python
#   INDEX_TTS_SOURCE_DIR=/path/to/official-index-tts
#   INSTALL_QWEN_TTS_REQUIREMENTS=...
#   QWEN_TTS_PYTHON=/path/to/qwen-env/bin/python
#   QWEN_TTS_VENV_DIR=$APP_DIR/.venv-qwen-tts
#   QWEN_TTS_DEVICE=cuda:0
#   QWEN_TTS_USE_FLASH_ATTENTION=1
#   INSTALL_MOSS_TTS_REQUIREMENTS=...
#   MOSS_TTS_PYTHON=/path/to/moss-env/bin/python
#   MOSS_TTS_VENV_DIR=$APP_DIR/.venv-moss-tts
#   USE_PM2=1                    # if set, install pm2 and daemonize

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"; }
have() { command -v "$1" >/dev/null 2>&1; }
die() {
  log "ERROR: $*"
  exit 1
}
resolve_executable() {
  local candidate="$1"
  if [[ "${candidate}" == */* ]]; then
    [ -x "${candidate}" ] || return 1
    (
      cd "$(dirname "${candidate}")"
      printf '%s/%s\n' "$(pwd -P)" "$(basename "${candidate}")"
    )
  else
    command -v "${candidate}"
  fi
}
python_environment_id() {
  local python_bin="$1"
  "${python_bin}" -c 'import os, sys; print(os.path.realpath(sys.prefix))'
}
is_isolated_python_environment() {
  local python_bin="$1"
  "${python_bin}" -c 'import os, sys; isolated = sys.prefix != sys.base_prefix or os.path.isdir(os.path.join(sys.prefix, "conda-meta")); raise SystemExit(0 if isolated else 1)'
}
assert_isolated_tts_pythons() {
  local index_python="${INDEX_TTS_PYTHON:-}"
  local qwen_python="${QWEN_TTS_PYTHON:-}"
  local moss_python="${MOSS_TTS_PYTHON:-}"
  local index_environment
  local qwen_environment
  local moss_environment

  if [[ -n "${index_python}" ]]; then
    index_python="$(resolve_executable "${index_python}")" || die "INDEX_TTS_PYTHON is not executable: ${INDEX_TTS_PYTHON}"
    index_environment="$(python_environment_id "${index_python}")" || die "Unable to inspect INDEX_TTS_PYTHON."
  fi
  if [[ -n "${qwen_python}" ]]; then
    qwen_python="$(resolve_executable "${qwen_python}")" || die "QWEN_TTS_PYTHON is not executable: ${QWEN_TTS_PYTHON}"
    qwen_environment="$(python_environment_id "${qwen_python}")" || die "Unable to inspect QWEN_TTS_PYTHON."
  fi
  if [[ -n "${moss_python}" ]]; then
    moss_python="$(resolve_executable "${moss_python}")" || die "MOSS_TTS_PYTHON is not executable: ${MOSS_TTS_PYTHON}"
    moss_environment="$(python_environment_id "${moss_python}")" || die "Unable to inspect MOSS_TTS_PYTHON."
  fi
  [[ -z "${index_environment:-}" || -z "${qwen_environment:-}" || "${index_environment}" != "${qwen_environment}" ]] || die "INDEX_TTS_PYTHON and QWEN_TTS_PYTHON must use separate environments."
  [[ -z "${index_environment:-}" || -z "${moss_environment:-}" || "${index_environment}" != "${moss_environment}" ]] || die "INDEX_TTS_PYTHON and MOSS_TTS_PYTHON must use separate environments."
  [[ -z "${qwen_environment:-}" || -z "${moss_environment:-}" || "${qwen_environment}" != "${moss_environment}" ]] || die "QWEN_TTS_PYTHON and MOSS_TTS_PYTHON must use separate environments."
}
validate_tts_configuration() {
  local configured_python
  local resolved_python

  if [[ -n "${INDEX_TTS_PYTHON:-}" ]]; then
    configured_python="${INDEX_TTS_PYTHON}"
    resolved_python="$(resolve_executable "${configured_python}")" || die "INDEX_TTS_PYTHON is not executable: ${configured_python}"
    python_environment_id "${resolved_python}" >/dev/null || die "INDEX_TTS_PYTHON does not point to a working Python interpreter."
    INDEX_TTS_PYTHON="${resolved_python}"
  fi
  if [[ -n "${QWEN_TTS_PYTHON:-}" ]]; then
    configured_python="${QWEN_TTS_PYTHON}"
    resolved_python="$(resolve_executable "${configured_python}")" || die "QWEN_TTS_PYTHON is not executable: ${configured_python}"
    python_environment_id "${resolved_python}" >/dev/null || die "QWEN_TTS_PYTHON does not point to a working Python interpreter."
    QWEN_TTS_PYTHON="${resolved_python}"
  fi
  if [[ -n "${MOSS_TTS_PYTHON:-}" ]]; then
    configured_python="${MOSS_TTS_PYTHON}"
    resolved_python="$(resolve_executable "${configured_python}")" || die "MOSS_TTS_PYTHON is not executable: ${configured_python}"
    python_environment_id "${resolved_python}" >/dev/null || die "MOSS_TTS_PYTHON does not point to a working Python interpreter."
    MOSS_TTS_PYTHON="${resolved_python}"
  fi
  if [[ -n "${INDEX_TTS_SOURCE_DIR:-}" ]]; then
    [[ -f "${INDEX_TTS_SOURCE_DIR}/indextts/infer_v2.py" ]] || \
      die "INDEX_TTS_SOURCE_DIR must be the official IndexTTS repository root containing indextts/infer_v2.py."
    INDEX_TTS_SOURCE_DIR="$(cd "${INDEX_TTS_SOURCE_DIR}" && pwd -P)"
  fi
  assert_isolated_tts_pythons
}
trim() {
  local var="$1"
  var="${var#"${var%%[![:space:]]*}"}"
  var="${var%"${var##*[![:space:]]}"}"
  printf '%s' "$var"
}
read_env_txt_value() {
  local wanted_key="$1"
  local line
  local key
  local value
  local found=""
  [[ -f "env.txt" ]] || return 0

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    if [[ "${line}" == export\ * ]]; then
      line="${line#export }"
    fi
    [[ "${line}" != *=* ]] && continue
    key="$(trim "${line%%=*}")"
    [[ "${key}" == "${wanted_key}" ]] || continue
    value="$(trim "${line#*=}")"
    found="${value}"
  done < env.txt
  printf '%s' "${found}"
}
prompt_yes_no() {
  local prompt="$1"
  local default="${2:-}"
  local answer
  if [ ! -t 0 ]; then
    case "${default}" in
      [Yy]) return 0 ;;
      [Nn]) return 1 ;;
      *) return 1 ;;
    esac
  fi
  while true; do
    if [[ "${default}" =~ ^[Yy]$ ]]; then
      if ! read -r -p "${prompt} [Y/n] " answer; then
        answer="${default}"
      fi
      answer="${answer:-$default}"
    elif [[ "${default}" =~ ^[Nn]$ ]]; then
      if ! read -r -p "${prompt} [y/N] " answer; then
        answer="${default}"
      fi
      answer="${answer:-$default}"
    else
      if ! read -r -p "${prompt} [y/n] " answer; then
        return 1
      fi
    fi
    case "${answer}" in
      [Yy]*) return 0 ;;
      [Nn]*) return 1 ;;
    esac
    echo "Please answer yes or no."
  done
}
is_truthy() {
  local value="${1:-}"
  value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|y|yes|true|on) return 0 ;;
    *) return 1 ;;
  esac
}
install_qwen_tts_requirements() {
  local qwen_venv="${QWEN_TTS_VENV_DIR:-${APP_DIR}/.venv-qwen-tts}"
  if [[ -z "${QWEN_TTS_PYTHON:-}" ]]; then
    if [ ! -x "${qwen_venv}/bin/python3" ]; then
      log "Creating isolated Qwen3 TTS environment at ${qwen_venv}..."
      python3 -m venv "${qwen_venv}"
    else
      log "Reusing isolated Qwen3 TTS environment at ${qwen_venv}."
    fi
    QWEN_TTS_PYTHON="${qwen_venv}/bin/python3"
  fi

  validate_tts_configuration
  is_isolated_python_environment "${QWEN_TTS_PYTHON}" || \
    die "QWEN_TTS_PYTHON must point to a virtual or Conda environment, not the system Python."
  log "Installing pinned Qwen3 TTS dependencies only in ${QWEN_TTS_PYTHON}..."
  "${QWEN_TTS_PYTHON}" -m pip install --upgrade pip >/dev/null
  "${QWEN_TTS_PYTHON}" -m pip install --upgrade --no-cache-dir "qwen-tts==0.1.1"
  log "Validating the isolated Qwen3 TTS environment..."
  "${QWEN_TTS_PYTHON}" - <<'PY'
import sys
import traceback

try:
    from server.python import qwen_tts_worker as worker
    worker.ensure_dependencies()
except Exception:
    traceback.print_exc()
    sys.exit(1)
PY
  log "Qwen3 TTS dependencies installed."
}
install_moss_tts_requirements() {
  local moss_venv="${MOSS_TTS_VENV_DIR:-${APP_DIR}/.venv-moss-tts}"
  if [[ -z "${MOSS_TTS_PYTHON:-}" ]]; then
    if [ ! -x "${moss_venv}/bin/python3" ]; then
      log "Creating isolated MOSS-TTS v1.5 environment at ${moss_venv}..."
      python3 -m venv "${moss_venv}"
    else
      log "Reusing isolated MOSS-TTS environment at ${moss_venv}."
    fi
    MOSS_TTS_PYTHON="${moss_venv}/bin/python3"
  fi

  validate_tts_configuration
  is_isolated_python_environment "${MOSS_TTS_PYTHON}" || \
    die "MOSS_TTS_PYTHON must point to a virtual or Conda environment, not the system Python."
  log "Installing pinned MOSS-TTS v1.5 dependencies only in ${MOSS_TTS_PYTHON}..."
  "${MOSS_TTS_PYTHON}" -m pip install --upgrade pip >/dev/null
  "${MOSS_TTS_PYTHON}" -m pip install --upgrade --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.9.1" "torchaudio==2.9.1" "torchcodec==0.8.1" \
    "transformers==5.0.0" "safetensors==0.6.2" "numpy==2.1.0" \
    "orjson==3.11.4" "tqdm==4.67.1" "PyYAML==6.0.3" "einops==0.8.1" \
    "scipy==1.16.2" "librosa==0.11.0" "tiktoken==0.12.0" \
    "soundfile==0.13.1" "huggingface_hub"
  log "Validating the isolated MOSS-TTS environment..."
  "${MOSS_TTS_PYTHON}" - <<'PY'
from importlib import metadata
import torch
import torchaudio
import torchcodec
import transformers

assert metadata.version("transformers") == "5.0.0"
assert tuple(map(int, torch.__version__.split("+", 1)[0].split(".")[:2])) >= (2, 9)
print(f"MOSS-TTS runtime ready: torch={torch.__version__}, transformers={transformers.__version__}")
PY
  log "MOSS-TTS v1.5 dependencies installed."
}
install_ollama() {
  if have ollama; then
    log "Ollama already installed; skipping."
    return 0
  fi
  log "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  log "Ollama installation finished."
}

export DEBIAN_FRONTEND=noninteractive

REPO_URL="${REPO_URL:-https://github.com/mikeymcfish/VoiceForge.git}"
APP_DIR="${APP_DIR:-$SCRIPT_DIR}"
PORT="${PORT:-5000}"
NODE_MAJOR="${NODE_MAJOR:-22}"

log "Installing base OS deps..."
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates curl git gnupg python3 python3-pip python3-venv build-essential openssl ffmpeg
rm -rf /var/lib/apt/lists/*

# Install Node.js ${NODE_MAJOR}.x if missing or too old.
need_node=true
if have node; then
  INSTALLED_MAJOR="$(node -v | sed 's/^v//' | cut -d. -f1)"
  if [ "$INSTALLED_MAJOR" -ge "$NODE_MAJOR" ]; then
    need_node=false
    log "Reusing existing Node.js $(node -v)"
  fi
fi
if [ "$need_node" = true ]; then
  log "Installing Node.js ${NODE_MAJOR}.x from NodeSource..."
  curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -
  apt-get install -y --no-install-recommends nodejs
  rm -rf /var/lib/apt/lists/*
  log "Node installed: $(node -v); npm $(npm -v)"
fi

# Fetch code without rewriting local work. Dirty checkouts are preserved as-is;
# clean checkouts may advance only through their configured upstream.
if [ -e "${APP_DIR}/.git" ]; then
  if [[ -n "$(git -C "${APP_DIR}" status --porcelain --untracked-files=normal)" ]]; then
    log "Repo at ${APP_DIR} has local changes; preserving them and skipping the update."
  elif git -C "${APP_DIR}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' >/dev/null 2>&1; then
    log "Clean repo found at ${APP_DIR}; applying a fast-forward-only update..."
    git -C "${APP_DIR}" pull --ff-only --prune
  else
    log "Repo at ${APP_DIR} has no configured upstream; preserving it and skipping the update."
  fi
else
  mkdir -p "${APP_DIR}"
  log "Bootstrapping ${REPO_URL} into ${APP_DIR}..."
  if [[ -z "$(find "${APP_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    git clone "${REPO_URL}" "${APP_DIR}"
  else
    die "${APP_DIR} is non-empty and is not a Git checkout. Choose an empty APP_DIR or clone VoiceForge there first; no files were changed."
  fi
fi

cd "${APP_DIR}"

# Load legacy backend settings before any optional Python installation. Explicit
# process environment variables take precedence over env.txt.
if [[ -f "env.txt" ]]; then
  [[ -n "${INDEX_TTS_PYTHON:-}" ]] || INDEX_TTS_PYTHON="$(read_env_txt_value "INDEX_TTS_PYTHON")"
  [[ -n "${INDEX_TTS_SOURCE_DIR:-}" ]] || INDEX_TTS_SOURCE_DIR="$(read_env_txt_value "INDEX_TTS_SOURCE_DIR")"
  [[ -n "${QWEN_TTS_PYTHON:-}" ]] || QWEN_TTS_PYTHON="$(read_env_txt_value "QWEN_TTS_PYTHON")"
  [[ -n "${MOSS_TTS_PYTHON:-}" ]] || MOSS_TTS_PYTHON="$(read_env_txt_value "MOSS_TTS_PYTHON")"
fi

if is_truthy "${INSTALL_TTS_REQUIREMENTS:-}"; then
  die "Automatic IndexTTS installation was removed. Prepare an isolated official IndexTTS environment, then set INDEX_TTS_PYTHON and INDEX_TTS_SOURCE_DIR."
fi
validate_tts_configuration

log "Installing npm dependencies..."
if [ -f package-lock.json ]; then
  npm ci
else
  log "package-lock.json is absent; using npm install to create one."
  npm install
fi

if [[ -z "${INSTALL_QWEN_TTS_REQUIREMENTS:-}" ]]; then
  if prompt_yes_no "Install Qwen3 TTS Python dependencies now?" "n"; then
    INSTALL_QWEN_TTS_REQUIREMENTS="yes"
  else
    INSTALL_QWEN_TTS_REQUIREMENTS="no"
  fi
fi
if is_truthy "${INSTALL_QWEN_TTS_REQUIREMENTS:-}"; then
  install_qwen_tts_requirements
fi
if [[ -z "${INSTALL_MOSS_TTS_REQUIREMENTS:-}" ]]; then
  if prompt_yes_no "Install MOSS-TTS v1.5 Python dependencies now?" "n"; then
    INSTALL_MOSS_TTS_REQUIREMENTS="yes"
  else
    INSTALL_MOSS_TTS_REQUIREMENTS="no"
  fi
fi
if is_truthy "${INSTALL_MOSS_TTS_REQUIREMENTS:-}"; then
  install_moss_tts_requirements
fi
assert_isolated_tts_pythons

if [[ -z "${INSTALL_OLLAMA:-}" ]]; then
  if prompt_yes_no "Install Ollama on this machine?" "n"; then
    INSTALL_OLLAMA="yes"
  else
    INSTALL_OLLAMA="no"
  fi
fi
if is_truthy "${INSTALL_OLLAMA:-}"; then
  install_ollama
fi

# Generate SESSION_SECRET if missing.
if [[ -z "${SESSION_SECRET:-}" && -f "env.txt" ]]; then
  SESSION_SECRET_FROM_ENV="$(grep -E '^SESSION_SECRET=' env.txt | tail -n 1 | cut -d'=' -f2- || true)"
  if [[ -n "${SESSION_SECRET_FROM_ENV:-}" ]]; then
    SESSION_SECRET="${SESSION_SECRET_FROM_ENV}"
    log "Using SESSION_SECRET from env.txt"
  fi
fi

if [[ -z "${SESSION_SECRET:-}" ]]; then
  if have python3; then
    SESSION_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  else
    SESSION_SECRET="$(openssl rand -hex 32)"
  fi
  log "Generated SESSION_SECRET."
fi

# Write .env (include PORT and NODE_ENV for consistency).
ENV_FILE=".env"
log "Preparing ${ENV_FILE}..."
declare -A env_values=(
  ["SESSION_SECRET"]="${SESSION_SECRET}"
  ["NODE_ENV"]="production"
  ["PORT"]="${PORT}"
)
[[ -n "${HUGGINGFACE_API_TOKEN:-}" ]] && env_values["HUGGINGFACE_API_TOKEN"]="${HUGGINGFACE_API_TOKEN}"
[[ -n "${OLLAMA_BASE_URL:-}" ]] && env_values["OLLAMA_BASE_URL"]="${OLLAMA_BASE_URL}"
[[ -n "${INDEX_TTS_PYTHON:-}" ]] && env_values["INDEX_TTS_PYTHON"]="${INDEX_TTS_PYTHON}"
[[ -n "${INDEX_TTS_SOURCE_DIR:-}" ]] && env_values["INDEX_TTS_SOURCE_DIR"]="${INDEX_TTS_SOURCE_DIR}"
[[ -n "${QWEN_TTS_PYTHON:-}" ]] && env_values["QWEN_TTS_PYTHON"]="${QWEN_TTS_PYTHON}"
[[ -n "${MOSS_TTS_PYTHON:-}" ]] && env_values["MOSS_TTS_PYTHON"]="${MOSS_TTS_PYTHON}"
[[ -n "${HF_PROVIDER:-}" ]] && env_values["HF_PROVIDER"]="${HF_PROVIDER}"
[[ -n "${LLM_DEBUG:-}" ]] && env_values["LLM_DEBUG"]="${LLM_DEBUG}"
[[ -n "${LLM_DEBUG_FULL:-}" ]] && env_values["LLM_DEBUG_FULL"]="${LLM_DEBUG_FULL}"
[[ -n "${LLM_DEBUG_FILE:-}" ]] && env_values["LLM_DEBUG_FILE"]="${LLM_DEBUG_FILE}"
[[ -n "${REUSE_PORT:-}" ]] && env_values["REUSE_PORT"]="${REUSE_PORT}"

if [[ -f "env.txt" ]]; then
  log "Found env.txt; merging values into ${ENV_FILE}"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    if [[ "${line}" == export\ * ]]; then
      line="${line#export }"
    fi
    [[ "${line}" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#${key}=}"
    key="$(trim "${key}")"
    value="$(trim "${value}")"
    [[ -z "${key}" ]] && continue
    if [[ -z "${env_values[$key]:-}" ]]; then
      env_values["$key"]="${value}"
    fi
  done < env.txt
fi

# env.txt may provide backend interpreter settings. Normalize and validate the
# final merged values before persisting or starting the application.
INDEX_TTS_PYTHON="${env_values[INDEX_TTS_PYTHON]:-}"
INDEX_TTS_SOURCE_DIR="${env_values[INDEX_TTS_SOURCE_DIR]:-}"
QWEN_TTS_PYTHON="${env_values[QWEN_TTS_PYTHON]:-}"
MOSS_TTS_PYTHON="${env_values[MOSS_TTS_PYTHON]:-}"
validate_tts_configuration
[[ -n "${INDEX_TTS_PYTHON}" ]] && env_values["INDEX_TTS_PYTHON"]="${INDEX_TTS_PYTHON}"
[[ -n "${INDEX_TTS_SOURCE_DIR}" ]] && env_values["INDEX_TTS_SOURCE_DIR"]="${INDEX_TTS_SOURCE_DIR}"
[[ -n "${QWEN_TTS_PYTHON}" ]] && env_values["QWEN_TTS_PYTHON"]="${QWEN_TTS_PYTHON}"
[[ -n "${MOSS_TTS_PYTHON}" ]] && env_values["MOSS_TTS_PYTHON"]="${MOSS_TTS_PYTHON}"

{
  for key in "${!env_values[@]}"; do
    printf '%s=%s\n' "$key" "${env_values[$key]}"
  done | sort
} > "${ENV_FILE}"

log "Building production assets..."
npm run build

# Optional: daemonize with pm2
if [[ -n "${USE_PM2:-}" ]]; then
  if ! have pm2; then
    log "Installing pm2 globally..."
    npm i -g pm2
  fi
  log "Starting with pm2 on port ${PORT}..."
  # Use name 'voiceforge' and inherit env (incl .env via npm start)
  pm2 start "npm start" --name voiceforge --update-env
  pm2 save || true
  log "pm2 started. View logs with: pm2 logs voiceforge"
else
  export PORT
  log "Starting VoiceForge in foreground on port ${PORT}..."
  exec npm start
fi
