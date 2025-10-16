#!/usr/bin/env bash
# One-shot VoiceForge setup for a fresh RunPod (PyTorch) image.
# Env vars you can set before running:
#   REPO_URL=https://github.com/mikeymcfish/VoiceForge.git
#   APP_DIR=$HOME/VoiceForge          # defaults to this script's directory
#   PORT=5000
#   NODE_MAJOR=20
#   SESSION_SECRET=...           # auto-generated if absent
#   HUGGINGFACE_API_TOKEN=...
#   OLLAMA_BASE_URL=...
#   INDEX_TTS_REPO=...
#   INDEX_TTS_PYTHON=...
#   USE_PM2=1                    # if set, install pm2 and daemonize

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"; }
have() { command -v "$1" >/dev/null 2>&1; }
trim() {
  local var="$1"
  var="${var#"${var%%[![:space:]]*}"}"
  var="${var%"${var##*[![:space:]]}"}"
  printf '%s' "$var"
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
install_tts_requirements() {
  if ! have python3; then
    log "Python 3 not found; skipping IndexTTS dependency installation."
    return 0
  fi
  log "Installing IndexTTS python dependencies..."
  python3 - <<'PY'
import sys
import traceback

try:
    from server.python import index_tts_worker as worker
    for spec in ("indextts", "huggingface_hub", "modelscope", "soundfile", "torch"):
        worker.ensure_package(spec)
    worker.ensure_runtime_dependencies()
except Exception:
    traceback.print_exc()
    sys.exit(1)
PY
  log "IndexTTS dependencies installed."
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
NODE_MAJOR="${NODE_MAJOR:-20}"

log "Installing base OS deps..."
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates curl git gnupg python3 python3-pip python3-venv build-essential openssl ffmpeg
rm -rf /var/lib/apt/lists/*

log "Installing Python utilities..."
python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install --no-cache-dir hf_transfer >/dev/null

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

# Fetch code (idempotent): pull if exists, else bootstrap into APP_DIR.
if [ -d "${APP_DIR}/.git" ]; then
  log "Repo exists at ${APP_DIR}; pulling latest..."
  git -C "${APP_DIR}" fetch --all --prune
  git -C "${APP_DIR}" reset --hard origin/HEAD || true
  git -C "${APP_DIR}" pull --ff-only || true
else
  mkdir -p "${APP_DIR}"
  log "Bootstrapping ${REPO_URL} into ${APP_DIR}..."
  if [[ -z "$(find "${APP_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    git clone "${REPO_URL}" "${APP_DIR}"
  else
    tmp_clone_root="$(mktemp -d)"
    git clone "${REPO_URL}" "${tmp_clone_root}/repo"
    if [[ "${SCRIPT_PATH}" == "${APP_DIR}/"* ]]; then
      find "${APP_DIR}" -mindepth 1 -maxdepth 1 ! -path "${SCRIPT_PATH}" -exec rm -rf {} +
    else
      find "${APP_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    fi
    cp -a "${tmp_clone_root}/repo/." "${APP_DIR}/"
    rm -rf "${tmp_clone_root}"
  fi
fi

cd "${APP_DIR}"

log "Installing npm dependencies..."
if [ -f package-lock.json ]; then
  npm ci || { log "npm ci failed; falling back to npm install"; npm install; }
else
  npm install
fi

if [[ -z "${INSTALL_TTS_REQUIREMENTS:-}" ]]; then
  if prompt_yes_no "Install IndexTTS Python dependencies now?" "n"; then
    INSTALL_TTS_REQUIREMENTS="yes"
  else
    INSTALL_TTS_REQUIREMENTS="no"
  fi
fi
if is_truthy "${INSTALL_TTS_REQUIREMENTS:-}"; then
  install_tts_requirements
fi

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
[[ -n "${INDEX_TTS_REPO:-}" ]] && env_values["INDEX_TTS_REPO"]="${INDEX_TTS_REPO}"
[[ -n "${INDEX_TTS_PYTHON:-}" ]] && env_values["INDEX_TTS_PYTHON"]="${INDEX_TTS_PYTHON}"

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
