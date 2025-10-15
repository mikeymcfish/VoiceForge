#!/usr/bin/env bash
# One-shot VoiceForge setup for a fresh RunPod (PyTorch) image.
# Env vars you can set before running:
#   REPO_URL=https://github.com/mikeymcfish/VoiceForge.git
#   APP_DIR=$HOME/VoiceForge
#   PORT=5000
#   NODE_MAJOR=20
#   SESSION_SECRET=...           # auto-generated if absent
#   HUGGINGFACE_API_TOKEN=...
#   OLLAMA_BASE_URL=...
#   INDEX_TTS_REPO=...
#   INDEX_TTS_PYTHON=...
#   USE_PM2=1                    # if set, install pm2 and daemonize

set -euo pipefail

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

export DEBIAN_FRONTEND=noninteractive

REPO_URL="${REPO_URL:-https://github.com/mikeymcfish/VoiceForge.git}"
APP_DIR="${APP_DIR:-$HOME/VoiceForge}"
PORT="${PORT:-5000}"
NODE_MAJOR="${NODE_MAJOR:-20}"

log "Installing base OS deps..."
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates curl git gnupg python3 python3-pip python3-venv build-essential openssl
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

# Fetch code (idempotent): pull if exists, else clone.
if [ -d "${APP_DIR}/.git" ]; then
  log "Repo exists at ${APP_DIR}; pulling latest..."
  git -C "${APP_DIR}" fetch --all --prune
  git -C "${APP_DIR}" reset --hard origin/HEAD || true
  git -C "${APP_DIR}" pull --ff-only || true
else
  log "Cloning ${REPO_URL} into ${APP_DIR}..."
  git clone "${REPO_URL}" "${APP_DIR}"
fi

cd "${APP_DIR}"

log "Installing npm dependencies..."
if [ -f package-lock.json ]; then
  npm ci || { log "npm ci failed; falling back to npm install"; npm install; }
else
  npm install
fi

# Generate SESSION_SECRET if missing.
if [[ -z "${SESSION_SECRET:-}" ]]; then
  if have python3; then
    SESSION_SECRET="$(python3 - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)"
  else
    SESSION_SECRET="$(openssl rand -hex 32)"
  fi
  log "Generated SESSION_SECRET."
fi

# Write .env (include PORT and NODE_ENV for consistency).
ENV_FILE=".env"
log "Writing ${ENV_FILE}..."
{
  echo "SESSION_SECRET=${SESSION_SECRET}"
  echo "NODE_ENV=production"
  echo "PORT=${PORT}"
  [[ -n "${HUGGINGFACE_API_TOKEN:-}" ]] && echo "HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN}"
  [[ -n "${OLLAMA_BASE_URL:-}" ]] && echo "OLLAMA_BASE_URL=${OLLAMA_BASE_URL}"
  [[ -n "${INDEX_TTS_REPO:-}" ]] && echo "INDEX_TTS_REPO=${INDEX_TTS_REPO}"
  [[ -n "${INDEX_TTS_PYTHON:-}" ]] && echo "INDEX_TTS_PYTHON=${INDEX_TTS_PYTHON}"
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
