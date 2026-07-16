# VoiceForge Studio

VoiceForge Studio is a local-first document-to-voice workspace. It turns pasted text, TXT, EPUB, or OCR-extracted PDF content into a reviewable narration script, then hands that script to local speech engines.

The supported workflow is:

1. **Import** — paste text, upload TXT/EPUB, or extract a PDF.
2. **Prepare** — edit the source and apply deterministic cleanup or optional AI repair.
3. **Cast** — preserve narration, detect dialogue, and map characters to speaker labels.
4. **Review** — compare the immutable source with an editable result and resolve partial-processing warnings.
5. **Generate** — continue directly to IndexTTS, VibeVoice, Qwen3-TTS, or MOSS-TTS v1.5 synthesis.

## What changed in 2.0

- A responsive React workspace replaces the old fixed three-column editor.
- Text can be pasted and edited; cleanup no longer overwrites the imported source.
- EPUB chapters follow the package spine instead of ZIP entry order.
- Sentence segmentation handles abbreviations, punctuation, and dialogue boundaries more reliably.
- Prompt rules now reflect the selected cleanup, narrator, speaker-count, and custom-instruction settings.
- Processing can be cancelled, and failed chunks are surfaced as warnings instead of reported as full success.
- PDF extraction and speech synthesis are part of the main navigation with direct handoffs between stages.
- The server binds to `127.0.0.1` by default.
- Qwen3-TTS and MOSS-TTS v1.5 can run from isolated local environments or through their official Hugging Face ZeroGPU Spaces.
- The JavaScript dependency set was pruned and upgraded; the current production audit is clean.

## Run the main app

Requirements:

- Node.js 22.12 or newer
- npm 10 or newer

### Windows launcher

Double-click [`VoiceForge.cmd`](./VoiceForge.cmd) for the normal Windows startup.
It checks Node and npm, installs the exact locked JavaScript dependencies only
when `package.json` or `package-lock.json` changes, builds production mode, starts
the local server, and opens the browser after the app responds. If Node.js is
missing or outdated, it can install the current LTS release through `winget`
after asking for confirmation. When the preferred/default port is occupied, the
launcher selects the next available localhost port; an explicit `--port` remains
strict and reports a conflict instead.

Useful terminal modes:

```powershell
.\VoiceForge.cmd                 # production startup (default)
.\VoiceForge.cmd dev             # development server
.\VoiceForge.cmd install         # install only when dependencies changed
.\VoiceForge.cmd repair          # force a clean npm ci
.\VoiceForge.cmd setup-index     # install/configure the official IndexTTS runtime
.\VoiceForge.cmd setup-qwen      # install the isolated Qwen3-TTS runtime
.\VoiceForge.cmd setup-moss      # install the isolated MOSS-TTS v1.5 runtime
.\VoiceForge.cmd --port 5050     # use another port for this launch
.\VoiceForge.cmd --no-browser    # keep the browser closed
```

The launcher never downloads speech models. Configure IndexTTS, VibeVoice, Qwen,
MOSS, or PDF OCR through their documented setup
steps when those backends are needed. Stop a running VoiceForge server before
using `repair` or updating `package.json`/`package-lock.json`, because `npm ci`
replaces the installed dependency tree.

### Manual startup

```powershell
npm ci
npm run dev
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Production build:

```powershell
npm run build
npm start
```

### RunPod/Linux installer safety

`install.sh` keeps Node on major version 22. Point `APP_DIR` at either an empty
directory for a fresh clone or an existing VoiceForge checkout:

```bash
sudo env APP_DIR=/opt/voiceforge ./install.sh
```

For an existing checkout, the installer updates only a clean branch with a
configured upstream, using `git pull --ff-only`. If the checkout is dirty or has
no upstream, it preserves the tree and skips the update. It refuses to bootstrap
into a non-empty, non-Git directory; it never resets the checkout or deletes the
directory's existing contents.

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `PORT` | HTTP port; defaults to `5000`. |
| `HOST` | Bind address; defaults to `127.0.0.1`. |
| `HUGGINGFACE_API_TOKEN` or `HF_TOKEN` | Optional Hugging Face inference token. |
| `HF_PROVIDER` | Hugging Face inference provider; defaults to `hf-inference`. |
| `OLLAMA_BASE_URL` | Ollama endpoint; defaults to `http://localhost:11434`. |
| `PDF_OCR_PYTHON` | Python executable used by the PDF OCR worker. |
| `INDEX_TTS_PYTHON` | Python executable for an operator-managed, isolated IndexTTS environment. |
| `INDEX_TTS_SOURCE_DIR` | Optional official IndexTTS repository root containing `indextts/infer_v2.py`; not needed when `indextts` is installed in the selected environment. |
| `QWEN_TTS_PYTHON` | Python executable for the isolated Qwen3-TTS environment. It must not share an environment with IndexTTS or MOSS. |
| `MOSS_TTS_PYTHON` | Python executable for the isolated MOSS-TTS v1.5 environment. It must not share an environment with Qwen or IndexTTS. |
| `MOSS_TTS_FFMPEG_BIN` | Windows-only path to an FFmpeg 4-7 shared-build `bin` directory used by MOSS/TorchCodec. The setup command discovers and writes this automatically. |
| `VIBEVOICE_PYTHON` | Python executable used by VibeVoice. |
| `VOICEFORGE_DEFAULT_VOICES_DIR` | Optional default voice-library directory; defaults to `<app>/default_voices`. |
| `VOICEFORGE_ENABLE_REMOTE_MCP` | Explicitly enable `/mcp` when `HOST` is not loopback. |
| `VOICEFORGE_MCP_BEARER_TOKEN` | Optional bearer token required by `/mcp` when set. |
| `VOICEFORGE_PUBLIC_URL` | Public HTTPS origin used in completed MCP audio links. |

Do not expose this app directly to an untrusted network. It is a personal local workspace, not a multi-user service: model setup, uploads, job files, and provider-token configuration do not have an authentication layer. If remote access is required, put it behind authentication and TLS and add upload/rate limits.

## Model choices

Text preparation supports:

- **Safe cleanup** — deterministic transformations that run without a model.
- **Hugging Face inference** — cloud processing with a configured token.
- **Ollama** — local inference using an explicitly installed model name.

Provider token counts and prices shown in the UI are estimates unless the provider reports exact usage. AI-repaired text should always be reviewed before synthesis. If any chunk fails validation after retries, VoiceForge preserves that source chunk and marks the result as partial.

## Speech backends

The React app exposes IndexTTS, VibeVoice, Qwen3-TTS, and MOSS-TTS v1.5 under
**Create audio**. Their model downloads are large and usually benefit from a
CUDA-capable GPU. Qwen and MOSS also show an explicit **Local GPU / Hugging Face
ZeroGPU Space** target selector. The “Agents” button on a Space is API discovery;
VoiceForge calls that same Gradio API when the hosted target is selected.
Local Qwen/MOSS jobs share a serialized GPU queue so two multi-gigabyte models
are never loaded into device memory at the same time.

The legacy Python/Gradio lab is still available for backend experiments:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r gradio_app\requirements.txt
python -m gradio_app
```

Use a separate isolated Python environment for each speech backend, and run only sources you trust. VibeVoice setup uses the community repository pinned to commit `07cb79feadd2d3fd7f47530d4c964a12857936a0`; neither the web app nor the Gradio lab accepts a repository or branch override.

### VibeVoice

VibeVoice setup checks out the reviewed community source revision in detached
mode. Its default model downloads are pinned as well:

- `microsoft/VibeVoice-1.5B` at `c00898d257e6b46004e3e2866a47534085fb685a`
- `aoi-ot/VibeVoice-Large` at `8229be00d7c036aa32321e4dae8a81d433f6413a`

Set `VIBEVOICE_PYTHON` to a dedicated environment with PyTorch 2.6 or newer.
Setup also downloads the matching Qwen tokenizer at a pinned revision, records
an atomic file-size inventory for every completed snapshot, and keeps synthesis
offline. Partial, modified, or unpinned snapshots are not selectable. The UI's
guidance scale maps directly to VibeVoice CFG (default `1.3`); an alternate CLI
is available only through an operator-reviewed `VIBEVOICE_COMMAND_TEMPLATE`.
Use only clean reference recordings you own or have permission to clone; the
model is intended for English and Chinese speech and can produce unstable or
unintended audio.

### IndexTTS2

VoiceForge downloads model data only from the official
[`IndexTeam/IndexTTS-2`](https://huggingface.co/IndexTeam/IndexTTS-2/tree/main)
repository, pinned to revision `740dcaff396282ffb241903d150ac011cd4b1ede`. The web API
does not accept a repository override.

VoiceForge does not download IndexTTS Python source, install packages, or change
Torch from a web request. On Windows, stop the app and run the explicit setup
helper once:

```powershell
.\VoiceForge.cmd setup-index
```

The helper checks out the [official IndexTTS repository](https://github.com/index-tts/index-tts)
at reviewed revision `13495845e3028f0bb6ca1462ad22aa0e76349e40`, creates its locked
Python 3.11 environment with the official `uv sync --frozen` workflow, validates
PyTorch and `indextts.infer_v2`, then writes the two runtime paths to the ignored
`.env` file. It preserves the separately downloaded model snapshot. Restart
VoiceForge after setup and select **Verify runtime**.

Advanced operators may instead prepare their own isolated environment from the
official source and configure the app before starting it:

```powershell
$env:INDEX_TTS_PYTHON = "C:\path\to\indextts-env\Scripts\python.exe"
$env:INDEX_TTS_SOURCE_DIR = "C:\path\to\official-index-tts"
npm run dev
```

`INDEX_TTS_SOURCE_DIR` can be omitted when the official `indextts` package is
already importable by `INDEX_TTS_PYTHON`. Loading and synthesis fail closed with a
setup error when the source is absent or the selected environment uses PyTorch
older than 2.6. A completed pinned model snapshot is recorded with a strict file
inventory and recognized after app restarts; runtime verification remains a
separate explicit step.

The Linux installer also does not install IndexTTS. Set `INDEX_TTS_PYTHON` and,
when needed, `INDEX_TTS_SOURCE_DIR` before running it. The legacy
`INSTALL_TTS_REQUIREMENTS=yes` mode now fails with a setup message instead of
modifying a shared Python environment.

### Qwen3-TTS voice cloning

The main app and Gradio lab use `qwen-tts==0.1.1` with these supported
voice-clone checkpoints:

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (default)
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

On Windows, stop VoiceForge and prepare the separate runtime once:

```powershell
.\VoiceForge.cmd setup-qwen
```

Restart VoiceForge, open **Create audio → Qwen3-TTS**, select **Local GPU**, and
download the pinned checkpoint. Local synthesis is then forced offline against
the verified file inventory. Hosted mode calls the official
[`Qwen/Qwen3-TTS`](https://huggingface.co/spaces/Qwen/Qwen3-TTS) Space and offers
voice cloning, voice design, and preset-speaker modes. Hosted calls have a short
per-request ZeroGPU time limit; use Local GPU for long scripts.

On Linux, the installer can prepare Qwen in its own virtual environment without
touching the IndexTTS interpreter:

```bash
INSTALL_QWEN_TTS_REQUIREMENTS=yes \
QWEN_TTS_VENV_DIR="$PWD/.venv-qwen-tts" \
./install.sh
```

The installer pins `qwen-tts==0.1.1`, validates the completed environment, and
writes the resulting `QWEN_TTS_PYTHON` path to `.env`. Install the Gradio lab
requirements and launch the lab with that same interpreter so Qwen stays
isolated:

```bash
set -a; source .env; set +a
"$QWEN_TTS_PYTHON" -m pip install -r gradio_app/requirements.txt
"$QWEN_TTS_PYTHON" -m gradio_app
```

Alternatively, set `QWEN_TTS_PYTHON` to an existing dedicated environment before
running the installer; optional installation refuses a system-wide interpreter.
The installer rejects configurations where
`QWEN_TTS_PYTHON` and `INDEX_TTS_PYTHON` resolve to the same Python environment.
Supply the exact transcript of the reference clip for best fidelity. Without it,
the UI warns that speaker-embedding-only cloning has lower expected quality.

Relevant settings include `QWEN_TTS_PYTHON`, `QWEN_TTS_MODEL_ID`, `QWEN_TTS_LANGUAGE`, `QWEN_TTS_DEVICE`, `QWEN_TTS_MAX_CHARS`, `QWEN_TTS_GAP_MS`, and `QWEN_TTS_USE_FLASH_ATTENTION=1`.

The integration follows the [official Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS) and [0.6B Base model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).

### MOSS-TTS v1.5

VoiceForge uses `OpenMOSS-Team/MOSS-TTS-v1.5` pinned to revision
`cdd3b911b1585e3f2dbc7775ef10f9926f58850a`, with its audio tokenizer pinned and
inventoried separately. MOSS is an 8B model with 31-language synthesis,
zero-shot voice cloning, continuation modes, and inline pauses such as
`[pause 1.5s]`. The standard PyTorch path is large; allow roughly 16 GB for the
model download and plan on a high-memory CUDA GPU for comfortable local use.

On Windows:

```powershell
.\VoiceForge.cmd setup-moss
```

MOSS uses TorchCodec for audio I/O. On Windows this requires an FFmpeg **shared**
build, not merely a working static `ffmpeg.exe`. The setup command discovers a
compatible FFmpeg 4-7 shared build and stores its directory in
`MOSS_TTS_FFMPEG_BIN`. If none is installed, use the pinned compatible build and
rerun setup:

```powershell
winget install --id Gyan.FFmpeg.Shared --exact --version 7.1.1 --scope user
.\VoiceForge.cmd setup-moss
```

Restart VoiceForge, open **Create audio → MOSS-TTS v1.5**, and download the
pinned snapshots. On Linux/RunPod, set `INSTALL_MOSS_TTS_REQUIREMENTS=yes` when
running `install.sh`, or point `MOSS_TTS_PYTHON` at a dedicated environment.
MOSS cannot share Qwen's environment: the official MOSS stack requires
Transformers 5.0.0 while Qwen pins Transformers 4.57.3.

Hosted mode calls the official
[`OpenMOSS-Team/MOSS-TTS-v1.5`](https://huggingface.co/spaces/OpenMOSS-Team/MOSS-TTS-v1.5)
ZeroGPU Space. VoiceForge validates the live endpoint contract before uploading
audio or submitting text, and rejects API drift rather than guessing.

### Hugging Face usage bars

The bar above the workspace reads the authenticated ZeroGPU balance from the
official Hub quota API in GPU-seconds. The second bar is deliberately labeled
**Included inference credit** rather than “tokens”: Inference Providers are
billed in dollars at model/provider-specific rates. That value is an estimate
of the plan's included monthly credit minus this month's HF-routed inference
spend; purchased credit, organization billing, and custom provider keys can make
the real billing balance different. Qwen/MOSS hosted synthesis consumes ZeroGPU,
not Inference Provider credit.

### Default voice library

Put reusable reference recordings in `default_voices/`. The folder is deliberately
Git-ignored because voice recordings and their transcripts may be private. Pair an
audio file with a same-name UTF-8 transcript when available:

```text
default_voices/
  narrator.wav
  narrator.txt
```

VoiceForge accepts WAV, MP3, FLAC, M4A, AAC, OGG, Opus, and WebM files up to
32 MB. The app and MCP server expose opaque voice IDs rather than paths, reject
links and nested files, and revalidate each file before use. A paired transcript
is supplied automatically to Qwen voice cloning. The **Create audio** page can use
library voices or custom uploads with all four engines; VibeVoice supports up to
four library/upload slots.

### ChatGPT/Codex MCP and skill

The running VoiceForge process exposes a stateless Streamable HTTP MCP endpoint:

```text
http://127.0.0.1:5000/mcp
```

The repo includes `.codex/config.toml` for local Codex discovery and the
repo-scoped `$voiceforge-tts` skill under `.agents/skills/voiceforge-tts`. Start
VoiceForge on port 5000 before opening a new Codex task so the MCP connection can
initialize. If you intentionally choose another port, update both MCP URLs.

The MCP tools can:

- list model capabilities/readiness and safe default voice IDs;
- recommend Qwen, MOSS, VibeVoice, or IndexTTS from text length and requirements;
- start an idempotent asynchronous synthesis job with an explicit `local` or
  `agent` target;
- poll/cancel jobs and return completed audio as an MCP resource/download URL.

`agent` means the official Hugging Face Space API: text and any selected voice
are uploaded to Hugging Face and ZeroGPU quota is consumed. The skill never
silently switches between Local and Agent. The deterministic length bands are
Qwen Agent through 1,200 characters, MOSS Agent through 5,000, and Local for
anything longer; multi-speaker work prefers local VibeVoice.

VoiceForge binds to loopback by default. For ChatGPT web, the MCP endpoint must be
behind HTTPS. Remote MCP stays disabled when the app binds to a non-loopback
`HOST` unless `VOICEFORGE_ENABLE_REMOTE_MCP=true` is set deliberately. You may
also set `VOICEFORGE_MCP_BEARER_TOKEN`; configure the same environment variable
in the MCP client. This bearer option is suitable for private clients, not a
replacement for OAuth on a broadly published ChatGPT app.

## Verification

```powershell
npm run check
npm run test:text
npm run test:speech
npm run test:hf-contract
npm run test:voices
npm run test:mcp
npm run test:prompts
npm run build
npm audit --omit=dev
python server/python/tts_workers_selftest.py
```

The text self-test covers sentence segmentation, TTS-sized chunking,
speaker-label preservation, and mojibake cleanup. The speech tests cover shared
contracts, ZeroGPU quota parsing, pinned worker manifests, tamper detection, and
the current public Space endpoint schemas. The contract test reads metadata only
and does not consume ZeroGPU inference quota. The prompt test checks
configuration-sensitive prompt construction.

## Project map

```text
client/                  React workspace
server/                  Express APIs, WebSockets, processing, OCR and TTS services
server/python/           Optional Python workers
shared/                  Shared schemas and text segmentation utilities
scripts/                 Regression and prompt smoke tests
gradio_app/              Optional Python backend lab
```

Project data and model caches remain local. The browser stores only UI settings and draft handoffs between workflow pages.
