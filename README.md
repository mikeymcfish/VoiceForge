# TTS Text Editor

A professional text preprocessing application for multi-speaker TTS (text-to-speech) systems with AI-powered text repair and intelligent dialogue parsing. Supports both cloud-based (HuggingFace API) and local offline models.

## 🌟 Features

### File Processing
- **File Support**: Upload `.txt` or `.epub` files (up to 10MB)
- **Real-time Stats**: Word count and character count display
- **EPUB Parsing**: Automatic extraction from EPUB files

### Text Cleaning Options
- Replace smart quotes and non-standard punctuation
- Fix OCR errors (spacing, merged words)
- Correct spelling and remove bad characters
- Strip URLs, footnotes, and metadata
- Add punctuation for better TTS prosody

### Multi-Speaker Modes
- **Mode 0 - Single Speaker**: Clean text without speaker tags
- **Mode 1 - Format Conversion**: Convert existing multi-speaker text to standardized format
- **Mode 2 - Intelligent Parsing**: AI-powered speaker detection and dialogue extraction
  - Extract character names from text samples
  - Customizable sample size (5-100 sentences)
  - Optional narrator as separate speaker
  - Character-to-speaker mapping management

### Dual Model Support
- **HuggingFace API** (Cloud):
  - Access to powerful models like Qwen/Qwen2.5-72B-Instruct
  - Requires API token
  - Best performance for complex tasks
  
- **Local Models** (Offline):
  - No API token required
  - Works completely offline once downloaded
  - Three model options:
    - **LaMini-Flan-T5-783M** (~800MB) - Best performance
    - **Flan-T5 Base** (~500MB) - Balanced
    - **Flan-T5 Small** (~300MB) - Fastest
  - Automatic download and caching

### Advanced Features
- **Custom Instructions**: Add specific instructions for the LLM
- **Prompt Preview**: View exact prompts before processing (supports Single‑Pass + Concise Prompts)
- **Test Mode**: Process one chunk to preview results
- **Real-time Progress**: Live WebSocket updates during processing
- **Activity Logging**: Detailed timestamped logs with export
- **Single‑Pass Processing**: Clean + speaker formatting in one LLM call per chunk (reduces tokens)
- **Concise Prompts**: Use shorter instruction prompts to reduce input tokens (toggleable)
- **Narrator Attribution Modes** (Intelligent Parsing):
  - Remove tags (default)
  - Narrator says tags (verbatim)
  - Narrator adds context (intelligent rewrite)
- **Fix Hyphenation**: Merge words split by line breaks or hyphens (PDF/EPUB artifacts)

## 📋 Prerequisites

- Node.js 18+ and npm
- (Optional) HuggingFace API token for cloud models
- Storage space for local models (300MB - 800MB per model)

## 🚀 Installation

### 1. Clone or Download the Project

If you're on Replit, the project is already set up. Otherwise:

```bash
git clone <your-repo-url>
cd tts-text-editor
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment Variables

You have two options depending on which model source you want to use:

#### Option A: Using HuggingFace API (Cloud Models)

1. Get a HuggingFace API token:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Give it a name (e.g., "TTS Text Editor")
   - Copy the token

2. **For Replit Users** (Recommended):
   - Open the "Secrets" tool in your Replit workspace (lock icon in sidebar)
   - Click "+ New Secret"
   - Set key: `HUGGINGFACE_API_TOKEN`
   - Paste your token as the value
   - Click "Add Secret"
   
   **Important for Published/Deployed Apps:**
   - When you publish your app, Replit automatically includes workspace secrets
   - If the published version shows "need to log in or provide a token", check:
     1. The secret `HUGGINGFACE_API_TOKEN` exists in your Replit Secrets
     2. The secret name is spelled exactly as shown (case-sensitive)
     3. Try redeploying your app after adding/updating the secret

3. **For Local Development**:
   Create a `.env` file in the project root:
   ```
   HUGGINGFACE_API_TOKEN=your_token_here
   SESSION_SECRET=your_random_secret_here
   ```

#### Option B: Using Local Models (No API Token Required)

If you don't want to use the HuggingFace API:
- No API token needed
- No environment variable setup required
- Models download automatically on first use
- Works completely offline once models are cached

### 4. Run the Application

#### Development Mode
```bash
npm run dev
```

The app will be available at `http://localhost:5000`

#### Production Mode
```bash
npm run build
npm start
```

## 🎯 Quick Start Guide

### Using Cloud Models (HuggingFace API)

1. **Upload a file**: Drag and drop a `.txt` or `.epub` file
2. **Configure cleaning options**: Select desired text cleaning features
3. **Choose speaker mode**: 
   - None (single speaker)
   - Format conversion
   - Intelligent parsing (with optional character extraction)
4. **Select Model Source**: Keep "HuggingFace API" selected
5. **Test first** (optional): Click "Test One Chunk" to preview
6. **Start Processing**: Click "Start Processing"
7. **Export**: Copy or download the processed text

### Using Local Models (Offline)

1. **Upload a file**: Drag and drop a `.txt` or `.epub` file
2. **Configure cleaning options**: Select desired text cleaning features
3. **Choose speaker mode**: Set your preferred mode
4. **Select Model Source**: 
   - Click "Local Model" option
   - Choose a model from the dropdown:
     - **Flan-T5 Small** (300MB) - Fastest, good for testing
     - **Flan-T5 Base** (500MB) - Balanced performance
     - **LaMini-Flan-T5-783M** (800MB) - Best quality
5. **First time**: Model will download automatically (progress shown in UI)
6. **Test and Process**: Same as cloud models

### Character Extraction (Intelligent Mode)

1. Select "Intelligent" speaker mode
2. Set sample size (5-100 sentences to analyze)
3. Toggle "Include Narrator" if needed
4. Click "Extract Characters"
5. Review and manage character mappings
6. Proceed with processing

You can choose how the Narrator handles dialogue attribution tags ("he said"): remove, verbatim, or convert into a concise context line.

## 🔧 Configuration Options

### Text Cleaning Options
- **Smart Quotes**: Replace curly quotes with straight quotes
- **OCR Fixes**: Fix common OCR errors
- **Spell Check**: Correct spelling mistakes
- **Remove URLs**: Strip web links
- **Add Punctuation**: Improve prosody for TTS

### Speaker Configuration
- **Mode**: None, Format Conversion, or Intelligent
- **Format**: "Speaker 1:" or "[1]:" label style
- **Number of Speakers**: 1-20

### Model Settings
- **Model Source**: HuggingFace API or Local Model
- **API Model**: Select from available HuggingFace models
- **Local Model**: Choose from downloaded/available ONNX models
- **Batch Size**: Number of sentences per processing chunk (default: 10)

### Advanced Options
- **Custom Instructions**: Add specific instructions for the LLM
- **Prompt Preview**: View exact prompts before processing
- **Test Mode**: Process one chunk for preview
- **Single‑Pass Processing**: Combine stages into a single call per chunk
- **Concise Prompts**: Shorter instructions to reduce tokens
- **Fix Hyphenation**: Merge words split across lines/hyphens

## Providers and Cost Controls

### Hugging Face Providers
- Set provider via env `HF_PROVIDER`:
  - `auto` (recommended): lets HF pick a provider for the model/task
  - `hf-inference`: Hugging Face Inference endpoints
  - `fireworks-ai`, `groq`, `together`, etc. (if available in your HF account)

Notes:
- We route via Hugging Face using your `HUGGINGFACE_API_TOKEN` (no provider-specific key required).
- If a provider does not support text-generation (e.g., Fireworks), the app auto‑falls back to chat‑completions for that request.
- Model ids are normalized (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct` → `meta-llama/Llama-3.1-8B-Instruct`).

### Ollama (Local)
- Set `OLLAMA_BASE_URL` (default `http://localhost:11434`).

### Token Usage Tips
- Enable Single‑Pass Processing to avoid sending two prompts per chunk.
- Use Concise Prompts to reduce fixed instruction tokens.
- Increase batch size to amortize instruction tokens across more sentences.
- Prefer local models via Ollama if usage costs are a concern.

### Debugging Requests
- Set `LLM_DEBUG=1` to log request/response URL, headers (redacted), and payload previews for HF and Ollama.
  - Useful to see provider mapping calls and router decisions.

## 🐛 Troubleshooting

### "Need to log in or provide a token" Error (Published Apps)

**Issue**: Published/deployed version shows HuggingFace authentication error like:
- "need to log in or provide a token for HF"
- "HuggingFace API authentication failed"

**Important**: You do NOT need to log in with your HuggingFace account! The API token should work automatically. This error means the token isn't being recognized.

**Solutions** (try in order):

1. **Verify Secret Exists** (Replit):
   - Open the **Secrets** tool (🔒 icon) in your workspace
   - Check that `HUGGINGFACE_API_TOKEN` exists
   - Secret names are **case-sensitive** - must be exactly: `HUGGINGFACE_API_TOKEN`
   - Click the eye icon to verify the token value is correct (starts with `hf_...`)

2. **Check Token is Valid**:
   - Go to https://huggingface.co/settings/tokens
   - Verify your token exists and is not expired
   - If needed, create a new token and update your secret

3. **Redeploy Your App**:
   - After adding or updating the secret
   - **Stop your current deployment** (if running)
   - Click **"Publish"** again to redeploy
   - Replit should automatically sync the secret to your deployment
   - Wait for deployment to complete

4. **Check Deployment Logs**:
   - In your published app, check the deployment logs
   - Look for the warning: "⚠️ HUGGINGFACE_API_TOKEN not found"
   - If you see this warning, the secret isn't syncing properly

5. **Alternative: Use Local Models** (No token needed!):
   - In your app, switch to **"Local Model"** option
   - Select any model (Flan-T5 Small is fastest)
   - Model downloads automatically on first use
   - Works completely offline - no API token required
   - **This is the easiest solution if you have deployment issues!**

**Why This Happens**:
- Replit Secrets are environment-specific
- Published deployments need secrets to sync from workspace
- If the secret was added after publishing, you must redeploy
- Secret names must match exactly (case-sensitive)

### Local Model Issues

**Model won't download**:
- Check internet connection (needed for first download)
- Ensure sufficient disk space (300MB - 800MB per model)
- Check server logs for error messages

**Model download is slow**:
- Models are 300MB-800MB, download may take time
- Progress is shown in the UI
- Once downloaded, models are cached

**Out of memory errors**:
- Try using a smaller model (Flan-T5 Small)
- Reduce batch size in settings
- Close other applications

### General Issues

**File upload fails**:
- Check file size (max 10MB)
- Verify file format (.txt or .epub)
- Try a different file

**Processing stuck or slow**:
- Check activity log for errors
- Try reducing batch size
- For local models, first run may be slower (model loading)
- Cloud models may be rate-limited

**WebSocket connection errors**:
- Refresh the page
- Check server logs
- Ensure port 5000 is not blocked

## 📁 Project Structure

```
├── client/                     # Frontend React application
│   ├── src/
│   │   ├── components/        # UI components
│   │   │   ├── character-extraction.tsx
│   │   │   ├── model-source-selector.tsx
│   │   │   └── ...
│   │   ├── pages/             # Page components
│   │   │   └── home.tsx
│   │   └── lib/               # Utilities
│       
├── server/                     # Backend Express application
│   ├── routes.ts              # API routes & WebSocket
│   ├── llm-service.ts         # HuggingFace API integration
│   ├── local-model-service.ts # Local ONNX model execution
│   └── text-processor.ts      # Text chunking & processing
│
├── shared/                     # Shared TypeScript types
│   └── schema.ts              # Data models
│
├── .cache/                     # Local model cache (auto-created)
└── design_guidelines.md       # UI/UX guidelines
```

## 🔐 Security Notes

- **Never commit API tokens** to version control
- Use Replit Secrets or `.env` files for sensitive data
- `.env` is already in `.gitignore`
- Local models run on your server (no external API calls)
- Session secrets should be random and unique

## 🚀 Deployment

### Deploying on Replit

1. **Configure Secrets**:
   - Open Secrets tool
   - Add `HUGGINGFACE_API_TOKEN` (if using API)
   - Add `SESSION_SECRET` (random string)

2. **Click "Publish"**:
   - Secrets automatically sync to deployment
   - App will be available at `<your-repl>.replit.app`

3. **Verify**:
   - Test with both API and local models
   - Check that secrets are working
   - Monitor activity logs for errors

### Deploying Elsewhere

1. Set environment variables:
   ```
   HUGGINGFACE_API_TOKEN=your_token
   SESSION_SECRET=random_secret
   NODE_ENV=production
   ```

2. Build and start:
   ```bash
   npm run build
   npm start
   ```

3. Ensure port 5000 is accessible

## 📊 Performance Tips

### For Best Performance:
- **Cloud Models**: Best for complex text and quality
- **Local Models**: Best for privacy and offline use
- **Batch Size**: 
  - Larger (15-20): Faster but may lose context
  - Smaller (5-10): Better quality, slower
- **Model Selection**:
  - Qwen 2.5-72B (API): Best overall quality
  - LaMini-Flan-T5 (Local): Best local quality
  - Flan-T5 Small (Local): Fastest local processing

### Resource Usage:
- **API Mode**: Minimal server resources, network required
- **Local Mode**: 
  - Memory: 2-4GB per model
  - Storage: 300MB-800MB per model
  - CPU: Higher usage during inference

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Support

For issues or questions:
- Check the troubleshooting section above
- Review activity logs for detailed error messages
- Ensure secrets are properly configured
- Try local models if API issues persist

---

**Built with**: React, TypeScript, Express.js, HuggingFace Transformers, WebSocket, Tailwind CSS
