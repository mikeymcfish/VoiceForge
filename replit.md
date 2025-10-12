# TTS Text Editor

A professional text preprocessing application for multi-speaker TTS (text-to-speech) systems with AI-powered text repair and intelligent dialogue parsing.

## Overview

This application helps prepare text for multi-speaker TTS systems by cleaning, formatting, and structuring text with the help of HuggingFace's LLM models. It supports both plain text and EPUB files, offering configurable text cleaning options and intelligent speaker detection.

## Features

### File Upload & Processing
- **File Support**: Upload .txt or .epub files (up to 10MB)
- **Real-time Stats**: Displays word count and character count
- **Visual Preview**: See file information before processing

### Text Cleaning Options
- Replace smart quotes and non-standard punctuation
- Fix OCR errors (spacing, merged words)
- Correct spelling and remove bad characters
- Strip URLs, footnotes, and metadata
- Add punctuation for better TTS prosody

### Multi-Speaker Functionality
- **Mode 0 - Single Speaker (No Tags)**: Clean text only, without adding speaker tags
- **Mode 1 - Format Conversion**: Convert existing multi-speaker text to standardized format
- **Mode 2 - Intelligent Parsing**: AI-powered speaker detection and dialogue extraction
  - **Character Extraction**: Extract character names from a text sample using LLM
  - **Customizable Sample Size**: Choose 5-100 sentences for character analysis
  - **Narrator Toggle**: Option to include narrator as a separate speaker
  - **Character Management**: View, remove, and renumber extracted characters
  - **Consistent Mapping**: Character-to-speaker assignments enforced throughout processing
- **Configurable Output**: Choose between "Speaker 1:" or "[1]:" label formats
- **Multiple Speakers**: Support for 1-20 speakers

### Model Options
- **HuggingFace API**: Use powerful cloud-based models via API (requires API token)
- **Local Models**: Download and run ONNX models locally on your server
  - LaMini-Flan-T5-783M (~800MB, best performance)
  - Flan-T5 Base (~500MB, balanced)
  - Flan-T5 Small (~300MB, fastest)
  - Automatic model download and caching on first use
  - Real-time download progress and status indicators

### Two-Stage LLM Processing
- **Stage 1**: Text cleaning and repair using HuggingFace models
- **Stage 2**: Speaker formatting and dialogue structuring
- **Validation**: Automatic output validation with retry logic
- **Real-time Progress**: Live progress bar and chunk-by-chunk updates

### Advanced Features
- **Character Extraction (Intelligent Mode)**: AI-powered extraction of character names from text sample
  - Configurable sample size (5-100 sentences)
  - Optional narrator inclusion as separate speaker
  - Character-to-speaker mapping for consistency
  - Remove and renumber characters as needed
- **Custom Instructions**: Add custom instructions for the LLM to follow during processing
- **Prompt Preview**: View the exact prompts being sent to the LLM for both processing stages
- **Test Mode**: Process a single chunk to preview results before full processing

### User Interface
- **Modern Design**: Clean, professional UI with dark/light theme support
- **Three-Column Layout**: Configuration, Output, and Activity Log panels
- **Real-time Updates**: WebSocket-powered live processing updates
- **Activity Log**: Detailed timestamped logs with export functionality
- **Output Management**: Copy to clipboard or download processed text

## Technology Stack

### Frontend
- React + TypeScript
- Tailwind CSS + Shadcn UI
- TanStack Query for data fetching
- WebSocket for real-time updates

### Backend
- Express.js server
- HuggingFace Inference API for cloud models
- Transformers.js for local model execution (ONNX format)
- WebSocket server for live processing
- adm-zip for EPUB parsing

## Configuration

### Environment Variables
- `HUGGINGFACE_API_TOKEN`: Your HuggingFace API token (required)
- `SESSION_SECRET`: Session secret for security

### Default Settings
- **Default Model**: Qwen/Qwen2.5-72B-Instruct
- **Batch Size**: 10 sentences per LLM request
- **Max File Size**: 10MB

## Usage

1. **Upload a File**: Drag and drop a .txt or .epub file
2. **Configure Options**: Select text cleaning options and speaker settings
3. **Extract Characters** (Intelligent Mode only, optional):
   - Set sample size for character analysis
   - Toggle narrator inclusion
   - Click "Extract Characters" to identify speakers
   - Review and manage extracted character mappings
4. **Select Model Source**:
   - Choose between HuggingFace API (cloud) or Local Model (offline)
   - If local, select model (downloads automatically on first use)
   - View real-time download progress and model status
5. **Add Custom Instructions** (optional): Provide additional instructions for the LLM
6. **Preview Prompts** (optional): Click "Load Preview" to see the exact prompts
7. **Test First** (optional): Click "Test One Chunk" to preview processing results
8. **Adjust Settings**: Set batch size and optionally change the LLM model
9. **Start Processing**: Click "Start Processing" to begin full processing
10. **Monitor Progress**: Watch real-time progress and activity logs
11. **Export Results**: Copy or download the processed text

## Project Structure

```
├── client/                 # Frontend application
│   ├── src/
│   │   ├── components/    # UI components
│   │   ├── pages/         # Page components
│   │   └── lib/           # Utilities
├── server/                # Backend application
│   ├── routes.ts          # API routes and WebSocket
│   ├── llm-service.ts     # HuggingFace API integration
│   ├── local-model-service.ts  # Local model execution (transformers.js)
│   └── text-processor.ts  # Text chunking and processing
├── shared/                # Shared TypeScript types
│   └── schema.ts          # Data models and schemas
└── design_guidelines.md   # UI/UX design guidelines
```

## Recent Changes

### Latest Updates (October 2025)
- **Local Model Support**: Download and run models locally on your server
  - Integrated transformers.js for ONNX model execution
  - Three pre-configured local models (Flan-T5 variants)
  - Automatic model download with progress tracking
  - Model status indicators and cache management
  - Works offline once models are downloaded
- **Compact UI**: Reduced spacing throughout application for information-dense layout
  - Smaller padding, headings, and component sizes
  - Better use of vertical space
  - Fixed sidebar scrolling to view all controls
- **Narrator Extraction Logic**: Improved prompt instructions for narrator mode
  - Preserves narrative when narrator is included
  - Removes dialogue attribution tags ("said", "replied")
  - Context-aware prompt branching based on narrator presence
- **Character Extraction**: AI-powered character name extraction for intelligent mode
  - Customizable sample size (5-100 sentences)
  - Narrator toggle option for including narrator as separate speaker
  - Character-to-speaker mapping with strict enforcement during processing
  - Character management (view, remove, renumber)
  - Activity log integration for extraction feedback
- **Single Speaker Mode**: Added "none" mode option for single-speaker text processing without tags
- **Custom Instructions**: Users can now add custom instructions for the LLM to follow
- **Prompt Preview**: New collapsible component to view exact LLM prompts before processing
- **Test Button**: One-chunk test functionality to preview processing results

### Previous Updates
- Implemented complete TTS text preprocessing application
- Added HuggingFace LLM integration for text repair and dialogue parsing
- Created beautiful UI with dark/light theme support
- Implemented WebSocket for real-time processing updates
- Added EPUB file support with proper parsing
- Built comprehensive activity logging system
- Added two-stage processing with validation and retry logic

## Future Enhancements

- Direct TTS engine integration for audio generation
- Batch processing queue for multiple files
- Additional export formats (JSON, SRT, custom formats)
- Enhanced EPUB parsing with OPF spine reading order
