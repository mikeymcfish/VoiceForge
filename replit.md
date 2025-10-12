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
- **Configurable Output**: Choose between "Speaker 1:" or "[1]:" label formats
- **Multiple Speakers**: Support for 1-20 speakers

### Two-Stage LLM Processing
- **Stage 1**: Text cleaning and repair using HuggingFace models
- **Stage 2**: Speaker formatting and dialogue structuring
- **Validation**: Automatic output validation with retry logic
- **Real-time Progress**: Live progress bar and chunk-by-chunk updates

### Advanced Features
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
- HuggingFace Inference API
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
3. **Add Custom Instructions** (optional): Provide additional instructions for the LLM
4. **Preview Prompts** (optional): Click "Load Preview" to see the exact prompts
5. **Test First** (optional): Click "Test One Chunk" to preview processing results
6. **Adjust Settings**: Set batch size and optionally change the LLM model
7. **Start Processing**: Click "Start Processing" to begin full processing
8. **Monitor Progress**: Watch real-time progress and activity logs
9. **Export Results**: Copy or download the processed text

## Project Structure

```
├── client/                 # Frontend application
│   ├── src/
│   │   ├── components/    # UI components
│   │   ├── pages/         # Page components
│   │   └── lib/           # Utilities
├── server/                # Backend application
│   ├── routes.ts          # API routes and WebSocket
│   ├── llm-service.ts     # HuggingFace integration
│   └── text-processor.ts  # Text chunking and processing
├── shared/                # Shared TypeScript types
│   └── schema.ts          # Data models and schemas
└── design_guidelines.md   # UI/UX design guidelines
```

## Recent Changes

### Latest Updates (October 2025)
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

- Local model download and execution from HuggingFace Hub
- Direct TTS engine integration for audio generation
- Batch processing queue for multiple files
- Additional export formats (JSON, SRT, custom formats)
- Enhanced EPUB parsing with OPF spine reading order
