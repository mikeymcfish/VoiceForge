---
name: voiceforge-tts
description: Draft or accept text and generate speech through VoiceForge from any Codex project. Use for TTS, narration, voiceovers, video audio, voice cloning, multi-speaker speech, Qwen3-TTS, MOSS-TTS, IndexTTS, VibeVoice, model recommendations by text length, Local versus Hugging Face Agent execution, audio enhancement, MP3 chapters, job monitoring, and copying completed WAV or MP3 files into project assets. The bundled MCP bridge starts the local VoiceForge app automatically.
---

# VoiceForge TTS

Use the current chat model to draft or revise narration when the user asks for writing. Use VoiceForge only to synthesize the resulting text.

## Workflow

1. Obtain an explicit execution target before generation:
   - `local` keeps inputs on this machine and persists job artifacts under VoiceForge.
   - `agent` uploads text and any reference voice to the model's official Hugging Face ZeroGPU Space and consumes the user's quota.
2. Call `voiceforge_list_voices` when cloning or multi-speaker generation needs reference voices. Use only returned `voice_id` values.
3. Preserve an explicit model choice. Otherwise call `voiceforge_recommend_model` with the actual text or character count, voice availability, speaker count, and target.
4. Call `voiceforge_generate_speech` with a unique stable `request_id`. Reuse it only for the exact same inputs, including every audio-format, chapter, and enhancement option.
5. Poll `voiceforge_get_job` until `completed`, `failed`, or `cancelled`.
6. For a project or video workflow, call `voiceforge_get_audio_path` after completion. Copy the returned WAV or MP3 into the current workspace without changing its extension; never move/delete the VoiceForge source or overwrite an existing asset without approval. Use the `voiceforge://` audio resource when a filesystem path is unsuitable.
7. Cancel only when the user asks.

The bridge discovers or starts VoiceForge automatically. Surface startup, runtime, model-readiness, and quota errors rather than substituting a different model or target.

## Selection guidance

- Agent, up to 1,200 characters: prefer Qwen3-TTS.
- Agent, 1,201-5,000 characters: prefer MOSS-TTS v1.5.
- More than 5,000 characters: use Local; never silently split an Agent request.
- Local long text may be split into chunks and recombined by VoiceForge.
- Two to four mapped speakers: prefer local VibeVoice.
- No reference voice: use MOSS direct generation, or Qwen Agent voice design when requested.
- Continuation or inline pause control: prefer MOSS.
- Treat IndexTTS as an explicit local cloning choice.

## Audio processing

Audio format, output-level normalization, chapters, and reference-audio enhancement are available only for Qwen3-TTS and MOSS-TTS. Do not send these options to IndexTTS or VibeVoice.

- `output_format` defaults to `wav`. Choose `mp3` only when the user requests MP3 or chapters; FFmpeg must be available.
- `normalize_levels` defaults to `true` for Qwen3-TTS and MOSS-TTS. It masters the assembled output to the VoiceForge speech loudness target with FFmpeg. Set it to `false` only when the user wants the model's raw output levels or FFmpeg is unavailable.
- For exact chapter timing, use Local with `output_format: "mp3"` and `use_chapters: true`. Mark each boundary in the narration as `[CHAPTER] Title`. `chapter_pause_ms` adds extra silence before each later chapter's spoken audio and defaults to `0`.
- `mp3_quality` controls non-chaptered MP3 VBR encoding from `0` (best) through `9` (smallest); it defaults to `2`.
- `reference_enhancement` defaults to `none` and requires a selected reference voice. `cleanup` applies gentle FFmpeg cleanup. `audiosr` uses the optional isolated AudioSR executable and is best reserved for short, degraded reference clips.
- AudioSR defaults are `audiosr_model: "speech"`, `audiosr_device: "auto"`, `audiosr_ddim_steps: 50`, `audiosr_guidance_scale: 3.5`, and `audiosr_seed: 42`. Change them only when the user asks or the execution environment requires a specific device.
