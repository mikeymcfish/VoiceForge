---
name: voiceforge-tts
description: Draft or accept text and generate speech through VoiceForge from any Codex project. Use for TTS, narration, voiceovers, video audio, voice cloning, multi-speaker speech, Qwen3-TTS, MOSS-TTS, IndexTTS, VibeVoice, model recommendations by text length, Local versus Hugging Face Agent execution, job monitoring, and copying completed WAV files into project assets. The bundled MCP bridge starts the local VoiceForge app automatically.
---

# VoiceForge TTS

Use the current chat model to draft or revise narration when the user asks for writing. Use VoiceForge only to synthesize the resulting text.

## Workflow

1. Obtain an explicit execution target before generation:
   - `local` keeps inputs on this machine and persists job artifacts under VoiceForge.
   - `agent` uploads text and any reference voice to the model's official Hugging Face ZeroGPU Space and consumes the user's quota.
2. Call `voiceforge_list_voices` when cloning or multi-speaker generation needs reference voices. Use only returned `voice_id` values.
3. Preserve an explicit model choice. Otherwise call `voiceforge_recommend_model` with the actual text or character count, voice availability, speaker count, and target.
4. Call `voiceforge_generate_speech` with a unique stable `request_id`. Reuse it only for the exact same inputs.
5. Poll `voiceforge_get_job` until `completed`, `failed`, or `cancelled`.
6. For a project or video workflow, call `voiceforge_get_audio_path` after completion. Copy the returned WAV into the current workspace; never move/delete the VoiceForge source or overwrite an existing asset without approval. Use the `voiceforge://` audio resource when a filesystem path is unsuitable.
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
