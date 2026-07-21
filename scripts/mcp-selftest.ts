import assert from "node:assert/strict";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import type { VoiceForgeModelStatus } from "../shared/speech-recommendation";
import {
  recommendAudioProcessingSpeechModel,
  recommendSpeechModel,
  resolveVoiceForgeMode,
} from "../shared/speech-recommendation";
import { createVoiceForgeMcpServer, type VoiceForgeMcpBackend } from "../server/mcp-server";
import type { VoiceForgeJob } from "../server/voiceforge-orchestrator";

const statuses: VoiceForgeModelStatus[] = [
  {
    id: "index-tts-2", label: "IndexTTS2", repositoryId: "IndexTeam/IndexTTS-2",
    targets: ["local"], localModes: ["clone"], agentModes: [],
    localReady: true, agentReady: false,
  },
  {
    id: "vibevoice-1.5b", label: "VibeVoice 1.5B", repositoryId: "microsoft/VibeVoice-1.5B",
    targets: ["local"], localModes: ["clone"], agentModes: [],
    localReady: true, agentReady: false,
  },
  {
    id: "vibevoice-large", label: "VibeVoice Large", repositoryId: "aoi-ot/VibeVoice-Large",
    targets: ["local"], localModes: ["clone"], agentModes: [],
    localReady: true, agentReady: false,
  },
  {
    id: "qwen3-tts-0.6b", label: "Qwen 0.6B", repositoryId: "Qwen/0.6B",
    targets: ["local", "agent"], localModes: ["clone"], agentModes: ["clone", "preset"],
    agentCharacterLimit: 1_200, localReady: true, agentReady: true,
  },
  {
    id: "qwen3-tts-1.7b", label: "Qwen 1.7B", repositoryId: "Qwen/1.7B",
    targets: ["local", "agent"], localModes: ["clone"], agentModes: ["clone", "design", "preset"],
    agentCharacterLimit: 1_200, localReady: true, agentReady: true,
  },
  {
    id: "moss-tts-v1.5", label: "MOSS", repositoryId: "OpenMOSS/MOSS",
    targets: ["local", "agent"], localModes: ["direct", "clone"],
    agentModes: ["direct", "clone", "continuation", "continuation-clone"],
    agentCharacterLimit: 5_000, localReady: true, agentReady: true,
  },
];

assert.equal(recommendSpeechModel({ characterCount: 1_200, target: "agent", hasVoice: true }, statuses).recommended?.model, "qwen3-tts-1.7b");
assert.equal(recommendSpeechModel({ characterCount: 1_201, target: "agent", hasVoice: true }, statuses).recommended?.model, "moss-tts-v1.5");
assert.equal(recommendSpeechModel({ characterCount: 5_000, target: "agent", hasVoice: true }, statuses).recommended?.model, "moss-tts-v1.5");
assert.equal(recommendSpeechModel({ characterCount: 5_001, target: "agent", hasVoice: true }, statuses).recommended, undefined);
assert.equal(recommendSpeechModel({ characterCount: 20_000, target: "local", hasVoice: true }, statuses).recommended?.model, "vibevoice-large");
assert.equal(
  recommendAudioProcessingSpeechModel(
    { characterCount: 20_000, target: "local", hasVoice: true },
    statuses
  ).recommended?.model,
  "moss-tts-v1.5"
);
assert.equal(recommendSpeechModel({ characterCount: 400, target: "local", speakerCount: 2, hasVoice: true }, statuses).recommended?.model, "vibevoice-large");
assert.equal(recommendSpeechModel({ characterCount: 400, target: "local", hasVoice: false }, statuses).recommended?.model, "moss-tts-v1.5");
assert.equal(recommendSpeechModel({ characterCount: 400, target: "auto", hasVoice: false }, statuses).recommended?.target, "local");
assert.equal(recommendSpeechModel({ characterCount: 400, target: "agent", hasVoice: false, mode: "design", preference: "speed" }, statuses).recommended?.model, "qwen3-tts-1.7b");
assert.equal(recommendSpeechModel({ characterCount: 400, target: "agent", hasVoice: false, mode: "preset", preference: "speed" }, statuses).recommended?.model, "qwen3-tts-0.6b");
assert.equal(recommendSpeechModel({ characterCount: 500_001, target: "local", hasVoice: true }, statuses).recommended?.model, "vibevoice-large");
assert.equal(resolveVoiceForgeMode("qwen3-tts-0.6b", "agent", "auto", false), "preset");
assert.equal(resolveVoiceForgeMode("qwen3-tts-1.7b", "agent", "auto", false), "design");

let capturedGenerateInput: Parameters<VoiceForgeMcpBackend["generate"]>[0] | undefined;
const completedMp3Job = {
  id: "vf-qwen-completed",
  model: "qwen3-tts-0.6b",
  target: "local",
  mode: "clone",
  status: "completed",
  progress: 100,
  createdAt: 1,
  updatedAt: 2,
  outputFormat: "mp3",
  outputMimeType: "audio/mpeg",
  chapterCount: 2,
  referenceEnhancement: "cleanup",
  levelNormalized: true,
  audioResourceUri: "voiceforge://speech/jobs/vf-qwen-completed/audio",
  audioPath: "/api/mcp/speech/jobs/vf-qwen-completed/audio",
} satisfies VoiceForgeJob;
const legacyCompletedWavJob = {
  id: "vf-qwen-legacy",
  model: "qwen3-tts-0.6b",
  target: "local",
  mode: "clone",
  status: "completed",
  progress: 100,
  createdAt: 1,
  updatedAt: 2,
  audioResourceUri: "voiceforge://speech/jobs/vf-qwen-legacy/audio",
  audioPath: "/api/mcp/speech/jobs/vf-qwen-legacy/audio",
} as VoiceForgeJob;
const backend: VoiceForgeMcpBackend = {
  getModelStatuses: () => statuses,
  listVoices: async () => [{
    id: `voice_${"A".repeat(43)}`,
    name: "Sample",
    format: "wav",
    mimeType: "audio/wav",
    sizeBytes: 100,
    hasTranscript: true,
    transcript: "Sample transcript",
  }],
  recommend: (input) => recommendSpeechModel(input, statuses),
  generate: async (input) => {
    capturedGenerateInput = input;
    return {
      job: {
        id: "vf-qwen-testjob",
        model: "qwen3-tts-0.6b",
        target: "local",
        mode: "clone",
        status: "queued",
        progress: 1,
        createdAt: 1,
        updatedAt: 1,
        outputFormat: input.outputFormat ?? "wav",
        outputMimeType: input.outputFormat === "mp3" ? "audio/mpeg" : "audio/wav",
        chapterCount: 0,
        referenceEnhancement: input.referenceEnhancement ?? "none",
        levelNormalized: input.normalizeLevels ?? true,
      },
      selectedByRecommendation: true,
      recommendationReasons: ["test"],
      warnings: [],
    };
  },
  getJob: (id) =>
    id === completedMp3Job.id
      ? completedMp3Job
      : id === legacyCompletedWavJob.id
        ? legacyCompletedWavJob
        : undefined,
  cancelJob: () => undefined,
  readJobAudio: async (id) =>
    id === completedMp3Job.id
      ? Buffer.from("selftest-mp3")
      : id === legacyCompletedWavJob.id
        ? Buffer.from("selftest-wav")
        : undefined,
  readVoiceAudio: async () => undefined,
};

const server = createVoiceForgeMcpServer({ backend, publicBaseUrl: "http://127.0.0.1:5000" });
const client = new Client({ name: "voiceforge-selftest", version: "1.0.0" });
const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);

const tools = await client.listTools();
assert.deepEqual(
  tools.tools.map((tool) => tool.name).sort(),
  [
    "voiceforge_cancel_job",
    "voiceforge_generate_speech",
    "voiceforge_get_job",
    "voiceforge_list_models",
    "voiceforge_list_voices",
    "voiceforge_recommend_model",
  ]
);
const generateTool = tools.tools.find((tool) => tool.name === "voiceforge_generate_speech");
assert.equal(generateTool?.annotations?.readOnlyHint, false);
assert.equal(generateTool?.annotations?.openWorldHint, true);

const recommendation = await client.callTool({
  name: "voiceforge_recommend_model",
  arguments: { character_count: 1_201, target: "agent", has_voice: true },
});
assert.equal(recommendation.isError, undefined);
assert.equal((recommendation.structuredContent as any).recommended.model, "moss-tts-v1.5");

const voices = await client.callTool({ name: "voiceforge_list_voices", arguments: {} });
const publicVoice = (voices.structuredContent as any).voices[0];
assert.equal(publicVoice.display_name, "Sample");
assert.equal("audioPath" in publicVoice, false);

const generated = await client.callTool({
  name: "voiceforge_generate_speech",
  arguments: {
    request_id: "selftest-generation-1",
    text: "[CHAPTER] Opening\nThis is a VoiceForge MCP self-test.",
    model: "qwen3-tts-0.6b",
    target: "local",
    voice_id: publicVoice.voice_id,
    output_format: "mp3",
    use_chapters: true,
    chapter_pause_ms: 275,
    mp3_quality: 1,
    normalize_levels: false,
    reference_enhancement: "audiosr",
    audiosr_model: "basic",
    audiosr_device: "cuda:1",
    audiosr_ddim_steps: 60,
    audiosr_guidance_scale: 4.25,
    audiosr_seed: -7,
  },
});
assert.equal(generated.isError, undefined);
assert.equal((generated.structuredContent as any).job.job_id, "vf-qwen-testjob");
assert.equal((generated.structuredContent as any).job.output_format, "mp3");
assert.equal((generated.structuredContent as any).job.output_mime_type, "audio/mpeg");
assert.equal((generated.structuredContent as any).job.reference_enhancement, "audiosr");
assert.equal((generated.structuredContent as any).job.level_normalized, false);
assert.equal(capturedGenerateInput?.voiceId, publicVoice.voice_id);
assert.equal(capturedGenerateInput?.outputFormat, "mp3");
assert.equal(capturedGenerateInput?.useChapters, true);
assert.equal(capturedGenerateInput?.chapterPauseMs, 275);
assert.equal(capturedGenerateInput?.mp3Quality, 1);
assert.equal(capturedGenerateInput?.normalizeLevels, false);
assert.equal(capturedGenerateInput?.referenceEnhancement, "audiosr");
assert.equal(capturedGenerateInput?.audioSrModel, "basic");
assert.equal(capturedGenerateInput?.audioSrDevice, "cuda:1");
assert.equal(capturedGenerateInput?.audioSrDdimSteps, 60);
assert.equal(capturedGenerateInput?.audioSrGuidanceScale, 4.25);
assert.equal(capturedGenerateInput?.audioSrSeed, -7);

const legacyGenerated = await client.callTool({
  name: "voiceforge_generate_speech",
  arguments: {
    request_id: "selftest-generation-legacy",
    text: "This call intentionally omits every new audio-processing option.",
    model: "qwen3-tts-0.6b",
    target: "local",
    voice_id: publicVoice.voice_id,
  },
});
assert.equal(legacyGenerated.isError, undefined);
assert.equal((legacyGenerated.structuredContent as any).job.output_format, "wav");
assert.equal((legacyGenerated.structuredContent as any).job.output_mime_type, "audio/wav");
assert.equal((legacyGenerated.structuredContent as any).job.chapter_count, 0);
assert.equal((legacyGenerated.structuredContent as any).job.reference_enhancement, "none");
assert.equal((legacyGenerated.structuredContent as any).job.level_normalized, true);
assert.equal(capturedGenerateInput?.outputFormat, undefined);
assert.equal(capturedGenerateInput?.useChapters, undefined);
assert.equal(capturedGenerateInput?.normalizeLevels, undefined);
assert.equal(capturedGenerateInput?.referenceEnhancement, undefined);

const invalidCudaDevice = await client.callTool({
  name: "voiceforge_generate_speech",
  arguments: {
    request_id: "selftest-invalid-cuda-device",
    text: "Reject an unsupported CUDA device index before creating a job.",
    model: "qwen3-tts-0.6b",
    target: "local",
    voice_id: publicVoice.voice_id,
    reference_enhancement: "audiosr",
    audiosr_device: "cuda:9999",
  },
});
assert.equal(invalidCudaDevice.isError, true);
assert.equal(capturedGenerateInput?.requestId, "selftest-generation-legacy");

const completed = await client.callTool({
  name: "voiceforge_get_job",
  arguments: { job_id: completedMp3Job.id },
});
assert.equal(completed.isError, undefined);
const completedPublicJob = (completed.structuredContent as any).job;
assert.equal(completedPublicJob.output_format, "mp3");
assert.equal(completedPublicJob.output_mime_type, "audio/mpeg");
assert.equal(completedPublicJob.chapter_count, 2);
assert.equal(completedPublicJob.reference_enhancement, "cleanup");
assert.equal(completedPublicJob.level_normalized, true);
const audioLink = (completed.content as any[]).find((content) => content.type === "resource_link");
assert.equal(audioLink.name, "qwen3-tts-0.6b-vf-qwen-completed.mp3");
assert.equal(audioLink.mimeType, "audio/mpeg");

const audioResource = await client.readResource({
  uri: completedMp3Job.audioResourceUri,
});
assert.equal(audioResource.contents[0]?.mimeType, "audio/mpeg");
assert.equal(audioResource.contents[0]?.blob, Buffer.from("selftest-mp3").toString("base64"));

const legacyCompleted = await client.callTool({
  name: "voiceforge_get_job",
  arguments: { job_id: legacyCompletedWavJob.id },
});
assert.equal(legacyCompleted.isError, undefined);
const legacyPublicJob = (legacyCompleted.structuredContent as any).job;
assert.equal(legacyPublicJob.output_format, "wav");
assert.equal(legacyPublicJob.output_mime_type, "audio/wav");
assert.equal(legacyPublicJob.chapter_count, 0);
assert.equal(legacyPublicJob.reference_enhancement, "none");
assert.equal(legacyPublicJob.level_normalized, false);
const legacyAudioResource = await client.readResource({
  uri: legacyCompletedWavJob.audioResourceUri!,
});
assert.equal(legacyAudioResource.contents[0]?.mimeType, "audio/wav");
assert.equal(
  legacyAudioResource.contents[0]?.blob,
  Buffer.from("selftest-wav").toString("base64")
);

const missingTarget = await client.callTool({
  name: "voiceforge_generate_speech",
  arguments: {
    request_id: "selftest-generation-2",
    text: "Target selection must be explicit.",
    model: "auto",
    voice_id: publicVoice.voice_id,
  },
});
assert.equal(missingTarget.isError, true);

await client.close();
await server.close();
console.log("VoiceForge MCP self-test passed.");
