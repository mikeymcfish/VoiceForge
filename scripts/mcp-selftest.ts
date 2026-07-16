import assert from "node:assert/strict";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import type { VoiceForgeModelStatus } from "../shared/speech-recommendation";
import { recommendSpeechModel, resolveVoiceForgeMode } from "../shared/speech-recommendation";
import { createVoiceForgeMcpServer, type VoiceForgeMcpBackend } from "../server/mcp-server";

const statuses: VoiceForgeModelStatus[] = [
  {
    id: "index-tts-2", label: "IndexTTS2", repositoryId: "IndexTeam/IndexTTS-2",
    targets: ["local"], localModes: ["clone"], agentModes: [],
    localCharacterLimit: 500_000, localReady: true, agentReady: false,
  },
  {
    id: "vibevoice-1.5b", label: "VibeVoice 1.5B", repositoryId: "microsoft/VibeVoice-1.5B",
    targets: ["local"], localModes: ["clone"], agentModes: [],
    localCharacterLimit: 500_000, localReady: true, agentReady: false,
  },
  {
    id: "vibevoice-large", label: "VibeVoice Large", repositoryId: "aoi-ot/VibeVoice-Large",
    targets: ["local"], localModes: ["clone"], agentModes: [],
    localCharacterLimit: 500_000, localReady: true, agentReady: false,
  },
  {
    id: "qwen3-tts-0.6b", label: "Qwen 0.6B", repositoryId: "Qwen/0.6B",
    targets: ["local", "agent"], localModes: ["clone"], agentModes: ["clone", "preset"],
    localCharacterLimit: 500_000, agentCharacterLimit: 1_200, localReady: true, agentReady: true,
  },
  {
    id: "qwen3-tts-1.7b", label: "Qwen 1.7B", repositoryId: "Qwen/1.7B",
    targets: ["local", "agent"], localModes: ["clone"], agentModes: ["clone", "design", "preset"],
    localCharacterLimit: 500_000, agentCharacterLimit: 1_200, localReady: true, agentReady: true,
  },
  {
    id: "moss-tts-v1.5", label: "MOSS", repositoryId: "OpenMOSS/MOSS",
    targets: ["local", "agent"], localModes: ["direct", "clone"],
    agentModes: ["direct", "clone", "continuation", "continuation-clone"],
    localCharacterLimit: 500_000, agentCharacterLimit: 5_000, localReady: true, agentReady: true,
  },
];

assert.equal(recommendSpeechModel({ characterCount: 1_200, target: "agent", hasVoice: true }, statuses).recommended?.model, "qwen3-tts-1.7b");
assert.equal(recommendSpeechModel({ characterCount: 1_201, target: "agent", hasVoice: true }, statuses).recommended?.model, "moss-tts-v1.5");
assert.equal(recommendSpeechModel({ characterCount: 5_000, target: "agent", hasVoice: true }, statuses).recommended?.model, "moss-tts-v1.5");
assert.equal(recommendSpeechModel({ characterCount: 5_001, target: "agent", hasVoice: true }, statuses).recommended, undefined);
assert.equal(recommendSpeechModel({ characterCount: 20_000, target: "local", hasVoice: true }, statuses).recommended?.model, "vibevoice-large");
assert.equal(recommendSpeechModel({ characterCount: 400, target: "local", speakerCount: 2, hasVoice: true }, statuses).recommended?.model, "vibevoice-large");
assert.equal(recommendSpeechModel({ characterCount: 400, target: "local", hasVoice: false }, statuses).recommended?.model, "moss-tts-v1.5");
assert.equal(recommendSpeechModel({ characterCount: 400, target: "auto", hasVoice: false }, statuses).recommended?.target, "local");
assert.equal(recommendSpeechModel({ characterCount: 400, target: "agent", hasVoice: false, mode: "design", preference: "speed" }, statuses).recommended?.model, "qwen3-tts-1.7b");
assert.equal(recommendSpeechModel({ characterCount: 400, target: "agent", hasVoice: false, mode: "preset", preference: "speed" }, statuses).recommended?.model, "qwen3-tts-0.6b");
assert.equal(recommendSpeechModel({ characterCount: 500_001, target: "local", hasVoice: true }, statuses).recommended, undefined);
assert.equal(resolveVoiceForgeMode("qwen3-tts-0.6b", "agent", "auto", false), "preset");
assert.equal(resolveVoiceForgeMode("qwen3-tts-1.7b", "agent", "auto", false), "design");

let capturedGenerateInput: Parameters<VoiceForgeMcpBackend["generate"]>[0] | undefined;
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
      },
      selectedByRecommendation: true,
      recommendationReasons: ["test"],
      warnings: [],
    };
  },
  getJob: () => undefined,
  cancelJob: () => undefined,
  readJobAudio: async () => undefined,
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
    text: "This is a VoiceForge MCP self-test.",
    model: "auto",
    target: "local",
    voice_id: publicVoice.voice_id,
  },
});
assert.equal(generated.isError, undefined);
assert.equal((generated.structuredContent as any).job.job_id, "vf-qwen-testjob");
assert.equal(capturedGenerateInput?.voiceId, publicVoice.voice_id);

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
