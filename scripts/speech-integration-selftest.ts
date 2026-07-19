import assert from "node:assert/strict";
import {
  huggingFaceTokenUpdateSchema,
  huggingFaceUsageStatusSchema,
  speechStatusSchema,
} from "../shared/schema";
import { parseZeroGpuQuotaError } from "../server/huggingface-usage-utils";
import {
  MOSS_DELAY_MODEL_ID,
  MOSS_DURATION_TOKENS_PLACEHOLDER,
  MOSS_LOCAL_CHECKPOINTS,
  MOSS_LOCAL_MODEL_ID,
  mossHostedDurationTokens,
} from "../shared/moss-tts";

assert.equal(MOSS_DURATION_TOKENS_PLACEHOLDER, 1);
assert.equal(mossHostedDurationTokens(false, 400), 1);
assert.equal(mossHostedDurationTokens(true, 400), 400);
assert.equal(MOSS_LOCAL_CHECKPOINTS[0]?.id, MOSS_DELAY_MODEL_ID);
assert.equal(MOSS_LOCAL_CHECKPOINTS[1]?.id, MOSS_LOCAL_MODEL_ID);
assert.notEqual(MOSS_DELAY_MODEL_ID, MOSS_LOCAL_MODEL_ID);
const localCheckpointCopy = [
  MOSS_LOCAL_CHECKPOINTS[1]?.label,
  MOSS_LOCAL_CHECKPOINTS[1]?.description,
].join(" ");
assert.match(localCheckpointCopy, /lower VRAM/i);
assert.doesNotMatch(localCheckpointCopy, /\bfaster\b/i);

const parsedQuota = parseZeroGpuQuotaError(
  "You have exceeded your Pro ZeroGPU quota (180s requested vs. 42s left). Try again in 1:02:03."
);
assert.equal(parsedQuota?.remaining, 42);
assert.ok(parsedQuota?.resetAt && parsedQuota.resetAt > Date.now());
assert.equal(parseZeroGpuQuotaError("ordinary network error"), undefined);

assert.equal(
  huggingFaceTokenUpdateSchema.safeParse({ token: `hf_${"A".repeat(32)}` }).success,
  true
);
assert.equal(
  huggingFaceTokenUpdateSchema.safeParse({ token: `hf_${"A".repeat(32)}\nSESSION_SECRET=injected` }).success,
  false
);
assert.equal(huggingFaceTokenUpdateSchema.safeParse({ token: "not-a-token" }).success, false);
assert.equal(huggingFaceTokenUpdateSchema.safeParse({ token: "" }).success, true);

speechStatusSchema.parse({
  tokenConfigured: true,
  audioProcessing: {
    ffmpegAvailable: true,
    audioSrAvailable: false,
  },
  engines: [
    {
      engine: "qwen",
      setupStatus: "completed",
      runtimeConfigured: true,
      modelsReady: true,
      availableModels: ["Qwen/Qwen3-TTS-12Hz-0.6B-Base"],
      modelsPath: "models",
      spaceId: "Qwen/Qwen3-TTS",
      hostedAvailable: true,
      localModes: ["clone"],
      hostedModes: ["clone", "design", "custom"],
    },
    {
      engine: "moss",
      setupStatus: "completed",
      runtimeConfigured: true,
      modelsReady: true,
      availableModels: [MOSS_DELAY_MODEL_ID, MOSS_LOCAL_MODEL_ID],
      modelsPath: "models",
      spaceId: "OpenMOSS-Team/MOSS-TTS-v1.5",
      hostedAvailable: true,
      localModes: ["direct", "clone"],
      hostedModes: ["direct", "clone", "continuation", "continuation-clone"],
    },
  ],
  jobs: [],
});

huggingFaceUsageStatusSchema.parse({
  tokenConfigured: true,
  accountName: "example",
  plan: "PRO",
  fetchedAt: Date.now(),
  zeroGpu: {
    status: "reported",
    authoritative: true,
    unit: "seconds",
    limit: 2400,
    used: 600,
    remaining: 1800,
    message: "Live from Hugging Face",
  },
  inferenceCredits: {
    status: "estimated",
    authoritative: false,
    unit: "usd",
    limit: 2,
    used: 0.4,
    remaining: 1.6,
    message: "Estimated included credit",
  },
});

console.log("Speech integration self-test passed.");
