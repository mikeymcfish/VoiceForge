import assert from "node:assert/strict";
import {
  huggingFaceTokenUpdateSchema,
  huggingFaceUsageStatusSchema,
  speechStatusSchema,
} from "../shared/schema";
import { parseZeroGpuQuotaError } from "../server/huggingface-usage-utils";

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
