import assert from "node:assert/strict";
import { Client } from "@gradio/client";
import {
  endpointParameters,
  exactParameterNames,
  publicGradioParameterNames,
} from "../server/gradio-api-contract";

type EndpointInfo = {
  parameters: Array<{ parameter_name: string }>;
};

type ApiInfo = {
  named_endpoints: Record<string, EndpointInfo>;
};

const contracts = [
  {
    spaceId: "Qwen/Qwen3-TTS",
    url: "https://qwen-qwen3-tts.hf.space/gradio_api/info",
    endpoints: {
      "/generate_voice_design": ["text", "language", "voice_description"],
      "/generate_voice_clone": ["ref_audio", "ref_text", "target_text", "language", "use_xvector_only", "model_size"],
      "/generate_custom_voice": ["text", "language", "speaker", "instruct", "model_size"],
    },
  },
  {
    spaceId: "OpenMOSS-Team/MOSS-TTS-v1.5",
    url: "https://openmoss-team-moss-tts-v1-5.hf.space/gradio_api/info",
    endpoints: {
      "/run_inference": [
        "text",
        "reference_audio",
        "mode_with_reference",
        "duration_control_enabled",
        "duration_tokens",
        "language_tag",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_new_tokens",
      ],
    },
  },
] as const;

for (const contract of contracts) {
  const response = await fetch(contract.url, { signal: AbortSignal.timeout(30_000) });
  assert.equal(response.ok, true, `${contract.url} returned ${response.status}`);
  const info = (await response.json()) as ApiInfo;
  for (const [endpointName, expectedParameters] of Object.entries(contract.endpoints)) {
    const endpoint = info.named_endpoints[endpointName];
    assert.ok(endpoint, `${endpointName} is missing from ${contract.url}`);
    assert.deepEqual(
      endpointParameters(info, endpointName),
      [...expectedParameters],
      `${endpointName} parameters changed`
    );
  }
}

const mossContract = contracts[1];
const mossClient = await Client.connect(mossContract.spaceId, { events: [] });
const originalConsoleError = console.error;
console.error = (...values: unknown[]) => {
  const [first] = values;
  if (first instanceof DOMException && first.name === "AbortError") return;
  originalConsoleError(...values);
};
try {
  const clientApi = await mossClient.view_api();
  assert.deepEqual(
    endpointParameters(clientApi, "/run_inference"),
    [...mossContract.endpoints["/run_inference"]],
    "The Gradio client metadata for MOSS contains an incompatible public contract"
  );
} finally {
  mossClient.close();
  await new Promise((resolve) => setTimeout(resolve, 50));
  console.error = originalConsoleError;
}

const visible = (name: string) => ({ parameter_name: name, component: "textbox", hidden: false });
const safeState = {
  parameter_name: null,
  component: "state",
  hidden: true,
  parameter_has_default: true,
  parameter_default: null,
};

assert.deepEqual(publicGradioParameterNames([visible("before"), safeState, visible("after")]), ["before", "after"]);
assert.throws(
  () => publicGradioParameterNames([visible("before"), { ...safeState, hidden: false }]),
  /not a safe internal Gradio state/
);
assert.throws(
  () => publicGradioParameterNames([visible("before"), { ...safeState, component: "textbox" }]),
  /not a safe internal Gradio state/
);
assert.throws(
  () => publicGradioParameterNames([visible("before"), { ...safeState, parameter_has_default: false }]),
  /not a safe internal Gradio state/
);
assert.throws(
  () => publicGradioParameterNames([visible("before"), { ...safeState, parameter_default: "caller-controlled" }]),
  /not a safe internal Gradio state/
);
const { parameter_name: _parameterName, ...stateWithoutName } = safeState;
assert.throws(
  () => publicGradioParameterNames([visible("before"), stateWithoutName]),
  /not a safe internal Gradio state/
);
assert.throws(
  () => publicGradioParameterNames([visible("before"), { ...safeState, parameter_name: "unexpected" }]),
  /unexpectedly hidden/
);
assert.equal(exactParameterNames(["before", "after"], ["before", "after"]), true);
assert.equal(exactParameterNames(["before|after"], ["before", "after"]), false);
assert.equal(exactParameterNames(["before", "after", "extra"], ["before", "after"]), false);

console.log("Official Hugging Face Space API contracts are compatible.");
