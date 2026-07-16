import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const url = process.env.VOICEFORGE_MCP_URL || "http://127.0.0.1:5000/mcp";
const client = new Client({ name: "voiceforge-http-smoke", version: "1.0.0" });
const transport = new StreamableHTTPClientTransport(new URL(url));

try {
  await client.connect(transport);
  const tools = await client.listTools();
  const voices = await client.callTool({ name: "voiceforge_list_voices", arguments: {} });
  const recommendation = await client.callTool({
    name: "voiceforge_recommend_model",
    arguments: { character_count: 1_201, target: "agent", has_voice: true },
  });
  const summary = {
    tools: tools.tools.map((tool) => tool.name),
    voiceCount: voices.structuredContent?.voices?.length,
    recommended: recommendation.structuredContent?.recommended,
  };
  if (summary.tools.length !== 6 || summary.voiceCount < 1 || summary.recommended?.model !== "moss-tts-v1.5") {
    throw new Error(`Unexpected VoiceForge MCP response: ${JSON.stringify(summary)}`);
  }
  console.log(JSON.stringify(summary, null, 2));
} finally {
  await client.close();
}

