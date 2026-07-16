export type LocalRequestPolicyResult = "allowed" | "invalid-host" | "invalid-origin";

export function isLoopbackBindHost(host: string): boolean {
  return ["127.0.0.1", "localhost", "::1"].includes(host.trim().toLowerCase());
}

export function evaluateLocalRequest(
  bindHost: string,
  port: number,
  requestHost: string | undefined,
  origin: string | undefined
): LocalRequestPolicyResult {
  if (!isLoopbackBindHost(bindHost)) return "allowed";
  const authorities = new Set([`127.0.0.1:${port}`, `localhost:${port}`, `[::1]:${port}`]);
  if (port === 80) {
    authorities.add("127.0.0.1");
    authorities.add("localhost");
    authorities.add("[::1]");
  }
  if (!requestHost || !authorities.has(requestHost.toLowerCase())) return "invalid-host";
  if (origin && !new Set([...authorities].map((value) => `http://${value}`)).has(origin.toLowerCase())) {
    return "invalid-origin";
  }
  return "allowed";
}
