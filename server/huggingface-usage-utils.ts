export type ZeroGpuReport = {
  remaining: number;
  resetAt?: number;
  observedAt: number;
};

export function parseZeroGpuQuotaError(message: string): ZeroGpuReport | undefined {
  const quotaMatch = message.match(
    /(?:\(|\b)(?:\d+(?:\.\d+)?)s\s+requested\s+vs\.?\s+(\d+(?:\.\d+)?)s\s+left/i
  );
  if (!quotaMatch) return undefined;

  const remaining = Number(quotaMatch[1]);
  if (!Number.isFinite(remaining) || remaining < 0) return undefined;

  const retryMatch = message.match(/try\s+again\s+in\s+(?:(\d+):)?(\d+):(\d+)/i);
  let resetAt: number | undefined;
  if (retryMatch) {
    const hours = Number(retryMatch[1] || 0);
    const minutes = Number(retryMatch[2] || 0);
    const seconds = Number(retryMatch[3] || 0);
    const delayMs = ((hours * 60 + minutes) * 60 + seconds) * 1000;
    if (Number.isFinite(delayMs) && delayMs >= 0) resetAt = Date.now() + delayMs;
  }

  return { remaining, resetAt, observedAt: Date.now() };
}
