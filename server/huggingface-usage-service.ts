import type { HuggingFaceUsageStatus } from "@shared/schema";
import { getHuggingFaceApiToken } from "./llm-service";
import { parseZeroGpuQuotaError, type ZeroGpuReport } from "./huggingface-usage-utils";

const FREE_ZERO_GPU_SECONDS = 5 * 60;
const PRO_ZERO_GPU_SECONDS = 40 * 60;
const FREE_INFERENCE_CREDITS_USD = 0.1;
const PRO_INFERENCE_CREDITS_USD = 2;
const CACHE_TTL_MS = 60_000;

type AccountSummary = {
  name?: string;
  plan?: string;
  zeroGpuLimit?: number;
  inferenceCreditLimit?: number;
};

type ZeroGpuQuotaResponse = {
  base: number;
  current: number;
  resetsAt: string | null;
  overquotaUsed?: number;
};

type InferenceUsageResponse = {
  currency: string;
  periods: Array<{
    period: string;
    sessions: Array<{ id: string; requestCount: number; costCents: number }>;
  }>;
};

class HuggingFaceUsageService {
  private accountCache?: { tokenKey: string; expiresAt: number; account: AccountSummary };
  private zeroGpuReport?: ZeroGpuReport;

  public invalidate(): void {
    this.accountCache = undefined;
    this.zeroGpuReport = undefined;
  }

  public noteZeroGpuError(message: string): void {
    const report = parseZeroGpuQuotaError(message);
    if (report) this.zeroGpuReport = report;
  }

  public noteZeroGpuSuccess(): void {
    // A successful Space call consumes quota, but Hugging Face does not return
    // the new balance. Discard any older scheduler snapshot instead of showing
    // it as current.
    this.zeroGpuReport = undefined;
  }

  private async getAccount(token: string): Promise<AccountSummary> {
    const tokenKey = `${token.length}:${token.slice(-8)}`;
    if (
      this.accountCache &&
      this.accountCache.tokenKey === tokenKey &&
      this.accountCache.expiresAt > Date.now()
    ) {
      return this.accountCache.account;
    }

    const response = await fetch("https://huggingface.co/api/whoami-v2", {
      headers: { Authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) {
      throw new Error(
        response.status === 401
          ? "The configured Hugging Face token was rejected."
          : `Hugging Face account lookup failed (${response.status}).`
      );
    }

    const payload = (await response.json()) as {
      name?: string;
      type?: string;
      isPro?: boolean;
      canPay?: boolean;
    };
    const isPro = payload.isPro === true;
    const account: AccountSummary = {
      name: payload.name,
      plan: isPro ? "PRO" : payload.type === "org" ? "Organization" : "Free",
      zeroGpuLimit: isPro ? PRO_ZERO_GPU_SECONDS : FREE_ZERO_GPU_SECONDS,
      inferenceCreditLimit: isPro
        ? PRO_INFERENCE_CREDITS_USD
        : FREE_INFERENCE_CREDITS_USD,
    };
    this.accountCache = {
      tokenKey,
      expiresAt: Date.now() + CACHE_TTL_MS,
      account,
    };
    return account;
  }

  private async getZeroGpuQuota(token: string): Promise<ZeroGpuQuotaResponse> {
    const response = await fetch("https://huggingface.co/api/spaces/zero-gpu/quota", {
      headers: { Authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) {
      throw new Error(
        response.status === 403
          ? "The configured token cannot read ZeroGPU quota; use a standard HF read token rather than an inference-only fine-grained token."
          : `ZeroGPU quota lookup failed (${response.status}).`
      );
    }
    const payload = (await response.json()) as Partial<ZeroGpuQuotaResponse>;
    if (
      typeof payload.base !== "number" ||
      !Number.isFinite(payload.base) ||
      typeof payload.current !== "number" ||
      !Number.isFinite(payload.current)
    ) {
      throw new Error("Hugging Face returned an invalid ZeroGPU quota response.");
    }
    return {
      base: Math.max(0, payload.base),
      current: Math.max(0, payload.current),
      resetsAt: typeof payload.resetsAt === "string" ? payload.resetsAt : null,
      overquotaUsed:
        typeof payload.overquotaUsed === "number" && Number.isFinite(payload.overquotaUsed)
          ? Math.max(0, payload.overquotaUsed)
          : undefined,
    };
  }

  private async getInferenceSpend(token: string): Promise<number> {
    const now = new Date();
    const start = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
    const url = new URL("https://huggingface.co/api/settings/billing/usage-by-inference-session");
    url.searchParams.set("startDate", start.toISOString());
    url.searchParams.set("endDate", now.toISOString());
    const response = await fetch(url, {
      headers: { Authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) {
      throw new Error(
        response.status === 403
          ? "The configured token cannot read billing usage; use a standard HF read token to populate this estimate."
          : `Inference usage lookup failed (${response.status}).`
      );
    }
    const payload = (await response.json()) as Partial<InferenceUsageResponse>;
    if (!Array.isArray(payload.periods)) throw new Error("Hugging Face returned invalid inference usage data.");
    const costCents = payload.periods.reduce(
      (periodTotal, period) =>
        periodTotal +
        (Array.isArray(period.sessions)
          ? period.sessions.reduce(
              (sessionTotal, session) =>
                sessionTotal +
                (typeof session.costCents === "number" && Number.isFinite(session.costCents)
                  ? Math.max(0, session.costCents)
                  : 0),
              0
            )
          : 0),
      0
    );
    return costCents / 100;
  }

  public async getStatus(): Promise<HuggingFaceUsageStatus> {
    const token = getHuggingFaceApiToken();
    const fetchedAt = Date.now();
    if (!token) {
      return {
        tokenConfigured: false,
        fetchedAt,
        zeroGpu: {
          status: "unavailable",
          authoritative: false,
          unit: "seconds",
          message: "Add a Hugging Face token to use and identify ZeroGPU quota.",
        },
        inferenceCredits: {
          status: "unavailable",
          authoritative: false,
          unit: "usd",
          message: "Add a Hugging Face token to identify the account credit tier.",
        },
      };
    }

    const [accountResult, zeroGpuResult, inferenceResult] = await Promise.allSettled([
      this.getAccount(token),
      this.getZeroGpuQuota(token),
      this.getInferenceSpend(token),
    ]);
    const account: AccountSummary =
      accountResult.status === "fulfilled" ? accountResult.value : {};
    const accountError =
      accountResult.status === "rejected"
        ? accountResult.reason instanceof Error
          ? accountResult.reason.message
          : String(accountResult.reason)
        : undefined;

    const quota = zeroGpuResult.status === "fulfilled" ? zeroGpuResult.value : undefined;
    const quotaError =
      zeroGpuResult.status === "rejected"
        ? zeroGpuResult.reason instanceof Error
          ? zeroGpuResult.reason.message
          : String(zeroGpuResult.reason)
        : undefined;
    const fallbackReport = this.zeroGpuReport;
    const hasFallbackReport = Boolean(
      fallbackReport &&
        (!fallbackReport.resetAt || fallbackReport.resetAt > fetchedAt) &&
        fetchedAt - fallbackReport.observedAt < 24 * 60 * 60 * 1000
    );
    const zeroLimit = quota?.base ?? account.zeroGpuLimit;
    const zeroRemaining = quota?.current ?? (hasFallbackReport ? fallbackReport!.remaining : undefined);
    const resetAt = quota?.resetsAt
      ? Date.parse(quota.resetsAt)
      : hasFallbackReport
        ? fallbackReport?.resetAt
        : undefined;

    const inferenceSpend = inferenceResult.status === "fulfilled" ? inferenceResult.value : undefined;
    const inferenceError =
      inferenceResult.status === "rejected"
        ? inferenceResult.reason instanceof Error
          ? inferenceResult.reason.message
          : String(inferenceResult.reason)
        : undefined;
    const inferenceLimit = account.inferenceCreditLimit;
    const estimatedInferenceRemaining =
      inferenceLimit !== undefined && inferenceSpend !== undefined
        ? Math.max(0, inferenceLimit - inferenceSpend)
        : undefined;

    return {
      tokenConfigured: true,
      accountName: account.name,
      plan: account.plan,
      fetchedAt,
      zeroGpu: quota || hasFallbackReport
        ? {
            status: "reported",
            authoritative: true,
            unit: "seconds",
            limit: zeroLimit,
            remaining: zeroRemaining,
            used:
              zeroLimit !== undefined && zeroRemaining !== undefined
                ? Math.max(0, zeroLimit - zeroRemaining)
                : undefined,
            resetAt: Number.isFinite(resetAt) ? resetAt : undefined,
            message: quota
              ? quota.overquotaUsed
                ? `Live Hugging Face balance; ${quota.overquotaUsed.toFixed(0)} paid overage GPU-seconds used.`
                : "Live remaining balance reported by Hugging Face."
              : "Last balance reported by the ZeroGPU scheduler after a quota check.",
          }
        : {
            status: "unavailable",
            authoritative: false,
            unit: "seconds",
            limit: zeroLimit,
            message:
              quotaError ?? accountError ?? "Live ZeroGPU quota is temporarily unavailable.",
          },
      inferenceCredits:
        inferenceSpend !== undefined
          ? {
              status: "estimated",
              authoritative: false,
              unit: "usd",
              limit: inferenceLimit,
              used: inferenceSpend,
              remaining: estimatedInferenceRemaining,
              message:
                "Estimate: included monthly credit minus HF-routed inference spend. Purchased credits, organization billing, and provider keys are not included.",
            }
          : {
              status: "unavailable",
              authoritative: false,
              unit: "usd",
              limit: inferenceLimit,
              message:
                inferenceError ??
                accountError ??
                "Monthly inference spend is temporarily unavailable.",
            },
    };
  }
}

export const huggingFaceUsageService = new HuggingFaceUsageService();
