import { useQuery } from "@tanstack/react-query";
import { Cloud, Coins, Gauge, RefreshCw } from "lucide-react";
import type { HuggingFaceUsageMetric, HuggingFaceUsageStatus } from "@shared/schema";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";

function metricPercent(metric: HuggingFaceUsageMetric): number {
  if (metric.limit === undefined || metric.limit <= 0 || metric.remaining === undefined) return 0;
  return Math.max(0, Math.min(100, (metric.remaining / metric.limit) * 100));
}

function formatMetric(metric: HuggingFaceUsageMetric): string {
  if (metric.remaining === undefined) return "Live balance unavailable";
  if (metric.unit === "seconds") {
    const minutes = Math.floor(metric.remaining / 60);
    const seconds = Math.round(metric.remaining % 60);
    return `${minutes}m ${seconds}s remaining`;
  }
  return `$${metric.remaining.toFixed(2)} estimated remaining`;
}

function UsageMetric({
  icon: Icon,
  label,
  metric,
}: {
  icon: typeof Gauge;
  label: string;
  metric: HuggingFaceUsageMetric;
}) {
  const percent = metricPercent(metric);
  return (
    <div className="min-w-0 flex-1" title={metric.message}>
      <div className="mb-1 flex items-center justify-between gap-3 text-[11px]">
        <span className="flex min-w-0 items-center gap-1.5 font-semibold">
          <Icon className="h-3.5 w-3.5 shrink-0 text-primary" />
          <span className="truncate">{label}</span>
        </span>
        <span className="shrink-0 text-muted-foreground">{formatMetric(metric)}</span>
      </div>
      <Progress
        value={percent}
        aria-label={`${label}: ${formatMetric(metric)}`}
        className={metric.status === "unavailable" ? "opacity-40" : undefined}
      />
      <p className="mt-1 truncate text-[10px] text-muted-foreground">
        {metric.status === "reported"
          ? "Live from Hugging Face"
          : metric.status === "estimated"
            ? "Estimate from monthly HF-routed spend"
            : metric.limit !== undefined
              ? `Allowance: ${metric.unit === "seconds" ? `${Math.round(metric.limit / 60)} min` : `$${metric.limit.toFixed(2)}`}`
              : metric.message}
      </p>
    </div>
  );
}

async function fetchUsage(): Promise<HuggingFaceUsageStatus> {
  const response = await fetch("/api/huggingface/usage");
  if (!response.ok) throw new Error("Hugging Face usage is unavailable");
  return response.json();
}

export function HuggingFaceUsageStrip() {
  const { data, isFetching, refetch } = useQuery({
    queryKey: ["huggingface-usage"],
    queryFn: fetchUsage,
    refetchInterval: 60_000,
    refetchOnWindowFocus: true,
    retry: 1,
  });

  const unavailable: HuggingFaceUsageMetric = {
    status: "unavailable",
    authoritative: false,
    unit: "seconds",
    message: "Configure a Hugging Face token in Prepare to load account usage.",
  };
  const creditsUnavailable: HuggingFaceUsageMetric = {
    ...unavailable,
    unit: "usd",
  };

  return (
    <section className="shrink-0 border-b bg-card/80 px-4 py-2 backdrop-blur" aria-label="Hugging Face usage">
      <div className="mx-auto flex max-w-7xl flex-col gap-2 lg:flex-row lg:items-center">
        <div className="flex shrink-0 items-center justify-between gap-3 lg:w-52">
          <div className="flex min-w-0 items-center gap-2">
            <Cloud className="h-4 w-4 shrink-0 text-primary" />
            <div className="min-w-0">
              <p className="truncate text-xs font-bold">Hugging Face</p>
              <p className="truncate text-[10px] text-muted-foreground">
                {data?.tokenConfigured
                  ? `${data.accountName ?? "Connected"}${data.plan ? ` · ${data.plan}` : ""}`
                  : "Token not configured"}
              </p>
            </div>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            onClick={() => void refetch()}
            disabled={isFetching}
            aria-label="Refresh Hugging Face usage"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isFetching ? "animate-spin" : ""}`} />
          </Button>
        </div>
        <div className="grid min-w-0 flex-1 gap-3 sm:grid-cols-2 lg:gap-5">
          <UsageMetric icon={Gauge} label="ZeroGPU remaining" metric={data?.zeroGpu ?? unavailable} />
          <UsageMetric
            icon={Coins}
            label="Included inference credit"
            metric={data?.inferenceCredits ?? creditsUnavailable}
          />
        </div>
        <a
          href="https://huggingface.co/settings/billing"
          target="_blank"
          rel="noreferrer"
          className="shrink-0 text-[10px] font-semibold text-primary hover:underline"
        >
          Open HF billing
        </a>
      </div>
    </section>
  );
}
