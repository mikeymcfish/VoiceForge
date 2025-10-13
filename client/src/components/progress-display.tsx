import { Progress } from "@/components/ui/progress";
import { Card } from "@/components/ui/card";

interface ProgressDisplayProps {
  progress: number;
  currentChunk: number;
  totalChunks: number;
  isProcessing: boolean;
  etaMs?: number;
  lastChunkMs?: number;
  avgChunkMs?: number;
  totalInputTokens?: number;
  totalOutputTokens?: number;
  totalCost?: number;
}

function formatMs(ms?: number): string {
  if (!ms && ms !== 0) return "";
  const s = Math.max(0, Math.round(ms / 1000));
  const m = Math.floor(s / 60);
  const ss = s % 60;
  return `${m}:${ss.toString().padStart(2, '0')}`;
}

export function ProgressDisplay({
  progress,
  currentChunk,
  totalChunks,
  isProcessing,
  etaMs,
  lastChunkMs,
  avgChunkMs,
  totalInputTokens,
  totalOutputTokens,
  totalCost,
}: ProgressDisplayProps) {
  if (!isProcessing && progress === 0) {
    return null;
  }

  return (
    <Card className="p-4">
      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">Processing Progress</span>
          <span className="text-muted-foreground" data-testid="text-progress">
            {progress.toFixed(0)}%
          </span>
        </div>
        <Progress value={progress} className="h-2" data-testid="progress-bar" />
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span data-testid="text-chunk-status">
            Chunk {currentChunk} of {totalChunks}
          </span>
          <span>
            {isProcessing ? "Processing..." : "Complete"}
          </span>
        </div>
        {(etaMs !== undefined || lastChunkMs !== undefined || avgChunkMs !== undefined) && (
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              ETA: {etaMs !== undefined ? formatMs(etaMs) : "—"}
            </span>
            <span>
              Last: {lastChunkMs !== undefined ? formatMs(lastChunkMs) : "—"} · Avg: {avgChunkMs !== undefined ? formatMs(avgChunkMs) : "—"}
            </span>
          </div>
        )}
        {(typeof totalInputTokens === 'number' || typeof totalOutputTokens === 'number' || typeof totalCost === 'number') && (
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              Tokens in/out: {typeof totalInputTokens === 'number' ? totalInputTokens : "-"}/{typeof totalOutputTokens === 'number' ? totalOutputTokens : "-"}
            </span>
            <span>
              Cost: {typeof totalCost === 'number' ? `$${totalCost.toFixed(4)}` : "-"}
            </span>
          </div>
        )}
      </div>
    </Card>
  );
}
