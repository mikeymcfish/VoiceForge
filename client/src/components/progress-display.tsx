import { Progress } from "@/components/ui/progress";
import { Card } from "@/components/ui/card";

interface ProgressDisplayProps {
  progress: number;
  currentChunk: number;
  totalChunks: number;
  isProcessing: boolean;
}

export function ProgressDisplay({
  progress,
  currentChunk,
  totalChunks,
  isProcessing,
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
      </div>
    </Card>
  );
}
