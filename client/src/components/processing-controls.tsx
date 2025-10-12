import { Play, Square, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useState } from "react";

interface ProcessingControlsProps {
  batchSize: number;
  onBatchSizeChange: (size: number) => void;
  modelName: string;
  onModelNameChange: (name: string) => void;
  onStart: () => void;
  onStop: () => void;
  isProcessing: boolean;
  canStart: boolean;
}

export function ProcessingControls({
  batchSize,
  onBatchSizeChange,
  modelName,
  onModelNameChange,
  onStart,
  onStop,
  isProcessing,
  canStart,
}: ProcessingControlsProps) {
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

  return (
    <Card className="p-4">
      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="batch-size" className="text-sm font-medium">
            Batch Size (sentences per LLM request)
          </Label>
          <Input
            id="batch-size"
            type="number"
            min={1}
            max={50}
            value={batchSize}
            onChange={(e) => onBatchSizeChange(parseInt(e.target.value) || 10)}
            disabled={isProcessing}
            data-testid="input-batch-size"
            className="h-10"
          />
          <p className="text-xs text-muted-foreground">
            Number of sentences to process in each LLM request
          </p>
        </div>

        <Collapsible open={isAdvancedOpen} onOpenChange={setIsAdvancedOpen}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className="w-full justify-start gap-2 px-0 h-auto"
              data-testid="button-advanced-settings"
            >
              <Settings className="h-4 w-4" />
              <span className="text-sm">Advanced Settings</span>
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-3 pt-3">
            <div className="space-y-2">
              <Label htmlFor="model-name" className="text-sm font-medium">
                LLM Model
              </Label>
              <Input
                id="model-name"
                value={modelName}
                onChange={(e) => onModelNameChange(e.target.value)}
                disabled={isProcessing}
                data-testid="input-model-name"
                className="h-10"
                placeholder="e.g., Qwen/Qwen2.5-72B-Instruct"
              />
              <p className="text-xs text-muted-foreground">
                HuggingFace model ID for text processing
              </p>
            </div>
          </CollapsibleContent>
        </Collapsible>

        <div className="flex gap-2 pt-2">
          {isProcessing ? (
            <Button
              onClick={onStop}
              variant="destructive"
              className="flex-1"
              data-testid="button-stop-processing"
            >
              <Square className="h-4 w-4 mr-2" />
              Stop Processing
            </Button>
          ) : (
            <Button
              onClick={onStart}
              disabled={!canStart}
              className="flex-1"
              data-testid="button-start-processing"
            >
              <Play className="h-4 w-4 mr-2" />
              Start Processing
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}
