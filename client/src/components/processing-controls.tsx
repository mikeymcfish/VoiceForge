import { Play, Square, Settings, TestTube } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useEffect, useState } from "react";

interface ProcessingControlsProps {
  batchSize: number;
  onBatchSizeChange: (size: number) => void;
  modelName: string;
  onModelNameChange: (name: string) => void;
  singlePass?: boolean;
  onSinglePassChange?: (val: boolean) => void;
  concisePrompts?: boolean;
  onConcisePromptsChange?: (val: boolean) => void;
  onStart: () => void;
  onStop: () => void;
  onTest: () => void;
  isProcessing: boolean;
  canStart: boolean;
  isTesting?: boolean;
}

export function ProcessingControls({
  batchSize,
  onBatchSizeChange,
  modelName,
  onModelNameChange,
  singlePass = false,
  onSinglePassChange,
  concisePrompts = true,
  onConcisePromptsChange,
  onStart,
  onStop,
  onTest,
  isProcessing,
  canStart,
  isTesting = false,
}: ProcessingControlsProps) {
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [goodModels, setGoodModels] = useState<string[] | null>(null);

  useEffect(() => {
    let isMounted = true;
    // Load known-good models from server (good_models.txt)
    fetch('/api/good-models')
      .then(async (res) => {
        if (!res.ok) throw new Error(`Failed to load models (${res.status})`);
        return res.json();
      })
      .then((data: { models?: string[] }) => {
        if (!isMounted) return;
        const list = Array.isArray(data?.models) ? data.models.filter(Boolean) : [];
        setGoodModels(list.length > 0 ? list : []);
      })
      .catch(() => {
        if (!isMounted) return;
        setGoodModels([]); // fall back to defaults below
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const normalizeModelId = (value: string) => {
    // Normalize common Llama 3.1 naming differences
    if (/^meta-llama\/Meta-Llama-3\.1-/i.test(value)) {
      return value.replace(/meta-llama\/Meta-Llama-3\.1-/i, "meta-llama/Llama-3.1-");
    }
    return value;
  };

  return (
    <Card className="p-3">
      <div className="space-y-3">
        <div className="space-y-1.5">
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
            className="h-9"
          />
          <p className="text-xs text-muted-foreground">
            Number of sentences to process in each LLM request
          </p>
        </div>

        <div className="flex items-center justify-between py-1">
          <div className="space-y-0.5">
            <Label className="text-sm font-medium">Single‑Pass Processing</Label>
            <p className="text-xs text-muted-foreground">
              Clean + speaker formatting in one LLM call per chunk
            </p>
          </div>
          <Switch
            checked={singlePass}
            onCheckedChange={(val) => onSinglePassChange?.(val === true)}
            disabled={isProcessing}
            data-testid="switch-single-pass"
          />
        </div>

        <div className="flex items-center justify-between py-1">
          <div className="space-y-0.5">
            <Label className="text-sm font-medium">Concise Prompts</Label>
            <p className="text-xs text-muted-foreground">
              Use shorter instructions to reduce input tokens
            </p>
          </div>
          <Switch
            checked={concisePrompts}
            onCheckedChange={(val) => onConcisePromptsChange?.(val === true)}
            disabled={isProcessing}
            data-testid="switch-concise-prompts"
          />
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
          <CollapsibleContent className="space-y-2 pt-2">
            <div className="space-y-1.5">
              <Label htmlFor="model-name" className="text-sm font-medium">
                LLM Model
              </Label>
          <Input
            id="model-name"
            value={modelName}
            onChange={(e) => onModelNameChange(normalizeModelId(e.target.value))}
            disabled={isProcessing}
            data-testid="input-model-name"
            className="h-9"
            placeholder="e.g., meta-llama/Llama-3.1-8B-Instruct"
          />
          {/^meta-llama\/Meta-Llama-3\.1-/i.test(modelName) && (
            <p className="text-xs text-amber-600 dark:text-amber-400">
              Tip: Use canonical id meta-llama/Llama-3.1-… (auto-normalized)
            </p>
          )}
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground">Quick Pick</Label>
                  <Select
                    value=""
                    onValueChange={(val) => onModelNameChange(normalizeModelId(val))}
                    disabled={isProcessing}
                  >
                    <SelectTrigger className="h-9">
                      <SelectValue placeholder="Choose a known-good model" />
                    </SelectTrigger>
                    <SelectContent>
                      {(goodModels && goodModels.length > 0
                        ? goodModels
                        : [
                            // Fallback defaults if file is missing/empty
                            'meta-llama/Llama-3.1-8B-Instruct',
                            'mistralai/Mistral-7B-Instruct-v0.2',
                            'Qwen/Qwen2.5-7B-Instruct',
                          ]
                      ).map((m) => (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                HuggingFace model ID for text processing
              </p>
            </div>
          </CollapsibleContent>
        </Collapsible>

        <div className="space-y-2 pt-1">
          <div className="flex gap-2">
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
          
          <Button
            onClick={onTest}
            disabled={!canStart || isProcessing || isTesting}
            variant="outline"
            className="w-full"
            data-testid="button-test-chunk"
          >
            <TestTube className="h-4 w-4 mr-2" />
            {isTesting ? "Testing..." : "Test One Chunk"}
          </Button>
        </div>
      </div>
    </Card>
  );
}
