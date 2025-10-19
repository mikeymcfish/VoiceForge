import { Play, Square, Settings, TestTube, CircleHelp } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { useEffect, useState } from "react";

interface ProcessingControlsProps {
  batchSize: number;
  onBatchSizeChange: (size: number) => void;
  modelName: string;
  onModelNameChange: (name: string) => void;  llmCleaningDisabled?: boolean;
  onLlmCleaningDisabledChange?: (val: boolean) => void;
  estimatedTotalCost?: number;
  singlePass?: boolean;
  onSinglePassChange?: (val: boolean) => void;
  extendedExamples?: boolean;
  onExtendedExamplesChange?: (val: boolean) => void;
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
  onModelNameChange,  llmCleaningDisabled = false,
  onLlmCleaningDisabledChange,
  estimatedTotalCost,
  singlePass = false,
  onSinglePassChange,
  extendedExamples = false,
  onExtendedExamplesChange,
  onStart,
  onStop,
  onTest,
  isProcessing,
  canStart,
  isTesting = false,
}: ProcessingControlsProps) {
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  type GoodModel = { id: string; display?: string; inCostPerM?: number; outCostPerM?: number; recommendedChunkSize?: number };
  const [goodModels, setGoodModels] = useState<GoodModel[] | null>(null);

  useEffect(() => {
    let isMounted = true;
    fetch('/api/good-models')
      .then(async (res) => {
        if (!res.ok) throw new Error(`Failed to load models (${res.status})`);
        return res.json();
      })
      .then((data: { models?: Array<string | { id: string; display?: string; inCostPerM?: number; outCostPerM?: number; recommendedChunkSize?: number }> }) => {
        if (!isMounted) return;
        const list = Array.isArray(data?.models)
          ? data.models
              .filter((m): m is string | { id: string; display?: string; inCostPerM?: number; outCostPerM?: number; recommendedChunkSize?: number } => Boolean(m))
              .map((m) => (typeof m === 'string' ? { id: m } : { id: m.id, display: m.display, inCostPerM: m.inCostPerM, outCostPerM: m.outCostPerM, recommendedChunkSize: m.recommendedChunkSize }))
          : [];
        setGoodModels(list.length > 0 ? list as GoodModel[] : []);
      })
      .catch(() => {
        if (!isMounted) return;
        setGoodModels([]);
      });
    return () => { isMounted = false; };
  }, []);

  const normalizeModelId = (value: string) => {
    if (/^meta-llama\/Meta-Llama-3\.1-/i.test(value)) {
      return value.replace(/meta-llama\/Meta-Llama-3\.1-/i, "meta-llama/Llama-3.1-");
    }
    return value;
  };

  return (
    <Card className="p-3">
      <div className="space-y-3">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Label htmlFor="batch-size" className="text-sm font-medium">Batch Size (sentences per LLM request)</Label>
            <Tooltip>
              <TooltipTrigger asChild>
                <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>Number of sentences to process in each LLM request</TooltipContent>
            </Tooltip>
          </div>
          <Input id="batch-size" type="number" min={1} max={50} value={batchSize} onChange={(e) => onBatchSizeChange(parseInt(e.target.value) || 10)} disabled={isProcessing} data-testid="input-batch-size" className="h-9" />
        </div>

        <div className="flex items-center justify-between py-1">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <Label className="text-sm font-medium">Singleâ€‘Pass Processing</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>Clean + speaker formatting in one LLM call per chunk</TooltipContent>
              </Tooltip>
            </div>
          </div>
          <Switch checked={singlePass} onCheckedChange={(val) => onSinglePassChange?.(val === true)} disabled={isProcessing} data-testid="switch-single-pass" />
        </div>

        <div className="flex items-center justify-between py-1">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <Label className="text-sm font-medium">Bypass LLM Cleaning</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>Skip LLM cleaning and use only local deterministic cleaning</TooltipContent>
              </Tooltip>
            </div>
          </div>
          <Switch checked={llmCleaningDisabled} onCheckedChange={(val) => onLlmCleaningDisabledChange?.(val === true)} disabled={isProcessing} data-testid="switch-bypass-llm-cleaning" />
        </div>

        <div className="flex items-center justify-between py-1">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <Label className="text-sm font-medium">Extended Prompt Examples</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>Add concrete examples to guide narrator speech handling</TooltipContent>
              </Tooltip>
            </div>
          </div>
          <Switch checked={extendedExamples} onCheckedChange={(val) => onExtendedExamplesChange?.(val === true)} disabled={isProcessing} data-testid="switch-extended-examples" />
        </div>

        <Collapsible open={isAdvancedOpen} onOpenChange={setIsAdvancedOpen}>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" className="w-full justify-start gap-2 px-0 h-auto" data-testid="button-advanced-settings">
              <Settings className="h-4 w-4" />
              <span className="text-sm">Advanced Settings</span>
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-2 pt-2">
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <Label htmlFor="model-name" className="text-sm font-medium">LLM Model</Label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>HuggingFace model ID for text processing</TooltipContent>
                </Tooltip>
              </div>
              <Input id="model-name" value={modelName} onChange={(e) => onModelNameChange(normalizeModelId(e.target.value))} disabled={isProcessing} data-testid="input-model-name" className="h-9" placeholder="e.g., meta-llama/Llama-3.1-8B-Instruct" />
              <div className="grid grid-cols-1 gap-2">
                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground">Quick Pick</Label>
                  <Select value="" onValueChange={(val) => {
                    const normalized = normalizeModelId(val);
                    onModelNameChange(normalized);
                    const gm = (goodModels || []).find(m => m.id === val);
                    if (gm && typeof gm.recommendedChunkSize === 'number') {
                      onBatchSizeChange(gm.recommendedChunkSize);
                    }
                  }} disabled={isProcessing}>
                    <SelectTrigger className="h-9">
                      <SelectValue placeholder="Choose a known-good model" />
                    </SelectTrigger>
                    <SelectContent>
                      {(goodModels && goodModels.length > 0 ? goodModels : [
                        { id: 'meta-llama/Llama-3.1-8B-Instruct' },
                        { id: 'mistralai/Mistral-7B-Instruct-v0.2' },
                        { id: 'Qwen/Qwen2.5-7B-Instruct' },
                      ]).map((m) => (
                        <SelectItem key={m.id} value={m.id}>{m.display || m.id}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>

        <div className="space-y-2 pt-1">
          <div className="flex gap-2">
            {isProcessing ? (
              <Button onClick={onStop} variant="destructive" className="flex-1" data-testid="button-stop-processing">
                <Square className="h-4 w-4 mr-2" />
                Stop Processing
              </Button>
            ) : (
              <Button onClick={onStart} disabled={!canStart} className="flex-1" data-testid="button-start-processing">
                <Play className="h-4 w-4 mr-2" />
                {`Start Processing${typeof estimatedTotalCost === 'number' ? ` â€” Est $${estimatedTotalCost.toFixed(4)}` : ''}`}
              </Button>
            )}
          </div>

          <Button onClick={onTest} disabled={!canStart || isProcessing || isTesting} variant="outline" className="w-full" data-testid="button-test-chunk">
            <TestTube className="h-4 w-4 mr-2" />
            {isTesting ? "Testing..." : "Test One Chunk"}
          </Button>
        </div>
      </div>
    </Card>
  );
}

