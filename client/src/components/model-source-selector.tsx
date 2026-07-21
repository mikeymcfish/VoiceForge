import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Cloud, HardDrive, CircleHelp } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { ModelSource } from "@shared/schema";
import { isThinkingOllamaModel } from "@shared/model-utils";
import { HuggingFaceTokenSettings } from "./huggingface-token-settings";

interface ModelSourceSelectorProps {
  modelSource: ModelSource;
  ollamaModelName?: string;
  onModelSourceChange: (source: ModelSource) => void;
  onOllamaModelChange: (modelId: string) => void;
  temperature?: number;
  onTemperatureChange?: (t: number) => void;
  disabled?: boolean;
}

export function ModelSourceSelector({
  modelSource,
  ollamaModelName,
  onModelSourceChange,
  onOllamaModelChange,
  temperature,
  onTemperatureChange,
  disabled,
}: ModelSourceSelectorProps) {
  const [installedModels, setInstalledModels] = useState<string[] | null>(null);
  const thinkingEnabled = modelSource === "ollama" && isThinkingOllamaModel(ollamaModelName);

  // Load installed Ollama models when switching to Ollama
  useEffect(() => {
    let active = true;
    if (modelSource !== 'ollama') return;
    setInstalledModels(null);
    fetch('/api/ollama/models')
      .then(async (res) => {
        if (!res.ok) throw new Error(`status ${res.status}`);
        return res.json();
      })
      .then((data: { models?: Array<string | { id: string }> }) => {
        if (!active) return;
        const list = Array.isArray(data?.models)
          ? data.models.map((m) => (typeof m === 'string' ? m : m.id)).filter(Boolean)
          : [];
        setInstalledModels(list);
      })
      .catch(() => {
        if (!active) return;
        setInstalledModels([]);
      });
    return () => { active = false; };
  }, [modelSource]);

  return (
    <Card className="rounded-xl p-3 shadow-none">
      <h3 className="text-sm font-bold mb-3">AI provider</h3>
      
      <div className="space-y-3">
        <RadioGroup
          value={modelSource}
          onValueChange={(value) => onModelSourceChange(value as ModelSource)}
          disabled={disabled}
        >
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="api" id="source-api" data-testid="radio-model-source-api" />
            <Label htmlFor="source-api" className="flex items-center gap-2 cursor-pointer">
              <Cloud className="w-4 h-4" />
              <span className="text-sm">Hugging Face API</span>
              <Badge variant="secondary" className="text-xs">Online</Badge>
            </Label>
          </div>
          
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="ollama" id="source-ollama" data-testid="radio-model-source-ollama" />
            <Label htmlFor="source-ollama" className="flex items-center gap-2 cursor-pointer">
              <HardDrive className="w-4 h-4" />
              <span className="text-sm">Ollama (local)</span>
              <Badge variant="secondary" className="text-xs">Local</Badge>
            </Label>
          </div>
        </RadioGroup>

        {modelSource === 'api' && (
          <div className="pl-6 space-y-3">
            <div className="flex items-center gap-2">
              <Label className="text-sm">Hugging Face API</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>Uses Hugging Face Inference with your API token. Best for large, powerful models.</TooltipContent>
              </Tooltip>
            </div>
            <p className="text-xs text-muted-foreground">
              Paste a Hugging Face API token to authenticate requests made from this server.
            </p>
            <HuggingFaceTokenSettings disabled={disabled} />
            <div className="space-y-1.5 pt-1">
              <div className="flex items-center gap-2">
                <Label htmlFor="api-temperature" className="text-xs text-muted-foreground">Temperature (0–2)</Label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>Controls randomness of the model output</TooltipContent>
                </Tooltip>
              </div>
              <Input
                id="api-temperature"
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={Number.isFinite(temperature as number) ? String(temperature) : "0.3"}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!onTemperatureChange) return;
                  if (Number.isFinite(v)) {
                    onTemperatureChange(Math.min(2, Math.max(0, v)));
                  }
                }}
                disabled={disabled}
                className="h-9 w-40"
              />
            </div>
          </div>
        )}

        

        {modelSource === 'ollama' && (
          <div className="pl-6 space-y-2">
            <div className="flex items-center gap-2">
              <Label htmlFor="ollama-model" className="text-sm">Ollama Model</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>Select or type an Ollama model name (requires Ollama running locally)</TooltipContent>
              </Tooltip>
            </div>
            <Input
              id="ollama-model"
              value={ollamaModelName || ''}
              onChange={(event) => onOllamaModelChange(event.target.value)}
              disabled={disabled}
              placeholder="e.g. qwen2.5:7b"
              data-testid="input-ollama-model"
            />
            {installedModels && installedModels.length > 0 ? (
              <Select value="" onValueChange={onOllamaModelChange} disabled={disabled}>
                <SelectTrigger data-testid="select-ollama-model">
                  <SelectValue placeholder="Choose an installed model" />
                </SelectTrigger>
                <SelectContent>
                  {installedModels.map((model) => <SelectItem key={model} value={model}>{model}</SelectItem>)}
                </SelectContent>
              </Select>
            ) : (
              <p className="text-[11px] leading-4 text-muted-foreground">
                {installedModels === null ? "Checking the local Ollama service…" : "No installed models were detected. Start Ollama, then enter an installed model name."}
              </p>
            )}
            {thinkingEnabled && (
              <p className="text-[11px] leading-4 text-muted-foreground" data-testid="ollama-thinking-status">
                Thinking mode is enabled for this model. Its reasoning stays separate from the edited text.
              </p>
            )}
            <div className="space-y-1.5 pt-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="ollama-temperature" className="text-xs text-muted-foreground">Temperature (0–2)</Label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>Controls randomness of the model output</TooltipContent>
                </Tooltip>
              </div>
              <Input
                id="ollama-temperature"
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={Number.isFinite(temperature as number) ? String(temperature) : "0.3"}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!onTemperatureChange) return;
                  if (Number.isFinite(v)) {
                    onTemperatureChange(Math.min(2, Math.max(0, v)));
                  }
                }}
                disabled={disabled}
                className="h-9 w-40"
              />
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}



