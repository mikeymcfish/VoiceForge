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
    <Card className="p-3">
      <h3 className="text-sm font-medium mb-3">Model Source</h3>
      
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
              <span className="text-sm">HuggingFace API</span>
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
              <Label className="text-sm">HuggingFace API</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>Uses HuggingFace Inference API with your API token. Best for large, powerful models.</TooltipContent>
              </Tooltip>
            </div>
            <p className="text-xs text-muted-foreground">
              Paste a HuggingFace API token to authenticate requests made from this server.
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
            <Select
              value={ollamaModelName || ''}
              onValueChange={onOllamaModelChange}
              disabled={disabled}
            >
              <SelectTrigger id="ollama-model" data-testid="select-ollama-model">
                <SelectValue placeholder="e.g., qwen2.5:7b or llama3.1:8b" />
              </SelectTrigger>
              <SelectContent>
                {((installedModels && installedModels.length > 0
                  ? installedModels
                  : (["qwen2.5:7b", "qwen2.5:14b", "llama3.1:8b", "mistral:7b"])) as string[]
                ).map((m: string) => (
                  <SelectItem key={m} value={m}>{m}</SelectItem>
                ))}
              </SelectContent>
            </Select>
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



