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
  disabled?: boolean;
}

export function ModelSourceSelector({
  modelSource,
  ollamaModelName,
  onModelSourceChange,
  onOllamaModelChange,
  disabled,
}: ModelSourceSelectorProps) {

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
                <SelectItem value="qwen2.5:7b">qwen2.5:7b</SelectItem>
                <SelectItem value="qwen2.5:14b">qwen2.5:14b</SelectItem>
                <SelectItem value="llama3.1:8b">llama3.1:8b</SelectItem>
                <SelectItem value="mistral:7b">mistral:7b</SelectItem>
              </SelectContent>
            </Select>
            <div className="space-y-1.5 pt-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="ollama-model-custom" className="text-sm">Or type a custom model</Label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>Examples: qwen2.5:32b-instruct, codellama:7b</TooltipContent>
                </Tooltip>
              </div>
              <Input
                id="ollama-model-custom"
                value={ollamaModelName || ''}
                onChange={(e) => onOllamaModelChange(e.target.value)}
                placeholder="e.g., qwen2.5:32b-instruct, codellama:7b"
                disabled={disabled}
                className="h-9"
              />
            </div>
            
          </div>
        )}
      </div>
    </Card>
  );
}
