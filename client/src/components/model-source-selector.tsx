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
import { Cloud, HardDrive } from "lucide-react";
import type { ModelSource } from "@shared/schema";

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
          <div className="pl-6 text-xs text-muted-foreground">
            Uses HuggingFace Inference API with your API token. Best for large, powerful models.
          </div>
        )}

        

        {modelSource === 'ollama' && (
          <div className="pl-6 space-y-2">
            <Label htmlFor="ollama-model" className="text-sm">Ollama Model</Label>
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
              <Label htmlFor="ollama-model-custom" className="text-sm">Or type a custom model</Label>
              <Input
                id="ollama-model-custom"
                value={ollamaModelName || ''}
                onChange={(e) => onOllamaModelChange(e.target.value)}
                placeholder="e.g., qwen2.5:32b-instruct, codellama:7b"
                disabled={disabled}
                className="h-9"
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Requires Ollama running locally at http://localhost:11434. Change with OLLAMA_BASE_URL.
            </p>
          </div>
        )}
      </div>
    </Card>
  );
}
