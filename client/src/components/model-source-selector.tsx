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
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Cloud, HardDrive, Download, CheckCircle2 } from "lucide-react";
import type { ModelSource } from "@shared/schema";

interface LocalModel {
  id: string;
  name: string;
  description: string;
  size: string;
  task: string;
}

interface ModelSourceSelectorProps {
  modelSource: ModelSource;
  localModelName?: string;
  onModelSourceChange: (source: ModelSource) => void;
  onLocalModelChange: (modelId: string) => void;
  disabled?: boolean;
}

export function ModelSourceSelector({
  modelSource,
  localModelName,
  onModelSourceChange,
  onLocalModelChange,
  disabled,
}: ModelSourceSelectorProps) {
  const { data: localModelsData } = useQuery<{ models: LocalModel[] }>({
    queryKey: ['/api/local-models'],
    enabled: true,
  });

  const { data: modelStatus } = useQuery<{ loaded: boolean; downloadProgress: number }>({
    queryKey: ['/api/local-model-status', localModelName],
    enabled: !!localModelName && modelSource === 'local',
    refetchInterval: localModelName && modelSource === 'local' ? 2000 : false,
  });

  const localModels = localModelsData?.models || [];
  const isModelLoaded = modelStatus?.loaded || false;
  const downloadProgress = modelStatus?.downloadProgress || 0;

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
            <RadioGroupItem value="local" id="source-local" data-testid="radio-model-source-local" />
            <Label htmlFor="source-local" className="flex items-center gap-2 cursor-pointer">
              <HardDrive className="w-4 h-4" />
              <span className="text-sm">Local Model</span>
              <Badge variant="secondary" className="text-xs">Offline</Badge>
            </Label>
          </div>
        </RadioGroup>

        {modelSource === 'api' && (
          <div className="pl-6 text-xs text-muted-foreground">
            Uses HuggingFace Inference API with your API token. Best for large, powerful models.
          </div>
        )}

        {modelSource === 'local' && (
          <div className="pl-6 space-y-2">
            <Label htmlFor="local-model" className="text-sm">Select Local Model</Label>
            <Select
              value={localModelName || ''}
              onValueChange={onLocalModelChange}
              disabled={disabled}
            >
              <SelectTrigger id="local-model" data-testid="select-local-model">
                <SelectValue placeholder="Choose a model..." />
              </SelectTrigger>
              <SelectContent>
                {localModels.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    <div className="flex items-center gap-2">
                      <span>{model.name}</span>
                      <span className="text-xs text-muted-foreground">({model.size})</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {localModelName && (
              <div className="space-y-1.5">
                {localModels.find(m => m.id === localModelName) && (
                  <p className="text-xs text-muted-foreground">
                    {localModels.find(m => m.id === localModelName)?.description}
                  </p>
                )}
                
                {isModelLoaded ? (
                  <div className="flex items-center gap-2 text-xs text-green-600 dark:text-green-400">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Model loaded and ready</span>
                  </div>
                ) : downloadProgress > 0 && downloadProgress < 100 ? (
                  <div className="flex items-center gap-2 text-xs text-blue-600 dark:text-blue-400">
                    <Download className="w-3.5 h-3.5 animate-pulse" />
                    <span>Downloading: {downloadProgress}%</span>
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    Model will download automatically on first use (~{localModels.find(m => m.id === localModelName)?.size})
                  </p>
                )}
              </div>
            )}

            <p className="text-xs text-muted-foreground pt-1">
              Local models run on your server. First use downloads the model (ONNX format).
            </p>
          </div>
        )}
      </div>
    </Card>
  );
}
