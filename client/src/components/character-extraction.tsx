import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Loader2, Users, Trash2, CircleHelp } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { clampCharacterSampleSize, getCharacterSampleCeiling, isThinkingOllamaModel } from "@shared/model-utils";

interface CharacterMapping {
  name: string;
  speakerNumber: number;
}

interface CharacterExtractionProps {
  text: string;
  modelSource?: "api" | "ollama";
  modelName: string;
  ollamaModelName?: string;
  characterMapping?: CharacterMapping[];
  sampleSize: number;
  includeNarrator: boolean;
  onSampleSizeChange: (size: number) => void;
  onIncludeNarratorChange: (include: boolean) => void;
  onCharactersExtracted: (characters: CharacterMapping[]) => void;
  onNarratorCharacterNameChange?: (name?: string) => void;
  disabled?: boolean;
}

export function CharacterExtraction({
  text,
  modelSource = "api",
  modelName,
  ollamaModelName,
  characterMapping = [],
  sampleSize,
  includeNarrator,
  onSampleSizeChange,
  onIncludeNarratorChange,
  onCharactersExtracted,
  onNarratorCharacterNameChange,
  disabled,
}: CharacterExtractionProps) {
  const [isExtracting, setIsExtracting] = useState(false);
  const { toast } = useToast();
  const [newCharacterName, setNewCharacterName] = useState("");
  const [sampleInput, setSampleInput] = useState(String(sampleSize));
  const maxSampleSize = getCharacterSampleCeiling(modelSource, ollamaModelName);
  const isThinking = modelSource === "ollama" && isThinkingOllamaModel(ollamaModelName);

  useEffect(() => {
    if (sampleSize > maxSampleSize) {
      onSampleSizeChange(maxSampleSize);
    }
  }, [maxSampleSize, onSampleSizeChange, sampleSize]);

  useEffect(() => {
    setSampleInput(String(sampleSize));
  }, [sampleSize]);

  const handleExtractCharacters = async () => {
    if (!text) {
      toast({
        title: "No text available",
        description: "Please upload a file first.",
        variant: "destructive",
      });
      return;
    }

    setIsExtracting(true);
    try {
      const res = await apiRequest("POST", "/api/extract-characters", {
        text,
        sampleSize,
        includeNarrator,
        modelSource,
        modelName,
        ollamaModelName,
      });

      const response = await res.json() as {
        characters: CharacterMapping[];
        narratorCharacterName?: string;
        sampleSentenceCount: number;
        sampleLimit?: number;
      };

      onCharactersExtracted(response.characters);
      if (onNarratorCharacterNameChange) {
        onNarratorCharacterNameChange(response.narratorCharacterName);
      }
      
      toast({
        title: "Characters Extracted",
        description: `Found ${response.characters.length} character(s) from ${response.sampleSentenceCount} sentence sample.` + (response.narratorCharacterName ? ` Narrator is '${response.narratorCharacterName}'.` : ""),
      });
    } catch (error) {
      console.error("Character extraction error:", error);
      toast({
        title: "Extraction Failed",
        description: error instanceof Error ? error.message : "Failed to extract characters",
        variant: "destructive",
      });
    } finally {
      setIsExtracting(false);
    }
  };

  const handleRemoveCharacter = (speakerNumber: number) => {
    const updated = characterMapping
      .filter((c) => c.speakerNumber !== speakerNumber)
      .map((c, index) => ({
        ...c,
        speakerNumber: index + 1,
      }));
    onCharactersExtracted(updated);
  };

  return (
    <Card className="p-3 space-y-3">
      <div className="flex items-center gap-2">
        <Users className="w-4 h-4" />
        <h4 className="text-sm font-medium">Character Extraction</h4>
      </div>

      <div className="space-y-3">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Label htmlFor="sample-size" className="text-sm">
              Sample Size (sentences)
            </Label>
            <Tooltip>
              <TooltipTrigger asChild>
                <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                Number of sentences to analyze for character names
              </TooltipContent>
            </Tooltip>
          </div>
          <Input
            id="sample-size"
            type="number"
            min={5}
            max={maxSampleSize}
            value={sampleInput}
            onChange={(e) => setSampleInput(e.target.value)}
            onBlur={() => {
              const parsed = parseInt(sampleInput, 10);
              if (Number.isNaN(parsed)) {
                setSampleInput(String(sampleSize));
                return;
              }
              const normalized = clampCharacterSampleSize(parsed, modelSource, ollamaModelName);
              setSampleInput(String(normalized));
              if (normalized !== sampleSize) {
                onSampleSizeChange(normalized);
              }
            }}
            disabled={disabled || isExtracting}
            data-testid="input-sample-size"
            className="h-9"
          />
          <p className="text-xs text-muted-foreground">
            {isThinking
              ? `Thinking Ollama models can scan up to ${maxSampleSize} sentences at once.`
              : `Maximum sample size: ${maxSampleSize} sentences.`}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Checkbox
            id="include-narrator"
            checked={includeNarrator}
            onCheckedChange={(checked) => onIncludeNarratorChange(checked === true)}
            disabled={disabled || isExtracting}
            data-testid="checkbox-include-narrator"
          />
          <Label htmlFor="include-narrator" className="text-sm cursor-pointer">
            Include Narrator as separate speaker
          </Label>
        </div>

        <Button
          onClick={handleExtractCharacters}
          disabled={disabled || isExtracting || !text}
          data-testid="button-extract-characters"
          className="w-full"
        >
          {isExtracting ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Extracting...
            </>
          ) : (
            "Extract Characters"
          )}
        </Button>

        {characterMapping.length > 0 && (
          <div className="space-y-1.5">
            <Label className="text-sm">Extracted Characters</Label>
            <div className="space-y-1.5">
              {characterMapping.map((char) => (
                <div
                  key={char.speakerNumber}
                  className="flex items-center justify-between p-1.5 border rounded-md"
                  data-testid={`character-mapping-${char.speakerNumber}`}
                >
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="font-mono text-xs">
                      Speaker {char.speakerNumber}
                    </Badge>
                    <span className="text-sm">{char.name}</span>
                  </div>
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => handleRemoveCharacter(char.speakerNumber)}
                    disabled={disabled}
                    data-testid={`button-remove-character-${char.speakerNumber}`}
                    className="h-7 w-7"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              These mappings will be used during processing to maintain character consistency
            </p>
          </div>
        )}

        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Label className="text-sm" htmlFor="new-character">Add Character</Label>
            <Tooltip>
              <TooltipTrigger asChild>
                <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                Manually add a known character. New entries are assigned the next speaker number.
              </TooltipContent>
            </Tooltip>
          </div>
          <div className="flex gap-2">
            <Input
              id="new-character"
              value={newCharacterName}
              onChange={(e) => setNewCharacterName(e.target.value)}
              placeholder="Character name"
              disabled={disabled}
              className="h-9"
            />
            <Button
              onClick={() => {
                const name = newCharacterName.trim();
                if (!name) return;
                if (characterMapping.some((c) => c.name.toLowerCase() === name.toLowerCase())) {
                  toast({ title: "Already exists", description: `${name} is already in the mapping.` });
                  return;
                }
                const updated = [
                  ...characterMapping,
                  { name, speakerNumber: characterMapping.length + 1 },
                ];
                onCharactersExtracted(updated);
                setNewCharacterName("");
              }}
              disabled={disabled || !newCharacterName.trim()}
              variant="secondary"
            >
              Add
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
}
