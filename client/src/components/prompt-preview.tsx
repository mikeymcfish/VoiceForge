import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { CleaningOptions, SpeakerConfig } from "@shared/schema";

interface PromptPreviewProps {
  sampleText: string;
  cleaningOptions: CleaningOptions;
  speakerConfig?: SpeakerConfig;
  customInstructions?: string;
  disabled?: boolean;
  singlePass?: boolean;
  llmCleaningDisabled?: boolean;
  extendedExamples?: boolean;
}

export function PromptPreview({
  sampleText,
  cleaningOptions,
  speakerConfig,
  customInstructions,
  disabled,
  singlePass,
  llmCleaningDisabled,
  extendedExamples,
}: PromptPreviewProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [prompts, setPrompts] = useState<{ stage1: string; stage2?: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleLoadPreview = async () => {
    if (!sampleText) return;

    setIsLoading(true);
    try {
      const response = await fetch("/api/preview-prompts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sampleText: sampleText.slice(0, 500), // Use first 500 chars for preview
          config: {
            cleaningOptions,
            speakerConfig,
            customInstructions: customInstructions || undefined,
            singlePass: singlePass ?? !!(speakerConfig && speakerConfig.mode !== 'none'),
            llmCleaningDisabled: llmCleaningDisabled === true,
            extendedExamples: extendedExamples ?? false,
          },
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to load preview");
      }

      const data = await response.json();
      setPrompts(data);
      setIsOpen(true);
    } catch (error) {
      console.error("Preview error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="p-3">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium">LLM Prompt Preview</h3>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleLoadPreview}
              disabled={disabled || isLoading || !sampleText}
              data-testid="button-load-preview"
            >
              {isLoading ? "Loading..." : "Load Preview"}
            </Button>
            <CollapsibleTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                disabled={!prompts}
                data-testid="button-toggle-preview"
              >
                {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </Button>
            </CollapsibleTrigger>
          </div>
        </div>

        <CollapsibleContent className="mt-3 space-y-3">
          {prompts && (
            <>
              <div className="space-y-1.5">
                <h4 className="text-sm font-medium">Stage 1 Prompt</h4>
                <pre className="text-xs bg-muted p-2.5 rounded-md overflow-x-auto whitespace-pre-wrap">
                  {prompts.stage1}
                </pre>
              </div>

              {prompts.stage2 && (
                <div className="space-y-1.5">
                  <h4 className="text-sm font-medium">Stage 2 Prompt</h4>
                  <pre className="text-xs bg-muted p-2.5 rounded-md overflow-x-auto whitespace-pre-wrap">
                    {prompts.stage2}
                  </pre>
                </div>
              )}
            </>
          )}
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}
