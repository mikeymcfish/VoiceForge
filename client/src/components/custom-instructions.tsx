import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { CircleHelp } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface CustomInstructionsProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

export function CustomInstructions({
  value,
  onChange,
  disabled,
}: CustomInstructionsProps) {
  return (
    <Card className="p-3">
      <h3 className="text-sm font-medium mb-3">Custom Instructions</h3>
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <Label htmlFor="custom-instructions" className="text-sm font-medium">
            Additional LLM Instructions
          </Label>
          <Tooltip>
            <TooltipTrigger asChild>
              <CircleHelp className="h-4 w-4 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent>Add custom instructions for the language model to follow during processing</TooltipContent>
          </Tooltip>
        </div>
        <Textarea
          id="custom-instructions"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          placeholder="e.g., Always preserve character names, Remove stage directions, etc."
          data-testid="textarea-custom-instructions"
          className="min-h-20 resize-none text-sm"
        />
      </div>
    </Card>
  );
}
