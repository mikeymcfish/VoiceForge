import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

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
    <Card className="p-4">
      <h3 className="text-base font-medium mb-4">Custom Instructions</h3>
      <div className="space-y-2">
        <Label htmlFor="custom-instructions" className="text-sm font-medium">
          Additional LLM Instructions
        </Label>
        <p className="text-xs text-muted-foreground mb-3">
          Add custom instructions for the language model to follow during processing
        </p>
        <Textarea
          id="custom-instructions"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          placeholder="e.g., Always preserve character names, Remove stage directions, etc."
          data-testid="textarea-custom-instructions"
          className="min-h-24 resize-none"
        />
      </div>
    </Card>
  );
}
