import { Copy, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";

interface OutputDisplayProps {
  text: string;
  fileName?: string;
}

export function OutputDisplay({ text, fileName }: OutputDisplayProps) {
  const { toast } = useToast();

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      toast({
        title: "Copied to clipboard",
        description: "Text has been copied successfully",
      });
    } catch (error) {
      toast({
        title: "Copy failed",
        description: "Failed to copy text to clipboard",
        variant: "destructive",
      });
    }
  };

  const handleDownload = () => {
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName
      ? `processed_${fileName.replace(/\.[^/.]+$/, "")}.txt`
      : "processed_text.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Download started",
      description: "File is being downloaded",
    });
  };

  return (
    <Card className="flex flex-col h-full">
      <div className="flex items-center justify-between gap-2 p-4 border-b">
        <h3 className="text-base font-medium">Output</h3>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            disabled={!text}
            data-testid="button-copy-output"
          >
            <Copy className="h-4 w-4 mr-2" />
            Copy
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDownload}
            disabled={!text}
            data-testid="button-download-output"
          >
            <Download className="h-4 w-4 mr-2" />
            Save
          </Button>
        </div>
      </div>
      <div className="flex-1 p-4 overflow-hidden">
        <Textarea
          value={text}
          readOnly
          placeholder="Processed text will appear here..."
          className="h-full resize-none font-mono text-sm border-0 focus-visible:ring-0"
          data-testid="textarea-output"
        />
      </div>
    </Card>
  );
}
