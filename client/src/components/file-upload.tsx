import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, File, X } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
  fileStats?: {
    wordCount: number;
    charCount: number;
  };
  isProcessing?: boolean;
}

export function FileUpload({
  onFileSelect,
  selectedFile,
  onClear,
  fileStats,
  isProcessing,
}: FileUploadProps) {
  const { toast } = useToast();
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles[0]) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/plain": [".txt"],
      "application/epub+zip": [".epub"],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
    onDropRejected: () => {
      toast({
        title: "File not accepted",
        description: "Choose one TXT or EPUB file no larger than 10 MB.",
        variant: "destructive",
      });
    },
    disabled: isProcessing,
  });

  if (selectedFile) {
    return (
      <Card className="rounded-xl border-primary/15 bg-primary/[0.035] p-4 shadow-none">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3 flex-1 min-w-0">
            <div className="p-2 rounded-md bg-primary/10 text-primary">
              <File className="h-5 w-5" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate" data-testid="text-filename">
                {selectedFile.name}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
              {fileStats && (
                <div className="flex gap-4 mt-2 text-xs text-muted-foreground">
                  <span data-testid="text-wordcount">
                    {fileStats.wordCount.toLocaleString()} words
                  </span>
                  <span data-testid="text-charcount">
                    {fileStats.charCount.toLocaleString()} characters
                  </span>
                </div>
              )}
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClear}
            disabled={isProcessing}
            data-testid="button-clear-file"
            aria-label="Clear file"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </Card>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={`
        border border-dashed rounded-xl p-6 text-center cursor-pointer
        transition-all
        ${isDragActive ? "border-primary bg-primary/5" : "border-border bg-card"}
        hover:border-primary/50 hover:bg-primary/[0.025]
      `}
      data-testid="dropzone-upload"
    >
      <input {...getInputProps()} data-testid="input-file" />
      <div className="flex flex-col items-center gap-2">
        <div className="p-3 rounded-full bg-primary/10 text-primary">
          <Upload className="h-6 w-6" />
        </div>
        <div>
          <p className="text-sm font-medium">
            {isDragActive ? "Drop file here" : "Upload text or EPUB file"}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Drag & drop or click to browse
          </p>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Supports .txt and .epub files
        </p>
      </div>
    </div>
  );
}
