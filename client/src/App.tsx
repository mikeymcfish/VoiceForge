import { Switch, Route, Link, useLocation } from "wouter";
import { lazy, Suspense } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import {
  AudioLines,
  FileInput,
  ScanText,
  ShieldCheck,
  Waves,
} from "lucide-react";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import { HuggingFaceUsageStrip } from "@/components/huggingface-usage-strip";
const Home = lazy(() => import("@/pages/home"));
const PdfOcrPage = lazy(() => import("@/pages/pdf-ocr"));
const TtsPage = lazy(() => import("@/pages/tts"));
const NotFound = lazy(() => import("@/pages/not-found"));

const navigation = [
  {
    href: "/",
    label: "Prepare",
    description: "Clean & structure",
    icon: FileInput,
  },
  {
    href: "/pdf-ocr",
    label: "Import PDF",
    description: "Extract with OCR",
    icon: ScanText,
  },
  {
    href: "/tts",
    label: "Create audio",
    description: "Cast & synthesize",
    icon: AudioLines,
  },
];

function Router() {
  return (
    <Suspense fallback={<div className="grid h-full place-items-center text-sm text-muted-foreground">Opening workspace…</div>}>
      <Switch>
        <Route path="/" component={Home} />
        <Route path="/pdf-ocr" component={PdfOcrPage} />
        <Route path="/tts" component={TtsPage} />
        <Route component={NotFound} />
      </Switch>
    </Suspense>
  );
}

function BrandMark() {
  return (
    <div className="relative grid h-10 w-10 place-items-center overflow-hidden rounded-2xl bg-primary text-primary-foreground shadow-[0_10px_28px_-10px_hsl(var(--primary)/0.8)]">
      <Waves className="h-5 w-5" strokeWidth={2.4} />
      <span className="absolute -right-1 -top-1 h-3 w-3 rounded-full bg-[#ffb86b]" />
    </div>
  );
}

function AppShell() {
  const [location] = useLocation();

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <aside className="hidden w-[264px] shrink-0 flex-col border-r border-sidebar-border bg-sidebar px-4 py-5 lg:flex">
        <Link href="/" className="flex items-center gap-3 px-2">
          <BrandMark />
          <div>
            <p className="text-[15px] font-bold tracking-[-0.02em]">VoiceForge</p>
            <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Studio 2.0</p>
          </div>
        </Link>

        <div className="mt-10 px-2">
          <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">Workspace</p>
        </div>
        <nav className="mt-3 space-y-1.5" aria-label="Primary navigation">
          {navigation.map((item) => {
            const Icon = item.icon;
            const isActive = location === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={isActive ? "page" : undefined}
                className={`group flex items-center gap-3 rounded-2xl px-3 py-3 transition-all ${
                  isActive
                    ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-sm"
                    : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground"
                }`}
              >
                <span
                  className={`grid h-9 w-9 shrink-0 place-items-center rounded-xl transition-colors ${
                    isActive ? "bg-primary text-primary-foreground" : "bg-background/70 text-muted-foreground group-hover:text-primary"
                  }`}
                >
                  <Icon className="h-[18px] w-[18px]" />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block text-sm font-semibold leading-tight">{item.label}</span>
                  <span className="mt-0.5 block text-[11px] text-muted-foreground">{item.description}</span>
                </span>
              </Link>
            );
          })}
        </nav>

        <div className="mt-auto rounded-2xl border border-border/70 bg-background/70 p-4 shadow-sm">
          <div className="flex items-center gap-2 text-xs font-semibold">
            <ShieldCheck className="h-4 w-4 text-emerald-500" />
            Local-first workspace
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
            Your source files stay on this machine unless you choose a cloud model.
          </p>
          <div className="mt-4 flex items-center justify-between border-t border-border/70 pt-3">
            <span className="text-[11px] font-medium text-muted-foreground">Appearance</span>
            <ThemeToggle />
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center justify-between border-b bg-background/90 px-4 backdrop-blur lg:hidden">
          <Link href="/" className="flex items-center gap-2.5">
            <BrandMark />
            <span className="font-bold tracking-tight">VoiceForge</span>
          </Link>
          <ThemeToggle />
        </header>

        <div className="border-b bg-card px-3 py-2 lg:hidden">
          <nav className="grid grid-cols-3 gap-1" aria-label="Mobile navigation">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = location === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center justify-center gap-1.5 rounded-xl px-2 py-2 text-xs font-semibold ${
                    isActive ? "bg-primary text-primary-foreground" : "text-muted-foreground"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>

        <HuggingFaceUsageStrip />

        <main className="min-h-0 flex-1 overflow-hidden">
          <Router />
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="light" storageKey="voiceforge-theme">
        <TooltipProvider delayDuration={250}>
          <AppShell />
          <Toaster />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
