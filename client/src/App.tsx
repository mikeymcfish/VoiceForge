import { Switch, Route, Link, useLocation } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import Home from "@/pages/home";
import PdfOcrPage from "@/pages/pdf-ocr";
import NotFound from "@/pages/not-found";
import { FileText, ScanText, Waves } from "lucide-react";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/pdf-ocr" component={PdfOcrPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  const [location] = useLocation();
  const navItems = [
    { href: "/", label: "Text Editor", icon: FileText },
    { href: "/pdf-ocr", label: "PDF OCR", icon: ScanText },
  ];

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="dark">
        <TooltipProvider>
          <div className="flex flex-col h-screen">
            <header className="sticky top-0 z-50 h-16 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
              <div className="flex items-center justify-between h-full px-6">
                <div className="flex items-center gap-6">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-md bg-primary/10 text-primary">
                      <Waves className="h-5 w-5" />
                    </div>
                    <h1 className="text-lg font-semibold">VoiceForge Studio</h1>
                  </div>
                  <nav className="flex items-center gap-2">
                    {navItems.map((item) => {
                      const Icon = item.icon;
                      const isActive = location === item.href;
                      return (
                        <Link
                          key={item.href}
                          href={item.href}
                          className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition ${
                            isActive
                              ? "bg-primary text-primary-foreground shadow"
                              : "text-muted-foreground hover:bg-muted hover:text-foreground"
                          }`}
                          aria-current={isActive ? "page" : undefined}
                        >
                          <Icon className="h-4 w-4" />
                          {item.label}
                        </Link>
                      );
                    })}
                  </nav>
                </div>
                <ThemeToggle />
              </div>
            </header>
            <main className="flex-1 overflow-hidden">
              <Router />
            </main>
          </div>
          <Toaster />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
