import {
  Activity,
  BarChart3,
  FileText,
  FlaskConical,
  KeyRound,
  LayoutDashboard,
  Moon,
  Play,
  ShieldCheck,
  Sun,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { api } from "../../api/client";
import type { HealthResponse } from "../../types/evalStudio";
import { Badge } from "../common";

interface NavItem {
  to: string;
  label: string;
  caption: string;
  icon: typeof LayoutDashboard;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Command Center", caption: "Overview & feed", icon: LayoutDashboard },
  { to: "/compare", label: "Compare Lab", caption: "Charts & stats", icon: BarChart3 },
  { to: "/run", label: "Run Launcher", caption: "Live SSE stream", icon: Play },
  { to: "/papers", label: "BYOP Papers", caption: "2-stage compiler", icon: FileText },
  { to: "/tests", label: "BYOT Tests", caption: "YAML & calibrate", icon: FlaskConical },
  { to: "/receipts", label: "Receipt Vault", caption: "Verify & rescore", icon: ShieldCheck },
  { to: "/settings", label: "BYOK Keys", caption: "Providers & cache", icon: KeyRound },
];

export interface StudioLayoutProps {
  children: ReactNode;
}

export function StudioLayout({ children }: StudioLayoutProps) {
  const location = useLocation();
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    const saved = typeof window !== "undefined" ? window.localStorage.getItem("eval-studio-theme") : null;
    return saved === "light" ? "light" : "dark";
  });
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("eval-studio-theme", theme);
  }, [theme]);

  useEffect(() => {
    const controller = new AbortController();
    api
      .getHealth(controller.signal)
      .then(setHealth)
      .catch(() => setHealth(null));
    return () => controller.abort();
  }, []);

  const currentNav = NAV_ITEMS.find((item) =>
    item.to === "/" ? location.pathname === "/" : location.pathname.startsWith(item.to)
  );

  return (
    <div
      className="flex min-h-screen w-full"
      style={{ backgroundColor: "var(--bg-canvas)", color: "var(--text-primary)" }}
    >
      {/* iOS-inspired frosted glass sidebar */}
      <aside
        className="sticky top-0 flex h-screen w-64 shrink-0 flex-col border-r backdrop-blur-xl"
        style={{
          backgroundColor: "color-mix(in srgb, var(--bg-surface) 88%, transparent)",
          borderColor: "var(--border-subtle)",
        }}
      >
        <div
          className="flex items-center gap-3 border-b px-5 py-4"
          style={{ borderColor: "var(--border-subtle)" }}
        >
          <div
            className="flex h-9 w-9 items-center justify-center rounded-xl shadow-sm"
            style={{
              background: "linear-gradient(135deg, var(--accent-cyan), var(--accent-violet))",
              color: "#090D16",
            }}
          >
            <Activity className="h-5 w-5" strokeWidth={2.4} />
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold tracking-tight">Evaluation Studio</div>
            <div
              className="truncate font-mono text-[11px]"
              style={{ color: "var(--text-secondary)" }}
            >
              {health ? `v${health.version}` : "Local-First Harness"}
            </div>
          </div>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4" aria-label="Primary Navigation">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) =>
                  `group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-all duration-200 ${
                    isActive ? "font-semibold shadow-sm" : "hover:translate-x-0.5"
                  }`
                }
                style={({ isActive }) => ({
                  backgroundColor: isActive ? "var(--accent-cyan-soft)" : "transparent",
                  color: isActive ? "var(--accent-cyan)" : "var(--text-secondary)",
                  border: isActive
                    ? "1px solid color-mix(in srgb, var(--accent-cyan) 35%, transparent)"
                    : "1px solid transparent",
                })}
              >
                <Icon className="h-4 w-4 shrink-0" />
                <div className="min-w-0 flex-1">
                  <div className="truncate leading-tight">{item.label}</div>
                  <div
                    className="truncate text-[11px] font-normal opacity-75"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {item.caption}
                  </div>
                </div>
              </NavLink>
            );
          })}
        </nav>

        <div
          className="space-y-3 border-t p-4"
          style={{ borderColor: "var(--border-subtle)" }}
        >
          <div className="flex items-center justify-between gap-2">
            <Badge variant={health ? "emerald" : "rose"} dot>
              {health ? "Engine Online" : "Offline"}
            </Badge>
            <button
              type="button"
              onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
              className="flex h-8 w-8 items-center justify-center rounded-lg border transition-colors hover:opacity-90"
              style={{
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--bg-elevated)",
                color: "var(--text-primary)",
              }}
              aria-label="Toggle light/dark theme"
              title="Toggle light/dark theme"
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
          <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
            BYOK · BYOT · BYOP · Portable Receipts
          </div>
        </div>
      </aside>

      {/* Main content area with smooth fade-in per route */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header
          className="sticky top-0 z-20 flex items-center justify-between border-b px-6 py-3.5 backdrop-blur-xl"
          style={{
            backgroundColor: "color-mix(in srgb, var(--bg-canvas) 82%, transparent)",
            borderColor: "var(--border-subtle)",
          }}
        >
          <div>
            <h1 className="text-base font-semibold tracking-tight">
              {currentNav?.label ?? "Evaluation Studio"}
            </h1>
            <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
              {currentNav?.caption ?? "Precision Clinical & Scientific LLM Evaluation"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {health && (
              <Badge variant={health.pdf_available ? "cyan" : "neutral"}>
                {health.pdf_available ? "PDF Ready" : "MD / TXT Mode"}
              </Badge>
            )}
          </div>
        </header>

        <main
          key={location.pathname}
          className="mx-auto w-full max-w-7xl flex-1 px-6 py-6 animate-[fadeSlideUp_220ms_ease-out]"
        >
          {children}
        </main>
      </div>
    </div>
  );
}
