import { Moon, Sun } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { api } from "../../api/client";
import type { HealthResponse } from "../../types/evalStudio";

interface NavItem {
  to: string;
  label: string;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Overview" },
  { to: "/compare", label: "Compare" },
  { to: "/run", label: "Run" },
  { to: "/papers", label: "Papers" },
  { to: "/tests", label: "Tests" },
  { to: "/receipts", label: "Receipts" },
  { to: "/settings", label: "Keys" },
];

export interface StudioLayoutProps {
  children: ReactNode;
}

export function StudioLayout({ children }: StudioLayoutProps) {
  const location = useLocation();
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    const saved =
      typeof window !== "undefined" ? window.localStorage.getItem("eval-studio-theme") : null;
    return saved === "light" ? "light" : "dark";
  });
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
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

  return (
    <div
      className="flex min-h-screen w-full flex-col"
      style={{ backgroundColor: "var(--bg-canvas)", color: "var(--text-primary)" }}
    >
      <header
        className="telemetry-glass sticky top-0 z-30 border-b"
        style={{ borderColor: "var(--border-subtle)" }}
      >
        <div className="mx-auto flex h-14 max-w-[1360px] items-center justify-between gap-4 px-4 sm:px-6">
          <div className="flex shrink-0 items-center gap-2.5">
            <svg
              width="22"
              height="22"
              viewBox="0 0 24 24"
              fill="none"
              aria-hidden="true"
              className="shrink-0"
            >
              <rect
                x="2.5"
                y="2.5"
                width="19"
                height="19"
                rx="3"
                stroke="var(--accent-cyan)"
                strokeWidth="1.75"
              />
              <path
                d="M12 6.5V17.5M6.5 12H17.5"
                stroke="var(--accent-cyan)"
                strokeWidth="1.75"
                strokeLinecap="square"
              />
              <circle cx="12" cy="12" r="2.25" fill="var(--accent-cyan)" />
            </svg>
            <NavLink
              to="/"
              className="text-[18px] font-bold tracking-[-0.025em] whitespace-nowrap"
              style={{ color: "var(--text-primary)" }}
            >
              Evaluation Studio
            </NavLink>
            {health && (
              <span
                className="font-mono-num text-ui-sm hidden sm:inline"
                style={{ color: "var(--text-muted)" }}
              >
                v{health.version}
              </span>
            )}
          </div>

          <nav
            className="flex min-w-0 items-center gap-1 overflow-x-auto"
            aria-label="Primary navigation"
          >
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className="telemetry-transition text-ui-base relative flex h-14 shrink-0 items-center px-2.5 whitespace-nowrap sm:px-3"
                style={({ isActive }) => ({
                  color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
                  fontWeight: isActive ? 600 : 500,
                })}
              >
                {({ isActive }) => (
                  <>
                    <span>{item.label}</span>
                    {isActive && (
                      <span
                        className="absolute inset-x-2 bottom-0 h-[2px]"
                        style={{ backgroundColor: "var(--accent-cyan)" }}
                      />
                    )}
                  </>
                )}
              </NavLink>
            ))}
          </nav>

          <div className="flex shrink-0 items-center gap-3">
            <span
              className="font-mono-num text-ui-sm hidden items-center gap-2 md:inline-flex"
              style={{
                color: health ? "var(--accent-emerald)" : "var(--accent-rose)",
              }}
            >
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{
                  backgroundColor: health ? "var(--accent-emerald)" : "var(--accent-rose)",
                }}
              />
              {health
                ? health.pdf_available
                  ? "Online / PDF"
                  : "Online / MD"
                : "Offline"}
            </span>
            <button
              type="button"
              onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
              className="telemetry-transition flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border hover:opacity-85"
              style={{
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--bg-surface)",
                color: "var(--text-primary)",
              }}
              aria-label="Toggle light or dark theme"
              title="Toggle light or dark theme"
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </header>

      <main
        key={location.pathname}
        className="animate-fade-slide-up mx-auto w-full max-w-[1360px] flex-1 px-4 py-7 sm:px-6"
      >
        {children}
      </main>
    </div>
  );
}
