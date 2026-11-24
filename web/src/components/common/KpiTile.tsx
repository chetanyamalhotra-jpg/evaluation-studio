import type { ReactNode } from "react";

export type AccentTone = "cyan" | "emerald" | "amber" | "rose" | "violet";

export interface KpiTileProps {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  delta?: string;
  accent?: AccentTone;
  icon?: ReactNode;
  loading?: boolean;
  className?: string;
}

const ACCENT_COLOR: Record<AccentTone, { fg: string; bg: string }> = {
  cyan: { fg: "var(--accent-cyan)", bg: "var(--accent-cyan-soft)" },
  emerald: { fg: "var(--accent-emerald)", bg: "var(--accent-emerald-soft)" },
  amber: { fg: "var(--accent-amber)", bg: "var(--accent-amber-soft)" },
  rose: { fg: "var(--accent-rose)", bg: "var(--accent-rose-soft)" },
  violet: { fg: "var(--accent-violet)", bg: "var(--accent-violet-soft)" },
};

export function KpiTile({
  label,
  value,
  hint,
  delta,
  accent = "cyan",
  icon,
  loading = false,
  className = "",
}: KpiTileProps) {
  const tone = ACCENT_COLOR[accent];

  return (
    <div
      className={`telemetry-card telemetry-transition relative overflow-hidden p-4 ${className}`}
    >
      <div
        className="absolute inset-x-0 top-0 h-[2px]"
        style={{ backgroundColor: tone.fg, opacity: 0.75 }}
      />
      <div className="flex items-start justify-between gap-3">
        <span
          className="text-[11px] font-semibold uppercase tracking-wider"
          style={{ color: "var(--text-secondary)" }}
        >
          {label}
        </span>
        {icon && (
          <span
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg"
            style={{ backgroundColor: tone.bg, color: tone.fg }}
          >
            {icon}
          </span>
        )}
      </div>

      <div className="mt-2.5 flex items-baseline gap-2">
        {loading ? (
          <div
            className="h-7 w-24 animate-pulse rounded-md"
            style={{ backgroundColor: "var(--bg-elevated)" }}
          />
        ) : (
          <div
            className="font-mono-num text-2xl font-semibold tracking-tight"
            style={{ color: "var(--text-primary)" }}
          >
            {value}
          </div>
        )}
        {delta && !loading && (
          <span
            className="font-mono-num rounded px-1.5 py-0.5 text-[11px] font-medium"
            style={{ backgroundColor: tone.bg, color: tone.fg }}
          >
            {delta}
          </span>
        )}
      </div>

      {hint && (
        <p
          className="mt-1.5 truncate text-xs"
          style={{ color: "var(--text-muted)" }}
        >
          {hint}
        </p>
      )}
    </div>
  );
}
