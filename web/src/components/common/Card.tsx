import type { ReactNode } from "react";

export interface CardProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  footer?: ReactNode;
  elevated?: boolean;
  padding?: "none" | "sm" | "md" | "lg";
  className?: string;
  children: ReactNode;
}

const PADDING_MAP: Record<NonNullable<CardProps["padding"]>, string> = {
  none: "",
  sm: "p-3.5",
  md: "p-5",
  lg: "p-6",
};

export function Card({
  title,
  subtitle,
  actions,
  footer,
  elevated = false,
  padding = "md",
  className = "",
  children,
}: CardProps) {
  const hasHeader = title !== undefined || subtitle !== undefined || actions !== undefined;

  return (
    <section
      className={`telemetry-transition rounded-xl border ${className}`}
      style={{
        backgroundColor: elevated ? "var(--bg-elevated)" : "var(--bg-surface)",
        borderColor: "var(--border-subtle)",
        boxShadow: elevated ? "var(--shadow-elevated)" : "var(--shadow-card)",
      }}
    >
      {hasHeader && (
        <header
          className="flex flex-wrap items-start justify-between gap-3 border-b px-5 py-3.5"
          style={{ borderColor: "var(--border-subtle)" }}
        >
          <div className="min-w-0 flex-1">
            {title !== undefined && (
              <h2
                className="text-sm font-semibold tracking-tight"
                style={{ color: "var(--text-primary)" }}
              >
                {title}
              </h2>
            )}
            {subtitle !== undefined && (
              <p
                className="mt-0.5 text-xs"
                style={{ color: "var(--text-secondary)" }}
              >
                {subtitle}
              </p>
            )}
          </div>
          {actions !== undefined && (
            <div className="flex shrink-0 items-center gap-2">{actions}</div>
          )}
        </header>
      )}
      <div className={PADDING_MAP[padding]}>{children}</div>
      {footer !== undefined && (
        <footer
          className="border-t px-5 py-3 text-xs"
          style={{
            borderColor: "var(--border-subtle)",
            color: "var(--text-secondary)",
          }}
        >
          {footer}
        </footer>
      )}
    </section>
  );
}
