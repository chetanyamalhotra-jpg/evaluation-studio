import type { ReactNode } from "react";

export type BadgeVariant =
  | "neutral"
  | "cyan"
  | "emerald"
  | "amber"
  | "rose"
  | "violet";

export interface BadgeProps {
  variant?: BadgeVariant;
  size?: "sm" | "md";
  dot?: boolean;
  mono?: boolean;
  className?: string;
  children: ReactNode;
}

const VARIANT_STYLES: Record<BadgeVariant, { bg: string; color: string; border: string; dot: string }> = {
  neutral: {
    bg: "var(--bg-elevated)",
    color: "var(--text-secondary)",
    border: "var(--border-subtle)",
    dot: "var(--text-muted)",
  },
  cyan: {
    bg: "var(--accent-cyan-soft)",
    color: "var(--accent-cyan)",
    border: "rgba(6, 182, 212, 0.28)",
    dot: "var(--accent-cyan)",
  },
  emerald: {
    bg: "var(--accent-emerald-soft)",
    color: "var(--accent-emerald)",
    border: "rgba(16, 185, 129, 0.28)",
    dot: "var(--accent-emerald)",
  },
  amber: {
    bg: "var(--accent-amber-soft)",
    color: "var(--accent-amber)",
    border: "rgba(245, 158, 11, 0.3)",
    dot: "var(--accent-amber)",
  },
  rose: {
    bg: "var(--accent-rose-soft)",
    color: "var(--accent-rose)",
    border: "rgba(244, 63, 94, 0.3)",
    dot: "var(--accent-rose)",
  },
  violet: {
    bg: "var(--accent-violet-soft)",
    color: "var(--accent-violet)",
    border: "rgba(139, 92, 246, 0.3)",
    dot: "var(--accent-violet)",
  },
};

export function Badge({
  variant = "neutral",
  size = "sm",
  dot = false,
  mono = false,
  className = "",
  children,
}: BadgeProps) {
  const tokens = VARIANT_STYLES[variant];
  const sizeCls =
    size === "sm"
      ? "px-2 py-0.5 text-[11px] leading-4"
      : "px-2.5 py-1 text-xs leading-4";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-medium tracking-tight ${
        mono ? "font-mono-num" : ""
      } ${sizeCls} ${className}`}
      style={{
        backgroundColor: tokens.bg,
        color: tokens.color,
        borderColor: tokens.border,
      }}
    >
      {dot && (
        <span
          className="h-1.5 w-1.5 shrink-0 rounded-full"
          style={{ backgroundColor: tokens.dot }}
        />
      )}
      {children}
    </span>
  );
}

export type SemanticStatus =
  | "APPROVED"
  | "REJECTED"
  | "VERIFIED"
  | "UNALIGNED"
  | "UNRESOLVED"
  | "PENDING"
  | "NEEDS_REVIEW"
  | "ALIGNED"
  | (string & {});

export function statusToBadgeVariant(status: string): BadgeVariant {
  switch (status.toUpperCase()) {
    case "APPROVED":
    case "VERIFIED":
    case "ALIGNED":
    case "COMPLETED":
    case "OK":
      return "emerald";
    case "REJECTED":
    case "UNALIGNED":
    case "FAILED":
    case "ERROR":
    case "TAMPERED":
      return "rose";
    case "UNRESOLVED":
    case "NEEDS_REVIEW":
    case "BUDGET_EXCEEDED":
    case "WARNING":
      return "amber";
    case "RUNNING":
    case "ACTIVE":
      return "cyan";
    case "PENDING":
    case "QUEUED":
      return "violet";
    default:
      return "neutral";
  }
}

export interface StatusBadgeProps {
  status: SemanticStatus;
  size?: "sm" | "md";
  dot?: boolean;
  className?: string;
}

export function StatusBadge({
  status,
  size = "sm",
  dot = true,
  className = "",
}: StatusBadgeProps) {
  const normalized = status.toUpperCase();
  return (
    <Badge
      variant={statusToBadgeVariant(normalized)}
      size={size}
      dot={dot}
      mono
      className={className}
    >
      {normalized}
    </Badge>
  );
}
