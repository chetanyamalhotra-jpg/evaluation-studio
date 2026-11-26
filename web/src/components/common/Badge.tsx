import type { ReactNode } from "react";

export type BadgeVariant =
  | "neutral"
  | "cyan"
  | "emerald"
  | "amber"
  | "rose";

export interface BadgeProps {
  variant?: BadgeVariant;
  size?: "sm" | "md";
  dot?: boolean;
  mono?: boolean;
  className?: string;
  children: ReactNode;
}

const VARIANT_COLORS: Record<BadgeVariant, { color: string; dot: string }> = {
  neutral: {
    color: "var(--text-secondary)",
    dot: "var(--text-muted)",
  },
  cyan: {
    color: "var(--accent-cyan)",
    dot: "var(--accent-cyan)",
  },
  emerald: {
    color: "var(--accent-emerald)",
    dot: "var(--accent-emerald)",
  },
  amber: {
    color: "var(--accent-amber)",
    dot: "var(--accent-amber)",
  },
  rose: {
    color: "var(--accent-rose)",
    dot: "var(--accent-rose)",
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
  const tokens = VARIANT_COLORS[variant];
  const sizeCls = size === "sm" ? "text-ui-sm" : "text-ui-base";

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-medium tracking-tight ${
        mono ? "font-mono-num" : ""
      } ${sizeCls} ${className}`}
      style={{ color: tokens.color }}
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
