import type { ReactNode } from "react";

export interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className = "",
}: EmptyStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center rounded-xl border border-dashed px-6 py-10 text-center ${className}`}
      style={{
        backgroundColor: "var(--bg-surface)",
        borderColor: "var(--border-strong)",
      }}
    >
      {icon && (
        <div
          className="mb-3 flex h-11 w-11 items-center justify-center rounded-xl"
          style={{
            backgroundColor: "var(--bg-elevated)",
            color: "var(--accent-cyan)",
          }}
        >
          {icon}
        </div>
      )}
      <h3
        className="text-sm font-semibold tracking-tight"
        style={{ color: "var(--text-primary)" }}
      >
        {title}
      </h3>
      {description && (
        <p
          className="mt-1 max-w-md text-xs leading-relaxed"
          style={{ color: "var(--text-secondary)" }}
        >
          {description}
        </p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
