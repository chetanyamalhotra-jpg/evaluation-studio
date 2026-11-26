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
      className={`flex flex-col items-center justify-center px-6 py-10 text-center ${className}`}
    >
      {icon && (
        <div
          className="mb-3 flex h-10 w-10 items-center justify-center"
          style={{ color: "var(--accent-cyan)" }}
        >
          {icon}
        </div>
      )}
      <h3
        className="text-ui-base font-semibold tracking-tight"
        style={{ color: "var(--text-primary)" }}
      >
        {title}
      </h3>
      {description && (
        <p
          className="text-ui-sm mt-1.5 max-w-[54ch]"
          style={{ color: "var(--text-secondary)" }}
        >
          {description}
        </p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
