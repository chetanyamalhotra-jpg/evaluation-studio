export interface LoadingSkeletonProps {
  variant?: "text" | "card" | "row" | "chart";
  lines?: number;
  className?: string;
}

export function LoadingSkeleton({
  variant = "card",
  lines = 3,
  className = "",
}: LoadingSkeletonProps) {
  if (variant === "chart") {
    return (
      <div
        className={`p-5 ${className}`}
        aria-busy="true"
        aria-label="Loading chart"
      >
        <div
          className="mb-4 h-4 w-40 animate-pulse rounded"
          style={{ backgroundColor: "var(--bg-elevated)" }}
        />
        <div
          className="h-56 w-full animate-pulse rounded"
          style={{ backgroundColor: "var(--bg-elevated)" }}
        />
      </div>
    );
  }

  if (variant === "row") {
    return (
      <div
        className={`divide-y ${className}`}
        style={{ borderColor: "var(--border-subtle)" }}
        aria-busy="true"
        aria-label="Loading rows"
      >
        {Array.from({ length: lines }).map((_, idx) => (
          <div key={idx} className="px-4 py-3">
            <div
              className="h-5 w-full animate-pulse rounded"
              style={{ backgroundColor: "var(--bg-elevated)" }}
            />
          </div>
        ))}
      </div>
    );
  }

  if (variant === "text") {
    return (
      <div
        className={`space-y-2.5 ${className}`}
        aria-busy="true"
        aria-label="Loading text"
      >
        {Array.from({ length: lines }).map((_, idx) => (
          <div
            key={idx}
            className="h-3.5 animate-pulse rounded"
            style={{
              backgroundColor: "var(--bg-elevated)",
              width: idx === lines - 1 ? "65%" : "100%",
            }}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={`space-y-3 p-5 ${className}`}
      aria-busy="true"
      aria-label="Loading content"
    >
      <div
        className="h-4 w-1/3 animate-pulse rounded"
        style={{ backgroundColor: "var(--bg-elevated)" }}
      />
      {Array.from({ length: lines }).map((_, idx) => (
        <div
          key={idx}
          className="h-3.5 animate-pulse rounded"
          style={{
            backgroundColor: "var(--bg-elevated)",
            width: idx === lines - 1 ? "72%" : "100%",
          }}
        />
      ))}
    </div>
  );
}
