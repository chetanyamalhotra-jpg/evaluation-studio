import { CopyButton } from "./CopyButton";

export interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
  maxHeight?: string;
  wrap?: boolean;
  copyable?: boolean;
  className?: string;
}

export function CodeBlock({
  code,
  language,
  title,
  maxHeight = "320px",
  wrap = false,
  copyable = true,
  className = "",
}: CodeBlockProps) {
  const showHeader = Boolean(title || language || copyable);

  return (
    <div
      className={`overflow-hidden rounded-xl border ${className}`}
      style={{
        backgroundColor: "var(--bg-code)",
        borderColor: "var(--border-subtle)",
      }}
    >
      {showHeader && (
        <div
          className="flex items-center justify-between gap-2 border-b px-3.5 py-2"
          style={{
            backgroundColor: "var(--bg-surface)",
            borderColor: "var(--border-subtle)",
          }}
        >
          <div className="flex min-w-0 items-center gap-2">
            {title && (
              <span
                className="truncate text-xs font-medium"
                style={{ color: "var(--text-primary)" }}
              >
                {title}
              </span>
            )}
            {language && (
              <span
                className="font-mono-num rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wider"
                style={{
                  backgroundColor: "var(--bg-elevated)",
                  color: "var(--text-muted)",
                }}
              >
                {language}
              </span>
            )}
          </div>
          {copyable && <CopyButton value={code} size="sm" />}
        </div>
      )}
      <pre
        className={`font-mono-num m-0 overflow-auto p-3.5 text-xs leading-relaxed ${
          wrap ? "whitespace-pre-wrap break-words" : "whitespace-pre"
        }`}
        style={{
          maxHeight,
          color: "var(--text-primary)",
        }}
      >
        <code>{code}</code>
      </pre>
    </div>
  );
}
