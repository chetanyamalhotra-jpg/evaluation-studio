import { CopyButton } from "./CopyButton";

export interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
  maxHeight?: string;
  wrap?: boolean;
  copyable?: boolean;
  flush?: boolean;
  className?: string;
}

export function CodeBlock({
  code,
  language,
  title,
  maxHeight = "340px",
  wrap = false,
  copyable = true,
  flush = false,
  className = "",
}: CodeBlockProps) {
  const showHeader = Boolean(title || language || copyable);

  return (
    <div
      className={`overflow-hidden ${flush ? "" : "rounded-md border"} ${className}`}
      style={{
        backgroundColor: "var(--bg-code)",
        borderColor: "var(--border-subtle)",
      }}
    >
      {showHeader && (
        <div
          className="flex items-center justify-between gap-2 border-b px-4 py-2"
          style={{
            backgroundColor: "var(--bg-surface)",
            borderColor: "var(--border-subtle)",
          }}
        >
          <div className="flex min-w-0 items-center gap-2.5">
            {title && (
              <span
                className="text-ui-sm truncate font-medium"
                style={{ color: "var(--text-primary)" }}
              >
                {title}
              </span>
            )}
            {language && (
              <span
                className="font-mono-num text-ui-sm"
                style={{ color: "var(--text-muted)" }}
              >
                {language}
              </span>
            )}
          </div>
          {copyable && <CopyButton value={code} size="sm" variant="ghost" />}
        </div>
      )}
      <pre
        className={`font-mono-num text-ui-sm m-0 overflow-auto p-4 leading-relaxed ${
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
