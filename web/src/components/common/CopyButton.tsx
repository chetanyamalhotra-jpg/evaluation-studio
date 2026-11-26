import { Check, Copy } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { copyToClipboard } from "../../utils/export";

export interface CopyButtonProps {
  value: string;
  label?: string;
  copiedLabel?: string;
  size?: "sm" | "md";
  variant?: "ghost" | "outline";
  className?: string;
  onCopyError?: (message: string) => void;
}

export function CopyButton({
  value,
  label = "Copy",
  copiedLabel = "Copied",
  size = "sm",
  variant = "outline",
  className = "",
  onCopyError,
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, []);

  const handleCopy = async () => {
    const ok = await copyToClipboard(value);
    if (!ok) {
      onCopyError?.("Clipboard access denied.");
      return;
    }
    setCopied(true);
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
    }
    timerRef.current = window.setTimeout(() => {
      setCopied(false);
      timerRef.current = null;
    }, 1800);
  };

  const sizeCls =
    size === "sm"
      ? "px-2.5 py-1 text-ui-sm gap-1.5"
      : "px-3 py-1.5 text-ui-sm gap-1.5";

  return (
    <button
      type="button"
      onClick={() => {
        void handleCopy();
      }}
      className={`telemetry-transition inline-flex cursor-pointer items-center rounded-md font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-focus)] ${
        variant === "outline" ? "border" : ""
      } ${sizeCls} ${className}`}
      style={{
        backgroundColor: copied
          ? "var(--accent-emerald-soft)"
          : variant === "outline"
            ? "var(--bg-elevated)"
            : "transparent",
        borderColor: copied
          ? "rgba(16, 185, 129, 0.4)"
          : "var(--border-subtle)",
        color: copied ? "var(--accent-emerald)" : "var(--text-secondary)",
      }}
      aria-label={copied ? copiedLabel : label}
    >
      {copied ? (
        <Check className="h-3.5 w-3.5 shrink-0" />
      ) : (
        <Copy className="h-3.5 w-3.5 shrink-0" />
      )}
      <span>{copied ? copiedLabel : label}</span>
    </button>
  );
}
