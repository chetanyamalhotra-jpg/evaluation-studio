import { X } from "lucide-react";
import { useEffect, type ReactNode } from "react";

export interface SlideDrawerProps {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  subtitle?: ReactNode;
  headerActions?: ReactNode;
  width?: "md" | "lg" | "xl";
  children: ReactNode;
}

const WIDTH_CLASS: Record<NonNullable<SlideDrawerProps["width"]>, string> = {
  md: "max-w-lg",
  lg: "max-w-2xl",
  xl: "max-w-4xl",
};

export function SlideDrawer({
  open,
  onClose,
  title,
  subtitle,
  headerActions,
  width = "lg",
  children,
}: SlideDrawerProps) {
  useEffect(() => {
    if (!open) return;

    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = prevOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end"
      role="dialog"
      aria-modal="true"
    >
      <div
        className="fixed inset-0 backdrop-blur-[2px] transition-opacity"
        style={{ backgroundColor: "var(--bg-backdrop)" }}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        className={`animate-drawer-in relative z-10 flex h-full w-full flex-col border-l ${WIDTH_CLASS[width]}`}
        style={{
          backgroundColor: "var(--bg-surface)",
          borderColor: "var(--border-subtle)",
          boxShadow: "var(--shadow-drawer)",
        }}
      >
        <header
          className="flex items-start justify-between gap-4 border-b px-5 py-4"
          style={{ borderColor: "var(--border-subtle)" }}
        >
          <div className="min-w-0 flex-1">
            <h2
              className="text-ui-lg truncate font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              {title}
            </h2>
            {subtitle && (
              <p
                className="text-ui-sm mt-0.5 truncate"
                style={{ color: "var(--text-secondary)" }}
              >
                {subtitle}
              </p>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {headerActions}
            <button
              type="button"
              onClick={onClose}
              aria-label="Close drawer"
              className="telemetry-transition cursor-pointer rounded-md border p-1.5 hover:opacity-85"
              style={{
                backgroundColor: "var(--bg-elevated)",
                borderColor: "var(--border-subtle)",
                color: "var(--text-secondary)",
              }}
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </header>
        <div className="flex-1 overflow-y-auto">{children}</div>
      </aside>
    </div>
  );
}
