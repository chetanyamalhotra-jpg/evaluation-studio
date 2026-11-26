import { FileText, Lock, Upload } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Card, CodeBlock, ErrorBanner, StatusBadge } from "../components/common";
import type {
  CompiledSuiteSummary,
  ExtractedItem,
  PaperCompileResponse,
  PaperLockResponse,
  PaperSampleSummary,
} from "../types/evalStudio";
import { shortHash } from "../utils/format";

export function PaperStudioPage() {
  const [samples, setSamples] = useState<PaperSampleSummary[]>([]);
  const [compiledSuites, setCompiledSuites] = useState<CompiledSuiteSummary[]>([]);
  const [paperPath, setPaperPath] = useState("");
  const [suiteName, setSuiteName] = useState("clinical_paper_suite");
  const [mode, setMode] = useState<"heuristic" | "mock" | "http-byok">("mock");
  const [compiled, setCompiled] = useState<PaperCompileResponse | null>(null);
  const [editableItems, setEditableItems] = useState<ExtractedItem[]>([]);
  const [locked, setLocked] = useState<PaperLockResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshPapers = async () => {
    try {
      const res = await api.listPapers();
      setSamples(res.samples);
      setCompiledSuites(res.compiled_suites);
      if (res.samples.length > 0 && !paperPath) {
        setPaperPath(res.samples[0]!.path);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to list papers.");
    }
  };

  useEffect(() => {
    void refreshPapers();
  }, []);

  const handleUpload = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const up = await api.uploadPaper(file);
      setPaperPath(up.paper_path);
      setSuiteName(up.filename.replace(/\.[^.]+$/, "") + "_suite");
      await refreshPapers();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Paper upload failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleCompile = async () => {
    if (!paperPath) return;
    setBusy(true);
    setError(null);
    setLocked(null);
    try {
      const res = await api.compilePaper({
        paper_path: paperPath,
        suite_name: suiteName,
        mode,
      });
      setCompiled(res);
      setEditableItems(res.approved_items);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Compilation failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleLock = async () => {
    if (!compiled || editableItems.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.lockPaperSuite({
        paper_path: compiled.source_path,
        suite_name: suiteName,
        extractor_model: compiled.extractor_model,
        verifier_model: compiled.verifier_model,
        items: editableItems,
      });
      setLocked(res);
      await refreshPapers();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "SHA-256 locking failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1
          className="text-[26px] font-semibold tracking-[-0.03em]"
          style={{ color: "var(--text-primary)" }}
        >
          Compile scientific papers into benchmarks
        </h1>
        <p className="text-ui-base mt-1 max-w-[64ch]" style={{ color: "var(--text-secondary)" }}>
          Extract citation-grounded questions, audit Stage 2 answerability verification, and lock immutable SHA-256 suites.
        </p>
      </div>

      {error && <ErrorBanner error={error} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <Card
          title="Source paper and compiler pipeline"
          subtitle="Upload .md, .txt, or .pdf manuscripts and select an extraction backend"
          className="lg:col-span-8"
        >
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <label className="text-ui-sm block space-y-1.5">
              <span style={{ color: "var(--text-secondary)" }}>Select paper</span>
              <select
                value={paperPath}
                onChange={(e) => setPaperPath(e.target.value)}
                className="input-control w-full px-3 py-2"
              >
                {samples.map((s) => (
                  <option key={s.path} value={s.path}>
                    {s.name} ({s.path})
                  </option>
                ))}
              </select>
            </label>

            <label className="text-ui-sm block space-y-1.5">
              <span style={{ color: "var(--text-secondary)" }}>Upload manuscript</span>
              <div
                className="input-control flex items-center gap-2 border-dashed px-3 py-1.5"
              >
                <Upload className="h-4 w-4 shrink-0" style={{ color: "var(--accent-cyan)" }} />
                <input
                  type="file"
                  accept=".md,.txt,.pdf"
                  onChange={(e) => void handleUpload(e.target.files?.[0])}
                  className="text-ui-sm w-full cursor-pointer"
                />
              </div>
            </label>

            <label className="text-ui-sm block space-y-1.5">
              <span style={{ color: "var(--text-secondary)" }}>Suite name</span>
              <input
                value={suiteName}
                onChange={(e) => setSuiteName(e.target.value)}
                className="input-control font-mono-num w-full px-3 py-2"
              />
            </label>

            <div className="text-ui-sm space-y-1.5">
              <span className="block" style={{ color: "var(--text-secondary)" }}>
                Compiler mode
              </span>
              <div
                role="radiogroup"
                aria-label="Compiler mode"
                className="inline-flex w-full divide-x overflow-hidden rounded-md border"
                style={{ borderColor: "var(--border-subtle)" }}
              >
                {(
                  [
                    { id: "mock", label: "Mock" },
                    { id: "heuristic", label: "Heuristic" },
                    { id: "http-byok", label: "HTTP BYOK" },
                  ] as const
                ).map((m) => {
                  const active = mode === m.id;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      role="radio"
                      aria-checked={active}
                      onClick={() => setMode(m.id)}
                      className="telemetry-transition text-ui-sm flex-1 cursor-pointer px-3 py-2 font-medium"
                      style={{
                        backgroundColor: active ? "var(--accent-cyan-soft)" : "var(--bg-elevated)",
                        color: active ? "var(--accent-cyan)" : "var(--text-secondary)",
                      }}
                    >
                      {m.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              type="button"
              disabled={busy || !paperPath}
              onClick={() => void handleCompile()}
              className="telemetry-transition text-ui-base inline-flex cursor-pointer items-center gap-2 rounded-md px-4 py-2 font-semibold hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              style={{ backgroundColor: "var(--accent-cyan)", color: "var(--text-inverse)" }}
            >
              <FileText className="h-4 w-4" /> Run two-stage compiler
            </button>
            {compiled && editableItems.length > 0 && (
              <button
                type="button"
                disabled={busy}
                onClick={() => void handleLock()}
                className="telemetry-transition text-ui-base inline-flex cursor-pointer items-center gap-2 rounded-md px-4 py-2 font-semibold hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                style={{ backgroundColor: "var(--accent-emerald)", color: "var(--text-inverse)" }}
              >
                <Lock className="h-4 w-4" /> Lock SHA-256 suite
              </button>
            )}
          </div>
        </Card>

        <Card
          title="Locked paper suites"
          subtitle="Immutable compiled suites in tasks/compiled/"
          padding="none"
          className="lg:col-span-4"
        >
          {compiledSuites.length === 0 ? (
            <div className="text-ui-sm p-5" style={{ color: "var(--text-secondary)" }}>
              No locked suites compiled yet.
            </div>
          ) : (
            <div className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
              {compiledSuites.map((s) => (
                <div key={s.path} className="px-5 py-3.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-ui-sm font-semibold">{s.name}</span>
                    <span className="font-mono-num text-ui-sm" style={{ color: "var(--accent-emerald)" }}>
                      {s.item_count} items
                    </span>
                  </div>
                  <div
                    className="font-mono-num text-ui-sm mt-1"
                    style={{ color: "var(--text-muted)" }}
                  >
                    SHA-256: {shortHash(s.suite_sha256)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {compiled && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card
            title={`Stage 2 approved items (${editableItems.length})`}
            subtitle="Verbatim citation-grounded items ready for SHA-256 locking"
            padding="none"
          >
            <div className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
              {editableItems.map((item, idx) => (
                <div key={item.item_id} className="space-y-2.5 p-5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono-num text-ui-sm font-semibold">{item.item_id}</span>
                    <StatusBadge status="APPROVED" />
                  </div>
                  <input
                    value={item.question}
                    onChange={(e) => {
                      const next = [...editableItems];
                      next[idx] = { ...item, question: e.target.value };
                      setEditableItems(next);
                    }}
                    className="input-control w-full px-3 py-2"
                  />
                  <div
                    className="font-mono-num text-ui-sm"
                    style={{ color: "var(--accent-cyan)" }}
                  >
                    Reference answer: {item.reference_answer} ({item.category})
                  </div>
                  {item.citations.map((c, ci) => (
                    <blockquote
                      key={ci}
                      className="text-ui-sm m-0 border-l-2 pl-3"
                      style={{
                        borderColor: "var(--accent-cyan)",
                        color: "var(--text-secondary)",
                      }}
                    >
                      p.{c.page} ({c.section}): &ldquo;{c.quote}&rdquo;
                    </blockquote>
                  ))}
                </div>
              ))}
            </div>
          </Card>

          <Card
            title={`Stage 2 rejected items (${compiled.rejected_items.length})`}
            subtitle="Filtered by answerability, leakage, or missing verbatim citation gate"
            padding="none"
          >
            {compiled.rejected_items.length === 0 ? (
              <div className="text-ui-sm p-5" style={{ color: "var(--text-secondary)" }}>
                Zero items rejected in this compilation pass.
              </div>
            ) : (
              <div className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                {compiled.rejected_items.map((rej, idx) => (
                  <div key={idx} className="space-y-2 p-5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono-num text-ui-sm font-semibold">
                        {rej.item.item_id}
                      </span>
                      <StatusBadge status="REJECTED" />
                    </div>
                    <p className="text-ui-sm" style={{ color: "var(--text-primary)" }}>
                      {rej.item.question}
                    </p>
                    <ul
                      className="text-ui-sm list-disc space-y-1 pl-5"
                      style={{ color: "var(--accent-rose)" }}
                    >
                      {rej.reasons.map((r, ri) => (
                        <li key={ri}>{r}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      )}

      {locked && (
        <Card
          title="Locked runnable task YAML"
          subtitle={`Saved to ${locked.yaml_path} (Suite SHA-256: ${shortHash(locked.suite_sha256)})`}
          padding="none"
        >
          <CodeBlock code={locked.yaml_text} language="yaml" flush />
        </Card>
      )}
    </div>
  );
}
