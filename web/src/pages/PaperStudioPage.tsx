import { CheckCircle2, FileText, Lock, Upload, XCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Badge, Card, CodeBlock, ErrorBanner } from "../components/common";
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
      {error && <ErrorBanner error={error} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card title="1. Source Paper & Extraction Mode" subtitle="Upload .md, .txt, or .pdf (with [pdf] extra)" className="lg:col-span-2">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Select Paper</span>
              <select
                value={paperPath}
                onChange={(e) => setPaperPath(e.target.value)}
                className="w-full rounded-lg border px-3 py-2 text-xs"
                style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
              >
                {samples.map((s) => (
                  <option key={s.path} value={s.path}>
                    {s.name} ({s.path})
                  </option>
                ))}
              </select>
            </label>

            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Or Upload Paper File</span>
              <div
                className="flex items-center gap-2 rounded-lg border border-dashed px-3 py-1.5"
                style={{ borderColor: "var(--border-subtle)" }}
              >
                <Upload className="h-3.5 w-3.5 shrink-0" style={{ color: "var(--accent-cyan)" }} />
                <input
                  type="file"
                  accept=".md,.txt,.pdf"
                  onChange={(e) => void handleUpload(e.target.files?.[0])}
                  className="w-full text-[11px]"
                />
              </div>
            </label>

            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Suite Name</span>
              <input
                value={suiteName}
                onChange={(e) => setSuiteName(e.target.value)}
                className="w-full rounded-lg border px-3 py-2 font-mono text-xs"
                style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
              />
            </label>

            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Compiler Mode</span>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as "heuristic" | "mock" | "http-byok")}
                className="w-full rounded-lg border px-3 py-2 text-xs"
                style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
              >
                <option value="mock">mock (demonstrates Stage 2 APPROVED + REJECTED)</option>
                <option value="heuristic">heuristic (deterministic offline Markdown)</option>
                <option value="http-byok">http-byok (separate Extractor + Verifier LLMs)</option>
              </select>
            </label>
          </div>

          <div className="mt-4 flex items-center gap-3">
            <button
              type="button"
              disabled={busy || !paperPath}
              onClick={() => void handleCompile()}
              className="inline-flex items-center gap-1.5 rounded-lg px-4 py-2 text-xs font-semibold disabled:opacity-50"
              style={{ backgroundColor: "var(--accent-cyan)", color: "#090D16" }}
            >
              <FileText className="h-3.5 w-3.5" /> Run 2-Stage Compile
            </button>
            {compiled && editableItems.length > 0 && (
              <button
                type="button"
                disabled={busy}
                onClick={() => void handleLock()}
                className="inline-flex items-center gap-1.5 rounded-lg px-4 py-2 text-xs font-semibold disabled:opacity-50"
                style={{ backgroundColor: "var(--accent-emerald)", color: "#090D16" }}
              >
                <Lock className="h-3.5 w-3.5" /> Lock SHA-256 & Export YAML
              </button>
            )}
          </div>
        </Card>

        <Card title="Compiled & Locked Suites" subtitle="SHA-256 immutable paper packs" elevated>
          {compiledSuites.length === 0 ? (
            <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
              No locked suites yet.
            </div>
          ) : (
            <div className="space-y-2.5 text-xs">
              {compiledSuites.map((s) => (
                <div
                  key={s.path}
                  className="rounded-lg border p-2.5"
                  style={{ borderColor: "var(--border-subtle)" }}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">{s.name}</span>
                    <Badge variant="emerald">{s.item_count} items</Badge>
                  </div>
                  <div className="mt-1 font-mono text-[10px]" style={{ color: "var(--text-muted)" }}>
                    SHA: {shortHash(s.suite_sha256)}
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
            title={`Stage 2 APPROVED (${editableItems.length})`}
            subtitle="Verbatim citation-grounded items ready to lock"
          >
            <div className="space-y-3">
              {editableItems.map((item, idx) => (
                <div
                  key={item.item_id}
                  className="space-y-2 rounded-xl border p-3 text-xs"
                  style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono font-semibold">{item.item_id}</span>
                    <Badge variant="emerald">
                      <CheckCircle2 className="mr-1 inline h-3 w-3" /> {item.category}
                    </Badge>
                  </div>
                  <input
                    value={item.question}
                    onChange={(e) => {
                      const next = [...editableItems];
                      next[idx] = { ...item, question: e.target.value };
                      setEditableItems(next);
                    }}
                    className="w-full rounded-md border px-2.5 py-1.5 text-xs"
                    style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-surface)" }}
                  />
                  <div className="font-mono text-[11px]" style={{ color: "var(--accent-cyan)" }}>
                    Gold: {item.reference_answer}
                  </div>
                  {item.citations.map((c, ci) => (
                    <div
                      key={ci}
                      className="rounded-md p-2 text-[11px]"
                      style={{ backgroundColor: "var(--bg-canvas)", color: "var(--text-secondary)" }}
                    >
                      p.{c.page} · {c.section}: “{c.quote}”
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </Card>

          <Card
            title={`Stage 2 REJECTED (${compiled.rejected_items.length})`}
            subtitle="Blocked by answerability, leakage, or missing verbatim citation gate"
          >
            <div className="space-y-3">
              {compiled.rejected_items.map((rej, idx) => (
                <div
                  key={idx}
                  className="space-y-2 rounded-xl border p-3 text-xs"
                  style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--accent-rose-soft)" }}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-semibold">{rej.item.item_id}</span>
                    <Badge variant="rose">
                      <XCircle className="mr-1 inline h-3 w-3" /> REJECTED
                    </Badge>
                  </div>
                  <div>{rej.item.question}</div>
                  <ul className="list-disc pl-4 text-[11px]" style={{ color: "var(--accent-rose)" }}>
                    {rej.reasons.map((r, ri) => (
                      <li key={ri}>{r}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {locked && (
        <Card
          title="Locked Runnable Task YAML"
          subtitle={`Saved to ${locked.yaml_path} · Suite SHA-256: ${shortHash(locked.suite_sha256)}`}
        >
          <CodeBlock code={locked.yaml_text} language="yaml" />
        </Card>
      )}
    </div>
  );
}
