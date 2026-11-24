import { CheckCircle2, FlaskConical, Save, Scale, Upload } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Badge, Card, CodeBlock, ErrorBanner } from "../components/common";
import type {
  CalibrateResponse,
  TaskSummary,
  ValidateTaskResponse,
} from "../types/evalStudio";

const STARTER_YAML = `name: custom_clinical_qa
version: "1.0.0"
trust_tier: tier2_declarative
output_type: generate_until
doc_to_text: "Clinical Question: {{ input_text }}\\nAnswer:"
doc_to_target: "{{ target }}"
metrics:
  - exact_match
metadata:
  items:
    - sample_id: q1
      input_text: "Which vitamin deficiency causes scurvy?"
      target: "Vitamin C"
`;

const STARTER_CALIBRATION = `{"judge_label": 1, "human_label": 1}
{"judge_label": 1, "human_label": 1}
{"judge_label": 0, "human_label": 0}
{"judge_label": 0, "human_label": 0}
{"judge_label": 1, "human_label": 0}`;

export function TestWorkbenchPage() {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [filename, setFilename] = useState("custom_clinical_qa.yaml");
  const [yamlText, setYamlText] = useState(STARTER_YAML);
  const [validation, setValidation] = useState<ValidateTaskResponse | null>(null);
  const [savedPath, setSavedPath] = useState<string | null>(null);

  const [calJsonl, setCalJsonl] = useState(STARTER_CALIBRATION);
  const [calResult, setCalResult] = useState<CalibrateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshTasks = async () => {
    try {
      const res = await api.listTasks();
      setTasks(res.tasks);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load tasks.");
    }
  };

  useEffect(() => {
    void refreshTasks();
  }, []);

  const handleLoadTask = async (path: string) => {
    setError(null);
    try {
      const detail = await api.getTaskDetail(path);
      setYamlText(detail.yaml_text);
      const base = path.split("/").pop() ?? "task.yaml";
      setFilename(base);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load task YAML.");
    }
  };

  const handleUploadDataset = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    try {
      const res = await api.uploadDataset(file);
      const qCol = res.columns[0] ?? "input_text";
      const tCol = res.columns[1] ?? "target";
      setYamlText(
        `name: ${file.name.replace(/\.[^.]+$/, "")}\nversion: "1.0.0"\ntrust_tier: tier1_zero_code\noutput_type: generate_until\ndataset_path: "${res.dataset_path}"\ndoc_to_text: "{{ ${qCol} }}"\ndoc_to_target: "{{ ${tCol} }}"\nmetrics:\n  - exact_match\n`
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Dataset upload failed.");
    }
  };

  const handleValidate = async () => {
    setError(null);
    try {
      const res = await api.validateTask({ yaml_text: yamlText });
      setValidation(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Validation request failed.");
    }
  };

  const handleSave = async () => {
    setError(null);
    setSavedPath(null);
    try {
      const res = await api.saveTask({ filename, yaml_text: yamlText });
      setSavedPath(res.path);
      await refreshTasks();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to save task.");
    }
  };

  const handleCalibrate = async () => {
    setError(null);
    try {
      const res = await api.calibrateJudge({ jsonl_text: calJsonl, threshold: 0.4 });
      setCalResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Calibration failed.");
    }
  };

  return (
    <div className="space-y-6">
      {error && <ErrorBanner error={error} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card
          title="BYOT Task YAML Editor (Tier 1 & Tier 2)"
          subtitle="Sandboxed Jinja2 validation with automatic SSTI & secret scanning"
          className="lg:col-span-2"
          actions={
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => void handleValidate()}
                className="inline-flex items-center gap-1 rounded-lg border px-3 py-1.5 text-xs font-semibold"
                style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
              >
                <CheckCircle2 className="h-3.5 w-3.5" /> Validate & Preview
              </button>
              <button
                type="button"
                onClick={() => void handleSave()}
                className="inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs font-semibold"
                style={{ backgroundColor: "var(--accent-cyan)", color: "#090D16" }}
              >
                <Save className="h-3.5 w-3.5" /> Save Task
              </button>
            </div>
          }
        >
          <div className="mb-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Task Filename (saved in tasks/)</span>
              <input
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
                className="w-full rounded-lg border px-3 py-1.5 font-mono text-xs"
                style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
              />
            </label>
            <label className="block space-y-1 text-xs">
              <span style={{ color: "var(--text-secondary)" }}>Upload CSV / JSONL / Parquet Dataset</span>
              <div
                className="flex items-center gap-2 rounded-lg border border-dashed px-3 py-1.5"
                style={{ borderColor: "var(--border-subtle)" }}
              >
                <Upload className="h-3.5 w-3.5 shrink-0" style={{ color: "var(--accent-cyan)" }} />
                <input
                  type="file"
                  accept=".csv,.jsonl,.json,.parquet"
                  onChange={(e) => void handleUploadDataset(e.target.files?.[0])}
                  className="w-full text-[11px]"
                />
              </div>
            </label>
          </div>

          <textarea
            rows={13}
            value={yamlText}
            onChange={(e) => setYamlText(e.target.value)}
            spellCheck={false}
            className="w-full rounded-xl border p-3 font-mono text-xs leading-relaxed"
            style={{
              borderColor: "var(--border-subtle)",
              backgroundColor: "var(--bg-canvas)",
              color: "var(--text-primary)",
            }}
          />

          {savedPath && (
            <div className="mt-2 text-xs" style={{ color: "var(--accent-emerald)" }}>
              Saved task YAML to <span className="font-mono">{savedPath}</span>
            </div>
          )}
        </Card>

        <div className="space-y-6">
          <Card title="Validation & Live Preview" subtitle="Sandboxed Jinja2 sample #0 output" elevated>
            {validation ? (
              <div className="space-y-3 text-xs">
                <div className="flex flex-wrap gap-2">
                  <Badge variant={validation.valid ? "emerald" : "rose"}>
                    {validation.valid ? "VALID" : "INVALID"}
                  </Badge>
                  <Badge variant={validation.secret_scan_clean ? "cyan" : "rose"}>
                    {validation.secret_scan_clean ? "Secret-Free" : "Secret Detected"}
                  </Badge>
                </div>
                {validation.errors.length > 0 && (
                  <ul className="list-disc pl-4 text-[11px]" style={{ color: "var(--accent-rose)" }}>
                    {validation.errors.map((e, i) => (
                      <li key={i}>{e}</li>
                    ))}
                  </ul>
                )}
                {validation.preview_prompt && (
                  <CodeBlock code={validation.preview_prompt} title="Rendered Sample #0 Prompt" />
                )}
              </div>
            ) : (
              <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
                Click <strong>Validate & Preview</strong> to inspect rendered prompts and schema
                diagnostics.
              </div>
            )}
          </Card>

          <Card title="Discovered Tasks" subtitle="Click to load into editor">
            <div className="max-h-52 space-y-1.5 overflow-y-auto">
              {tasks.map((t) => (
                <button
                  key={t.path}
                  type="button"
                  onClick={() => void handleLoadTask(t.path)}
                  className="flex w-full items-center justify-between rounded-lg border px-2.5 py-2 text-left text-xs transition-colors hover:opacity-90"
                  style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-elevated)" }}
                >
                  <span className="flex items-center gap-1.5 truncate font-medium">
                    <FlaskConical className="h-3.5 w-3.5 shrink-0" style={{ color: "var(--accent-cyan)" }} />
                    {t.name}
                  </span>
                  <Badge variant="neutral">{t.source}</Badge>
                </button>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {/* Judge Calibration Calculator */}
      <Card
        title="LLM Judge Alignment Calibration (Cohen's Kappa)"
        subtitle="Audit judge vs. human labels; warns UNALIGNED when κ < 0.40"
        actions={
          <button
            type="button"
            onClick={() => void handleCalibrate()}
            className="inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-semibold"
            style={{ backgroundColor: "var(--accent-violet)", color: "#FFFFFF" }}
          >
            <Scale className="h-3.5 w-3.5" /> Compute κ
          </button>
        }
      >
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <textarea
            rows={6}
            value={calJsonl}
            onChange={(e) => setCalJsonl(e.target.value)}
            className="w-full rounded-xl border p-3 font-mono text-xs"
            style={{ borderColor: "var(--border-subtle)", backgroundColor: "var(--bg-canvas)" }}
          />
          <div className="flex flex-col justify-center rounded-xl border p-4 text-xs" style={{ borderColor: "var(--border-subtle)" }}>
            {calResult ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-semibold">Alignment Status</span>
                  <Badge variant={calResult.passed_threshold ? "emerald" : "amber"}>
                    {calResult.status}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: "var(--text-secondary)" }}>Cohen&apos;s κ</span>
                  <span className="font-mono text-sm font-semibold">{calResult.cohens_kappa.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: "var(--text-secondary)" }}>Raw Agreement</span>
                  <span className="font-mono">{(calResult.raw_agreement * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: "var(--text-secondary)" }}>Pairs Evaluated</span>
                  <span className="font-mono">{calResult.n_samples}</span>
                </div>
              </div>
            ) : (
              <div style={{ color: "var(--text-secondary)" }}>
                Edit judge vs. human JSONL pairs on the left and click <strong>Compute κ</strong>.
              </div>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
