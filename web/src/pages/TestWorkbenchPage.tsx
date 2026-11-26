import { CheckCircle2, FlaskConical, Save, Scale, Upload } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Badge, Card, CodeBlock, ErrorBanner, StatusBadge } from "../components/common";
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
      <div>
        <h1
          className="text-[26px] font-semibold tracking-[-0.03em]"
          style={{ color: "var(--text-primary)" }}
        >
          Author benchmarks and calibrate judges
        </h1>
        <p className="text-ui-base mt-1 max-w-[64ch]" style={{ color: "var(--text-secondary)" }}>
          Build Tier 1 datasets and Tier 2 YAML specifications, inspect sandboxed Jinja2 prompts, and verify judge alignment.
        </p>
      </div>

      {error && <ErrorBanner error={error} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <Card
          title="Declarative task YAML"
          subtitle="Sandboxed Jinja2 validation with automatic SSTI and secret scanning"
          padding="none"
          className="lg:col-span-7"
          actions={
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => void handleValidate()}
                className="telemetry-transition text-ui-sm inline-flex cursor-pointer items-center gap-1.5 rounded-md border px-3 py-1.5 font-medium hover:bg-[var(--bg-elevated)]"
                style={{ borderColor: "var(--border-subtle)", color: "var(--text-primary)" }}
              >
                <CheckCircle2 className="h-4 w-4" /> Validate preview
              </button>
              <button
                type="button"
                onClick={() => void handleSave()}
                className="telemetry-transition text-ui-sm inline-flex cursor-pointer items-center gap-1.5 rounded-md px-3.5 py-1.5 font-semibold hover:opacity-90"
                style={{ backgroundColor: "var(--accent-cyan)", color: "var(--text-inverse)" }}
              >
                <Save className="h-4 w-4" /> Save task
              </button>
            </div>
          }
        >
          <div
            className="grid grid-cols-1 gap-3 border-b p-4 sm:grid-cols-2"
            style={{ borderColor: "var(--border-subtle)" }}
          >
            <label className="text-ui-sm block space-y-1">
              <span style={{ color: "var(--text-secondary)" }}>Filename (saved in tasks/)</span>
              <input
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
                className="input-control font-mono-num w-full px-3 py-1.5"
              />
            </label>
            <label className="text-ui-sm block space-y-1">
              <span style={{ color: "var(--text-secondary)" }}>Upload CSV / JSONL / Parquet</span>
              <div className="input-control flex items-center gap-2 border-dashed px-3 py-1.5">
                <Upload className="h-4 w-4 shrink-0" style={{ color: "var(--accent-cyan)" }} />
                <input
                  type="file"
                  accept=".csv,.jsonl,.json,.parquet"
                  onChange={(e) => void handleUploadDataset(e.target.files?.[0])}
                  className="text-ui-sm w-full cursor-pointer"
                />
              </div>
            </label>
          </div>

          <textarea
            rows={14}
            value={yamlText}
            onChange={(e) => setYamlText(e.target.value)}
            spellCheck={false}
            className="font-mono-num text-ui-sm block w-full resize-y border-0 p-4 leading-relaxed focus:outline-none"
            style={{
              backgroundColor: "var(--bg-code)",
              color: "var(--text-primary)",
            }}
          />

          {savedPath && (
            <div
              className="text-ui-sm border-t px-4 py-2.5"
              style={{
                borderColor: "var(--border-subtle)",
                color: "var(--accent-emerald)",
              }}
            >
              Saved task YAML to <span className="font-mono-num">{savedPath}</span>
            </div>
          )}
        </Card>

        <div className="space-y-6 lg:col-span-5">
          <Card
            title="Sandboxed Jinja2 preview"
            subtitle="Rendered Sample #0 prompt and static checks"
            padding="none"
          >
            {validation ? (
              <div className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                <div className="flex flex-wrap items-center gap-4 px-5 py-3">
                  <StatusBadge status={validation.valid ? "APPROVED" : "REJECTED"} />
                  <Badge variant={validation.secret_scan_clean ? "cyan" : "rose"} dot mono>
                    {validation.secret_scan_clean ? "SECRET_CLEAN" : "SECRET_DETECTED"}
                  </Badge>
                </div>
                {validation.errors.length > 0 && (
                  <ul
                    className="text-ui-sm list-disc space-y-1 px-8 py-3"
                    style={{ color: "var(--accent-rose)" }}
                  >
                    {validation.errors.map((e, i) => (
                      <li key={i}>{e}</li>
                    ))}
                  </ul>
                )}
                {validation.preview_prompt && (
                  <CodeBlock
                    code={validation.preview_prompt}
                    title="Rendered Sample #0 prompt"
                    flush
                  />
                )}
              </div>
            ) : (
              <div className="text-ui-sm p-5" style={{ color: "var(--text-secondary)" }}>
                Select Validate preview to inspect rendered Jinja2 prompts and schema diagnostics.
              </div>
            )}
          </Card>

          <Card
            title="Workspace task directory"
            subtitle="Select any benchmark to load into the editor"
            padding="none"
          >
            <div
              className="max-h-64 divide-y overflow-y-auto"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              {tasks.map((t) => (
                <button
                  key={t.path}
                  type="button"
                  onClick={() => void handleLoadTask(t.path)}
                  className="telemetry-transition text-ui-sm flex w-full cursor-pointer items-center justify-between gap-3 px-5 py-3 text-left hover:bg-[var(--bg-elevated)]"
                >
                  <span className="flex min-w-0 items-center gap-2 font-medium">
                    <FlaskConical
                      className="h-4 w-4 shrink-0"
                      style={{ color: "var(--accent-cyan)" }}
                    />
                    <span className="truncate">{t.name}</span>
                  </span>
                  <span
                    className="font-mono-num shrink-0"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {t.source}
                  </span>
                </button>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <Card
        title="LLM judge alignment calibration"
        subtitle="Audit judge versus human labels; flags UNALIGNED when Cohen's kappa falls below 0.40"
        padding="none"
        actions={
          <button
            type="button"
            onClick={() => void handleCalibrate()}
            className="telemetry-transition text-ui-sm inline-flex cursor-pointer items-center gap-1.5 rounded-md px-3.5 py-1.5 font-semibold hover:opacity-90"
            style={{ backgroundColor: "var(--accent-cyan)", color: "var(--text-inverse)" }}
          >
            <Scale className="h-4 w-4" /> Compute Cohen&apos;s kappa
          </button>
        }
      >
        <div
          className="grid grid-cols-1 divide-y lg:grid-cols-2 lg:divide-x lg:divide-y-0"
          style={{ borderColor: "var(--border-subtle)" }}
        >
          <textarea
            rows={7}
            value={calJsonl}
            onChange={(e) => setCalJsonl(e.target.value)}
            spellCheck={false}
            className="font-mono-num text-ui-sm block w-full resize-none border-0 p-4 leading-relaxed focus:outline-none"
            style={{ backgroundColor: "var(--bg-code)", color: "var(--text-primary)" }}
          />
          <div>
            {calResult ? (
              <div className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                <div className="text-ui-sm flex items-center justify-between px-5 py-3.5">
                  <span style={{ color: "var(--text-secondary)" }}>Alignment status</span>
                  <StatusBadge status={calResult.status} />
                </div>
                <div className="text-ui-sm flex justify-between px-5 py-3.5">
                  <span style={{ color: "var(--text-secondary)" }}>Cohen&apos;s kappa</span>
                  <span className="font-mono-num font-semibold">
                    {calResult.cohens_kappa.toFixed(4)}
                  </span>
                </div>
                <div className="text-ui-sm flex justify-between px-5 py-3.5">
                  <span style={{ color: "var(--text-secondary)" }}>Raw agreement</span>
                  <span className="font-mono-num">
                    {(calResult.raw_agreement * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="text-ui-sm flex justify-between px-5 py-3.5">
                  <span style={{ color: "var(--text-secondary)" }}>Pairs evaluated</span>
                  <span className="font-mono-num">{calResult.n_samples}</span>
                </div>
              </div>
            ) : (
              <div className="text-ui-sm p-5" style={{ color: "var(--text-secondary)" }}>
                Edit judge versus human JSONL pairs on the left and select Compute Cohen&apos;s kappa.
              </div>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
