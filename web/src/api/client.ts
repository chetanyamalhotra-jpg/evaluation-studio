import type {
  BudgetExceededEvent,
  ByokPresetsResponse,
  ByokStatusResponse,
  CalibrateRequest,
  CalibrateResponse,
  CompareRequest,
  CompareResponse,
  CostEstimateResponse,
  DeleteKeyResponse,
  HealthResponse,
  KeyDeleteStorage,
  NoiseFloorRequest,
  NoiseFloorResponse,
  PaperCompileRequest,
  PaperCompileResponse,
  PaperLockRequest,
  PaperLockResponse,
  PapersListResponse,
  PaperUploadResponse,
  ReceiptDetailResponse,
  ReceiptSamplesResponse,
  ReceiptsListResponse,
  RegistriesResponse,
  RescoreRequest,
  RescoreResponse,
  RunCompletedEvent,
  RunErrorEvent,
  RunLaunchRequest,
  RunLaunchResponse,
  RunSseEvent,
  RunStartedEvent,
  RunStatusResponse,
  SampleScoredEvent,
  SaveKeyRequest,
  SaveKeyResponse,
  SaveTaskRequest,
  SaveTaskResponse,
  SecretRole,
  SubtaskStartedEvent,
  TaskDetailResponse,
  TasksListResponse,
  TestConnectionRequest,
  TestConnectionResponse,
  UploadDatasetResponse,
  ValidateTaskRequest,
  ValidateTaskResponse,
  VerifyReceiptResponse,
} from "../types/evalStudio";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  readonly url: string;

  constructor(status: number, detail: string, url: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.url = url;
  }
}

async function parseErrorDetail(res: Response): Promise<string> {
  const fallback = `HTTP ${res.status}${res.statusText ? ` ${res.statusText}` : ""}`;
  try {
    const contentType = res.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const payload = (await res.json()) as { detail?: unknown };
      if (typeof payload.detail === "string" && payload.detail.trim()) {
        return payload.detail;
      }
      if (Array.isArray(payload.detail)) {
        const msgs = payload.detail
          .map((item) => {
            if (typeof item === "string") return item;
            if (item && typeof item === "object" && "msg" in item) {
              const loc =
                "loc" in item && Array.isArray(item.loc)
                  ? `${item.loc.join(".")}: `
                  : "";
              return `${loc}${String(item.msg)}`;
            }
            return JSON.stringify(item);
          })
          .filter(Boolean);
        if (msgs.length > 0) {
          return msgs.join("; ");
        }
      }
      return JSON.stringify(payload);
    }
    const text = await res.text();
    return text.trim() || fallback;
  } catch {
    return fallback;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }
  const body = init?.body;
  if (typeof body === "string" && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(path, {
    ...init,
    headers,
  });

  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new ApiError(res.status, detail, path);
  }

  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

async function uploadMultipart<T>(
  path: string,
  fileOrFormData: File | FormData,
  fieldName = "file",
  extraFields?: Record<string, string>,
  signal?: AbortSignal
): Promise<T> {
  let formData: FormData;
  if (fileOrFormData instanceof FormData) {
    formData = fileOrFormData;
  } else {
    formData = new FormData();
    formData.append(fieldName, fileOrFormData);
    if (extraFields) {
      for (const [k, v] of Object.entries(extraFields)) {
        formData.append(k, v);
      }
    }
  }

  // Never set Content-Type manually for FormData; browser adds multipart boundary.
  return request<T>(path, {
    method: "POST",
    body: formData,
    signal,
  });
}

function withQuery(
  path: string,
  params?: Record<string, string | number | boolean | undefined | null>
): string {
  if (!params) return path;
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  }
  const qs = search.toString();
  return qs ? `${path}?${qs}` : path;
}

export interface RunEventHandlers {
  onRunStarted?: (event: RunStartedEvent) => void;
  onSubtaskStarted?: (event: SubtaskStartedEvent) => void;
  onSampleScored?: (event: SampleScoredEvent) => void;
  onBudgetExceeded?: (event: BudgetExceededEvent) => void;
  onCompleted?: (event: RunCompletedEvent) => void;
  onError?: (err: { message: string; event?: RunErrorEvent }) => void;
  onOpen?: () => void;
  onEvent?: (event: RunSseEvent) => void;
}

export function subscribeRunEvents(
  runId: string,
  handlers: RunEventHandlers
): () => void {
  const url = `/api/runs/${encodeURIComponent(runId)}/events`;
  const source = new EventSource(url);
  let closed = false;

  const cleanup = () => {
    if (closed) return;
    closed = true;
    source.close();
  };

  const dispatchPayload = (rawData: string, explicitType?: RunSseEvent["event"]) => {
    if (closed || !rawData) return;
    try {
      const parsed = JSON.parse(rawData) as Partial<RunSseEvent> & Record<string, unknown>;
      const eventType = (parsed.event ?? explicitType) as RunSseEvent["event"] | undefined;
      if (!eventType) return;

      const normalized = { ...parsed, event: eventType } as RunSseEvent;
      handlers.onEvent?.(normalized);

      switch (normalized.event) {
        case "run_started":
          handlers.onRunStarted?.(normalized);
          break;
        case "subtask_started":
          handlers.onSubtaskStarted?.(normalized);
          break;
        case "sample_scored":
          handlers.onSampleScored?.(normalized);
          break;
        case "budget_exceeded":
          handlers.onBudgetExceeded?.(normalized);
          cleanup();
          break;
        case "completed":
          handlers.onCompleted?.(normalized);
          cleanup();
          break;
        case "error":
          handlers.onError?.({
            message: normalized.error || "Run stream reported an error.",
            event: normalized,
          });
          cleanup();
          break;
      }
    } catch (err) {
      handlers.onError?.({
        message: err instanceof Error ? err.message : "Failed to parse SSE event payload.",
      });
    }
  };

  source.onopen = () => {
    if (!closed) {
      handlers.onOpen?.();
    }
  };

  source.onmessage = (ev: MessageEvent<string>) => {
    dispatchPayload(ev.data);
  };

  const namedEvents: ReadonlyArray<RunSseEvent["event"]> = [
    "run_started",
    "subtask_started",
    "sample_scored",
    "budget_exceeded",
    "completed",
    "error",
  ];

  for (const eventName of namedEvents) {
    source.addEventListener(eventName, (ev: Event) => {
      const msg = ev as MessageEvent<string>;
      if (typeof msg.data === "string") {
        dispatchPayload(msg.data, eventName);
      }
    });
  }

  source.onerror = () => {
    if (closed) return;
    if (source.readyState === EventSource.CLOSED) {
      cleanup();
      return;
    }
    handlers.onError?.({
      message: "SSE connection interrupted while streaming run telemetry.",
    });
  };

  return cleanup;
}

export const api = {
  // Health
  getHealth: (signal?: AbortSignal) =>
    request<HealthResponse>("/api/health", { signal }),

  // BYOK
  getByokPresets: (signal?: AbortSignal) =>
    request<ByokPresetsResponse>("/api/byok/presets", { signal }),

  getByokStatus: (signal?: AbortSignal) =>
    request<ByokStatusResponse>("/api/byok/status", { signal }),

  saveByokKey: (payload: SaveKeyRequest) =>
    request<SaveKeyResponse>("/api/byok/keys", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  deleteByokKey: (role: SecretRole, storage: KeyDeleteStorage = "all") =>
    request<DeleteKeyResponse>(
      withQuery(`/api/byok/keys/${encodeURIComponent(role)}`, { storage }),
      { method: "DELETE" }
    ),

  testByokConnection: (payload: TestConnectionRequest) =>
    request<TestConnectionResponse>("/api/byok/test", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // BYOT
  listTasks: (signal?: AbortSignal) =>
    request<TasksListResponse>("/api/byot/tasks", { signal }),

  getTaskDetail: (path: string, signal?: AbortSignal) =>
    request<TaskDetailResponse>(withQuery("/api/byot/tasks/detail", { path }), {
      signal,
    }),

  uploadDataset: (file: File, signal?: AbortSignal) =>
    uploadMultipart<UploadDatasetResponse>(
      "/api/byot/upload-dataset",
      file,
      "file",
      undefined,
      signal
    ),

  saveTask: (payload: SaveTaskRequest) =>
    request<SaveTaskResponse>("/api/byot/save", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  validateTask: (payload: ValidateTaskRequest) =>
    request<ValidateTaskResponse>("/api/byot/validate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getRegistries: (signal?: AbortSignal) =>
    request<RegistriesResponse>("/api/byot/registries", { signal }),

  calibrateJudge: (payload: CalibrateRequest | File, threshold?: number) => {
    if (payload instanceof File) {
      return uploadMultipart<CalibrateResponse>(
        "/api/byot/calibrate",
        payload,
        "file",
        threshold !== undefined ? { threshold: String(threshold) } : undefined
      );
    }
    return request<CalibrateResponse>("/api/byot/calibrate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  // BYOP
  listPapers: (signal?: AbortSignal) =>
    request<PapersListResponse>("/api/byop/papers", { signal }),

  uploadPaper: (file: File, signal?: AbortSignal) =>
    uploadMultipart<PaperUploadResponse>(
      "/api/byop/upload",
      file,
      "file",
      undefined,
      signal
    ),

  compilePaper: (payload: PaperCompileRequest) =>
    request<PaperCompileResponse>("/api/byop/compile", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  lockPaperSuite: (payload: PaperLockRequest) =>
    request<PaperLockResponse>("/api/byop/lock", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // Runs & SSE
  estimateRun: (payload: RunLaunchRequest) =>
    request<CostEstimateResponse>("/api/runs/estimate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  launchRun: (payload: RunLaunchRequest) =>
    request<RunLaunchResponse>("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getRunStatus: (runId: string, signal?: AbortSignal) =>
    request<RunStatusResponse>(`/api/runs/${encodeURIComponent(runId)}`, {
      signal,
    }),

  subscribeRunEvents,

  // Receipts & Analytics
  listReceipts: (signal?: AbortSignal) =>
    request<ReceiptsListResponse>("/api/receipts", { signal }),

  getReceipt: (receiptId: string, signal?: AbortSignal) =>
    request<ReceiptDetailResponse>(
      `/api/receipts/${encodeURIComponent(receiptId)}`,
      { signal }
    ),

  getReceiptSamples: (
    receiptId: string,
    params?: { offset?: number; limit?: number; q?: string },
    signal?: AbortSignal
  ) =>
    request<ReceiptSamplesResponse>(
      withQuery(`/api/receipts/${encodeURIComponent(receiptId)}/samples`, params),
      { signal }
    ),

  verifyReceipt: (receiptId: string) =>
    request<VerifyReceiptResponse>(
      `/api/receipts/${encodeURIComponent(receiptId)}/verify`,
      { method: "POST" }
    ),

  rescoreReceipt: (receiptId: string, payload: RescoreRequest) =>
    request<RescoreResponse>(
      `/api/receipts/${encodeURIComponent(receiptId)}/rescore`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    ),

  measureNoiseFloor: (payload: NoiseFloorRequest) =>
    request<NoiseFloorResponse>("/api/receipts/noise-floor", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  compareReceipts: (payload: CompareRequest) =>
    request<CompareResponse>("/api/receipts/compare", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
