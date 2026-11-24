// Self-contained SVG, PNG, CSV, and clipboard export utilities.

function ensureExtension(filename: string, ext: string): string {
  const normalized = ext.startsWith(".") ? ext : `.${ext}`;
  return filename.toLowerCase().endsWith(normalized.toLowerCase())
    ? filename
    : `${filename}${normalized}`;
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  try {
    link.click();
  } finally {
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

function resolveCssVars(
  raw: string,
  rootStyles: CSSStyleDeclaration
): string {
  return raw.replace(/var\((--[a-zA-Z0-9_-]+)(?:,\s*([^)]+))?\)/g, (_, name: string, fallback?: string) => {
    const resolved = rootStyles.getPropertyValue(name).trim();
    if (resolved) {
      return resolved;
    }
    return fallback ? fallback.trim() : "#94A3B8";
  });
}

function resolveSurfaceBackground(container: HTMLElement, rootStyles: CSSStyleDeclaration): string {
  const containerBg = getComputedStyle(container).backgroundColor;
  if (
    containerBg &&
    containerBg !== "transparent" &&
    containerBg !== "rgba(0, 0, 0, 0)"
  ) {
    return containerBg;
  }
  const tokenSurface = rootStyles.getPropertyValue("--bg-surface").trim();
  if (tokenSurface) {
    return tokenSurface;
  }
  const theme = document.documentElement.dataset.theme;
  return theme === "light" ? "#FFFFFF" : "#0F1623";
}

function findExportSvg(element: HTMLElement): SVGSVGElement {
  if (element instanceof SVGSVGElement) {
    return element;
  }
  const svgs = Array.from(element.querySelectorAll("svg"));
  if (svgs.length === 0) {
    throw new Error("No SVG chart element found to export.");
  }
  // Pick the largest SVG (avoids picking small Lucide icons inside chart headers).
  let best = svgs[0]!;
  let bestArea = 0;
  for (const svg of svgs) {
    const rect = svg.getBoundingClientRect();
    const area = rect.width * rect.height;
    if (area > bestArea) {
      bestArea = area;
      best = svg;
    }
  }
  return best;
}

function prepareResolvedSvg(
  sourceSvg: SVGSVGElement,
  container: HTMLElement
): { clone: SVGSVGElement; width: number; height: number; bgColor: string } {
  const rootStyles = getComputedStyle(document.documentElement);
  const rect = sourceSvg.getBoundingClientRect();
  const vb = sourceSvg.viewBox?.baseVal;

  const width = Math.max(
    1,
    Math.round(rect.width || Number(sourceSvg.getAttribute("width")) || vb?.width || 800)
  );
  const height = Math.max(
    1,
    Math.round(rect.height || Number(sourceSvg.getAttribute("height")) || vb?.height || 420)
  );

  const clone = sourceSvg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  if (!clone.getAttribute("viewBox")) {
    clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
  }

  const bgColor = resolveSurfaceBackground(container, rootStyles);
  const sansFont =
    rootStyles.getPropertyValue("--font-sans").trim() ||
    "Inter, -apple-system, BlinkMacSystemFont, sans-serif";
  const monoFont =
    rootStyles.getPropertyValue("--font-mono").trim() ||
    "JetBrains Mono, ui-monospace, monospace";

  // Resolve CSS variables on attributes & inline styles across all cloned nodes.
  const allNodes = [clone, ...Array.from(clone.querySelectorAll("*"))];
  for (const node of allNodes) {
    if (!(node instanceof Element)) continue;
    for (const attr of ["fill", "stroke", "color", "stop-color"]) {
      const val = node.getAttribute(attr);
      if (val && val.includes("var(")) {
        node.setAttribute(attr, resolveCssVars(val, rootStyles));
      }
    }
    const styleAttr = node.getAttribute("style");
    if (styleAttr && styleAttr.includes("var(")) {
      node.setAttribute("style", resolveCssVars(styleAttr, rootStyles));
    }
    if (node.tagName.toLowerCase() === "style" && node.textContent) {
      node.textContent = resolveCssVars(node.textContent, rootStyles);
    }
    if (node.tagName.toLowerCase() === "text" || node.tagName.toLowerCase() === "tspan") {
      if (!node.getAttribute("font-family")) {
        const isMono = node.classList.contains("font-mono-num");
        node.setAttribute("font-family", isMono ? monoFont : sansFont);
      }
    }
  }

  // Inject opaque surface background rect as first child so exported SVG/PNG is crisp.
  const bgRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bgRect.setAttribute("x", "0");
  bgRect.setAttribute("y", "0");
  bgRect.setAttribute("width", String(width));
  bgRect.setAttribute("height", String(height));
  bgRect.setAttribute("fill", bgColor);
  clone.insertBefore(bgRect, clone.firstChild);

  return { clone, width, height, bgColor };
}

export function exportElementToSvg(
  element: HTMLElement,
  filename: string
): void {
  const sourceSvg = findExportSvg(element);
  const { clone } = prepareResolvedSvg(sourceSvg, element);
  const serializer = new XMLSerializer();
  const svgXml = `<?xml version="1.0" encoding="UTF-8"?>\n${serializer.serializeToString(clone)}`;
  const blob = new Blob([svgXml], { type: "image/svg+xml;charset=utf-8" });
  triggerDownload(blob, ensureExtension(filename, ".svg"));
}

export async function exportElementToPng(
  element: HTMLElement,
  filename: string,
  scale = 2
): Promise<void> {
  const sourceSvg = findExportSvg(element);
  const { clone, width, height, bgColor } = prepareResolvedSvg(sourceSvg, element);
  const serializer = new XMLSerializer();
  const svgString = serializer.serializeToString(clone);
  const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  const svgUrl = URL.createObjectURL(svgBlob);

  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("Failed to rasterize SVG chart to PNG."));
      image.src = svgUrl;
    });

    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Canvas 2D context is unavailable.");
    }

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0, width, height);

    const pngBlob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error("Canvas PNG serialization returned null."));
        }
      }, "image/png");
    });

    triggerDownload(pngBlob, ensureExtension(filename, ".png"));
  } finally {
    URL.revokeObjectURL(svgUrl);
  }
}

function escapeCsvCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  const str =
    typeof value === "object" ? JSON.stringify(value) : String(value);
  if (/[",\r\n]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function downloadCsv(
  rows: ReadonlyArray<Record<string, unknown>>,
  filename: string,
  columns?: string[]
): void {
  const headers: string[] = columns ? [...columns] : [];
  if (headers.length === 0) {
    const seen = new Set<string>();
    for (const row of rows) {
      for (const key of Object.keys(row)) {
        if (!seen.has(key)) {
          seen.add(key);
          headers.push(key);
        }
      }
    }
  }

  const lines: string[] = [];
  if (headers.length > 0) {
    lines.push(headers.map((h) => escapeCsvCell(h)).join(","));
    for (const row of rows) {
      lines.push(headers.map((h) => escapeCsvCell(row[h])).join(","));
    }
  }

  const csvContent = lines.join("\r\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8" });
  triggerDownload(blob, ensureExtension(filename, ".csv"));
}

export async function copyToClipboard(text: string): Promise<boolean> {
  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fallback below when clipboard permission is denied or non-HTTPS context.
    }
  }

  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    textarea.style.top = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(textarea);
    return ok;
  } catch {
    return false;
  }
}
