const API_KEY_STORAGE = "kefu_admin_api_key";

export function getApiKey() {
  return localStorage.getItem(API_KEY_STORAGE) || window.__KEFU_CONFIG__?.apiKey || "";
}

export function setApiKey(key) {
  localStorage.setItem(API_KEY_STORAGE, key);
}

export async function apiFetch(path, options = {}) {
  const headers = {
    ...(options.headers || {}),
    "X-API-Key": getApiKey(),
  };
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  const resp = await fetch(path, { ...options, headers });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

export async function fetchHealth() {
  const resp = await fetch("/health");
  return resp.json();
}

export async function fetchHealthDeep() {
  const resp = await fetch("/health/deep");
  return resp.json();
}

export async function fetchMetricsText() {
  const headers = { "X-API-Key": getApiKey() };
  const resp = await fetch("/metrics", { headers });
  if (!resp.ok) throw new Error("Failed to load metrics");
  return resp.text();
}

export function parsePrometheusMetrics(text) {
  const counters = {};
  const lines = text.split("\n");
  for (const line of lines) {
    if (!line || line.startsWith("#")) continue;
    const m = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)/);
    if (!m) continue;
    const name = m[1];
    const value = parseFloat(m[3]);
    if (!counters[name]) counters[name] = 0;
    counters[name] += value;
  }
  return counters;
}

export async function listKnowledge() {
  return apiFetch("/api/v1/admin/knowledge");
}

export async function deleteKnowledge(filename) {
  return apiFetch(`/api/v1/admin/knowledge/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  });
}

export async function rebuildKnowledge() {
  return apiFetch("/api/v1/knowledge/rebuild", { method: "POST" });
}

export async function uploadKnowledge(file) {
  const form = new FormData();
  form.append("file", file);
  const headers = { "X-API-Key": getApiKey() };
  const resp = await fetch("/api/v1/knowledge/upload", {
    method: "POST",
    headers,
    body: form,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

export async function listComplaints() {
  return apiFetch("/api/v1/admin/complaints");
}

export async function updateComplaintStatus(id, status) {
  return apiFetch(`/api/v1/admin/complaints/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });
}

export async function runInference(messages, maxNewTokens) {
  return apiFetch("/inference", {
    method: "POST",
    body: JSON.stringify({ messages, max_new_tokens: maxNewTokens }),
  });
}
