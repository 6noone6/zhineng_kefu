const API_BASE = window.location.origin;
const WS_BASE = API_BASE.replace(/^http/, "ws");
const CONFIG = window.__KEFU_CONFIG__ || { wsPath: "/api/v1/ws/chat" };

const SESSION_STORAGE_KEY = "kefu_session_id";
const HISTORY_STORAGE_KEY = "kefu_session_history";
const MAX_HISTORY = 30;

let sessionId = localStorage.getItem(SESSION_STORAGE_KEY);
let ws = null;
let isConnected = false;
let connState = "connecting";
let authToken = localStorage.getItem("kefu_token") || "";
let currentUser = null;
let lastUserMessage = "";
let pendingAssistantEl = null;
let abortRequested = false;
let reconnectDelayMs = 1000;
let reconnectTimer = null;
const RECONNECT_MAX_MS = 30000;

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const cancelBtn = document.getElementById("cancel");
const newSessionBtn = document.getElementById("newSession");
const sessionInfoEl = document.getElementById("sessionInfo");
const sessionListEl = document.getElementById("sessionList");
const authStatusEl = document.getElementById("authStatus");
const loginEmailEl = document.getElementById("loginEmail");
const loginBtn = document.getElementById("loginBtn");
const logoutBtn = document.getElementById("logoutBtn");
const ordersPanelEl = document.getElementById("ordersPanel");
const ordersListEl = document.getElementById("ordersList");
const connStatusEl = document.getElementById("connStatus");
const quickPromptsEl = document.getElementById("quickPrompts");

function setCancelEnabled(enabled) {
  if (cancelBtn) cancelBtn.disabled = !enabled;
}

function cancelPending() {
  abortRequested = true;
  if (pendingAssistantEl) {
    pendingAssistantEl.classList.remove("typing");
    const body = pendingAssistantEl.querySelector(".message-body");
    if (body && !(pendingAssistantEl.dataset.full || "").trim()) {
      body.textContent = t("cancelled") || "已取消";
    }
    pendingAssistantEl = null;
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      // message intentionally omitted — server must accept cancel without it
      ws.send(JSON.stringify({ type: "cancel", session_id: sessionId || null }));
    } catch {
      /* ignore */
    }
  }
  setCancelEnabled(false);
  if (sendBtn) sendBtn.disabled = false;
}

const TOOL_LABELS = {
  fetch_logistics_information: "物流查询",
  record_user_complaint: "投诉记录",
  create_return_request: "退换货",
  customer_chat: "知识库",
  query_order: "订单查询",
  query_my_orders: "我的订单",
};

function authHeaders() {
  const headers = { "Content-Type": "application/json" };
  if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
  return headers;
}

function updateConnStatus() {
  if (!connStatusEl) return;
  const labels = {
    connected: t("connConnected"),
    connecting: t("connConnecting"),
    disconnected: t("connDisconnected"),
  };
  connStatusEl.textContent = labels[connState] || connState;
  connStatusEl.dataset.state = connState;
}

function getSessionHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveSessionHistory(history) {
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)));
}

function upsertSessionHistory(id, title) {
  if (!id) return;
  const preview = (title || "").trim().slice(0, 40) || id.slice(0, 8);
  const history = getSessionHistory().filter((item) => item.id !== id);
  history.unshift({ id, title: preview, updatedAt: Date.now() });
  saveSessionHistory(history);
  renderSessionList();
}

function persistSessionId(id) {
  sessionId = id;
  if (id) {
    localStorage.setItem(SESSION_STORAGE_KEY, id);
  } else {
    localStorage.removeItem(SESSION_STORAGE_KEY);
  }
  if (sessionInfoEl) {
    sessionInfoEl.textContent = id ? `${t("session")}: ${id}` : "";
  }
  renderSessionList();
}

function removeSessionFromHistory(id) {
  const history = getSessionHistory().filter((item) => item.id !== id);
  saveSessionHistory(history);
  if (sessionId === id) {
    persistSessionId(null);
    clearMessages();
  }
  renderSessionList();
}

async function deleteSession(id, event) {
  if (event) {
    event.stopPropagation();
    event.preventDefault();
  }
  if (!window.confirm(t("confirmDeleteSession"))) return;

  try {
    const resp = await fetch(`${API_BASE}/api/v1/sessions/${id}`, {
      method: "DELETE",
      headers: authHeaders(),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || resp.statusText);
    }
  } catch (e) {
    console.warn("deleteSession failed", e);
  }
  removeSessionFromHistory(id);
}

function renderSessionList() {
  if (!sessionListEl) return;
  const history = getSessionHistory();
  sessionListEl.innerHTML = "";

  if (history.length === 0) {
    const empty = document.createElement("li");
    empty.className = "session-list-empty";
    empty.textContent = t("noHistory");
    sessionListEl.appendChild(empty);
    return;
  }

  history.forEach((item) => {
    const li = document.createElement("li");
    li.className = "session-list-item";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `session-open-btn${item.id === sessionId ? " active" : ""}`;
    btn.innerHTML = `
      <span class="session-preview">${escapeHtml(item.title)}</span>
      <span class="session-id">${escapeHtml(item.id.slice(0, 8))}…</span>
    `;
    btn.addEventListener("click", () => loadSession(item.id));

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "session-delete-btn";
    delBtn.title = t("deleteSession");
    delBtn.setAttribute("aria-label", t("deleteSession"));
    delBtn.textContent = "×";
    delBtn.addEventListener("click", (e) => deleteSession(item.id, e));

    li.appendChild(btn);
    li.appendChild(delBtn);
    sessionListEl.appendChild(li);
  });
}

function renderQuickPrompts() {
  if (!quickPromptsEl) return;
  quickPromptsEl.innerHTML = "";
  const label = document.createElement("span");
  label.className = "quick-label";
  label.textContent = t("quickPrompts");
  quickPromptsEl.appendChild(label);

  getPrompts().forEach((p) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "quick-btn";
    btn.textContent = p.label;
    btn.title = p.text;
    btn.addEventListener("click", () => {
      if (inputEl) {
        inputEl.value = p.text;
        inputEl.focus();
      }
    });
    quickPromptsEl.appendChild(btn);
  });
}

async function loadOrders() {
  if (!ordersPanelEl || !ordersListEl) return;
  if (!currentUser || !authToken) {
    ordersPanelEl.hidden = true;
    return;
  }
  ordersPanelEl.hidden = false;
  ordersListEl.innerHTML = `<li class="orders-loading">…</li>`;
  try {
    const resp = await fetch(`${API_BASE}/api/v1/users/me/orders`, {
      headers: authHeaders(),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);
    const orders = data.data?.orders || [];
    ordersListEl.innerHTML = "";
    if (orders.length === 0) {
      ordersListEl.innerHTML = `<li class="orders-empty">${t("noOrders")}</li>`;
      return;
    }
    orders.forEach((o) => {
      const li = document.createElement("li");
      li.className = "order-item";
      li.innerHTML = `
        <span class="order-id">${escapeHtml(o.order_id || "")}</span>
        <span class="order-status">${escapeHtml(o.status || "")}</span>
        <span class="order-meta">${escapeHtml(o.carrier || "")} ${escapeHtml(o.tracking_number || "")}</span>
      `;
      li.addEventListener("click", () => {
        if (inputEl) {
          inputEl.value = `查询订单 ${o.order_id} 的物流信息`;
          inputEl.focus();
        }
      });
      ordersListEl.appendChild(li);
    });
  } catch {
    ordersListEl.innerHTML = `<li class="orders-empty">${t("ordersLoadFailed")}</li>`;
  }
}

function clearMessages() {
  if (messagesEl) messagesEl.innerHTML = "";
}

function renderMessages(messages) {
  clearMessages();
  messages.forEach((m) => {
    if (m.role === "user" || m.role === "assistant") {
      appendMessage(m.role, m.content || "", m.citations || [], m.tools_used || []);
    }
  });
}

async function loadSession(id) {
  if (!id) return;
  try {
    const resp = await fetch(`${API_BASE}/api/v1/sessions/${id}`, {
      headers: authHeaders(),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);
    persistSessionId(id);
    renderMessages(data.messages || []);
    const firstUser = (data.messages || []).find((m) => m.role === "user");
    upsertSessionHistory(id, firstUser?.content || id);
  } catch (e) {
    removeSessionFromHistory(id);
    renderSessionList();
    console.warn("loadSession failed", e);
  }
}

function updateAuthUI() {
  if (!authStatusEl) return;
  if (currentUser) {
    authStatusEl.textContent = `${t("loggedInAs")}: ${currentUser.email}`;
    if (loginBtn) loginBtn.hidden = true;
    if (logoutBtn) logoutBtn.hidden = false;
    if (loginEmailEl) loginEmailEl.hidden = true;
  } else {
    authStatusEl.textContent = t("notLoggedIn");
    if (loginBtn) loginBtn.hidden = false;
    if (logoutBtn) logoutBtn.hidden = true;
    if (loginEmailEl) loginEmailEl.hidden = false;
    if (ordersPanelEl) ordersPanelEl.hidden = true;
  }
}

async function fetchMe() {
  try {
    const resp = await fetch(`${API_BASE}/api/v1/auth/me`, { headers: authHeaders() });
    const data = await resp.json();
    currentUser = data.authenticated ? data.user : null;
  } catch {
    currentUser = null;
  }
  updateAuthUI();
  await loadOrders();
}

async function login() {
  const email = (loginEmailEl?.value || "").trim();
  if (!email) return;
  try {
    const resp = await fetch(`${API_BASE}/api/v1/auth/login`, {
      method: "POST",
      headers: authHeaders(),
      credentials: "include",
      body: JSON.stringify({ email, name: email.split("@")[0] }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(formatApiDetail(data.detail) || resp.statusText);
    authToken = data.access_token;
    localStorage.setItem("kefu_token", authToken);
    currentUser = data.user;
    updateAuthUI();
    await loadOrders();
    if (ws) {
      ws.close();
      ws = null;
    }
    connectWebSocket();
  } catch (e) {
    alert(`${t("error")}: ${e.message}`);
  }
}

async function logout() {
  try {
    await fetch(`${API_BASE}/api/v1/auth/logout`, {
      method: "POST",
      headers: authHeaders(),
      credentials: "include",
    });
  } catch {
    /* ignore */
  }
  authToken = "";
  localStorage.removeItem("kefu_token");
  currentUser = null;
  updateAuthUI();
  if (ws) {
    ws.close();
    ws = null;
  }
  connectWebSocket();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function sanitizeAssistantContent(text) {
  if (!text) return "";
  let s = String(text);
  s = s.replace(/`[\s\S]*?`/gi, "");
  s = s.replace(/`[\s\S]*?`/gi, "");
  s = s.replace(/<think>[\s\S]*?<\/redacted_thinking>/gi, "");
  s = s.replace(/<think[\s\S]*?<\/redacted_thinking>/gi, "");
  s = s.replace(/<think[\s\S]*?<\/think>/gi, "");
  s = s.replace(/`[\s\S]*$/i, "");
  s = s.replace(/<think>[\s\S]*$/i, "");
  s = s.replace(/<think[\s\S]*$/i, "");
  const leak = s.search(/\n(?:user|assistant)\s*(?:\n|$)/i);
  if (leak >= 0) s = s.slice(0, leak);
  const chatml = s.search(/<\|im_start\|>(?:user|assistant)\b/i);
  if (chatml >= 0) s = s.slice(0, chatml);
  return s.trim();
}

function formatMessageContent(text, role = "user") {
  const clean = role === "assistant" ? sanitizeAssistantContent(text) : text;
  return escapeHtml(clean).replace(/\n/g, "<br>");
}

function formatToolLabel(name) {
  return TOOL_LABELS[name] || name;
}

function buildWsUrl() {
  // Prefer HttpOnly cookie auth — do not put JWT in the query string.
  return `${WS_BASE}${CONFIG.wsPath || "/api/v1/ws/chat"}`;
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectWebSocket();
  }, reconnectDelayMs);
  reconnectDelayMs = Math.min(reconnectDelayMs * 2, RECONNECT_MAX_MS);
}

function connectWebSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  if (ws && ws.readyState === WebSocket.CONNECTING) return;

  connState = "connecting";
  updateConnStatus();
  ws = new WebSocket(buildWsUrl());

  ws.onopen = () => {
    isConnected = true;
    connState = "connected";
    reconnectDelayMs = 1000;
    updateConnStatus();
    if (sendBtn) sendBtn.disabled = false;
  };

  ws.onmessage = (event) => {
    try {
      handleWsMessage(JSON.parse(event.data));
    } catch (e) {
      console.warn("Invalid WS payload", e);
    }
  };

  ws.onclose = () => {
    isConnected = false;
    connState = "disconnected";
    updateConnStatus();
    if (sendBtn && !pendingAssistantEl) sendBtn.disabled = false;
    setCancelEnabled(false);
    scheduleReconnect();
  };

  ws.onerror = () => {
    isConnected = false;
    connState = "disconnected";
    updateConnStatus();
  };
}

function appendToolsBadge(parent, tools) {
  if (!tools || tools.length === 0) return;
  const wrap = document.createElement("div");
  wrap.className = "tools-badge";
  wrap.textContent = `${t("toolsUsed")}: `;
  tools.forEach((name) => {
    const tag = document.createElement("span");
    tag.className = "tool-tag";
    tag.textContent = formatToolLabel(name);
    wrap.appendChild(tag);
  });
  parent.appendChild(wrap);
}

function appendCitations(parent, citations) {
  if (!citations || citations.length === 0) return;
  const ul = document.createElement("ul");
  ul.className = "citations";
  citations.forEach((c) => {
    const li = document.createElement("li");
    li.textContent = `${t("ref")}: ${c}`;
    ul.appendChild(li);
  });
  parent.appendChild(ul);
}

function appendFeedbackControls(parent) {
  if (!parent || parent.querySelector(".feedback-bar")) return;
  const bar = document.createElement("div");
  bar.className = "feedback-bar";
  const good = document.createElement("button");
  good.type = "button";
  good.className = "feedback-btn";
  good.dataset.rating = "5";
  good.textContent = t("feedbackGood");
  const bad = document.createElement("button");
  bad.type = "button";
  bad.className = "feedback-btn";
  bad.dataset.rating = "1";
  bad.textContent = t("feedbackBad");
  bar.appendChild(good);
  bar.appendChild(bad);
  bar.querySelectorAll(".feedback-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (bar.dataset.submitted === "1" || !sessionId) return;
      const rating = Number(btn.dataset.rating);
      try {
        const resp = await fetch(`${API_BASE}/api/v1/chat/feedback`, {
          method: "POST",
          headers: authHeaders(),
          credentials: "include",
          body: JSON.stringify({ session_id: sessionId, rating }),
        });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          throw new Error(formatApiDetail(data.detail) || resp.statusText);
        }
        bar.dataset.submitted = "1";
        bar.querySelectorAll(".feedback-btn").forEach((b) => {
          b.disabled = true;
          b.classList.toggle("selected", b === btn);
        });
      } catch (e) {
        console.warn("feedback failed", e);
      }
    });
  });
  parent.appendChild(bar);
}

function appendMessage(role, content, citations = [], toolsUsed = []) {
  const div = document.createElement("div");
  div.className = `message ${role}`;

  const body = document.createElement("div");
  body.className = "message-body";
  body.innerHTML = formatMessageContent(content, role);
  div.appendChild(body);
  if (role === "assistant") {
    appendToolsBadge(div, toolsUsed);
    appendCitations(div, citations);
    if ((content || "").trim()) appendFeedbackControls(div);
  } else {
    appendCitations(div, citations);
  }

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function handleWsMessage(data) {
  if (data.type === "session") {
    persistSessionId(data.session_id);
    upsertSessionHistory(data.session_id, lastUserMessage);
  } else if (data.type === "chunk" && pendingAssistantEl) {
    if (abortRequested) return;
    const current = pendingAssistantEl.dataset.full || "";
    const full = current + data.content;
    pendingAssistantEl.dataset.full = full;
    const body = pendingAssistantEl.querySelector(".message-body");
    if (body) {
      body.innerHTML = formatMessageContent(full, "assistant");
    }
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } else if (data.type === "cancelled") {
    abortRequested = true;
    setCancelEnabled(false);
    if (pendingAssistantEl) {
      pendingAssistantEl.classList.remove("typing");
      const body = pendingAssistantEl.querySelector(".message-body");
      if (body && !(pendingAssistantEl.dataset.full || "").trim()) {
        body.textContent = t("cancelled") || "已取消";
      } else if (pendingAssistantEl && (pendingAssistantEl.dataset.full || "").trim()) {
        appendFeedbackControls(pendingAssistantEl);
      }
      pendingAssistantEl = null;
    }
    if (sendBtn) sendBtn.disabled = false;
  } else if (data.type === "done") {
    if (abortRequested) {
      setCancelEnabled(false);
      if (sendBtn) sendBtn.disabled = false;
      pendingAssistantEl = null;
      return;
    }
    persistSessionId(data.session_id);
    upsertSessionHistory(data.session_id, lastUserMessage);
    setCancelEnabled(false);
    if (pendingAssistantEl) {
      pendingAssistantEl.classList.remove("typing");
      const body = pendingAssistantEl.querySelector(".message-body");
      if (body) {
        body.innerHTML = formatMessageContent(data.answer || "", "assistant");
      }
      const tools = data.tools_used?.length
        ? data.tools_used
        : data.tool_name
          ? [data.tool_name]
          : [];
      appendToolsBadge(pendingAssistantEl, tools);
      appendCitations(pendingAssistantEl, data.citations);
      if ((data.answer || "").trim()) appendFeedbackControls(pendingAssistantEl);
      pendingAssistantEl = null;
    }
    if (sendBtn) sendBtn.disabled = false;
  } else if (data.type === "error" && pendingAssistantEl) {
    pendingAssistantEl.classList.remove("typing");
    const body = pendingAssistantEl.querySelector(".message-body");
    if (body) {
      body.textContent = `${t("error")}: ${data.error || "Unknown error"}`;
    }
    pendingAssistantEl = null;
    if (sendBtn) sendBtn.disabled = false;
    setCancelEnabled(false);
  }
}

async function sendMessage() {
  const message = inputEl.value.trim();
  if (!message) return;

  abortRequested = false;
  lastUserMessage = message;
  inputEl.value = "";
  sendBtn.disabled = true;
  setCancelEnabled(true);
  appendMessage("user", message);

  pendingAssistantEl = appendMessage("assistant", "");
  pendingAssistantEl.classList.add("typing");
  pendingAssistantEl.dataset.full = "";

  // New chats / cleared session send null so REST creates a session (avoid stale UUID 404).
  const outboundSessionId = sessionId || null;

  if (isConnected && ws) {
    ws.send(JSON.stringify({ message, session_id: outboundSessionId }));
  } else {
    try {
      const resp = await fetch(`${API_BASE}/api/v1/chat`, {
        method: "POST",
        headers: authHeaders(),
        credentials: "include",
        body: JSON.stringify({ message, session_id: outboundSessionId }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(formatApiDetail(data.detail) || resp.statusText);
      if (abortRequested) return;
      persistSessionId(data.session_id);
      upsertSessionHistory(data.session_id, message);
      if (pendingAssistantEl) {
        pendingAssistantEl.classList.remove("typing");
        pendingAssistantEl.remove();
        pendingAssistantEl = null;
      }
      const tools = data.tools_used?.length
        ? data.tools_used
        : data.tool_name
          ? [data.tool_name]
          : [];
      appendMessage("assistant", data.answer, data.citations, tools);
    } catch (e) {
      if (pendingAssistantEl) {
        pendingAssistantEl.classList.remove("typing");
        const body = pendingAssistantEl.querySelector(".message-body");
        if (body) {
          body.textContent = `${t("requestFailed")}: ${e.message}`;
        }
        pendingAssistantEl = null;
      }
    }
    sendBtn.disabled = false;
    setCancelEnabled(false);
  }
}

function formatApiDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return String(detail);
}

function startNewSession() {
  persistSessionId(null);
  clearMessages();
  lastUserMessage = "";
  if (ws) {
    ws.close();
    ws = null;
  }
  connectWebSocket();
}

if (newSessionBtn) {
  newSessionBtn.addEventListener("click", startNewSession);
}

if (loginBtn) loginBtn.addEventListener("click", login);
if (logoutBtn) logoutBtn.addEventListener("click", logout);
if (loginEmailEl) {
  loginEmailEl.value = "demo@gulf.ae";
  loginEmailEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") login();
  });
}

if (sendBtn) {
  sendBtn.addEventListener("click", sendMessage);
}
if (cancelBtn) {
  cancelBtn.addEventListener("click", cancelPending);
}
if (inputEl) {
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

async function initApp() {
  renderSessionList();
  renderQuickPrompts();
  updateConnStatus();
  await fetchMe();
  if (sessionId) {
    await loadSession(sessionId);
  }
  connectWebSocket();
}

initLanguageSwitcher();
initApp();
