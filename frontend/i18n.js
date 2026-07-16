const I18N = {
  zh: {
    title: "智能客服",
    newSession: "新对话",
    send: "发送",
    placeholder: "输入您的问题…",
    session: "会话",
    ref: "参考",
    error: "错误",
    requestFailed: "请求失败",
    cancelled: "已取消",
    feedbackGood: "有用",
    feedbackBad: "无用",
    login: "登录",
    logout: "退出",
    notLoggedIn: "未登录",
    loggedInAs: "已登录",
    loginPlaceholder: "email@gulf.ae",
    historySessions: "历史会话",
    noHistory: "暂无历史",
    deleteSession: "删除",
    confirmDeleteSession: "确定删除该会话？此操作不可恢复。",
    myOrders: "我的订单",
    noOrders: "暂无订单",
    ordersLoadFailed: "订单加载失败",
    connConnected: "已连接",
    connConnecting: "连接中…",
    connDisconnected: "已断开，重连中…",
    toolsUsed: "调用工具",
    quickPrompts: "快捷提问",
    adminLink: "管理后台",
    prompts: [
      { label: "质保政策", text: "手机质保多久？" },
      { label: "我的订单", text: "我的订单有哪些？" },
      { label: "查物流", text: "帮我查一下运单号 ARX123456789 的物流" },
      { label: "退货退款", text: "我想申请退货退款，流程是什么？" },
      { label: "投诉", text: "我要投诉，客服态度很差" },
    ],
  },
  en: {
    title: "Customer Service",
    newSession: "New Chat",
    send: "Send",
    placeholder: "Type your question…",
    session: "Session",
    ref: "Ref",
    error: "Error",
    requestFailed: "Request failed",
    cancelled: "Cancelled",
    feedbackGood: "Helpful",
    feedbackBad: "Unhelpful",
    login: "Log in",
    logout: "Log out",
    notLoggedIn: "Not signed in",
    loggedInAs: "Signed in",
    loginPlaceholder: "email@gulf.ae",
    historySessions: "History",
    noHistory: "No history yet",
    deleteSession: "Delete",
    confirmDeleteSession: "Delete this conversation? This cannot be undone.",
    myOrders: "My Orders",
    noOrders: "No orders",
    ordersLoadFailed: "Failed to load orders",
    connConnected: "Connected",
    connConnecting: "Connecting…",
    connDisconnected: "Disconnected, reconnecting…",
    toolsUsed: "Tools",
    quickPrompts: "Quick prompts",
    adminLink: "Admin",
    prompts: [
      { label: "Warranty", text: "What is the warranty period for phones?" },
      { label: "My orders", text: "What are my orders?" },
      { label: "Tracking", text: "Track shipment ARX123456789" },
      { label: "Returns", text: "How do I request a return or refund?" },
      { label: "Complaint", text: "I want to file a complaint about poor service" },
    ],
  },
  ar: {
    title: "خدمة العملاء",
    newSession: "محادثة جديدة",
    send: "إرسال",
    placeholder: "اكتب سؤالك…",
    session: "الجلسة",
    ref: "مرجع",
    error: "خطأ",
    requestFailed: "فشل الطلب",
    cancelled: "تم الإلغاء",
    feedbackGood: "مفيد",
    feedbackBad: "غير مفيد",
    login: "تسجيل الدخول",
    logout: "خروج",
    notLoggedIn: "غير مسجل",
    loggedInAs: "مسجل",
    loginPlaceholder: "email@gulf.ae",
    historySessions: "السجل",
    noHistory: "لا يوجد سجل",
    deleteSession: "حذف",
    confirmDeleteSession: "حذف هذه المحادثة؟ لا يمكن التراجع.",
    myOrders: "طلباتي",
    noOrders: "لا توجد طلبات",
    ordersLoadFailed: "فشل تحميل الطلبات",
    connConnected: "متصل",
    connConnecting: "جاري الاتصال…",
    connDisconnected: "غير متصل، إعادة الاتصال…",
    toolsUsed: "الأدوات",
    quickPrompts: "أسئلة سريعة",
    adminLink: "الإدارة",
    prompts: [
      { label: "الضمان", text: "ما مدة ضمان الهاتف؟" },
      { label: "طلباتي", text: "ما هي طلباتي؟" },
      { label: "الشحن", text: "تتبع الشحنة ARX123456789" },
      { label: "الإرجاع", text: "كيف أطلب إرجاع أو استرداد؟" },
      { label: "شكوى", text: "أريد تقديم شكوى بسبب سوء الخدمة" },
    ],
  },
};

let currentLang = localStorage.getItem("kefu-lang") || "zh";

function t(key) {
  return (I18N[currentLang] || I18N.zh)[key] || I18N.zh[key] || key;
}

function getPrompts() {
  return (I18N[currentLang] || I18N.zh).prompts || I18N.zh.prompts;
}

function applyLanguage(lang) {
  currentLang = lang;
  localStorage.setItem("kefu-lang", lang);
  document.documentElement.lang = lang === "ar" ? "ar" : lang === "en" ? "en" : "zh-CN";
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  document.body.classList.toggle("rtl", lang === "ar");

  const titleEl = document.querySelector(".sidebar h1");
  if (titleEl) titleEl.textContent = t("title");
  const newBtn = document.getElementById("newSession");
  if (newBtn) newBtn.textContent = t("newSession");
  const sendBtn = document.getElementById("send");
  if (sendBtn) sendBtn.textContent = t("send");
  const input = document.getElementById("input");
  if (input) input.placeholder = t("placeholder");
  const loginBtn = document.getElementById("loginBtn");
  if (loginBtn) loginBtn.textContent = t("login");
  const logoutBtn = document.getElementById("logoutBtn");
  if (logoutBtn) logoutBtn.textContent = t("logout");
  const loginEmail = document.getElementById("loginEmail");
  if (loginEmail) loginEmail.placeholder = t("loginPlaceholder");
  const historyTitle = document.querySelector(".session-list-wrap .sidebar-subtitle");
  if (historyTitle) historyTitle.textContent = t("historySessions");
  const ordersTitle = document.getElementById("ordersTitle");
  if (ordersTitle) ordersTitle.textContent = t("myOrders");
  const adminLink = document.querySelector(".admin-link");
  if (adminLink) adminLink.title = t("adminLink");
  if (typeof renderSessionList === "function") renderSessionList();
  if (typeof renderQuickPrompts === "function") renderQuickPrompts();
  if (typeof updateConnStatus === "function") updateConnStatus();

  document.querySelectorAll(".lang-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.lang === lang);
  });
}

function initLanguageSwitcher() {
  const sidebar = document.querySelector(".sidebar");
  if (!sidebar || document.getElementById("langSwitcher")) return;

  const wrap = document.createElement("div");
  wrap.id = "langSwitcher";
  wrap.className = "lang-switcher";
  wrap.innerHTML = `
    <button type="button" class="lang-btn" data-lang="zh">中文</button>
    <button type="button" class="lang-btn" data-lang="en">EN</button>
    <button type="button" class="lang-btn" data-lang="ar">عربي</button>
  `;
  const newSession = sidebar.querySelector("#newSession");
  if (newSession) sidebar.insertBefore(wrap, newSession);

  wrap.querySelectorAll(".lang-btn").forEach((btn) => {
    btn.addEventListener("click", () => applyLanguage(btn.dataset.lang));
  });

  applyLanguage(currentLang);
}
