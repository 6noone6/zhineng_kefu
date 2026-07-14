from prometheus_client import Counter, Histogram

CHAT_REQUESTS = Counter(
    "kefu_chat_requests_total",
    "Total chat requests",
    ["method"],
)
TOOL_CALLS = Counter(
    "kefu_tool_calls_total",
    "Tool invocations",
    ["tool_name"],
)
TOOL_SELECTION = Counter(
    "kefu_tool_selection_total",
    "Tool selected by agent",
    ["tool_name"],
)
SESSIONS_CREATED = Counter(
    "kefu_sessions_created_total",
    "New chat sessions created",
)
COMPLAINTS_RECORDED = Counter(
    "kefu_complaints_recorded_total",
    "User complaints recorded",
)
RAG_QUERIES = Counter(
    "kefu_rag_queries_total",
    "RAG knowledge queries",
)
RAG_HIT_RATE = Counter(
    "kefu_rag_hit_total",
    "RAG retrieval hit/miss",
    ["hit"],
)
KNOWLEDGE_UPLOADS = Counter(
    "kefu_knowledge_uploads_total",
    "Knowledge document uploads",
)
RESPONSE_LATENCY = Histogram(
    "kefu_response_latency_seconds",
    "End-to-end agent response latency",
    ["method"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
TOOL_LATENCY = Histogram(
    "kefu_tool_latency_seconds",
    "Tool execution latency",
    ["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
RAG_LATENCY = Histogram(
    "kefu_rag_latency_seconds",
    "RAG retrieval latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
LLM_LATENCY = Histogram(
    "kefu_llm_latency_seconds",
    "LLM API call latency",
    ["operation"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
KIMI_TOKENS = Counter(
    "kefu_kimi_tokens_total",
    "Kimi API token usage",
    ["token_type"],
)
