# 智能客服（Intelligent Customer Service）

生产级跨境智能客服系统，采用**混合 LLM 架构**：Kimi K2.6 负责 Agent 编排（ReAct 工具调用、答案合成），本地 **Qwen3.5-2B + LoRA** 或云端 Kimi 负责 RAG 知识问答（可选 GPU），配合 FastAPI REST/WebSocket API、管理后台与 Web 聊天界面。

**多语言支持**：客服根据用户输入语言自动匹配回复（中/英/阿拉伯语等）。知识库含中文主文档及 `_en` / `_ar` 后缀多语言副本；检索层支持跨语言查询扩展与语言偏好重排，生成层将知识概括为用户语言后作答。

## 系统架构

```
用户 (Web / API / Admin)
       │
       ▼
  FastAPI (src/main.py)
       │
       ▼
  AgentService
       ├── 问候短路（正则 + Kimi）
       ├── 退换货 Workflow（Redis 状态机，优先于 ReAct）
       └── ReAct 循环（Kimi K2.6 Function Calling，最多 AGENT_MAX_STEPS 步）
              │
              ├── fetch_logistics_information   物流查询
              ├── record_user_complaint         投诉记录 → PostgreSQL
              ├── create_return_request         退换货（内部 RAG）
              ├── customer_chat                 RAG 知识问答
              ├── query_order                   订单查询
              └── query_my_orders               我的订单（需登录）
              │
              ├── 单轮 RAG / 多工具结果 → Kimi 合成
              └── customer_chat / create_return_request → 检索 + 生成
                        │
                        ├── Hybrid Retriever (BM25 + ChromaDB + RRF)
                        ├── 跨语言查询扩展 (multilingual.py)
                        └── 生成: Local Qwen+LoRA（中文）或 Kimi（云端/非中文）
       │
       ▼
  PostgreSQL（会话/消息/投诉/反馈）+ Redis（历史缓存/摘要/Workflow）
```

| 组件 | 技术 | 职责 |
|------|------|------|
| Agent 编排 | Kimi K2.6（Moonshot API） | ReAct 工具选择、多步并行调用、答案合成 |
| 业务 Workflow | Redis 状态机 | 退换货等多轮引导（订单号收集等） |
| RAG 生成 | Qwen3.5-2B + LoRA 或 Kimi | 基于知识库回答产品/政策咨询 |
| 检索 | BM25 + ChromaDB（Hybrid + RRF） | 多语言 chunk 召回与重排 |
| Embedding | BAAI/bge-m3 | 向量检索 |
| 后端 | FastAPI + Uvicorn | REST / WebSocket API |
| 持久化 | PostgreSQL 16 + Redis 7 | 会话、消息、投诉、反馈、知识文档 |
| 前端 | HTML/JS + React Admin | Web 聊天 + 管理后台 |
| 观测 | structlog + Prometheus + OTel（可选） | 日志、指标、链路追踪 |

## 快速开始

### 环境要求

- Python **>= 3.10**
- Docker（PostgreSQL + Redis）
- GPU（可选，本地 RAG 推理，建议 24GB+ 显存）
- [Moonshot API Key](https://platform.moonshot.cn/)（Agent 编排必需）

### 1. 安装依赖

```bash
pip install -e ".[lint,test]"
# 本地 GPU RAG：
pip install -e ".[gpu]"
# 可选链路追踪：
pip install -e ".[otel]"
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写 `MOONSHOT_API_KEY`。本地 RAG 建议：

```bash
QWEN_MODEL_PATH=Qwen/Qwen3.5-2B
LORA_PATH=./models/qwen35_2b_lora
RAG_BACKEND=local
```

### 3. 启动基础设施

```bash
cd docker
docker compose up -d postgres redis
```

### 4. 构建知识库索引

新增或更新 `data/knowledge/` 文档后执行（写入 `lang` 等多语言元数据）：

```bash
python scripts/rebuild_index.py
```

### 5. 启动 API 服务

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
# 或使用 CLI 入口：
kefu
```

启动日志中 `Uvicorn running on http://0.0.0.0:8000` 表示监听所有网卡，浏览器请访问 **http://localhost:8000**，勿使用 `http://0.0.0.0:8000`。

### 6. 访问界面

| 地址 | 说明 |
|------|------|
| http://localhost:8000 | Web 聊天（WebSocket 流式） |
| http://localhost:8000/admin | 管理后台（需先 `frontend-admin` 构建） |
| http://localhost:8000/health | 健康检查 |
| http://localhost:8080 | Nginx 全栈部署入口 |

## API 端点

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/health` | 无 | 健康检查 |
| GET | `/health/deep` | 无 | 深度探针（PostgreSQL / Redis / Kimi） |
| GET | `/config.js` | 无 | 前端运行时配置 |
| POST | `/api/v1/chat` | `X-API-Key` | 同步聊天（限流 30/min） |
| POST | `/api/v1/chat/feedback` | `X-API-Key` | 对话质量反馈（1–5 分） |
| WS | `/api/v1/ws/chat` | `api_key` 或 `X-API-Key` | 流式聊天 |
| GET | `/api/v1/sessions/{id}` | `X-API-Key` | 会话历史 |
| DELETE | `/api/v1/sessions/{id}` | `X-API-Key` | 删除会话 |
| POST | `/api/v1/auth/login` | 无 | 邮箱登录（JWT） |
| GET | `/api/v1/users/me/orders` | Bearer JWT | 我的订单 |
| POST | `/api/v1/knowledge/upload` | `X-API-Key` | 上传知识文档 |
| POST | `/api/v1/knowledge/rebuild` | `X-API-Key` | 重建向量索引 |
| GET | `/api/v1/admin/complaints` | `X-API-Key` | 投诉列表 |
| PATCH | `/api/v1/admin/complaints/{id}` | `X-API-Key` | 投诉状态流转 |
| GET | `/metrics` | `X-API-Key`（可配置） | Prometheus 指标 |

**聊天请求示例：**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me-in-production" \
  -d '{"message": "手机的质保是多久？"}'
```

生产部署：设置 `ENV=production` 与非默认 `API_KEY`；可用 `API_KEY_PREVIOUS` 做 Key 轮换宽限期；可选 `QWEN_INFERENCE_URL` 连接 rag-worker。

### Agent 工具（6 个）

| 工具名 | 说明 |
|--------|------|
| `fetch_logistics_information` | 根据物流单号查询运输状态 |
| `record_user_complaint` | 记录服务质量投诉 → PostgreSQL + 工单号 |
| `create_return_request` | 退换货/退款政策咨询（RAG）；可由 Workflow 直接触发 |
| `customer_chat` | 一般产品/政策咨询（Hybrid RAG） |
| `query_order` | 根据订单号查询详情 |
| `query_my_orders` | 登录用户查询全部订单 |

ReAct 支持**单轮多工具并行调用**（如同时 `query_order` + `fetch_logistics_information`）。

## 项目结构

```
zhineng_kefu/
├── src/
│   ├── main.py                  # FastAPI 入口、OTel、生命周期
│   ├── core/                    # 配置、鉴权、限流、指标、输入防护、追踪
│   ├── api/routes/              # chat / auth / feedback / admin / knowledge …
│   ├── models/                  # session / message / complaint / feedback / user …
│   ├── services/
│   │   ├── agent_service.py     # ReAct Agent 编排核心
│   │   ├── session_service.py   # 会话、Redis 摘要、Workflow 状态
│   │   ├── complaint_service.py # 投诉工单状态流转
│   │   ├── workflows/           # 退换货等业务 Workflow
│   │   └── llm/                 # Kimi、本地/远程 Qwen
│   ├── rag/                     # 分块、Hybrid 检索、多语言、Prompt
│   ├── tools/                   # 6 个业务工具实现
│   └── eval/                    # Golden QA 数据集加载
├── tests/                       # pytest 单元/集成测试
├── data/
│   ├── knowledge/               # 知识库（含 _en / _ar 多语言副本）
│   ├── chroma/                  # ChromaDB 向量索引
│   └── eval/golden_qa.jsonl     # Agent 路由评测用例
├── scripts/
│   ├── rebuild_index.py         # 重建向量索引
│   └── eval_agent.py            # 离线/在线 Agent 评测
├── frontend/                    # Web 聊天 UI
├── frontend-admin/              # React 管理后台
├── models/                      # LoRA 权重（gitignore）
├── docker/                      # Docker Compose、Nginx
└── alembic/                     # 数据库迁移
```

## 数据与模型

### 知识库

| 路径 | 说明 |
|------|------|
| `data/knowledge/` | 主文档（中文）及 `*_en.txt`、`*_ar.txt` 多语言副本 |

核心主题文档包括：退换货、物流、支付、质保、清关、海湾专项、FAQ 等。索引 chunk 携带 `lang` 元数据（`rebuild_index.py` 写入 Chroma）。

### LoRA 权重

位于 `models/qwen35_2b_lora/`，通过 `LORA_PATH` 加载，存在时自动 `merge_and_unload()`。体积较大，已 `.gitignore`，部署时需单独拷贝或挂载。

## 配置参考

配置加载：`src/core/config.py`（Pydantic Settings，读取 `.env`）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MOONSHOT_API_KEY` | — | Kimi API 密钥（别名 `KIMI_API_KEY`） |
| `MOONSHOT_MODEL` | `kimi-k2.6` | Agent 模型（temperature 固定为 1） |
| `QWEN_MODEL_PATH` | `Qwen/Qwen3.5-2B` | 本地 RAG 基座 |
| `LORA_PATH` | `./models/qwen35_2b_lora` | LoRA 权重路径 |
| `RAG_BACKEND` | `local` | `local` 或 `cloud` |
| `RETRIEVER_TYPE` | `hybrid` | `bm25` / `vector` / `hybrid` |
| `RAG_TOP_K` | `3` | 检索条数 |
| `AGENT_MAX_STEPS` | `4` | ReAct 最大步数 |
| `KNOWLEDGE_DIR` | `./data/knowledge` | 知识库目录 |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB 路径 |
| `DATABASE_URL` | `postgresql+asyncpg://…` | PostgreSQL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis |
| `API_KEY` | `change-me-in-production` | REST 鉴权 |
| `API_KEY_PREVIOUS` | 空 | Key 轮换过渡期旧 Key |
| `SESSION_MAX_HISTORY` | `6` | Redis 会话历史条数 |
| `OTEL_ENABLED` | `false` | 启用 OpenTelemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | 空 | OTLP HTTP 端点 |
| `LOGISTICS_API_URL` 等 | 空 | 未配置时开发环境用 Mock |

**数据库迁移：**

```bash
alembic upgrade head
```

应用启动时也会自动 `create_all`。

## Docker 全栈部署

```bash
cd docker
docker compose up -d                    # API + Postgres + Redis + Nginx
docker compose --profile gpu up -d      # 额外 GPU RAG Worker（8001）
```

| 服务 | 端口 | 说明 |
|------|------|------|
| api | 8000 | FastAPI，默认 `RAG_BACKEND=cloud` |
| rag-worker | 8001 | GPU 容器（需 NVIDIA GPU） |
| nginx | 8080 | 静态前端 + API 反向代理 |
| postgres | 内部 | PostgreSQL 16 |
| redis | 内部 | Redis 7 |

远程 RAG：`QWEN_INFERENCE_URL=http://rag-worker:8001`

## 开发与运维

```bash
# 重建向量索引（多语言 lang 元数据）
python scripts/rebuild_index.py

# Agent 路由离线评测
python scripts/eval_agent.py
python scripts/eval_agent.py --live   # 需 MOONSHOT_API_KEY

# 测试
pytest tests -v

# 代码检查
ruff check src tests
```

### 依赖分组（pyproject.toml）

| 分组 | 安装命令 | 主要包 |
|------|----------|--------|
| 核心 | `pip install -e .` | fastapi, sqlalchemy, chromadb, openai, redis … |
| gpu | `pip install -e ".[gpu]"` | torch, transformers, peft, modelscope |
| test | `pip install -e ".[test]"` | pytest, httpx, aiosqlite |
| lint | `pip install -e ".[lint]"` | ruff |
| otel | `pip install -e ".[otel]"` | opentelemetry-sdk, instrumentation |

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| Web | FastAPI + Uvicorn |
| Agent | Kimi K2.6（ReAct + Function Calling） |
| 本地 LLM | Qwen3.5-2B + PEFT LoRA |
| 检索 | rank-bm25 + ChromaDB + RRF 融合 |
| Embedding | BAAI/bge-m3 |
| 数据库 | PostgreSQL 16 + SQLAlchemy 2.0 async |
| 缓存 | Redis 7（历史、摘要、Workflow） |
| 观测 | structlog + Prometheus + OpenTelemetry（可选） |
| 安全 | API Key 轮换、输入注入检测、JWT、slowapi 限流 |

## 生产就绪检查清单

| 类别 | 项目 | 状态 |
|------|------|------|
| **安全** | API Key + 轮换宽限期（`API_KEY_PREVIOUS`） | ✅ |
| | Prompt 注入启发式检测（`input_guard`） | ✅ |
| | WebSocket / REST 鉴权、限流 | ✅ |
| | 生产环境强制非默认 `API_KEY` | ✅ |
| **Agent** | ReAct 多步 + 并行工具调用 | ✅ |
| | 退换货 Workflow 状态机 | ✅ |
| | Few-shot 工具选择提示 | ✅ |
| | Golden QA + `eval_agent.py` | ✅ |
| **RAG** | Hybrid 检索 + 跨语言查询扩展 | ✅ |
| | chunk `lang` 元数据 + `_en`/`_ar` 文档 | ✅ |
| | 本地 Qwen 真流式 / Kimi 流式 RAG | ✅ |
| **会话** | 多轮历史 + Redis 滚动摘要 | ✅ |
| | 对话反馈 API | ✅ |
| **业务** | 6 工具 + Mock/生产 fail-fast | ✅ |
| | 投诉工单状态流转 + Webhook | ✅ |
| **运维** | `/health/deep`、Prometheus、可选 OTel | ✅ |
| | pytest 测试套件（`tests/`） | ✅ |
| | Docker 全栈 + GPU Worker | ✅ |

## 知识库文档（`data/knowledge/`）

除中文主文档外，以下主题提供 **英文**（`*_en.txt`）与 **阿拉伯语**（`*_ar.txt`）副本：

- `returns_refund`、`shipping_cross_border`、`gulf_warranty`
- `payment_currency`、`faq_contact`、`customs_clearance`

另有中文专题文档：订单追踪、投诉流程、质保除外、海湾专项、各类 mined FAQ 等。

新增文档后执行 `python scripts/rebuild_index.py` 重建索引。

## 注意事项

1. **模型版本**：生产默认 **Qwen3.5-2B + LoRA**；`RAG_BACKEND=cloud` 时 RAG 由 Kimi 生成。
2. **WebSocket 鉴权**：`?api_key=` 或 `X-API-Key`，与 REST 一致；可选 `token` 绑定用户。
3. **流式输出**：问候与 Kimi 路径为原生 token 流；本地 Qwen 中文 RAG 为 `TextIteratorStreamer` 真流式；Workflow 回复按 20 字符分块推送。
4. **LoRA 权重**：约 6GB，`.gitignore`；部署需单独挂载 `models/`。
5. **OpenTelemetry**：设置 `OTEL_ENABLED=true` 并安装 `[otel]` 依赖后生效。

## License

MIT
