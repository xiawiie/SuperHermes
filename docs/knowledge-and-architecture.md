# SuperHermes 项目完整知识梳理与架构解析

## 一、项目定位

SuperHermes 是一个**企业级 RAG（检索增强生成）文档问答助手**，核心能力是：用户上传 PDF/Word/Excel 文档，系统自动分块、向量化、索引到 Milvus，然后通过多阶段检索流水线（混合检索 + 重排序 + 置信度门控）回答用户问题，支持实时 SSE 流式输出。

## 二、技术栈全景

| 层级 | 技术 |
|------|------|
| **后端框架** | FastAPI + Uvicorn（ASGI） |
| **RAG 框架** | LangChain + LangGraph（StateGraph） |
| **Embedding** | BAAI/bge-m3（dense 1024维）+ 自定义 BM25 sparse（jieba 分词） |
| **向量数据库** | Milvus v2.5（HNSW dense + SPARSE_INVERTED_INDEX sparse，RRF 融合） |
| **重排序** | BAAI/bge-reranker-v2-m3 CrossEncoder（本地 GPU/CPU） |
| **关系数据库** | PostgreSQL 15（用户/会话/父块存储）+ SQLite 降级 |
| **缓存** | Redis 7（会话缓存/父文档缓存/重排序缓存/文件名注册表） |
| **LLM** | OpenAI 兼容 API（ARK API Key） |
| **前端** | Vue 3 CDN + marked + highlight.js |
| **包管理** | uv（Python 3.12） |

## 三、系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Vue 3 CDN)                    │
│              index.html / script.js / style.css              │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / SSE
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                        │
│  routers/: auth.py / chat.py / sessions.py / documents.py   │
│                         / rag.py                             │
└───────────┬─────────────────────────────────────┬───────────┘
            │                                     │
            ▼                                     ▼
┌───────────────────────┐           ┌─────────────────────────┐
│   Chat Agent 层        │           │   Document Service      │
│  agent.py (会话管理)    │           │  document_service.py    │
│  tools.py (工具定义)    │           │  loader.py (文档分块)    │
│  deep_mode.py (深度)    │           │                         │
└───────────┬───────────┘           └────────────┬────────────┘
            │                                      │
            ▼                                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Engine (核心)                          │
│                                                             │
│  pipeline.py ── LangGraph StateGraph                        │
│       │   retrieve_initial → grade → rewrite → expand       │
│       ▼                                                      │
│  utils.py ── 1900 行检索编排器                                │
│       │                                                      │
│       ├── modes.py          模式分类 (FAST/STANDARD/DEEP)     │
│       ├── query_plan.py     查询规划 (文件匹配/作用域路由)      │
│       ├── retrieval.py      RRF 融合/文件名加权/去重          │
│       ├── rerank.py         CrossEncoder 重排序+分数融合      │
│       ├── rerank_policy.py  神经重排序策略路由                 │
│       ├── layered_rerank.py 4 层分层重排序                    │
│       ├── confidence.py     置信度门控                        │
│       ├── context.py        自动合并/结构重排序               │
│       └── diagnostics.py    失败分类                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure 层                          │
│                                                             │
│  embedding.py     Dense(BAAI/bge-m3) + Sparse(BM25+jieba)   │
│  milvus_client.py Milvus 混合检索 + 重连机制                  │
│  parent_chunk_store.py  父块存储 (PostgreSQL + Redis)        │
│  conversation_storage.py 会话持久化                          │
│  cache.py         Redis 缓存封装                             │
│  database.py      SQLAlchemy 引擎                            │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 目录结构

```
SuperHermes/
  .env / .env.example          -- 环境变量配置
  pyproject.toml / uv.lock     -- 依赖管理 (uv)
  docker-compose.yml            -- PostgreSQL, Redis, Milvus (etcd+minio+standalone+attu)

  backend/
    app.py                      -- ASGI 入口 (uvicorn)
    api.py                      -- 路由注册 (auth, chat, sessions, documents, rag)
    config.py                   -- 环境变量加载 + RAG Profile 解析

    application/
      main.py                   -- FastAPI app 工厂, CORS, 静态挂载, lifespan

    chat/
      agent.py                  -- LangChain Agent, 流式/非流式聊天, 会话记忆
      deep_mode.py              -- Deep Mode 规划控制器 (多跳查询)
      tools.py                  -- @tool 定义: weather, knowledge_base; RAG 步骤发射器

    contracts/
      schemas.py                -- 所有 API 请求/响应的 Pydantic 模型

    documents/
      loader.py                 -- DocumentLoader: PDF/Word/Excel 解析, 3 级分块

    evaluation/
      answer_eval.py            -- 离线答案评估辅助

    infra/
      cache.py                  -- Redis 缓存封装
      embedding.py              -- EmbeddingService: dense (BAAI/bge-m3) + BM25 sparse
      db/
        database.py             -- SQLAlchemy engine/session
        models.py               -- ORM: User, ChatSession, ChatMessage, ParentChunk
        conversation_storage.py -- 聊天历史持久化 + Redis 读穿透缓存
      vector_store/
        milvus_client.py        -- MilvusManager: hybrid/split/dense 检索, 重连逻辑
        milvus_writer.py        -- MilvusWriter: 向量写入编排
        parent_chunk_store.py   -- ParentChunkStore: L1/L2 父文档 (PostgreSQL+Redis)

    rag/                        -- RAG 核心引擎
      pipeline.py               -- LangGraph StateGraph
      utils.py                  -- ~1900 行编排器: retrieve_documents(), embedding, rerank
      modes.py                  -- 模式分类器: FAST/STANDARD/DEEP
      query_plan.py             -- QueryPlan 解析, 文件名匹配, scope 路由
      retrieval.py              -- 加权 RRF 合并, 文件名加权, 标题词法评分, 去重
      rerank.py                 -- CrossEncoder 重排序, pair enrichment, 分数融合
      rerank_policy.py          -- 神经重排序策略: auto/force/off, 规则+LLM 路由
      layered_rerank.py         -- 3-slot 分层 L1 预过滤, L2 自适应 K, L3 root cap
      confidence.py             -- 检索置信度门控, 锚点匹配, post-CE 风险评分
      context.py                -- 自动合并 (L3→L1), 结构重排序 (root 加权, per-root cap)
      diagnostics.py            -- 失败分类 (file_recall_miss, page_miss, ranking_miss 等)
      profiles.py               -- 索引 Profile 规范化, chunk ID 前缀
      profile_config.py         -- K/I/M/A Profile 解析, 设备策略 (GPU/CPU/auto)
      trace.py                  -- 候选身份, 文本哈希, 黄金追踪签名
      types.py                  -- StageError dataclass

    routers/
      auth.py                   -- /auth/register, /auth/login, /auth/me
      chat.py                   -- /chat, /chat/stream (SSE)
      documents.py              -- /documents (GET/POST/DELETE)
      rag.py                    -- /rag/status (管理诊断)
      sessions.py               -- /sessions (GET/DELETE)

    security/
      auth.py                   -- JWT, PBKDF2+bcrypt 密码哈希, RBAC 依赖

    services/
      document_service.py       -- 上传/列表/删除编排 (staging, BM25 同步)

    shared/
      filename_normalization.py -- 文件名规范化
      filename_utils.py         -- 去重, 最大数量辅助
      json_utils.py             -- 从 LLM 输出提取 JSON

  frontend/
    index.html                  -- Vue 3 CDN SPA
    script.js                   -- 聊天 UI, SSE 流式, thinking 状态机
    style.css                   -- 样式
    src/
      api.js                    -- API 客户端
      messages.js               -- 消息格式化

  docs/                         -- 文档
  scripts/                      -- 15+ 评估/分析脚本
  tests/                        -- 50+ 测试文件
  data/                         -- BM25 state JSON, SQLite 降级 DB
  eval/                         -- 评估数据集、报告、基线
  volumes/                      -- Docker 卷挂载 (minio, postgres 等)
```

## 四、文档摄入流水线（Ingestion Pipeline）

```
PDF/Word/Excel 文件
       │
       ▼
  PyPDFLoader / Docx2txtLoader / UnstructuredExcelLoader
       │
       ▼
  DocumentLoader._detect_profile()  ←── 采样前3页，检测文档类型
       │                            ←── ≥3个标题命中 or ≥8%标题率 → "structured"
       │                            ←── 否则 → "generic"
       ▼
┌─────────────────────────────────────────────────┐
│            3 级分层分块 (Hierarchy Chunking)       │
│                                                   │
│  L1 (root):   ~1200+ 字符，代表整个章节/页面        │
│  L3 (leaf):   ~320+ 字符，是 root 的子切片         │
│                                                   │
│  每个 leaf 通过 parent_chunk_id / root_chunk_id   │
│  关联到其父块和根块                                  │
│                                                   │
│  structured 模式:                                  │
│    检测中文/十进制标题层级 → 维护 heading_stack      │
│    → 按 section 切分 → root + leaf                 │
│                                                   │
│  generic 模式:                                     │
│    每页独立切分为 root + leaf                       │
└───────────┬──────────────────────┬────────────────┘
            │                      │
            ▼                      ▼
    L1/L2 父块              L3 叶块
            │                      │
            ▼                      ▼
  PostgreSQL + Redis        EmbeddingService
  (ParentChunkStore)        ├─ Dense: BAAI/bge-m3 (1024维)
                             └─ Sparse: BM25 + jieba 分词
                                      │
                                      ▼
                               Milvus 向量库
                          (HNSW dense + SPARSE_INVERTED sparse)
```

### 4.1 Milvus Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT64 | 主键, auto_id |
| `dense_embedding` | FLOAT_VECTOR | 1024维 dense 向量 |
| `sparse_embedding` | SPARSE_FLOAT_VECTOR | BM25 稀疏向量 |
| `text` | VARCHAR(2000) | 原始块文本 |
| `retrieval_text` | VARCHAR(4000) | 增强检索文本 |
| `filename` | VARCHAR(255) | 源文件名 |
| `page_number`, `page_start`, `page_end` | INT64 | 页码位置 |
| `chunk_id`, `parent_chunk_id`, `root_chunk_id` | VARCHAR(512) | 块层级关系 |
| `chunk_level` | INT64 | 层级 (1=根, 3=叶) |
| `section_title`, `section_path` | VARCHAR | 标题/路径 |
| `anchor_id` | VARCHAR | 结构锚点 |

**索引配置**: Dense = HNSW (M=16, efConstruction=256, IP); Sparse = SPARSE_INVERTED_INDEX (drop_ratio_build=0.2, IP)

**关键设计**: 只有 L3 叶块进入 Milvus 检索；L1/L2 父块存储在 PostgreSQL + Redis 中，用于检索后的自动合并扩展。

## 五、RAG 工作流（核心重点）

### 5.1 LangGraph 状态图总览

```
                    ┌──────────────────┐
                    │  retrieve_initial │
                    │  (初始检索)       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ grade_documents  │
                    │ (文档相关性评分)   │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    │  conditional     │
                    │  route?          │
                    └───┬─────────┬───┘
                        │         │
           "generate_answer"   "rewrite_question"
                        │         │
                        ▼         ▼
                      [END]  ┌──────────────────┐
                             │ rewrite_question  │
                             │ (查询改写)         │
                             └────────┬─────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │ retrieve_expanded │
                             │ (扩展检索)         │
                             └────────┬─────────┘
                                      │
                                      ▼
                                    [END]
```

### 5.2 阶段 1: retrieve_initial（初始检索）

这是最复杂的阶段，内部有完整的子流水线：

```
用户问题 (raw_query)
       │
       ▼
  ① QueryPlan 解析 (query_plan.py)
       │  ├── 提取《》书名 → doc_hints
       │  ├── 提取型号/锚点/章节引用
       │  ├── 匹配 Milvus 文件名注册表（三级缓存: 进程→Redis→Milvus）
       │  ├── 计算 scope_mode:
       │  │     "filter" → 精确匹配到单文件（硬约束）
       │  │     "boost"  → 多文件或模糊匹配（软加权）
       │  │     "none"   → 无匹配（全局检索）
       │  └── 生成 semantic_query（去除书名前缀，保留语义核心）
       │
       ▼
  ② 模式分类 (modes.py) — 零 LLM 纯规则分类器
       │  ├── 提取 15 维查询特征 (QueryFeatures)
       │  │     实体数、比较意图、时间意图、否定、锚点等
       │  ├── 计算 fast_score / deep_score
       │  └── 分类结果:
       │        FAST:     单事实/定位/引用，fast_score >= 0.75
       │        STANDARD: 默认，均衡检索
       │        DEEP:     比较/穷举/时间演进，deep_score >= 0.40
       │
       ▼
  ③ 向量化 (embedding.py) — 并行执行
       │  ├── Dense:  BAAI/bge-m3 → 1024维向量
       │  └── Sparse: BM25 + jieba → 稀疏向量
       │        (ThreadPoolExecutor 并行计算)
       │
       ▼
  ④ Milvus 检索 — 三条路径
       │
       ├─── [Scoped 路径] scope_mode == "filter"
       │    ├── 并行执行: scoped检索(带filter) + global检索(无filter)
       │    ├── 比较 scoped vs global 的 top_score
       │    ├── 选择更优集合 或 加权 RRF 合并
       │    └── 优先使用 scoped，但如果全局有更好的结果则不固执
       │
       ├─── [Global 路径] scope_mode == "none" 或 "boost"
       │    ├── hybrid_retrieve(): dense + sparse AnnSearchRequest
       │    │   → Milvus RRFRanker(k=60) 融合
       │    └── 如果 scope == "boost": 应用文件名加权
       │
       └─── [Layered 候选策略] RAG_CANDIDATE_STRATEGY=layered 时
            ├── split_retrieve(): 分离 dense/sparse 检索
            │   各自独立返回带分数的候选集
            ├── 添加 hybrid_guarantee 候选（确保混合策略的覆盖）
            └── 添加 fallback_pool（兜底候选集）
       │
       ▼
  ⑤ RRF 融合 (retrieval.py)
       │  weighted_rrf_merge(): 多路结果按权重融合
       │  score = Σ weight_i / (rrf_k + rank_i)
       │  可选: filename_boost / heading_lexical_scoring
       │
       ▼
  ⑥ 候选策略 + 共享重排序流水线
       │
       ├─── [候选层] standard / layered
       │    │
       │    ├── layered L1 Prefilter (3-Slot 架构):
       │    │   ├── Slot C (保证槽): 锚点块 + scope 匹配文件块 (≤20)
       │    │   ├── Slot A (文件感知槽): 按文件聚合评分，top12 文件各取 top-N 块 (≥18)
       │    │   └── Slot B (路径保证槽): dense-only + sparse-only + metadata 块 (≥6)
       │    │   → 合并去重 → L1 候选集 (30~40 个)
       │    │
       │    └── standard 直接输出 scoped/global hybrid 候选池
       │
       └─── [共享 L2/L3]
            ├── rerank.py 是唯一 CrossEncoder 重排序实现
            ├── shared-rerank-v2 统一 pair enrichment / cache / 分数融合
            └── shared-postprocess-v1 统一结构 rerank / confidence / trace 收口
       │
       ▼
  ⑦ 置信度门控 (confidence.py)
       │  ├── Pre-CE 评估:
       │  │   top_margin / dominant_root_share / anchor_match
       │  │   → fallback_required = True/False
       │  │
       │  ├── Post-CE 风险评分:
       │  │   top_score_low (2.0) + ce_margin_small (1.5)
       │  │   + evidence_count_low (2.0) + scope_unmatched (2.0)
       │  │
       │  └── 裁决:
       │       risk < 2.0 → 高置信，可回答
       │       risk < 4.0 → 可回答
       │       risk < 6.0 → 不可回答，建议 DEEP 模式
       │       risk ≥ 6.0 → 不可回答，需要用户澄清
       │
       ▼
  ⑧ 可选: FAST → STANDARD 降级重试
       │  如果 FAST 模式下置信度不足
       │  自动用 STANDARD 参数重新检索
       │
       ▼
  返回 {docs, rag_trace}  → 进入 grade_documents
```

### 5.3 阶段 2: grade_documents（文档评分决策）

```
grade_documents_node(state)
       │
       ├── RAG_FALLBACK_ENABLED == False? → 直接 route="generate_answer"
       │
       ├── rag_trace.fallback_required == False? → route="generate_answer"
       │
       ├── 高置信检测:
       │   top_score >= 0.85 AND top_margin >= 0.08 AND scope干净
       │   OR 唯一文件名匹配 AND top_score >= 0.75
       │   → route="generate_answer"
       │
       ├── fallback_required == True? → route="rewrite_question"
       │
       └── LLM 评分（默认路径）:
            GRADE_PROMPT + GradeDocuments 结构化输出
            ├── binary_score="yes"   → route="generate_answer"
            └── binary_score="no"    → route="rewrite_question"
```

### 5.4 阶段 3: rewrite_question（查询改写）

```
rewrite_question_node(state)
       │
       ├── Deadline 管理 (默认 6 秒超时)
       │
       ├── 策略选择 LLM:
       │   "step_back" → 生成更广泛的问题+答案
       │   "hyde"      → 生成假设性文档
       │   "complex"   → 同时执行 step_back + hyde
       │
       ├── 执行改写 (ThreadPoolExecutor + deadline):
       │   step_back_expand(query) → {step_back_question, step_back_answer}
       │   generate_hypothetical_document(query) → 假设性文档文本
       │
       └── 返回 {expansion_type, expanded_query, ...}
```

### 5.5 阶段 4: retrieve_expanded（扩展检索）

```
retrieve_expanded(state)
       │
       ├── strategy == "timeout" → 降级到初始检索结果
       │
       ├── strategy == "complex":
       │   并行执行 HyDE检索 + Step-Back检索
       │   → 合并去重
       │
       ├── strategy == "hyde":
       │   用假设性文档作为 query 检索
       │
       ├── strategy == "step_back":
       │   用 step-back 问题检索
       │
       └── 所有路径最终: _retrieve_with_rerank_mode()
           → 重新走一遍 embedding → Milvus → rerank 流水线
```

## 六、Agentic RAG 工作流（深度模式）

### 6.1 两种 RAG 触发路径

SuperHermes 有两条 RAG 触发路径，对应不同程度的"Agentic"：

```
┌─────────────────────────────────────────────────────┐
│              用户消息 (user_text)                      │
└──────────────────────┬──────────────────────────────┘
                       │
               ┌───────┴───────┐
               │ context_files? │
               └───┬───────┬───┘
                   │       │
          [有文件附件]   [无文件附件]
                   │       │
                   ▼       ▼
        ┌──────────┐  ┌──────────────────┐
        │ 路径 A    │  │ 路径 B            │
        │ 直接调用  │  │ Agent 自主决策     │
        │          │  │                  │
        │ RAG→System│  │ Agent + Tools    │
        │ Message   │  │ search_knowledge │
        │ →LLM生成  │  │ _base (自主调用)  │
        └──────────┘  └──────────────────┘
```

**路径 A**（有 context_files）：检索结果直接注入 SystemMessage，不经过 Agent 工具循环，LLM 直接基于上下文生成答案。

**路径 B**（无 context_files）：Agent 自主决定是否调用 `search_knowledge_base` 工具。Agent 可以先思考，然后主动检索，再基于检索结果回答。这是真正的 **Agentic RAG**。

### 6.2 Deep Mode 控制器（Agentic RAG 核心）

当查询被分类为 DEEP 模式时（比较/穷举/时间演进类问题），触发 Deep Mode 控制器：

```
用户问题: "A产品和B产品的性能参数对比"
       │
       ▼
  ① 查询分解 (decompose_query)
       │
       ├── 比较 (compare):
       │   SubQuery 1: "A产品的性能参数" → target_entity=A, aspect=性能参数
       │   SubQuery 2: "B产品的性能参数" → target_entity=B, aspect=性能参数
       │
       ├── 时间演进 (temporal):
       │   SubQuery 1: "X最早的状态" (earliest_state)
       │   SubQuery 2: "X最新的状态" (latest_state)
       │   SubQuery 3: "X的变化信号" (change_signal) [可选]
       │
       ├── 穷举 (exhaustive):
       │   SubQuery 1: 广泛查询 (broad)
       │   SubQuery 2..N: 每个实体的定向查询 [可选]
       │
       └── 兜底: 单个广泛查询
       │
       ▼
  ② 预算管理 (BudgetTracker)
       │  ├── max_retrieval_calls: 3 (最多 3 次检索调用)
       │  ├── max_wall_time_ms: 6000 (6 秒总时限)
       │  ├── max_evidence_items: 15
       │  └── max_chars_per_item: 600
       │
       ▼
  ③ 逐个子查询执行 (循环)
       │  for each subquery:
       │    ├── 检查预算 (时间/调用次数)
       │    ├── retrieve_fn(sq.query, mode_override="STANDARD", allow_deep=False)
       │    │   ↑ 注意: allow_deep=False 防止递归
       │    │   ↑ 每个子查询走完整的 RAG 流水线
       │    └── accumulate_evidence(): 去重合并到证据池
       │       dedupe_key = (filename, page, text_hash)
       │       合并 target_entities/aspects/route_sources
       │
       ▼
  ④ 覆盖率评估 (evaluate_coverage)
       │  ├── 检查每个 target_entity 是否有证据
       │  ├── 检查每个 target_aspect 是否有证据
       │  ├── 计算 coverage_ratio
       │  └── 识别 missing_entities / missing_aspects
       │
       ▼
  ⑤ 返回 DeepModeResult
       {final_answer: "", citations: [], evidence_items: [...],
        coverage: 0.xx, missing_entities: [...], partial: True/False,
        budgets: {...}, trace: {...}}
       │
       ▼
  ⑥ 引用验证 (verify_citations) [可选]
       检查最终答案中的引用是否与证据一致
```

### 6.3 完整的 Agentic 交互流程

```
┌──────────────────────────────────────────────────────────────┐
│                   SSE 流式交互架构                              │
│                                                              │
│  Frontend (script.js)                                        │
│       │                                                      │
│       │  POST /chat/stream                                   │
│       ▼                                                      │
│  chat_with_agent_stream()                                    │
│       │                                                      │
│       ├── set_rag_context_files()   ← 设置上下文文件            │
│       ├── set_rag_step_queue()      ← 捕获 async event loop   │
│       │                                                      │
│       ├── _agent_worker (后台任务):                            │
│       │     │                                                │
│       │     ├── Agent 自主推理                                │
│       │     │     │                                          │
│       │     │     ├── 决定调用 search_knowledge_base           │
│       │     │     │     │                                    │
│       │     │     │     ▼                                    │
│       │     │     │   run_rag_graph()                        │
│       │     │     │     │                                    │
│       │     │     │     ├── retrieve_initial                  │
│       │     │     │     │     │  ← emit_rag_step()           │
│       │     │     │     │     │    ↓                         │
│       │     │     │     │     │  loop.call_soon_threadsafe() │
│       │     │     │     │     │    ↓                         │
│       │     │     │     │     │  queue.put_nowait(step)      │
│       │     │     │     │                                    │
│       │     │     │     ├── grade_documents                   │
│       │     │     │     └── [可能] rewrite → retrieve_expanded│
│       │     │     │                                          │
│       │     │     └── 基于检索结果生成最终答案                   │
│       │     │                                                │
│       │     └── 所有内容事件 → queue                          │
│       │                                                      │
│       └── 主循环: yield SSE 事件                               │
│             ├── RAG 步骤事件 (thinking 指示器)                  │
│             ├── 内容片段                                       │
│             ├── trace 事件                                    │
│             └── [DONE] sentinel                               │
│                                                              │
│  关键: 跨线程事件调度                                          │
│  tools.py 的 emit_rag_step() 在 ThreadPoolExecutor 中运行      │
│  通过 loop.call_soon_threadsafe() 安全地注入到 async SSE 流     │
└──────────────────────────────────────────────────────────────┘
```

### 6.4 模式路由决策树

```
classify_mode(features)
  → user_explicit_deep?        → DEEP
  → context_reference?         → STANDARD
  → deep_score >= 0.40?        → DEEP (除非用户请求 fast)
  → user_explicit_fast?        → FAST (如果 fast_score >= 0.50)
  → fast_score >= 0.75?        → FAST
  → else                       → STANDARD (安全默认)

resolve_effective_mode_and_plan()
  → shadow_enabled?            → STANDARD (仅记录)
  → routing_disabled?          → STANDARD
  → DEEP?                      → STANDARD (deep_suggest_only 默认开启)
  → FAST + fast_disabled?      → STANDARD
  → FAST + low confidence?     → STANDARD (降级)
  → FAST + valid plan?         → FAST (如果 plan_execution_enabled)
  → STANDARD                   → STANDARD (有或无 plan)
```

所有路径最终 fail-closed 到 STANDARD。DEEP 模式当前为 suggest-only — 它分类但不执行深度检索，而是在 trace 中标注 `suggested_mode=DEEP` 并在工具输出中附加说明。

## 七、配置系统（K/I/M/A 四轴体系）

### 7.1 四个维度

| 维度 | 含义 | 级别 | 影响 |
|------|------|------|------|
| **K** | 检索质量 | K1(基础) / K2(全面) / K3(中等) / K4(深度) | 候选数、重排序参数、特征开关 |
| **I** | 索引存储 | I1(legacy) / I2(当前) | Collection 名、Profile、文本模式 |
| **M** | 模式路由 | M0(关) / M1(影子) / M2(激活) | FAST/STANDARD/DEEP 是否生效 |
| **A** | 设备策略 | A0(CPU) / A1(auto) / A2(GPU) | Embedding/Rerank 设备选择 |

默认配置 `K2/I2/M0/A1` = 全面检索质量 + 当前索引 + 模式路由关闭 + 自动设备。

### 7.2 K 维度详细参数

| 参数 | K1 | K2 | K3 | K4 |
|------|----|----|----|----|
| QUERY_PLAN_ENABLED | false | true | true | true |
| RERANK_SCORE_FUSION_ENABLED | false | true | true | true |
| RAG_CANDIDATE_K | 50 | 120 | 80 | 160 |
| RERANK_TOP_N | - | 30 | - | 40 |
| SEARCH_EF | - | 160 | 128 | - |
| DEEP_MODE | off | off | off | on (suggest_only=false) |
| 分数融合权重 | - | CE:0.65 RRF:0.20 SCOPE:0.10 META:0.05 | 同K2 | 同K2 |

### 7.3 I 维度详细参数

| 参数 | I1 (legacy) | I2 (current) |
|------|-------------|--------------|
| Collection | embeddings_collection | embeddings_collection_v3_quality |
| Index Profile | legacy | v3_quality |
| Text Mode | title_context | title_context_filename |

### 7.4 M 维度详细参数

| 参数 | M0 | M1 (shadow) | M2 (active) |
|------|----|----|----|
| Shadow | off | on | off |
| Routing | off | off | on |
| Fast | off | off | off |
| Plan Execution | off | off | on |

### 7.5 A 维度详细参数

| 参数 | A0 | A1 (auto) | A2 |
|------|----|----|----|
| Embedding Device | cpu | auto (GPU优先) | cuda (硬性) |
| Rerank Device | cpu | auto (GPU优先) | cuda (硬性) |
| CUDA 不可用时 | 正常运行 | 降级到 CPU | 报错 |

## 八、核心 Embedding 与检索算法

### 8.1 EmbeddingService

**Dense Embedding:**
- 模型: BAAI/bge-m3
- 维度: 1024
- 提供: HuggingFaceEmbeddings (local) 或 OllamaEmbeddings (ollama)

**Sparse Embedding (BM25):**
- 分词: jieba (中文, 最小2字符) + 正则 (ASCII 字母数字)
- BM25 参数: k1=1.5, b=0.75
- IDF 公式: `log((N - df + 0.5) / (df + 0.5) + 1)`
- 词汇外 token: IDF = `log((N + 1) / 1)`
- 持久化: JSON 文件，原子写入 (tmp + rename)

### 8.2 Milvus 混合检索

**Hybrid Retrieve (RRF):**
- 两个 AnnSearchRequest: dense (HNSW, IP) + sparse (SPARSE_INVERTED_INDEX, IP)
- 融合: `RRFRanker(k=60)`, score = `1 / (k + rank)`
- 搜索 limit: `top_k * 2`

**Split Retrieve:**
- Dense 和 Sparse 各自独立搜索
- 返回带 per-path score/rank 的候选集
- 按 chunk_id 合并为统一池
- 允许下游 score fusion

**重连机制:**
- 每次操作创建新 client
- 可恢复错误列表 (8 种 RPC 错误模式)
- 最多重试 2 次，指数退避

### 8.3 加权 RRF 融合

```python
score = Σ (weight_i / (rrf_k + rank_i))
# 默认 rrf_k = 60
```

支持多路结果加权融合，每路可配不同权重。

### 8.4 分数融合 (Score Fusion)

```
final_score = (w_rerank * rerank_norm + w_rrf * rrf_norm
             + w_scope * scope + w_metadata * metadata) / total_weight

# K2 默认权重:
# rerank:   0.65 (CrossEncoder 分数, min-max 归一化)
# rrf:      0.20 (RRF 排名分数, min-max 归一化)
# scope:    0.10 (文件名匹配分数, 0-1)
# metadata: 0.05 (元数据匹配分数, 0-1)
```

### 8.5 置信度门控算法

**Pre-CE 评估:**
- `top_margin` = top1_score - top2_score
- `dominant_root_share` = 最强 root 在 top5 中的分数占比
- `anchor_match` = 查询锚点是否匹配 top2 文档

**Post-CE 风险评分:**
| 信号 | 权重 | 触发条件 |
|------|------|----------|
| top_score_low | 2.0 | top doc CE score < 0.15 |
| ce_margin_small | 1.5 | top1-top2 margin < 0.05 |
| evidence_count_low | 2.0 | 文档数 < 2 |
| precise_scope_unmatched | 2.0 | 有 scope 但 top3 无匹配文件 |

**裁决阈值:**
| risk 范围 | 判定 |
|-----------|------|
| < 2.0 | 高置信，可回答 |
| < 4.0 | 可回答，无需升级 |
| < 6.0 | 不可回答，建议 DEEP |
| ≥ 6.0 | 不可回答，需用户澄清 |

`confidence_score = max(0, 1 - risk/10)`

## 九、API 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/auth/register` | 用户注册 |
| POST | `/auth/login` | 用户登录 |
| GET | `/auth/me` | 获取当前用户信息 |
| POST | `/chat` | 同步聊天 |
| POST | `/chat/stream` | SSE 流式聊天 |
| GET | `/sessions` | 获取会话列表 |
| GET | `/sessions/{id}` | 获取会话详情 |
| DELETE | `/sessions/{id}` | 删除会话 |
| PUT | `/sessions/{id}/rename` | 重命名会话 |
| GET | `/documents` | 获取文档列表 |
| POST | `/documents/upload` | 上传文档 |
| DELETE | `/documents/{filename}` | 删除文档 |
| GET | `/rag/status` | RAG 系统状态（管理） |

## 十、失败诊断系统 (diagnostics.py)

### 10.1 失败分类

| 类别 | 含义 | 判断条件 |
|------|------|----------|
| `insufficient_trace` | 数据不足 | 缺少 rag_trace 或 retrieved_chunks |
| `hard_negative_confusion` | 硬负例混淆 | top5 文件全在 hard_negative_files 中 |
| `file_recall_miss` | 召回缺失 | 期望文件不在 pre-rerank 候选中 |
| `ranking_miss` | 排序缺失 | 期望文件在候选中但不在 top5 |
| `page_miss` | 页码缺失 | 期望文件在 top5 但期望页码不匹配 |
| `ok` | 成功 | 期望文件在 top5 且页码匹配 |
| `low_confidence` | 低置信 | top1 score < 0.20 或 fallback required |

### 10.2 锚点匹配算法

使用正则表达式匹配结构锚点（如 "1.2.3", "第三章", "（二）"），并防止部分匹配（"1.2" 不会匹配 "11.2"）。

## 十一、核心创新点总结

1. **混合检索双降级**: Dense + BM25 Sparse RRF 融合，sparse 失败自动降级到 dense-only
2. **3 级分块 + 自动合并**: 只有 L3 叶块进 Milvus，L1/L2 在 PostgreSQL；当同一父块的子块被检索到 >=2 个时自动合并为父块
3. **4 层分层重排序**: L0 拆分检索 → L1 三槽预过滤(保证/文件感知/路径保证) → L2 自适应 K → L3 根多样性上限
4. **神经重排序策略路由**: 纯规则 + 可选 LLM 路由器，决定是否/如何使用 CrossEncoder
5. **零 LLM 模式分类器**: 纯规则 FAST/STANDARD/DEEP 分类，FAST 失败自动降级到 STANDARD
6. **跨线程 SSE 事件流**: RAG 步骤从同步线程通过 `call_soon_threadsafe` 注入到异步 SSE 流
7. **预算制 Deep Mode**: 子查询分解 + 时间/调用次数预算 + 覆盖率评估 + 防递归
8. **置信度门控 + 查询改写**: Pre-CE/Post-CE 双重评估，失败自动触发 Step-Back/HyDE/Complex 改写
9. **K/I/M/A 四轴 Profile**: 统一管理检索质量/索引存储/模式路由/设备策略
10. **三级文件名缓存**: 进程 LRU → Redis → Milvus 查询，平衡性能与一致性

## 十二、端到端数据流总结

```
用户提问
  → QueryPlan(文件匹配 + scope 路由)
  → Mode 分类(FAST/STANDARD/DEEP)
  → Embedding(Dense + Sparse 并行)
  → Milvus 检索(Hybrid/Split/Scoped)
  → RRF 融合
  → 分层重排序(3-Slot → 自适应 K → CE → 结构重排)
  → 置信度门控
      → [通过] → 返回文档 → LLM 生成答案
      → [未通过] → Grade → [改写] → Step-Back/HyDE → 重新检索
                                    → [通过] → 返回文档 → LLM 生成答案
```
