# SuperHermes

SuperHermes 是一个面向私有文档知识库的 RAG 问答系统。项目以 FastAPI 后端为核心，连接 PostgreSQL、Redis、Milvus、Embedding 模型和 CrossEncoder reranker，提供账号认证、会话管理、文档上传索引、流式问答和可追溯检索证据。

它现在已经是独立演进的项目：核心目标不是做一个普通聊天壳，而是把“文档解析 -> 结构化索引 -> 多阶段检索 -> 证据重排 -> 可诊断回答”做成一条稳定、可评测、可维护的本地知识库流水线。

## 项目定位

SuperHermes 适合这些场景：

| 场景 | SuperHermes 的处理方式 |
| --- | --- |
| 私有文档问答 | 管理员上传 PDF、Word、Excel，系统自动解析并索引到 Milvus |
| 技术手册检索 | 通过文件名、页码、标题、章节和 chunk/root 证据做多层定位 |
| 可追溯回答 | 回答链路保留 `rag_trace`，便于查看检索、重排、上下文和降级情况 |
| 本地或内网部署 | Docker 管理 PostgreSQL、Redis、Milvus、etcd、MinIO、Attu |
| RAG 质量调优 | 内置过一套基于 gold dataset 的评测口径，README 固化当前关键数据 |

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 认证与权限 | 用户注册、登录、JWT Bearer token，管理员邀请码控制管理权限 |
| 静态前端 | `frontend/` 是 Vue CDN 静态界面，后端启动后自动挂载到 `/` |
| 会话系统 | 支持会话列表、消息历史、会话重命名和删除 |
| 文档管理 | 管理员可上传、查看、删除知识库文档 |
| 文档解析 | 支持 PDF、Word、Excel，解析后写入父 chunk 存储和向量库 |
| 混合检索 | Milvus dense + sparse 检索，配合 RRF、文件名增强和 QueryPlan |
| CrossEncoder 重排 | 使用 `BAAI/bge-reranker-v2-m3` 做强证据候选排序 |
| RAG trace | 返回检索模式、候选、重排状态、上下文文件和错误降级信息 |
| 流式输出 | `/chat/stream` 使用 SSE 输出增量响应 |

## 总体架构

```text
浏览器静态前端
  -> FastAPI 应用入口 backend.app
  -> HTTP 路由 backend.routers.*
  -> 认证 / 会话 / 文档服务 / Chat Agent
  -> RAG 流水线 backend.rag.*
  -> PostgreSQL + Redis + Milvus + 本地 Embedding/Rerank 模型
```

后端按职责拆分为几个稳定边界：

| 模块 | 职责 |
| --- | --- |
| `backend/application/` | FastAPI 应用工厂、生命周期、CORS 和静态资源挂载 |
| `backend/routers/` | HTTP 层，包含 auth、chat、sessions、documents |
| `backend/contracts/` | Pydantic 请求和响应模型 |
| `backend/security/` | 密码哈希、JWT、用户鉴权、管理员鉴权 |
| `backend/services/` | 文档上传、索引、删除等用例编排 |
| `backend/chat/` | Chat Agent、工具调用、会话存储接入和 RAG 执行 |
| `backend/rag/` | QueryPlan、检索、重排、上下文、引用、诊断、trace |
| `backend/infra/` | 数据库、Redis、Embedding、Milvus 读写和父 chunk 存储 |
| `backend/documents/` | 文档解析、分块和元数据抽取 |
| `frontend/` | Web UI、知识库管理、会话历史、流式对话展示 |
| `tests/` | 后端入口、API、RAG 核心模块和前端静态检查相关测试 |

本地运行时还会使用一些不上传到 GitHub 的维护目录：`data/`、`docs/`、`eval/`、`scripts/`、`volumes/`。其中评测和历史设计材料的关键结论已经整理进本文档。

## 请求链路

普通对话请求：

```text
POST /chat
  -> 校验 JWT
  -> 读取用户和 session_id
  -> Chat Agent 判断是否需要文档检索
  -> 执行 RAG 工具
  -> 组织上下文并调用模型
  -> 返回 response + rag_trace
```

流式对话请求：

```text
POST /chat/stream
  -> SSE event stream
  -> 增量返回模型输出
  -> 出错时以 error event 返回结构化错误
```

文档上传请求：

```text
POST /documents/upload
  -> 管理员鉴权
  -> 文件类型检查
  -> DocumentLoader 解析 PDF / Word / Excel
  -> 生成 chunk、页码、标题和文件元数据
  -> 写入 Milvus collection 和父 chunk 存储
```

## RAG 设计

SuperHermes 的 RAG 不是单一向量召回，而是多阶段证据筛选：

1. **QueryPlan 与文件范围识别**：从问题中提取可能的文件名、章节、页码或标题提示，优先缩小搜索空间。
2. **混合召回**：使用 dense embedding 与 sparse/BM25 信息共同召回候选，再用 RRF 合并。
3. **文件名与标题增强**：对设备手册、产品文档这类强文件名线索做加权，降低同系列 hard negative 混淆。
4. **CrossEncoder rerank**：对候选 chunk 使用 `BAAI/bge-reranker-v2-m3` 重排，GPU 默认走 fp16。
5. **Score fusion**：融合 rerank、RRF、scope 和 metadata 信号，避免单一分数支配最终排名。
6. **上下文组装**：按父 chunk、章节/root 和 token 预算组织最终上下文。
7. **Trace 与诊断**：保留检索候选、重排状态、上下文文件、错误降级和关键开关，便于定位失败原因。

当前默认产品档位：

```text
K2 / I2 / M0 / A1 / fp16
```

| 维度 | 含义 |
| --- | --- |
| `K2` | 强证据检索档：QueryPlan + CrossEncoder rerank + score fusion |
| `I2` | 当前结构化索引与 `v3_quality` collection 意图 |
| `M0` | 模式路由关闭，Deep Mode 不默认执行 |
| `A1` | 自动设备选择：优先 GPU，无 GPU 时回退 CPU |
| `fp16` | GPU rerank 默认半精度，兼顾质量和延迟 |

常用档位：

| 档位 | 适用场景 |
| --- | --- |
| `K1` | 低延迟稳定基线，关闭强证据实验开关 |
| `K2` | 默认质量档，适合需要更强证据定位的问答 |
| `K3` | 快速证据档，降低候选数和 CrossEncoder 成本 |
| `M1` | shadow 模式路由，只记录判断，不影响结果 |
| `M2` | active 模式路由，显式启用后才改变链路 |

## RAG 效果数据

以下只展示当前默认质量档的一组最佳测试结果。数据来自本地 gold dataset 评测快照，样本规模为 125 条，包含文件、页码、chunk/root、关键词和 hard negative 信息。后续更换模型、重建索引、修改 chunk 规则或开启缓存后都需要重新评测。

评测口径：gold dataset，`RERANK_CACHE_ENABLED=false`，跳过重建索引，使用当前 `v3_quality` collection。

| 规范档位 | 数据类型 | File@5 | File+Page@5 | Chunk@5 | Root@5 | P50 | P95 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `K2/I2/M0/A1/fp16` | fp16 | 0.992 | 0.768 | 0.720 | 0.793 | 1116 ms | 1283 ms | 当前默认质量档 |

这组结果说明：默认档在文件召回、页码定位、chunk 命中和 root/章节证据命中之间最均衡；`fp16` 在 GPU rerank 场景下兼顾质量、显存占用和延迟。

## 数据集与评测口径

当前 README 固化的 gold dataset 口径：

| 项目 | 数值 |
| --- | ---: |
| 样本数 | 125 |
| 覆盖源文件 | 60 |
| operation 样本 | 45 |
| specification 样本 | 42 |
| troubleshooting 样本 | 36 |
| safety 样本 | 2 |
| 带结构锚点样本 | 50 |
| 平均 source coverage | 0.983 |
| 平均质量分 | 6.086 |

主要指标含义：

| 指标 | 含义 |
| --- | --- |
| `File@5` | top5 中是否包含正确文件 |
| `File+Page@5` | top5 中是否同时命中正确文件和页码 |
| `Chunk@5` | top5 中是否命中 gold chunk |
| `Root@5` | top5 中是否命中 gold root/章节证据 |
| `ChunkMRR` / `RootMRR` | 正确 chunk/root 首次出现排名的倒数 |
| `P50` / `P95` | 检索链路延迟分位 |

## HTTP 接口

| 方法 | 路径 | 权限 | 说明 |
| --- | --- | --- | --- |
| `POST` | `/auth/register` | 公开 | 注册用户，可通过管理员邀请码创建管理员 |
| `POST` | `/auth/login` | 公开 | 登录并获取 Bearer token |
| `GET` | `/auth/me` | 登录用户 | 获取当前用户信息 |
| `POST` | `/chat` | 登录用户 | 普通问答 |
| `POST` | `/chat/stream` | 登录用户 | SSE 流式问答 |
| `GET` | `/sessions` | 登录用户 | 获取当前用户会话列表 |
| `GET` | `/sessions/{session_id}` | 登录用户 | 获取指定会话消息 |
| `PATCH` | `/sessions/{session_id}` | 登录用户 | 重命名会话 |
| `DELETE` | `/sessions/{session_id}` | 登录用户 | 删除会话 |
| `GET` | `/documents` | 管理员 | 查看已索引文档 |
| `POST` | `/documents/upload` | 管理员 | 上传并索引文档 |
| `DELETE` | `/documents/{filename}` | 管理员 | 删除文档及索引 |

除注册和登录外，请求需要携带：

```text
Authorization: Bearer <token>
```

## 运行依赖

| 依赖 | 用途 |
| --- | --- |
| Python 3.12+ | 后端运行环境 |
| uv | Python 依赖管理 |
| Docker Compose | 启动 PostgreSQL、Redis、Milvus、etcd、MinIO、Attu |
| PostgreSQL | 用户、会话和业务数据 |
| Redis | 缓存和部分检索辅助状态 |
| Milvus | 向量库和混合检索 |
| 本地 embedding 模型 | 默认 `BAAI/bge-m3` |
| 本地 reranker 模型 | 默认 `BAAI/bge-reranker-v2-m3` |
| OpenAI 兼容模型服务 | 生成最终回答和部分辅助判断 |

## 快速启动

安装依赖：

```powershell
uv sync
```

创建环境配置：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少配置模型服务、JWT、管理员邀请码和向量库相关变量。然后启动依赖服务：

```powershell
docker compose up -d
```

启动应用：

```powershell
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

也可以使用脚本入口：

```powershell
uv run python backend/app.py
```

访问前端：

```text
http://127.0.0.1:8000/
```

## 配置说明

`.env.example` 中保留了完整配置模板。常用配置如下：

| 配置组 | 关键变量 |
| --- | --- |
| LLM | `ARK_API_KEY`、`MODEL`、`BASE_URL`、`FAST_MODEL`、`GRADE_MODEL` |
| Embedding | `EMBEDDING_PROVIDER`、`EMBEDDING_MODEL`、`EMBEDDING_DEVICE` |
| Rerank | `RERANK_MODEL`、`RERANK_DEVICE`、`RERANK_TORCH_DTYPE` |
| Milvus | `MILVUS_HOST`、`MILVUS_PORT`、`MILVUS_COLLECTION`、`RAG_INDEX_PROFILE` |
| 数据库 | `DATABASE_URL`、`FALLBACK_DATABASE_URL` |
| Redis | `REDIS_URL` |
| 认证 | `JWT_SECRET_KEY`、`ADMIN_INVITE_CODE`、`JWT_EXPIRE_MINUTES` |
| RAG 档位 | `RAG_K`、`RAG_I`、`RAG_M`、`RAG_A` |
| RAG 开关 | `QUERY_PLAN_ENABLED`、`HEADING_LEXICAL_ENABLED`、`RERANK_SCORE_FUSION_ENABLED` |

推荐默认 RAG 配置意图：

```env
RAG_K=K2
RAG_I=I2
RAG_M=M0
RAG_A=A1
MILVUS_COLLECTION=embeddings_collection_v3_quality
EVAL_RETRIEVAL_TEXT_MODE=title_context_filename
QUERY_PLAN_ENABLED=true
HEADING_LEXICAL_ENABLED=true
RERANK_PAIR_ENRICHMENT_ENABLED=true
RERANK_SCORE_FUSION_ENABLED=true
RAG_CANDIDATE_K=120
RERANK_INPUT_K_GPU=80
RERANK_INPUT_K_CPU=30
RERANK_TOP_N=30
BM25_STATE_PATH=data/bm25_state_v3_quality.json
```

## 上传范围

本仓库上传到 GitHub 的范围以核心应用为主：后端、前端、测试、配置模板、Docker 编排和锁文件。

以下目录属于本地运行数据、历史文档、评测材料或维护脚本，不上传到 GitHub：

```text
data/
docs/
eval/
scripts/
volumes/
```

此外，虚拟环境、依赖目录、IDE 配置缓存、pytest/ruff 缓存和 `.env` 也不会作为新内容上传。

## 验证

核心静态检查：

```powershell
uv run python -m compileall backend tests
node --check frontend\script.js
node frontend\ui-redesign.test.mjs
git diff --check
```

后端核心测试可按需运行：

```powershell
uv run pytest tests\test_application_entrypoints.py tests\test_api_routes.py tests\test_document_service.py tests\test_rag_pipeline.py -q
```

手动 smoke：

1. 启动 Docker 依赖和后端。
2. 注册管理员账号或使用已有管理员登录。
3. 在知识库界面上传一份小文档。
4. 发起问题，确认回答包含该文档相关证据。
5. 查看会话历史是否保存。
6. 删除测试文档，并确认 `/documents` 不再返回该文档。

## 设计取舍

- **先保证检索证据，再追求生成花活**：评测数据显示，文件、页码、chunk/root 命中比复杂 fallback 更直接影响答案可信度。
- **默认关闭高风险模式路由**：`M0` 是默认，Deep Mode 和 active routing 只在显式启用时参与链路。
- **优先可诊断性**：RAG trace 中保留候选、重排、上下文和错误状态，方便定位是召回失败、页码失败还是重排掉点。
- **本地优先部署**：依赖服务全部可通过 Docker Compose 在本机或内网启动。
- **文档与评测材料本地维护**：GitHub README 固化关键结论，完整历史评测材料保留在本地目录，不随仓库上传。

## 后续方向

| 方向 | 目标 |
| --- | --- |
| 页码感知排序 | 进一步提升 `File+Page@5` 和 `Chunk@5` |
| qrel 人工复核 | 提高 chunk/root 指标可信度 |
| 选择性 fallback | 只在低置信度问题上触发更昂贵链路 |
| 答案评测 | 增加 groundedness、citation coverage 和 answer relevance |
| 业务语料微调 | 对 hard negative 和同系列产品文档做更强区分 |
