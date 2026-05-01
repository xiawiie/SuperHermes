# SuperHermes

SuperHermes 是一个 FastAPI + RAG 文档问答项目。当前收尾目标不是继续刷准确率，而是让运行口径、评测口径、诊断入口和文档入口保持一致。

## 当前运行口径

默认口径使用：

```text
K2 / I2 / M0 / A1 / fp16
```

含义：

| 维度 | 含义 |
| --- | --- |
| `K2` | 强证据档：开启 QueryPlan、CrossEncoder rerank、score fusion |
| `I2` | 当前结构化索引 |
| `M0` | 不启用模式路由 |
| `A1` | 自动设备：优先 GPU，无 GPU 时回退 CPU |
| `fp16` | GPU 推理默认半精度；CPU 回退时按运行时能力处理 |

命名规则见 [docs/rag-profile-naming.md](docs/rag-profile-naming.md)。

## RAG 现状

最新评测入口以短名为准，旧名仅作为兼容别名保留：

| Variant | File@5 | File+Page@5 | Chunk@5 | Root@5 | P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `K1` | 0.992 | 0.688 | 0.589 | 0.678 | 57.1 ms |
| `K2` | 0.984 | 0.712 | 0.720 | 0.747 | 1123.4 ms |
| `K3` | 0.944 | 0.696 | 0.673 | 0.724 | 730.0 ms |
| `M1` | 0.984 | 0.712 | 0.720 | 0.747 | 1129.2 ms |
| `M2` | 0.984 | 0.712 | 0.720 | 0.747 | 111.4 ms |

评测报告和运行说明放在 `eval/docs/`；历史实验和原始报告不要直接当作当前生产口径。

## 快速启动

安装依赖：

```powershell
uv sync
```

启动依赖服务：

```powershell
docker compose up -d
```

启动后端：

```powershell
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

或使用脚本入口：

```powershell
uv run python backend/app.py
```

## 关键接口

| 接口 | 用途 |
| --- | --- |
| `POST /chat` | 普通问答 |
| `POST /chat/stream` | 流式问答 |
| `GET /documents` | 查看已索引文档 |
| `POST /documents/upload` | 上传并索引文档 |
| `DELETE /documents/{filename}` | 删除文档及索引 |

计划中的 RAG 诊断接口应只返回运行口径、设备、collection、BM25 路径和关键开关，不能返回 secret。

## 配置要点

最终 `.env` 应保持短名口径和实际旧变量兼容。核心意图如下：

```env
RAG_K=K2
RAG_I=I2
RAG_M=M0
RAG_A=A1
RAG_DTYPE=fp16

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

设备语义：

| 名称 | 行为 |
| --- | --- |
| `A1` / `auto` | 优先 GPU，无 GPU 自动 CPU |
| `A2` / `cuda` | 只允许 GPU，无 GPU 直接失败 |
| `A0` / `cpu` | 只使用 CPU |

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `backend/` | 后端应用、RAG、文档处理、数据库、向量库和认证 |
| `frontend/` | 前端静态界面 |
| `scripts/` | 评测、诊断、维护脚本 |
| `eval/datasets/` | 评测数据集 |
| `eval/qrels/` | 评测标注 |
| `eval/docs/` | 当前评测说明和报告 |
| `eval/reports/` | 原始评测输出 |
| `docs/` | 架构、命名规则和历史设计文档 |

## 后端导入约定

支持入口：

```powershell
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
uv run python backend/app.py
```

支持 `backend.*` 包路径导入：

```python
from backend.rag.utils import retrieve_documents
from backend.rag.pipeline import run_rag_graph
from backend.infra.db.database import init_db
from backend.documents.loader import DocumentLoader
```

不支持旧的裸导入：

```python
import rag_utils
import database
from document_loader import DocumentLoader
```

## 收尾验证

常规验证命令：

```powershell
uv run pytest -q
uv run ruff check backend scripts tests
uv run python -m compileall backend scripts tests
node --check frontend\script.js
node frontend\ui-redesign.test.mjs
git diff --check
```

手动 smoke 建议：

1. 启动 Docker 依赖和后端。
2. 登录管理员账号。
3. 上传一份小文档。
4. 提问并确认回答引用上传文档。
5. 删除文档。
6. 再查 `/documents` 和 `/rag/status`，确认索引与运行口径正常。
