# RAG Profile Naming

本文档固定 SuperHermes RAG 的短命名规则，避免继续混用 `S1_linear`、`V3Q`、`V3Q_OPT`、`V4_ACTIVE` 这类历史实验名。

## 命名目标

新代码、新文档、新评测入口统一使用短数字名：

```text
K2 / I2 / M0 / A1 / fp16
```

旧名字不批量删除，只保留为兼容 alias，方便旧报告和脚本继续可读。

## 迁移结论

先整理历史信息和评测数据，再替换代码入口中的名字，这个方案可行，也是推荐顺序。

原因是 RAG 评测名不是普通变量名。`V3Q`、`V3Q_OPT`、`S1_linear` 这类历史名已经出现在旧报告、旧命令、run id、对比文档和诊断结论里。如果直接全局替换为 `K2`、`K3`、`K1`，旧报告会失去上下文，前后指标也会看起来像来自不同实验。正确做法是：

1. 先在本文档固定新旧名称映射和报告阅读口径。
2. 再把历史报告中的关键数据按新命名重新整理。
3. 然后让新评测入口优先接受 `K1/K2/K3`。
4. 最后把新文档和默认命令中的主名字替换成短名，旧名仅保留为 alias。

这意味着：新名字是主语言，旧名字是可追溯标签。旧名字不再作为新方案标题，但旧报告中出现时必须能被解释。

## 标准维度

| 名称 | 含义 |
| --- | --- |
| `K1` | 默认稳定档，低延迟，少实验开关 |
| `K2` | 强证据档，QueryPlan + CrossEncoder + fusion |
| `K3` | 轻量证据档，降低候选和 CrossEncoder 成本 |
| `K4` | 深度或多跳档，预留给未来 Deep Mode |
| `I1` | 旧索引 |
| `I2` | 当前结构化索引 |
| `M0` | 不启用模式路由 |
| `M1` | shadow，只记录模式路由判断，不影响结果 |
| `M2` | active，真正影响路由 |
| `A1` | auto，优先 GPU，无 GPU 走 CPU |
| `A2` | GPU only，无 GPU 直接失败 |
| `A0` | CPU only |

`fp16`、`bf16`、`fp32` 是 dtype 维度，不写进 `K/I/M/A` 名称里。

## 旧名映射

| 旧名字 | 新归属 |
| --- | --- |
| `S1_linear` | `K1` |
| `V3Q` | `K2` |
| `V3Q_OPT` | `K3` |
| `V3Q_LAYERED` / `EXP_C*` | `K2_LAYERED` 兼容实验档，不作为默认产品名 |
| `V4_SHADOW` | `M1` |
| `V4_ACTIVE` | `M2` |
| `FP16` / `BF16` / `FP32` | dtype 维度 |

## 历史报告阅读口径

历史报告中的旧名按以下规则阅读：

| 历史写法 | 新规范读法 | 说明 |
| --- | --- | --- |
| `S1_linear` | `K1` | 默认稳定档，关闭 QueryPlan 和强 CE 路径，适合低延迟基线 |
| `V3Q` | `K2` | 当前强证据档，包含 QueryPlan、heading lexical、pair enrichment、score fusion |
| `V3Q_OPT` | `K3` | `K2` 的轻量版，降低 candidate 和 CrossEncoder 成本 |
| `V3Q_LAYERED` | `K2_LAYERED` 实验档 | 分层重排兼容入口，不作为默认产品名 |
| `EXP_C*` | `K2_LAYERED` 实验档 | 参数探索兼容入口，不作为默认产品名 |
| `V4_SHADOW` | `M1` | 只记录模式路由判断，不影响结果 |
| `V4_ACTIVE` | `M2` | 模式路由真正影响链路，未接入前不能描述为默认 active |

阅读历史报告时，不能只看旧名或新名。若报告同时记录了 collection、BM25 state、dtype、candidate K、rerank K、rerank top N、关键开关，则以报告内的实际配置为准。短名只表示产品档位，不替代完整运行参数。

## 历史数据整理

以下数据用于把已有评测结果迁移到新命名语义下。它们是当前评测快照，不是永久基准；后续重新建索引、换模型、换数据集或调整缓存策略后，需要重新生成。

### 离线 retrieval 口径

评测口径：gold dataset，125 条样本，`RERANK_CACHE_ENABLED=false`，`--skip-reindex`，使用现有 `v3_quality` collection。

| 历史名 | 新规范名 | dtype | File@5 | File+Page@5 | Chunk@5 | Root@5 | P50 | P95 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `V3Q` | `K2/I2/M0/A1/fp16` | `fp16` | 0.992 | 0.768 | 0.720 | 0.793 | 1116 ms | 1283 ms | 推荐默认质量档 |
| `V3Q_OPT` | `K3/I2/M0/A1/fp16` | `fp16` | 0.968 | 0.704 | 0.673 | 0.713 | 718 ms | 816 ms | 快速备选档 |
| `V3Q` | `K2/I2/M0/A1/bf16` | `bf16` | 0.984 | 0.768 | 0.720 | 0.793 | 1051 ms | 1190 ms | 可灰度观察 |
| `V3Q` | `K2/I2/M0/A1/fp32` | `fp32` | 0.992 | 0.768 | 0.720 | 0.793 | 4483 ms | 5118 ms | 仅适合调试或数值基线 |

### 在线 graph 口径

评测口径：gold dataset，125 条样本，`RERANK_CACHE_ENABLED=false`，`--mode graph`，不包含最终外部 LLM 答案生成。

| 历史名 | 新规范名 | dtype | File@5 | File+Page@5 | Chunk@5 | Root@5 | P50 | P95 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `V3Q` | `K2/I2/M0/A1/fp16` | `fp16` | 0.992 | 0.768 | 0.720 | 0.793 | 1148 ms | 1306 ms | 在线 graph 与离线 retrieval 基本一致 |
| `V3Q_OPT` | `K3/I2/M0/A1/fp16` | `fp16` | 0.968 | 0.704 | 0.673 | 0.713 | 740 ms | 850 ms | 在线快速档可用，但质量低于 K2 |

### dtype 结论

| dtype | 运行含义 | 当前结论 |
| --- | --- | --- |
| `fp16` | 映射到 `RERANK_TORCH_DTYPE=float16` | 推荐默认，质量和延迟最均衡 |
| `bf16` | 映射到 `RERANK_TORCH_DTYPE=bfloat16` | 本机支持，可灰度，但本轮 File@5 略有波动 |
| `fp32` | 映射到 `RERANK_TORCH_DTYPE=float32` | 无明显质量收益，延迟明显升高，不建议在线生产 |

## 新报告写法

新评测报告必须同时写清楚短名、旧 alias 和实际参数，避免只有一个标签导致结果不可复现。

推荐报告字段：

```json
{
  "rag_profile": "K2/I2/M0/A1/fp16",
  "rag_k": "K2",
  "rag_i": "I2",
  "rag_m": "M0",
  "rag_a": "A1",
  "rag_dtype": "fp16",
  "legacy_variant": "V3Q",
  "collection": "embeddings_collection_v3_quality",
  "bm25_state_path": "data/bm25_state_v3_quality.json",
  "retrieval_text_mode": "title_context_filename",
  "candidate_k": 120,
  "rerank_input_k_gpu": 80,
  "rerank_input_k_cpu": 30,
  "rerank_top_n": 30,
  "rerank_torch_dtype": "float16"
}
```

报告标题和汇总表优先显示新规范名，例如：

```text
K2/I2/M0/A1/fp16 (legacy: V3Q)
K3/I2/M0/A1/fp16 (legacy: V3Q_OPT)
```

旧名只能放在括号、alias 字段或历史说明中，不再作为新报告的主标题。

## 替换顺序

命名替换按以下顺序进行：

1. 文档层：本文档先固定映射、历史数据和新报告写法。
2. 评测层：`evaluate_rag_matrix.py` 接受 `K1/K2/K3`，旧名解析到对应短名。
3. 报告层：新报告主显示完整短名，同时记录 legacy alias、collection、BM25 state 和配置指纹。
4. 运行层：`.env` 中的 `RAG_DTYPE` 被实际 rerank 链路消费；`RAG_A` 被设备解析链路消费。
5. 清理层：新文档、新命令和新 PR 描述使用短名；旧名只保留在 alias 表、历史报告说明和兼容入口里。

替换完成的判断标准：

- `--variants K2,K3` 可以直接运行。
- `--variants V3Q,V3Q_OPT` 仍可运行，并被标记为 legacy alias。
- 新报告能看出 `K2` 对应的 collection、BM25 state、dtype 和关键开关。
- `RAG_DTYPE=fp16/bf16/fp32` 被实际运行链路消费，而不是只停留在 `.env` 文本里。
- `M1/M2` 在真正接入前只作为 shadow/active 语义说明，不宣称默认可用。

## 当前默认

生产默认口径：

```text
K2 / I2 / M0 / A1 / fp16
```

含义：

- `K2`：使用当前强证据检索配置。
- `I2`：使用当前结构化索引和 `v3_quality` collection 意图。
- `M0`：模式路由关闭，Deep Mode 不默认执行。
- `A1`：优先 GPU，无 GPU 时自动 CPU fallback。
- `fp16`：GPU 上使用半精度以降低显存与延迟成本。

## 评测语义

| 组合 | 语义 |
| --- | --- |
| `K1/I2` | 当前结构化索引，关闭 QueryPlan、fallback 和强 CE 路径 |
| `K2/I2` | QueryPlan、heading lexical、pair enrichment、score fusion、CE 80 candidates |
| `K3/I2` | K2 的轻量版，候选和 CE 成本更低 |
| `M1` | shadow 模式路由，只记录判断 |
| `M2` | active 模式路由，显式启用后才影响链路 |

历史报告中出现 `V3Q` 时，先按 `K2` 理解；出现 `V3Q_OPT` 时，先按 `K3` 理解。若报告中同时记录了 collection、BM25 state、dtype、candidate K、rerank K，则以报告内的实际配置为准。

## 配置示例

```env
RAG_K=K2
RAG_I=I2
RAG_M=M0
RAG_A=A1
RAG_DTYPE=fp16
```

兼容旧变量时，应保持语义一致：

```env
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
```

## 修改规则

1. 新增运行档时优先使用下一个短数字名，不把实验细节写进名字。
2. 实验细节写入文档或配置字段，例如 collection、candidate K、rerank K、dtype。
3. 旧名只做 alias，不再作为新文档标题或默认入口。
4. 评测报告必须同时记录短名、旧 alias、collection、BM25 path、dtype 和关键开关。
5. Deep Mode 在真正接入执行链路前，只能标记为 `M1` shadow 或 suggest-only，不能描述为默认 active。

## 项目约束

- `scripts/rag_eval/variants.py` 中属于 RAG 产品档的公开配置使用规范名；`S1_linear`、`V3Q`、`V3Q_OPT`、`V3Q_LAYERED`、`EXP_C*` 只能出现在 `LEGACY_VARIANT_ALIASES`、历史说明或兼容测试中。
- 默认评测入口使用 `K2,K3`；旧命令 `--variants V3Q,V3Q_OPT` 必须解析为 `K2,K3`，并且结果去重。
- 新报告和 fingerprint 必须输出 `rag_profile`、`rag_k`、`rag_i`、`rag_m`、`rag_a`、`rag_dtype`、`legacy_variant`、collection、BM25 path、dtype 和关键检索参数。
- 新增测试、评测脚本和项目文档默认写 `K/I/M/A/dtype` 命名；需要提旧名时必须明确它是 legacy alias 或历史报告上下文。
