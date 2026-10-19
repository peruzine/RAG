# ragkb · 企业知识库问答（RAG）

一个**能跑、能测、能上线、能说清为什么这么做**的 RAG 参考实现。
核心链路**零第三方依赖**（只用 Python 标准库），离线即可端到端跑通；把 `.env` 改成任何
OpenAI 兼容网关即可切换到真实模型。

```bash
python -m ragkb.cli demo                      # 入库示例语料（增量、幂等）
python -m ragkb.cli ask "P0 故障的解决时限是多久？" --show-context
python -m ragkb.cli serve                     # 起服务 + 内置 Web UI
```

> 需要 Python 3.11+。**不需要** `pip install` 任何东西（`numpy`/`python-docx`/`openpyxl`/`pypdf` 都是可选加速项）。

<details>
<summary><b>Windows 用户：先看这三条</b>（<code>python</code> 打不开、脚本被禁、BOM 问题）</summary>

1. **`python` 可能是 Microsoft Store 的占位程序**：`python --version` 看着正常，但
   `python -m pip` 报 `9009`、`python -c "..."` 没有输出。改用 Python 启动器：
   ```powershell
   py -3 -m ragkb.cli demo
   ```
   实在没有可用解释器就建虚拟环境：`py -3 -m venv .venv`，之后 `run.ps1` 会优先使用它。
2. **Windows 默认禁止运行 `.ps1`**（`无法加载文件 … 因为在此系统上禁止运行脚本`）：
   ```powershell
   .\scripts\ragkb.cmd demo                        # .cmd 包装器，不必改策略
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run.ps1 demo
   # 或只放行当前窗口：Set-ExecutionPolicy -Scope Process Bypass
   ```
3. **`.ps1` 需要 UTF-8 BOM**：Windows PowerShell 5.1 会把没有 BOM 的脚本按 ANSI 读取，
   中文 here-string 会解析失败（报"意外的标记 }"）。仓库里的 `run.ps1` 已带 BOM，改的时候
   别去掉；`ragkb.cmd` 刻意只用 ASCII，因为 cmd.exe 按 OEM 代码页读批处理。
</details>

### 在 VS Code 里跑

仓库带了 `.vscode/` 配置，打开项目后**先选对解释器**（`Ctrl+Shift+P` → `Python: Select
Interpreter` → 选 `.\.venv\Scripts\python.exe`；`.env` 选了 ONNX embedding，只有这个环境装了
onnxruntime）。

| 操作 | 效果 |
| --- | --- |
| `Ctrl+Shift+B` | 运行默认生成任务 **ragkb: serve**，在集成终端里起服务（自带访问日志，Ctrl+C 退出） |
| `Ctrl+Shift+P` → `Tasks: Run Task` | 选 `ragkb: test` / `ragkb: eval` / `ragkb: demo` / `ragkb: compare` |
| `F5` | 调试当前选中的配置：**serve（可打断点）**、**ask（会弹框让你输入问题）**、**eval**、**当前测试文件** |
| 测试面板（烧瓶图标） | 直接跑 171 个单测，可单点重跑/调试（已配好 `unittest`） |

> 调试配置用的是 `"type": "debugpy"`（新版 Python 扩展）。如果你的扩展较老、F5 报
> "configured debug type is not supported"，把 `launch.json` 里的 `debugpy` 全部改成 `python` 即可。

也可以在集成终端里直接跑（PowerShell / cmd / Git Bash 都行）：

```powershell
cd C:\Users\86135\Documents\deepseek-harness\default-workspace\rag-kb
.\.venv\Scripts\python.exe -m ragkb.cli serve        # 起服务 → http://127.0.0.1:8000/
# 或者激活环境后用 run 脚本：
.\.venv\Scripts\Activate.ps1
.\scripts\ragkb.cmd ask "酒店一晚上最多能报多少钱？" --show-context
```

---

## 目录

- [为什么值得看](#为什么值得看)
- [架构与请求链路](#架构与请求链路)
- [关键设计决策](#关键设计决策)
- [评测：用数据说话](#评测用数据说话)
- [可观测性：线上出问题怎么查](#可观测性线上出问题怎么查)
- [HTTP API 与 Web UI](#http-api-与-web-ui)
- [权限模型](#权限模型)
- [配置](#配置)
- [目录结构](#目录结构)
- [测试](#测试)
- [已知限制](#已知限制)
- [容量与演进](#容量与演进)
- [部署](#部署)
- [面试怎么讲这个项目](#面试怎么讲这个项目)

---

## 为什么值得看

绝大多数 RAG demo 停在"向量库 + prompt 套壳"，一旦问它这五个问题就答不上来：

| 面试/评审会问 | 本项目的回答位置 |
| --- | --- |
| 效果怎么度量？改一版 prompt 是变好还是变坏？ | [`eval/`](eval/) + [`ragkb/evaluate.py`](ragkb/evaluate.py)：22 条评测集、5 类指标、CI 门禁 |
| 线上答错了，怎么定位是检索还是生成？ | 结构化 trace 落到 JSONL，含每路召回排名、阶段耗时、覆盖率、越界引用 |
| 为什么用 RRF 而不是加权求和？ | [混合检索](#3-用-rrf-融合而不是分数加权) 一节，含量纲分析 |
| 用户问的东西语料里没有，会不会编？ | IDF 加权覆盖率门禁 + 模型哨兵双层拒答，**误答率 0**（实测） |
| 换个 embedding 模型要注意什么？ | 向量按 `embed_model` 隔离 + 启动告警，避免跨模型比较向量 |

另外这些通常是"上线才会想到"的东西，这里也都实现了：
**流式输出（SSE）**、**行级权限过滤（检索阶段生效）**、**增量入库（内容 hash 幂等）**、
**成本折算**、**降级策略**（向量服务挂了自动退化纯 BM25，模型挂了退化为原文摘录）、**请求体限流**、**健康检查**。

---

## 架构与请求链路

```
                       ┌──────────────────────── 入库 ────────────────────────┐
  文档 md/txt/html/csv │  loaders ─► chunking ─► embedding ─► SqliteStore     │
  （docx/xlsx/pdf 可选）│  结构化解析   标题栈/表格/   词法向量或 API  文档+切块+倒排+向量+权限
                       │              代码块原子切分                        │
                       └──────────────────────────────────────────────────────┘

                       ┌──────────────────────── 问答 ────────────────────────┐
  问题 ─► QueryProcessor ─┬─► BM25 倒排检索 ─┐                                  │
         （按需改写/路由）  └─► 向量相似检索 ─┴─► RRF 融合 ─► rerank ─► 覆盖率门禁 │
                                                                      │          │
                                       拒答 ◄──────────────────────────┤          │
                                                                      ▼          │
                                                       build_answer_messages      │
                                                                      │          │
                              Answer（引用/用量/成本/trace）◄── LLM 生成 ─► SSE 流式│
                       └──────────────────────────────────────────────────────┘

  全链路 trace ──► data/traces.jsonl ──► /metrics（p50/p95、tokens、成本、拒答率）
```

一次 `ask()` 的 trace 实际长这样（`python -m ragkb.cli ask ... --json` 或 `data/traces.jsonl`）：

```json
{
  "latency_ms": 5.2,
  "spans": [
    {"name": "condense", "ms": 0.01, "rewritten": false, "query": "P0 故障的解决时限是多久？"},
    {"name": "retrieve", "ms": 1.9, "n_hits": 6, "best_score": 0.812, "channels_only": {"both": 3, "vector_only": 10}},
    {"name": "generate", "ms": 2.8}
  ],
  "usage": {"prompt_tokens": 1287, "completion_tokens": 58, "total_tokens": 1345},
  "cost_usd": 0.0, "refused": false, "citation_ids": [1],
  "n_context": 6, "invalid_citations": [], "route": "kb"
}
```

---

## 关键设计决策

每条都是**权衡**，不是"最佳实践"抄来的。

### 1. 零第三方依赖（核心层）

用 `sqlite3` 做存储与倒排索引、`http.server` 做 API 与 SSE、`urllib` 调模型、`html.parser` 解析 HTML。

- **好处**：`git clone` 后立刻能跑；单测 162 个用例 20 秒内跑完（无需下载模型/镜像）；依赖不会腐烂；面试时能讲清每一层。
- **代价**：没有现成的 FAISS/pgvector，向量检索是暴力扫描（10 万 chunk 内 p95 < 80ms，够用）；文档解析不如 `unstructured` 全面。
- **结论**：把"可替换点"设计清楚（`Embedder` / `Reranker` / `ChatModel` / `lexical_search` / `vector_search`），比一开始就上重型框架更重要。

### 2. 中文用字级二元切分（bigram），不引入分词器

中文没有空格。这里不依赖 jieba 等词典分词，而是：

- **建索引**：一元 + 二元（保留词频）——保留一元是为了让"库"这种单字查询也能命中；
- **查询**：只产出二元（长度 ≥2 的连续汉字串），一元噪声大、会稀释 BM25 的区分度；只有单字串才回退到一元；
- 两者共用同一份停用词表，保证词汇表一致。

实测对"pgvector""混合检索"这类未登录词/专业术语很稳，且没有词典加载成本和 OOV 问题。
**代价**：跨词边界会产生"障是""是怎"这类伪词条（df=0）。它在检索里无害（IDF 自然处理），
但在"关键词承接"和覆盖率计算里必须先过滤 `df>0`——这个坑在代码里有注释标记。

### 3. 用 RRF 融合，而不是分数加权

BM25 分数无上界，余弦相似度在 0~1，直接加权就必须先归一化，而归一化区间随查询变化，非常脆弱。
RRF 只用**排名**：`score = Σ 1/(k + rank)`。

- 对量纲免疫，无需调参；
- **天然可扩展**：后来加"子问题拆解"（多路 BM25/向量）时直接并进来，不用重新调权重；
- 每个通道只取前 `fuse_top_k` 名参与融合：否则长尾候选的 `1/(k+rank)` 与第一名同量级，噪声会把相关的挤下去。

### 4. 结构化切块，而不是按字数硬切

先按 Markdown 解析出语义块（标题栈 → `heading_path`；**表格/代码块/列表原子化**），再贪心装箱：

- 跨章节且当前块已过半 → 提前收尾，避免一个块横跨两节；
- 块间 overlap 从**上一块结尾的完整句子**里取，不出现半截句子；
- 太短的尾块合并，避免"只有一行"的碎块污染检索；
- 入库打印切块质量报告（`chunks/blocks/avg/min/max/oversized/tiny`），参数调整有依据。

向量化时把 `标题 > 章节路径` 拼进前缀（`Chunk.embedding_text`），相当于免费拿到层级上下文；
而返回给用户的仍是原文，不污染展示。

### 5. 拒答做成两层，并且可度量

企业知识库里**编造答案比答不出来严重得多**。两层门禁：

1. **便宜的前置规则**：零召回 / 最高分过低 / **IDF 加权覆盖率**不足 → 直接拒答，省掉一次模型调用；
2. **模型哨兵**：prompt 要求"资料不足时只输出 `NO_ANSWER`"，由上层翻译成统一话术。

覆盖率用 **IDF 加权**而不是平均覆盖率：中文二元切分会产生大量"公司/数据"这类高频词，
命中与否几乎不含信息量；按 IDF 加权后，"问题里的关键实体到底在不在语料里"才可跨语料复用。
覆盖率同时把章节标题算进去（标题也会进 prompt）。

> 阈值不是拍脑袋定的，见 [`scripts/sweep_refusal.py`](scripts/sweep_refusal.py) 与[下一节](#门禁阈值是扫出来的)。

### 6. 引用可验证，越界引用会被抓住

答案里的 `[n]` 会逐一对齐到检索结果。越界编号（只给了 5 条上下文却引用 `[7]`）会被剔除并计入
`invalid_citations`——它是**幻觉的早期代理指标**，也是 `citation_validity` 指标的来源。

### 7. 降级优先于报错

- 向量通道异常 → 记录 `reranker_error`，退化纯 BM25（可用性优先）；
- 重排 HTTP 服务失败 → 自动降级为本地词法重排；
- 生成模型失败 → 输出"最相关原文摘录 + 说明"，用户至少能自助；
- 单篇文档解析失败 → 记入入库报告，不中断整批。

### 8. 只在必要时调用模型

多轮改写先用规则判断（有历史 + 出现指代/省略词），否则直接用原问题，**省掉一次 LLM 调用**。
改写结果还要过 `_valid_rewrite` 校验：一段"回答"或"拒绝话术"被当成查询送进检索，是 RAG 里
很隐蔽的故障（检索用垃圾查 → 召回全错 → 用户看到"答非所问"）。这个坑本项目实际踩过，见
[`ragkb/query.py`](ragkb/query.py) 的注释。

---

## 评测：用数据说话

### 指标口径

| 类别 | 指标 | 口径与理由 |
| --- | --- | --- |
| 检索 | `hit@k` | 前 k 条里有没有相关块——回答"能不能答" |
| 检索 | `recall@k` | gold 片段被覆盖的比例——比 hit 严格 |
| 检索 | `MRR` / `nDCG@k` | 排得够不够靠前 |
| 检索 | `precision@k` | 上下文里混进多少无关内容（旋钮是 `final_top_k`） |
| 生成 | `answer_f1` / `answer_contains` | 与参考答案的字符级 F1 / 包含关系 |
| 生成 | `groundedness` | 答案内容词落在检索上下文里的比例——**无标注**的忠实度代理 |
| 生成 | `citation_validity` | 引用编号是否越界、是否"引用了但没给角标" |
| 拒答 | `refusal_accuracy` | 该拒的拒了、不该拒的答了，**两个方向都算** |

两个刻意的口径选择：

- **检索指标只在有 gold 标注的样本上平均**（18/22）。拒答样本本来就没有可召回的目标，
  算进去只会稀释指标、掩盖真实问题。
- **相关性判定从严格到宽松**：`chunk_id` > `gold_snippets` > `gold_sources`。
  如果先按文件判相关，"检索到同一篇文档但完全不相干的那一段"也算命中，`precision@k` 会虚高。

### 实测结果（离线基线：hashing 向量 + 抽取式生成 + 词法重排）

```
指标                     本版       阈值    结果
hit@k                  1.000      0.850   pass      # 22 条用例，18 条有 gold
recall@k               1.000      0.850   pass
precision@k            0.222        -     pass      # 见下方说明
mrr                    0.872      0.700   pass
ndcg@k                 0.963      0.700   pass
refusal_accuracy       0.955      0.850   pass
answer_f1              0.452      0.250   pass
groundedness           0.931      0.800   pass
citation_validity      1.000      0.950   pass
latency_ms_avg         3.4          -     pass      # 检索 p95 = 1.9ms，生成 = 本地抽取基线
```

- `precision@k = 0.22` 偏低是**预期**：示例语料每篇文档只有 1~2 个相关块，而 top-5 固定返回 5 条。
  它衡量的是"上下文里有多少无关内容"，控制它的旋钮是 `final_top_k`。
- `latency` 只反映**检索与编排**开销（毫秒级）；生成用的是本地抽取式基线，不代表线上 LLM 延迟。
  接真实模型后这个数字由模型决定，而 trace 会把两者分开统计（`retrieve` vs `generate` span）。
- 分标签报告能看出哪类问题在拖后腿，例如 `multi_turn`（指代改写）和 `acl` 的 MRR 明显低于 `fact`。

### 门禁阈值是扫出来的

`RAGKB_REFUSE_COVERAGE_FLOOR` 直接决定"误答"与"误拒"的平衡。用
[`scripts/sweep_refusal.py`](scripts/sweep_refusal.py) 在评测集上扫一遍（检索与生成只跑一次，
按阈值重算门禁结果）：

| floor | 拒答准确率 | 误拒率 | 误答率 |
| --- | --- | --- | --- |
| 0.15 | 0.955 | 0.000 | 0.250 |
| 0.20 | 0.909 | 0.056 | 0.250 |
| **0.25** | **0.955** | **0.056** | **0.000** |
| 0.28 | 0.864 | 0.167 | 0.000 |
| 0.30 | 0.818 | 0.222 | 0.000 |

取 **0.25**：在误答率为 0 的前提下误拒率最低。5.6% 的误拒集中在一类"措辞与文档偏差较大"的
问题上（如"未达标"vs 文档里的"未达到"），这正是**真实语义 embedding 能补上、而词法兜底补不上**
的部分——一个具体、可验证的"为什么需要真 embedding"的论据。

### 评测集自检

改过文档之后 gold 片段过期，会导致指标突然掉 20 个点却找不到原因。所以 `eval --check` 会校验
每个 `gold_source` 是否在库、每个 `gold_snippet` 是否还能在 top-20 里找到：

```bash
python -m ragkb.cli eval --check          # 自检 + 评测 + 门禁
python -m ragkb.cli check                 # 体检：配置一致性 + 评测集 + 语料
```

---

## 实测：换语义 embedding 到底有没有用

这一节是**跑出来的数据，不是经验之谈**。本地跑真实中文语义模型（BGE-small-zh-v1.5，
ONNX 量化版 23MB，无需 torch）：

```bash
python scripts/fetch_model.py                    # 下载模型（约 23MB）
pip install "onnxruntime>=1.19.2,<1.20" tokenizers
python scripts/compare_configs.py               # 多配置对比评测
python scripts/diagnose_case.py --config onnx   # 逐案看"谁把 gold 挤下去了"
```

### 整体指标（26 条评测集，其中 4 条是改写提问）

| 配置 | hit@5 | recall@5 | MRR | nDCG@5 | 拒答准确 | 误拒率 | 误答率 | 检索p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **baseline**（hashing 词法 + 词法重排） | 0.955 | 0.955 | 0.827 | 0.908 | 0.808 | 0.227 | 0.000 | 1.5ms |
| **baseline-norerank**（关掉重排） | 0.955 | 0.955 | 0.873 | 0.941 | 0.808 | 0.227 | 0.000 | 1.2ms |
| **onnx**（语义，门禁仍是纯词法） | 0.955 | 0.955 | 0.841 | 0.917 | 0.769 | 0.273 | 0.000 | 4.0ms |
| **onnx-semantic-gate**（语义 + 语义拒答豁免） | 0.955 | 0.955 | 0.841 | 0.917 | **0.885** | **0.136** | **0.000** | 4.2ms |
| **onnx-norerank**（语义 + 关词法重排） | 0.955 | 0.955 | **0.886** | **0.950** | **0.885** | **0.136** | **0.000** | 3.5ms |

整体 `hit@5` 四个配置完全一样——**只看这张表会得出"语义 embedding 没用"的错误结论**。

### 分开看改写提问（`paraphrase` 标签，用户按自己的说法问）

| 配置 | hit@k | MRR | nDCG@k |
| --- | --- | --- | --- |
| baseline（纯词法） | 0.750 | 0.625 | 0.658 |
| onnx（语义） | **1.000** | 0.750 | 0.815 |
| onnx-norerank | **1.000** | **0.875** | **0.908** |

差距全在这里：**纯词法有 1/4 的改写提问完全找不到，语义模型全部能找到。**
逐条看余弦相似度（同一对「查询 / 文档片段」）：

| 查询 → 文档片段 | 语义(BGE) | 词法(hashing) |
| --- | --- | --- |
| 出差住宿一线城市每晚上限是多少？ → 一线城市住宿每晚上限 700 元 | 0.835 | 0.641 |
| … → **酒店每晚最多能报多少钱**（同义改写，无共同词条） | **0.656** | 0.182 |
| … → P0 故障的解决时限是 4 小时（无关） | 0.419 | 0.093 |
| **SLA 未达标最多能赔多少？** → 未达到解决时限的按比例赔付 | **0.661** | **0.074** |

最后一行就是纯词法门禁误拒的那个 case：词法认为 0.074（几乎无关），语义认为 0.661（明确相关）。

### 三个反直觉的发现

1. **语义 embedding 单独加进来会让指标变差**（拒答准确 0.808 → 0.769）。原因是稠密检索的
   典型失效：`sla-ticket`（提交工单要提供哪些信息）被语义模型判成最像员工入职的「需提交哪些材料」
   （cos 0.558，向量排名第 1）——**功能/句式相似、实体完全不同**。RRF 把这种错块顶上来，
   反而挤掉了 BM25 本来排第 1 的正确块。
2. **词法重排在和语义召回打架**：关掉它，语义配置的 MRR 从 0.841 升到 0.886。
   手写的词法重排和语义召回目标不一致；正确做法不是关掉重排，而是换成真正的交叉编码器。
3. **纯词法门禁会把语义检索的收益吃掉**：所以加了「语义豁免」（`RAGKB_REFUSE_VECTOR_SIMILARITY_FLOOR`）。
   这是换 embedding 时必须一起改的一环，否则检索侧的提升在拒答环节被抵消。

### 结论：什么时候该上真实 embedding

- **语料小、且用户问法贴近文档原文**（关键词/编号/术语查询）→ BM25 已经够好，语义通道只带来噪声。
  本项目的示例语料就是这个情况。
- **用户会用自己的话提问**（这是真实场景的常态）→ 语义 embedding 是刚需，纯词法会丢掉 1/4 的问题。
- **无论哪种情况，都要上语义门禁**，否则误拒率会比纯词法更差。
- 本评测集只有 4 条改写样本，**样本量不足以做统计结论**，但机制清楚、方向一致。

### 语义门禁阈值必须在自己的语料上定（而且它调不完美）

把每个样本的「向量最高相似度」排出来看，会发现**两个分布是重叠的**：

| 群体 | 相似度区间 |
| --- | --- |
| 可答样本（22 条） | 0.487 ~ 0.792 |
| 应拒样本（4 条） | 0.399 ~ **0.521** |

`acl-salary-public`（0.521，权限被挡但语义相邻）和 `refuse-competitor`（0.518）比 3 条可答样本
（`short-term` 0.487、`para-onboarding` 0.500、`para-lateclaim` 0.517）还要高。所以阈值只能取舍：

| floor | 误拒 | 误答 |
| --- | --- | --- |
| 0.52 | 3/22 | 1/4 |
| 0.55 | 5/22 | 0/4 |
| **0.58**（默认） | 7/22 | **0/4** |

默认取 0.58：**先保证不编造**。这个重叠说明"单靠一个阈值调不好"——真正的解法是更强的模型
（bge-base / bge-m3）、加一个真正的交叉编码器重排，或者让 LLM 兜第二层判断，而不是继续拧阈值。
用 `scripts/diagnose_case.py` 可以逐条看相似度与各通道排名。

---

## 可观测性：线上出问题怎么查

"答得不对"至少有四种原因，没有 trace 只能猜：

| 现象 | trace 里的判据 |
| --- | --- |
| 检索没召回 | `lexical_hits=0` 且 `vector_hits=0`，或 `coverage` 很低 |
| 召回了但排序靠后 | `channels` 里有该块但 `rank` 靠后；`best_score` 低 |
| 上下文不相关（噪声多） | `channels_only.vector_only` 占多数（语义召回带进噪声） |
| 模型没按上下文作答 | `invalid_citations` 非空，或 `groundedness` 低于阈值 |

线上指标聚合（`/metrics` 或 `stats --online`）：

```json
{
  "count": 226,
  "latency_ms": {"p50": 3.8, "p95": 4.8, "max": 7.8},
  "stage_ms_p95": {"condense": 0.0, "retrieve": 1.9, "generate": 2.8},
  "tokens": {"avg_prompt": 1416.6, "avg_completion": 47.5},
  "cost_usd": {"total": 0.0, "avg_per_query": 0.0},
  "refusal_rate": 0.2522,
  "models": ["scripted-extractive"]
}
```

阶段耗时分开统计是关键：**p95 涨了，是检索变慢还是模型变慢**，一眼可辨。
成本按 `price_table` 里的单价折算（USD/1K tokens），换成真实模型后自动生效。

---

## HTTP API 与 Web UI

`python -m ragkb.cli serve` 后访问 `http://127.0.0.1:8000/`（内置单页 UI：流式问答、引用溯源、
文档上传/删除、运行指标）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查（含语料与组件信息，**免鉴权**，供探针） |
| GET | `/stats` `/metrics` | 语料/组件统计；trace 与反馈聚合 |
| GET | `/documents` | 文档列表（按角色过滤） |
| POST | `/documents` | 上传入库 `{filename, content, acl}` |
| DELETE | `/documents/<doc_id>` | 删除文档及其全部切块、倒排、向量、权限 |
| POST | `/ask` | 问答，返回答案 + 引用 + **检索调试信息** |
| POST | `/chat` | 问答（SSE：`meta` → `delta`\* → `done`） |
| POST | `/feedback` | 反馈 `{trace_id, rating, comment}` |

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
     -H 'X-User-Roles: public' \
     -d '{"question":"P0 故障的解决时限是多久？"}' | python -m json.tool

# SSE 流式
curl -N localhost:8000/chat -H 'Content-Type: application/json' \
     -d '{"question":"出差住宿一线城市上限是多少？"}'
```

流式协议刻意不用 `EventSource`（只能 GET，且断线会自动重连导致重复提问），
前端用 `fetch` + `ReadableStream` 读取。

---

## 权限模型

- 每篇文档入库时打 ACL 标签（默认 `public`），**检索阶段就按角色过滤**（不是查出来再遮）；
- 请求通过 `X-User-Roles` 传角色，`none`/`all` 表示不过滤（仅管理员调试用）；
- 未打标签的块默认对所有人可见；`documents` 列表也按角色过滤；
- 上传时校验**不能给自己没有的标签**（防越权上传 `internal` 内容），但默认角色始终允许；
- **改权限不需要重算向量**：内容未变而 ACL 变化时，入库只更新权限表并返回
  `acl-only public -> internal` 的报告（`status=updated`、`chunks=0`）。这一点必须走独立路径——
  否则"内容 hash 相同就 skip"会把权限改动静默丢弃，等于把该收紧的文档继续公开。

演示（同一问题、不同角色）：

```bash
python -m ragkb.cli ask "P6 职级的薪酬带宽上限是多少？" --roles public
#   → 知识库中没有找到相关依据（internal 文档被检索阶段过滤掉）
python -m ragkb.cli ask "P6 职级的薪酬带宽上限是多少？" --roles public,internal
#   → 84.0 万元
```

---

## 配置

全部通过 `RAGKB_*` 环境变量或 `.env` 文件（进程环境优先）。
**[`.env.example`](.env.example) 是完整的配置清单与调参建议**，重点几个：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAGKB_EMBED_PROVIDER` | `hashing` | `hashing`（离线词法兜底）/ `openai`（任何兼容网关） |
| `RAGKB_LLM_PROVIDER` | `scripted` | `scripted`（离线抽取基线）/ `openai` |
| `RAGKB_RERANK_PROVIDER` | `lexical` | `none` / `lexical` / `http`（Cohere、Jina 风格） |
| `RAGKB_FINAL_TOP_K` | `6` | 送进模型的上下文条数（**context 预算的主旋钮**） |
| `RAGKB_REFUSE_COVERAGE_FLOOR` | `0.25` | 拒答门禁，宁高勿低（误答比误拒更伤信任） |
| `RAGKB_VECTOR_MIN_SIMILARITY` | `0.0` | 真实 embedding 建议 0.15~0.25 |

切换真实模型只需两组变量：

```bash
RAGKB_EMBED_PROVIDER=openai
RAGKB_EMBED_MODEL=BAAI/bge-m3
RAGKB_EMBED_BASE_URL=https://api.siliconflow.cn/v1
RAGKB_EMBED_API_KEY=sk-xxx
RAGKB_EMBED_DIM=1024

RAGKB_LLM_PROVIDER=openai
RAGKB_LLM_MODEL=deepseek-chat
RAGKB_LLM_BASE_URL=https://api.deepseek.com/v1
RAGKB_LLM_API_KEY=sk-xxx
```

> ⚠️ **换 embedding 模型后必须 `ingest --force` 重建**。旧向量与新向量不在同一空间，混用会得到
> "看起来能跑但排序全乱"的结果。本项目按 `embed_model` 隔离向量并在启动时告警
> （`RAGKB_*` 之外的目录/端口等也都可配，见 `.env.example`）。

---

## 目录结构

```
ragkb/                  核心包（零第三方依赖）
  config.py             配置（.env 解析，零依赖）
  types.py              数据结构（Document/Chunk/Retrieved/Citation/Answer/EvalCase）
  text.py               中英文归一化、字级二元分词、句子切分、F1
  chunking.py           结构化切块 + 切块质量报告
  loaders.py            md/txt/html/csv/docx/xlsx/pdf 加载（可选依赖惰性导入）
  store.py              SQLite：文档/切块/倒排(postings+df)/向量/权限 + BM25 + 余弦检索
  embedding.py          Embedder 抽象：HashingEmbedder / OpenAICompatEmbedder
  llm.py                ChatModel 抽象：ScriptedChat（抽取基线）/ OpenAICompatChat
  rerank.py             Reranker 抽象：Noop / Lexical / Http（失败自动降级）
  retrieve.py           混合召回 + RRF 融合 + 重排（含多子查询）
  query.py              多轮改写（含语料感知的关键词承接）、KB/闲聊路由、子问题拆解
  prompts.py            Prompt 集中管理（含 NO_ANSWER 哨兵协议）
  answer.py             生成、引用抽取、双层拒答、流式输出、降级兜底
  ingest.py             入库管线（增量幂等）
  obs.py                trace + 成本 + 线上指标聚合
  metrics.py            评测指标（检索/答案/拒答/引用）
  evaluate.py           评测执行器 + 基线回归门禁 + 评测集自检
  server.py             stdlib HTTP API + SSE
  static/index.html     内置 Web UI（单文件，无构建步骤）
  cli.py                命令行入口
eval/
  dataset.jsonl         22 条评测集（含拒答、ACL、多轮、表格类样本）
  thresholds.json       CI 门禁阈值与最大跌幅
  baseline.json         基线（`eval --update-baseline` 生成）
tests/                  162 个单元/端到端测试
scripts/
  smoke.py              端到端冒烟（离线、临时库）
  sweep_refusal.py      拒答阈值扫描
  run.ps1               Windows 常用命令封装
data/corpus/            示例语料（虚构公司"云雀科技"的制度文档）
```

---

## 测试

```bash
python -m unittest discover -s tests -t . -v      # 162 个用例，约 20 秒
python scripts/smoke.py                          # 端到端冒烟（离线，不污染正式库）
python -m ragkb.cli eval --check                 # 评测集自检 + 评测 + 门禁（退出码即 CI 结果）
```

覆盖范围（全部离线，不联网、不需要密钥）：

- **文本层**：二元切分的建索引/查询差异、停用词一致性、全角折叠、中文标点保留、句子切分、F1；
- **切块**：标题栈、表格/代码块原子性、超长块按句切分与 overlap、碎块合并、chunk_id 回填；
- **存储**：幂等版本递增、删除清理（切块/倒排/权限）、BM25 打分、向量余弦、维度不匹配报错、
  ACL 过滤、df 剪枝、统计口径；
- **检索**：RRF 数学（含单通道/双通道/截断）、召回明细、多子查询合并、向量通道故障降级、重排语义；
- **生成**：引用抽取与越界识别、双层拒答、多轮改写、模型故障降级、流式事件顺序与拼接一致性；
- **入库**：增量跳过 / 变更更新 / 强制重建 / 坏文件不中断；
- **HTTP**：真实 socket 上的鉴权、SSE、上传删除、413/403/404、角色越权拦截、静态目录穿越防护；
- **评测层**：指标数学、聚合口径（无 gold 样本不参与检索平均）、门禁逻辑、评测集自检。

> 测试不写系统临时目录，也不用 `tempfile.mkdtemp`——在 Windows 受限令牌/沙箱下它创建的目录缺少
> 可写 ACE（实测 `WinError 5`），所以统一用 `Path.mkdir` 在项目内建隔离目录。

---

## 已知限制

诚实列出没做或做得浅的部分——这些也是继续做的方向：

1. **默认 embedding 是词法兜底，不是语义模型**。`HashingEmbedder` 用特征哈希投影，能跑通全链路，
   但同义改写基本失效（实测同一对文本 0.074 vs 语义模型 0.661）。**已经有本地语义方案了**：
   `RAGKB_EMBED_PROVIDER=onnx` + `scripts/fetch_model.py`（BGE-small-zh，23MB，无需 torch），
   代价是检索 p95 从 1.2ms 涨到 4.2ms。要更强就接 `bge-m3` 一类的云端模型（接口已留好）。
2. **默认生成是抽取式基线**，不是生成式。它的价值是确定性和对照实验（能量化"换成 LLM 到底带来多少
   增益"），不要拿它当产品体验。
3. **`eval/dataset.jsonl` 只有 26 条**，够做门禁和演示，不够做统计显著的结论——尤其是只有 4 条
   改写提问，语义 embedding 的收益方向清楚但量级不可信。真实项目应该 200+ 条并按业务分层抽样，
   且定期从线上 badcase 回流。
4. **`groundedness` 是代理指标**，非事实验证。生产上应叠加 LLM-as-judge（接口留了位置）与人工抽检。
5. **PDF/扫描件**：`pypdf` 只抽文本层，扫描件（图片型 PDF）需要接 OCR；表格类 PDF 解析质量一般。
6. **多租户隔离**只做到 ACL 标签级别，没有按租户物理隔离索引（SQLite 单文件）。
7. **没有对话状态持久化**：`history` 由调用方传入，服务端不存会话。
8. **单进程多线程**服务。多进程部署时每个 worker 各持连接（WAL 支持多读单写），写并发高时
   SQLite 会成为瓶颈——届时应换 PostgreSQL，`store.py` 是唯一要改的文件。

---

## 容量与演进

| 规模 | 现状 | 建议 |
| --- | --- | --- |
| ≤ 1 万 chunk | 直接可用（BM25 p95 < 5ms，向量 p95 < 10ms） | 打开 rerank 与真实 embedding |
| 1 万 ~ 10 万 | 可用（向量暴力扫描 p95 < 80ms，numpy 加速） | 加 `prune_postings` 剪高频词、定期 `recompute_stats` |
| > 10 万 | 需要替换 | 向量 → Milvus/Qdrant；词法 → OpenSearch；存储 → PostgreSQL |

替换点是设计好的：`SqliteStore.lexical_search` / `vector_search` 两个方法，以及
`Embedder` / `Reranker` / `ChatModel` 三个抽象。上层 `RagApp` 装配处不用改。

召回率优化建议**按顺序**做（很多人一上来就换模型，其实顺序错了）：

1. 先补数据——很多"检索不准"其实是语料缺失；
2. 打开 rerank——性价比最高的一步；
3. 调 `chunk_size`（技术文档 600–900，制度类 800–1200）；
4. 最后才换 embedding 模型（要重建全库索引，成本最高）。

---

## 部署

```bash
cp .env.example .env          # 按需填 API Key
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

或直接跑（无依赖）：

```bash
python -m ragkb.cli demo                  # 幂等，已入库会全部 skipped
python -m ragkb.cli serve --host 0.0.0.0 --port 8000
```

对外暴露前务必：设 `RAGKB_API_KEY`、置于反向代理之后、开启 TLS 与限流。
备份只需复制 `data/ragkb.sqlite3`（WAL 模式下先 `PRAGMA wal_checkpoint(TRUNCATE)` 或停服）。

---

## 面试怎么讲这个项目

**一句话**：从零实现的 RAG 系统，重点不在"接了哪个模型"，而在**检索质量工程**与**可度量、可观测**。

简历条目可以直接用（数字都能在本仓库复现）：

> - 从零实现企业知识库问答系统（Python，核心零第三方依赖）：混合检索（SQLite 倒排 BM25 + 向量）
>   + RRF 融合 + 重排；自建 22 条评测集与 CI 门禁，`hit@5=1.00`、`MRR=0.87`、引用有效率 100%。
> - 设计 IDF 加权覆盖率拒答门禁（阈值由扫描实验确定），把"语料外提问"的**误答率压到 0**，
>   代价是 5.6% 误拒；配合模型哨兵形成双层拒答。
> - 建立全链路 trace 与成本统计（阶段耗时 p95、tokens、单次成本），线上"答错"可定位到
>   检索 / 排序 / 上下文 / 生成四个环节中的具体一环。
> - 完成增量幂等入库（内容 hash）、行级权限检索过滤、SSE 流式输出、降级策略（向量故障退 BM25、
>   模型故障退原文摘录）与 stdlib HTTP 服务 + Web UI；162 个离线单测全部通过。

**高频追问与答法**（都在本仓库里有实现或数据支撑）：

| 追问 | 怎么答 |
| --- | --- |
| 为什么用 RRF 不用加权求和？ | 量纲不同（BM25 无上界、余弦 0~1），归一化区间随查询变化很脆弱；RRF 只用排名，且加通道时不用重调权重。 |
| 怎么防止模型编造？ | 三层：prompt 强约束 + `NO_ANSWER` 哨兵 + 覆盖率门禁；并用 `citation_validity` / `groundedness` 持续度量。 |
| 怎么知道换了 embedding 变好了？ | 固定评测集 + CI 门禁（绝对阈值 + 相对基线最大跌幅），`eval --retrieval-only` 可只测检索、不花模型钱。 |
| 一次问答多少 token、多少钱、p95 多少？ | trace 里有 `usage` / `cost_usd` / 分阶段耗时；`/metrics` 聚合 p50/p95、平均 prompt tokens、拒答率。 |
| 并发上来怎么办？ | 现在是单进程多线程 + SQLite（WAL）；写并发高时换 PostgreSQL，检索层换 Milvus/OpenSearch，替换点已隔离在两个方法里。 |
| 中文为什么要自己写分词？ | 不引入词典依赖，用一元+二元切分（建索引）与二元优先（查询）；对未登录词/专业术语稳。踩过的坑是跨词边界伪词条 df=0，必须过滤后再做关键词承接。 |

---

## License

MIT
