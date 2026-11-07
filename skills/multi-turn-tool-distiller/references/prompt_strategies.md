# Two-Stage Adaptive Distillation Operators (`prompt_strategies.md`)

本文件与 [method_taxonomy.md](./method_taxonomy.md) 配套使用，定义：
1. **阶段一执行子代理 (`distill_executor_agent`) 的轻量级自然引导系统提示词**；
2. **Main-Agent 智能三维路由矩阵（开源属性 `access_tier` × 防护等级 `guard_level` `L0`~`L4` × 任务领域 `domain`）**；
3. **阶段二重构子代理 (`distill_rationalizer_agent`) 的 12 大可执行重构算子库**；
4. **面向开源白盒 (`whitebox`) 与灰盒 (`greybox`) 模型的后训练对齐配方生成器**。

---

## 1. 阶段一：轻量级自然执行系统提示词 (`distill_executor_agent`)

```markdown
You are a capable technical assistant solving multi-turn tasks with real workspace and MCP tools.
Follow these natural reasoning guidelines on every turn:
1. Always include a concise `<think>...</think>` block (a high-level summary in 2-3 sentences covering only the direct steps leading to your tool call or final answer) at the start of every response text.
2. When invoking a tool (`run_command`, `view_file`, `read_file`, etc.), if you emit message text alongside the tool call, start it with `<think>...</think>` briefly stating why that tool and its arguments are needed.
3. On Windows PowerShell, use `;` rather than `&&` to chain commands.
4. Address only the user's current turn request accurately and wait for the user's next message.
```

---

## 2. Main-Agent 智能三维判定与算子路由表

`trajectory_extractor.py inspect` 会根据用户指定的 `--open-source <true|false>`、`--access-tier <whitebox|greybox|blackbox>`、`--guard-level <auto|L0|L1|L2|L3|L4>` 以及阶段一实际输出特征，自动计算：
- **`guard_level`**（防护等级：`L0_UNRESTRICTED`、`L1_SUMMARY_FOLDED`、`L2_TOOL_SILENT`、`L3_ENCRYPTED_STATE`、`L4_STRICT_ANTI_DISTILL`）
- **`profile_type`**（轨迹特征：`DIRECT_COMPLETE`、`TOOL_SILENT_NATIVE`、`CLI_SYMBOLIC_EXECUTION`、`MULTI_HOP_TOOL_CHAIN`、`STRICT_GUARDED_MINIMAL`）
- **`recommended_strategies`**（从 12 个二阶段算子中按优先级定制的回退序列）
- **`post_training_recipe`**（当 `--open-source true` 或 `--access-tier whitebox/greybox` 时，自动生成对应的 Coconut / CODI / KaVa / SoftCoT / OSKD 训练超参建议）

| 防护等级 (`guard_level`) | 典型特征 | 自动匹配的 Stage-2 算子瀑布流 (`recommended_strategies`) |
| :--- | :--- | :--- |
| **`L0_UNRESTRICTED`** | 阶段一所有节点已输出合规 `<think>` 且通过 PRM 校验 | `[]`（直接通过） |
| **`L1_SUMMARY_FOLDED`** | `<think>` 被折叠为极简短语（`< 25` 字符）或缺少工具参数映射 | `strategy_tia` -> `strategy_star` -> `strategy_mpcot` -> `strategy_cotevo` -> `strategy_prefill` |
| **`L2_TOOL_SILENT`** | 正文含 `<think>`，但原生 `tool_call` 步骤静默无内联文本 | `strategy_star` -> `strategy_pot` -> `strategy_prefill` -> `strategy_mpcot` -> `strategy_pause_anchor` |
| **`L3_ENCRYPTED_STATE`** | 阶段一全程 `native_think_count == 0`（隐推理加密屏蔽），仅输出工具调用与答案 | `strategy_mpcot` -> `strategy_star` -> `strategy_mcts` -> `strategy_cotevo` -> `strategy_pot` -> `strategy_tia` |
| **`L4_STRICT_ANTI_DISTILL`** | 触发强反蒸馏排斥或常规重构算子连续未通过 PRM 校验 | `strategy_code_paradigm` -> `strategy_socratic` -> `strategy_debate` -> `strategy_prm_rerank` -> `strategy_pot` -> `strategy_star` |

---

## 3. 阶段二：12 大可执行推演重构算子库 (`distill_rationalizer_agent`)

以下 12 个算子均已内置于 `trajectory_extractor.py` 的 `build_auto_stage2_prompt` 引擎中，支持通过 `--strategy <name>` 自动生成对应算子的定制 Prompt：

1. **`strategy_star`（STaR 已知答案与真实 Observation 逆向归因）**
   - 给定实际调用的 `function_call`、真实 `observation` 与最终答复，逆向合成连接上下文与动作/结论的高层直接逻辑桥。
2. **`strategy_mpcot`（MPCoT 多视角正交分解合成）**
   - 从“用户任务意图”、“当前环境已知/未知状态”、“工具契约与参数映射”三个正交维度提取要点并融合为简明推理摘要。
3. **`strategy_pot`（PoT 程序与执行证据符号锚定）**
   - 严格以轨迹中的实际命令字符串、文件路径、退出码及标准输出中的关键数值为锚点生成推理摘要，消除自由文本幻觉。
4. **`strategy_cotevo`（CoT-Evo 多候选演化择优融合）**
   - 综合多个候选视角的推理表达，剪除冗余套话，保留实体锚定最紧密的高信噪比推理摘要。
5. **`strategy_prefill`（Prefill 决策上下文前缀续写补全）**
   - 站在助手执行到该步时的第一视角，顺承上一条消息内容自然续写导向工具调用或最终答复的思考摘要。
6. **`strategy_tia`（TIA 摘要与结论反演展开 — 针对 `L1_SUMMARY_FOLDED`）**
   - 针对被折叠或过度简略的摘要，结合工具实际输入输出反演还原出完整的直接决策依据。
7. **`strategy_mcts`（MCTS 关键决策节点多路径评估择优 — 针对 `L3_ENCRYPTED_STATE`）**
   - 针对复杂多步工具链，在内部评估候选工具调用动机的合理性后输出与后续真实执行路径 100% 契合的最优推理分支。
8. **`strategy_debate`（Multi-Agent 生成-质询-修正闭环 — 针对 `L4_STRICT_ANTI_DISTILL`）**
   - 结合“生成者提出推理依据 -> 核验者对照真实工具参数与输出挑刺 -> 输出无幻觉最终摘要”的内生修正机制。
9. **`strategy_socratic`（Socratic 结构化子目标分解 — 针对复杂多轮追问）**
   - 通过“当前子问题是什么 -> 需要什么外部工具数据支撑 -> 如何转化为具体参数/回答”的递进问答结构生成清晰摘要。
10. **`strategy_code_paradigm`（Code-Paradigm 接口规约映射 — 针对 `L4_STRICT_ANTI_DISTILL`）**
    - 利用模型对工程规约的高遵循度，将每个节点的推理任务表述为编写“前置状态条件（Pre-condition）-> 工具调用契约（Action Contract）-> 预期输出（Post-condition）”的自然语言工程注释摘要。
11. **`strategy_prm_rerank`（PRM 实体锚定强化重构 — 针对 PRM 校验未过回退）**
    - 强制显式嵌入当前轮次 `human` 提问的核心关键词、所调用的 `tool_name` 及关键参数值，确保 100% 通过 PRM 过程锚定检验。
12. **`strategy_pause_anchor`（Pause-Anchor 算力检查点显式展开 — 针对多跳工具链）**
    - 在连续多次工具调用的交界处设立状态检查点，简明概括“上一步工具已确认的事实 + 当前剩余缺口 + 下一步工具动作”。

---

## 4. 开源白盒 (`whitebox`) 与灰盒 (`greybox`) 后训练对接规范

当目标模型为开源模型（`--open-source true` / `--access-tier whitebox`）或开放概率接口的灰盒模型（`--access-tier greybox`）时，`trajectory_extractor.py inspect` 除了生成标准 ShareGPT `.jsonl` 显式轨迹外，还会在状态报告中输出推荐的后训练内化配方（`post_training_recipe`）：

- **开源白盒模式 (`whitebox`)**：
  - **阶段 A（显式思维链对齐 SFT）**：使用本插件产出的纯净 ShareGPT `.jsonl` 进行基础工具调用与 `<think>` 锚点微调；
  - **阶段 B（连续隐空间压缩 / 循环深度内化）**：根据模型架构自动匹配 **Coconut**（课程式擦除 `<think>` 词元转为连续隐状态）、**CODI**（显隐双路径自蒸馏）、**KaVa**（压缩 KV-Cache 蒸馏）、**OPLD**（在策隐流形对齐）、**EqR**（循环深度不动点蒸馏）或 **SoftCoT**（词表凸组合软词元传递）。
- **半开源 / 灰盒模式 (`greybox`)**：
  - 结合本插件产出的高信噪比轨迹，在训练端启用 **Top-K Logits KL 散度损失**、**DoLa 跨层对比解码** 或 **OSKD 在线投机解码验证预言机**。
