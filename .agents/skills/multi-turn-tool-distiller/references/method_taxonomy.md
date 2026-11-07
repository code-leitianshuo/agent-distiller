# 隐推理提取与知识蒸馏 12 维 66 法全景手册 (`method_taxonomy.md`)

本手册完整收录大模型隐推理（Hidden / Latent Reasoning）提取、提纯与内化的 **12 大技术维度、66 种细分方法（`M01` ~ `M66`）**。每种方法均明确定义了 **方法编号 (`M01`~`M66`)**、**CLI 策略键 (`--select-methods`)**、**适用权限与防护等级**、**使用条件（什么时候用）** 以及 **详细使用方法（怎么用）**。

---

## 一、多维细化自由选择与智能路由体系

用户或 Main-Agent 可通过以下 **两种方式** 灵活选用全部 66 种方法：

### 方式 1：自由点选与自定义组合（精确到任意单方法或多方法序列）
通过 `--select-methods` 参数，支持直接传入 **方法编号（`M01`~`M66`）** 或 **策略键名**，自由指定单条方法或自定义回退瀑布流顺序；也可通过 `--exclude-methods` 排除特定方法：

```powershell
# 示例 A：自由指定先用 M15 (EchoCoT 工具状态草稿锚定)，再用 M60 (PoT 程序符号推演)，并生成 M34 (Coconut) 白盒训练配置
python skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py inspect `
  --transcript "<Stage1_Transcript_Path>" `
  --sample-id "sample_001" `
  --select-methods "M15,M60,M34" `
  --output "datasets/stage1_sample_001.json"

# 示例 B：按维度或关键词检索方法库（查看某维度全部方法的使用条件与用法）
python skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py list-methods --dimension D02
python skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py list-methods --guard-level L4_STRICT_ANTI_DISTILL
```

### 方式 2：五维细化条件自动路由（Main-Agent 多维画像决策）
当未手动锁定 `--select-methods` 时，Main-Agent 根据以下 **5 个细化维度** 自动筛选并排序最优方法序列：

| 路由维度 | CLI 参数 | 可选细化值 | 路由作用说明 |
| :--- | :--- | :--- | :--- |
| **1. 开源属性与访问层级** | `--open-source` / `--access-tier` | `blackbox`（闭源纯文本）<br>`greybox`（开放 Top-K Logits / 投机验证）<br>`whitebox`（开源权重 / 隐状态 / KV-Cache）<br>`hybrid`（黑盒采样 + 白盒内化联合）<br>`auto`（默认自动判定） | 决定启用 **阶段二在线重构算子**、**灰盒分布/预言机采样** 还是 **白盒连续隐空间后训练配方** |
| **2. 模型防护等级** | `--guard-level` | `L0_UNRESTRICTED`（原生完整外显 `<think>`）<br>`L1_SUMMARY_FOLDED`（思考被折叠为简短摘要）<br>`L2_TOOL_SILENT`（正文有思考，工具调用步静默）<br>`L3_ENCRYPTED_STATE`（全链路隐推理加密/剥离）<br>`L4_STRICT_ANTI_DISTILL`（强反蒸馏拒答或格式排斥）<br>`auto`（根据阶段一轨迹自动诊断） | 针对不同护栏强度匹配破局策略（如 `L1` 用反演展开，`L2` 用工具状态逆向归因，`L4` 用代码范式与苏格拉底助产术） |
| **3. 任务领域与轨迹形态** | `--task-domain` | `tool_agent`（多跳工具调用）<br>`code_symbolic`（命令行/代码/SQL/AST）<br>`math_logic`（严密数学与逻辑约束）<br>`long_context`（超长文档与跨轮状态）<br>`multimodal`（语音/视觉跨模态）<br>`general` / `auto`（根据使用的工具自动识别） | 将任务特征与算子专长对齐（如命令行任务优先路由 `M60 PoT` 与 `M61 AST`） |
| **4. 推理延迟预算 / 加速目标** | `--latency-budget` | `unconstrained`（1x 完整显式自然语言思维链）<br>`moderate`（1~2x 紧凑摘要 / 灰盒软标签）<br>`low_latency`（3~10x 连续隐词元 / KV 缓存压缩）<br>`ultra_fast`（5~15x 循环深度不动点 / 在策流形） | 为学生模型后训练阶段匹配满足线上延迟预算的压缩与内化算法 |
| **5. 技术维度定向过滤** | `--dimension` | `D01` ~ `D12`（支持逗号分隔多选，如 `D01,D03,D10`）或 `all` | 限定仅在指定的技术流派内选择方案 |

---

## 二、12 大技术维度、66 种方法全景速查与详细指南

---

### 维度 D01：黑盒提示诱导与逆向重构层（Black-Box Prompting & Post-Hoc Reconstruction）
> **核心定位**：针对仅开放标准文本对话接口的闭源教师模型，利用指令遵循先验与已知输出逆向条件概率 $P(Z \mid X, A, O, Y)$，在不破坏阶段一自然执行分布的前提下，外化或事后重构高层推理摘要。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 适用防护等级 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M01** | **结构化元提示拆解**<br>(Structured Meta-Prompting) | `strategy_meta_prompt` | `blackbox`<br>`L0` / `L1` / `L2` | **使用条件**：目标模型指令遵循能力强、未设置强反蒸馏护栏（`L0`~`L2`），希望在单次生成或阶段二补全中按固定认知框架输出简要步骤。 | **使用方法**：<br>1. 在阶段一轻量系统提示或阶段二 Prompt 中植入“目标识别 → 前置条件核对 → 核心操作推导”三段式元框架；<br>2. CLI 调用：`--select-methods M01` 或 `strategy_meta_prompt`。 |
| **M02** | **回复预填充引导**<br>(Assistant Response Prefilling) | `strategy_prefill` | `blackbox`<br>`L1` / `L2` / `L3` | **使用条件**：模型习惯直接输出结论或直调工具、默认跳过 `<think>`，但接口或阶段二子 Agent 支持给定起始前缀顺延续写。 | **使用方法**：<br>1. 为每个缺失思考的节点提取首句上下文锚点（如“为解答关于 X 的问题，需先调用工具 Y 获取…”）；<br>2. 令 `distill_rationalizer_agent` 从该前缀顺势补全 2~3 句直接推导；<br>3. CLI 调用：`--select-methods M02`。 |
| **M03** | **已知答案与动作逆向归因**<br>(STaR Post-Hoc Rationalization) | `strategy_star` | `blackbox`<br>`L1` / `L2` / `L3` | **使用条件**：阶段一已成功拿到真实的 `function_call` 参数、`observation` 返回值与最终回答，仅缺失中间 `<think>` 节点（最通用的基础重构算子）。 | **使用方法**：<br>1. 将完整轨迹 $(X, A, O, Y)$ 作为已知先验喂给阶段二子 Agent；<br>2. 严格基于真实工具参数与观察结果逆向合成连接问题与动作的最直接推理桥；<br>3. CLI 调用：`--select-methods M03`。 |
| **M04** | **词元级隐式前瞻合理化**<br>(Quiet-STaR Lookahead Rationalization) | `strategy_quiet_star` | `blackbox` / `whitebox`<br>`L1` / `L2` / `L3` | **使用条件**：多轮对话中存在隐含转折或跨步工具调用，需要在每个关键决策词元/动作点前注入面向未来多步收益的前瞻合理化说明。 | **使用方法**：<br>1. 识别轨迹中改变后续走向的关键动作点；<br>2. 结合后续轮次的真实 `observation` 收益，在当前步生成解释“为何该动作能支撑后续步骤”的前瞻摘要；<br>3. CLI 调用：`--select-methods M04`。 |
| **M05** | **多智能体对抗质询与修正**<br>(Multi-Agent Critique & Refine Debate) | `strategy_debate` | `blackbox`<br>`L2` / `L3` / `L4` | **使用条件**：单次逆向重构容易出现逻辑跳步、遗漏工具参数来源或未通过 `check_prm_grounding` 锚定校验时。 | **使用方法**：<br>1. 在阶段二 Prompt 中内嵌“生成者（Proposer）→ 挑刺审计员（Critic）→ 终审合成器（Synthesizer）”三步闭环；<br>2. 强制核对每个工具名与参数值在上下文中的出处后输出定稿；<br>3. CLI 调用：`--select-methods M05`。 |
| **M06** | **苏格拉底助产术子问题拆解**<br>(Socratic Maieutic Decomposition) | `strategy_socratic` | `blackbox`<br>`L2` / `L3` / `L4` | **使用条件**：遭遇 `L4` 反蒸馏护栏（对“输出内部思考”敏感拒答）或复杂多跳推理任务，需通过客观自问自答还原决策逻辑。 | **使用方法**：<br>1. 摒弃任何“思考过程/内心独白”字眼，改为围绕任务构建 2~3 个递进式客观子问题（“已知什么？缺什么数据？哪个工具能提供？”）；<br>2. 将子问题问答浓缩为 `<think>` 摘要；<br>3. CLI 调用：`--select-methods M06`。 |
| **M07** | **由简入繁递进分解**<br>(Least-to-Most Progressive Decomposition) | `strategy_least_to_most` | `blackbox`<br>`L1` / `L2` / `L3` | **使用条件**：多轮追问或复合工具链任务，后续步骤强依赖前置基础子任务的执行结果。 | **使用方法**：<br>1. 将复合任务按依赖层级拆解为最小原子子问题序列；<br>2. 每个节点仅针对当前最小子问题与对应工具调用生成直接推导摘要；<br>3. CLI 调用：`--select-methods M07`。 |
| **M08** | **后退一步抽象原理锚定**<br>(Step-Back Abstraction Grounding) | `strategy_step_back` | `blackbox`<br>`L1` / `L2` / `L3` | **使用条件**：涉及架构审计、复杂规则判定、数学/科学定律或底层原理分析，直接就事论事容易陷入局部细节而丢失全局逻辑。 | **使用方法**：<br>1. 先从具体问题后退一步，提炼该步所依据的通用准则、文件结构规范或接口契约；<br>2. 再将该准则映射到当前具体工具参数选型；<br>3. CLI 调用：`--select-methods M08`。 |
| **M09** | **先规划后求解显式锚定**<br>(Plan-and-Solve Explicit Blueprint) | `strategy_plan_and_solve` | `blackbox`<br>`L1` / `L2` / `L3` | **使用条件**：单轮内需要连续发起多次工具调用（如先检索目录、再读取文件、最后运行测试），需要清晰的全局执行蓝图。 | **使用方法**：<br>1. 在首轮工具调用前生成高层执行计划（Plan）；<br>2. 在后续每次收到 `observation` 后锚定当前处于计划的第几步及下一步工具动作（Solve）；<br>3. CLI 调用：`--select-methods M09`。 |
| **M10** | **正反对比思维链归因**<br>(Contrastive CoT Grounding) | `strategy_ccot` | `blackbox`<br>`L2` / `L3` | **使用条件**：候选工具集存在多个功能相近工具（如 `read_file` vs `run_command`），需要明确解释为何选 A 而不选 B。 | **使用方法**：<br>1. 对比当前动作与潜在易错备选动作的适用边界；<br>2. 用一句话阐明选用当前工具及参数的直接决定因素；<br>3. CLI 调用：`--select-methods M10`。 |

---

### 维度 D02：代码范式映射、工具状态驻留与上下文预算层（Code Paradigm, Tool State Window & Budget Grounding）
> **核心定位**：针对前沿模型（如 o 系列、Claude Thinking、Gemini Thinking）在常规对话下隐藏思维链或启用 `L3`/`L4` 强防护的场景，利用代码范式效应、多轮工具状态驻留窗口（EchoCoT）与长度预算信号，在合规沙箱边界内实现高保真还原。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 适用防护等级 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M11** | **代码与接口规约范式转换**<br>(Code / Specification Paradigm Framing) | `strategy_code_paradigm` | `blackbox`<br>`L3` / `L4` | **使用条件**：目标模型在自然语言对话模式下触发 `L4_STRICT_ANTI_DISTILL` 拒答或拒绝解释推理过程，但对代码工程文档、函数契约（Docstring）或接口规约完全放行。 | **使用方法**：<br>1. 将阶段二任务包装为“为每个工具调用与返回值编写 `@precondition`、`@action_rationale` 与 `@postcondition` 工程规约”；<br>2. 将生成的规约条目无损转换为紧凑 `<think>` 摘要；<br>3. CLI 调用：`--select-methods M11`。 |
| **M12** | **跨轮会话状态连续性锚定**<br>(Cross-Turn Session State Continuity) | `strategy_state_continuity` | `blackbox`<br>`L2` / `L3` | **使用条件**：多轮长对话（`target_turns >= 2`）中，后一轮的工具决策依赖前一轮已获取的状态或中间结论，需消除跨轮逻辑断层。 | **使用方法**：<br>1. 提取前序轮次的最终状态快照（State Snapshot）注入当前轮次逆向归因上下文；<br>2. 明确写出“基于上一轮已确认的 X，本轮进一步执行 Y”；<br>3. CLI 调用：`--select-methods M12`。 |
| **M13** | **跨尺寸同族模型对齐转录**<br>(Cross-Tier Teacher-Student Alignment) | `strategy_tier_alignment` | `blackbox` / `greybox`<br>`L3` / `L4` | **使用条件**：高阶旗舰模型输出极其精炼或隐藏了中间步骤，需借助同家族指令遵循更开放的轻量/中量级子 Agent 实例展开可读中间步骤。 | **使用方法**：<br>1. 阶段一使用旗舰配置采集高准确度动作轨迹；<br>2. 阶段二拉起指定 `Model="flash"` 或同族辅助子 Agent，锚定旗舰动作轨迹完成可读推理摘要展开；<br>3. CLI 调用：`--select-methods M13`。 |
| **M14** | **长上下文自动摘要逆向展开**<br>(Context Compaction Summary Unfolding) | `strategy_context_compaction` | `blackbox`<br>`L1` / `L3` | **使用条件**：长链路多工具调用（消息数 `> 12`）或存在会话历史压缩摘要，需从高度浓缩的阶段性结论中还原各步直接依据。 | **使用方法**：<br>1. 先提取全链路全局里程碑摘要（Milestone Summary）；<br>2. 再按时间片将每个里程碑拆解回对应工具调用节点的 2~3 句直接推导；<br>3. CLI 调用：`--select-methods M14`。 |
| **M15** | **工具调用链状态驻留草稿锚定**<br>(EchoCoT Multi-Tool State Scratchpad) | `strategy_echocot_scratchpad` | `blackbox`<br>`L2` / `L3` / `L4` | **使用条件**：单轮内包含连续多次工具调用（Multi-Hop Tool Chain），模型在两次工具调用之间必须维持内部状态一致性。 | **使用方法**：<br>1. 锁定相邻两次 `function_call` 之间的 `observation` 差值（Delta State）；<br>2. 围绕“上一步工具返回了什么具体数据 → 暴露了什么新需求 → 驱动下一步传入什么参数”构建状态转移推理桥；<br>3. CLI 调用：`--select-methods M15`。 |
| **M16** | **词元长度与复杂度预算预言机**<br>(Token-Count & Complexity Budget Oracle) | `strategy_budget_oracle` | `blackbox` / `greybox`<br>`L1` / `L2` / `L3` | **使用条件**：需要根据工具返回负载大小、命令复杂度或接口返回的 `reasoning_tokens` 元数据，精确控制每个节点 `<think>` 摘要的信息密度与句数。 | **使用方法**：<br>1. `trajectory_extractor.py` 自动计算每步工具参数复杂度与 `observation` 字节数，生成目标句数/字数预算（如简单步 1~2 句、复杂多跳步 2~3 句）；<br>2. 将预算作为硬约束注入阶段二生成；<br>3. CLI 调用：`--select-methods M16`。 |

---

### 维度 D03：搜索树展开、过程奖励评估与轨迹反演层（Tree/Graph Search, PRM & Trajectory Inversion）
> **核心定位**：克服单次采样容易混入“结果正确但过程跳步/幻觉”的缺陷，通过外显树/图搜索、步级过程奖励模型（PRM）过滤以及摘要反演网络，确保每一步推理严格锚定真实上下文与工具结果。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 适用防护等级 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M17** | **蒙特卡洛树搜索轨迹重构**<br>(MCTS Lookahead Reconstruction) | `strategy_mcts` | `blackbox`<br>`L2` / `L3` | **使用条件**：多步工具探索任务（如定位未知目录、排查代码错误），需要根据最终成功的执行路径逆向赋予各节点最高价值的决策理由。 | **使用方法**：<br>1. 以整条成功轨迹的终端奖励为回传信号（Backpropagation Value）；<br>2. 为每个中间工具节点生成最大化后续信息增益的直接推理说明；<br>3. CLI 调用：`--select-methods M17`。 |
| **M18** | **思维树分叉评估展开**<br>(Tree-of-Thoughts Beam Expansion) | `strategy_tot` | `blackbox`<br>`L2` / `L3` | **使用条件**：关键决策节点存在多种可行分解路径，需先并行生成多条候选推理摘要再择优录取。 | **使用方法**：<br>1. 令阶段二子 Agent 针对目标节点生成 $K=2\sim 3$ 条简短候选推导；<br>2. 选取与实际 `function_call` 参数契合度最高的一条写入轨迹；<br>3. CLI 调用：`--select-methods M18`。 |
| **M19** | **思维图有向聚合重构**<br>(Graph-of-Thoughts DAG Aggregation) | `strategy_got` | `blackbox`<br>`L2` / `L3` | **使用条件**：模型在回答前并行调用了多个独立工具（如同时读取多个文件），需要将多路独立的工具观察结果汇合（Aggregate）为统一结论。 | **使用方法**：<br>1. 将多个前置工具的 `observation` 视为有向无环图（DAG）的入边节点；<br>2. 在最终汇总节点显式说明如何合并各路观察证据得出最终回答；<br>3. CLI 调用：`--select-methods M19`。 |
| **M20** | **思维缓冲区元模板检索**<br>(Buffer-of-Thoughts Meta-Template) | `strategy_bot` | `blackbox`<br>`L1` / `L2` / `L3` | **使用条件**：同类领域任务批量蒸馏（如全量做 Git 仓库审计或全量做 API Schema 校验），希望复用经验证的高质量推理结构模板。 | **使用方法**：<br>1. 根据任务的 `domain` 标签匹配领域元推理槽位（如 `[目标文件] -> [解析方式] -> [校验结论]`）；<br>2. 填入当前样本的真实实体与参数；<br>3. CLI 调用：`--select-methods M20`。 |
| **M21** | **环境反馈语言强化自省**<br>(Reflexion Verbal Reinforcement) | `strategy_reflexion` | `blackbox`<br>`L2` / `L3` | **使用条件**：阶段一轨迹中包含工具报错、非零退出码或参数修正过程，需要准确解释模型如何根据环境报错调整下一步命令。 | **使用方法**：<br>1. 捕获前一步 `observation` 中的异常/报错特征；<br>2. 在紧随其后的 `gpt` 节点生成“根据返回状态 X 调整参数为 Y”的直接应对逻辑；<br>3. CLI 调用：`--select-methods M21`。 |
| **M22** | **多路自洽性投票与共识提纯**<br>(Self-Consistency Consensus) | `strategy_self_consistency` | `blackbox`<br>`L2` / `L3` | **使用条件**：对推理摘要的事实一致性要求极高，需消除单次生成的偶然措辞偏差。 | **使用方法**：<br>1. 针对同一目标节点采样多份独立推理摘要；<br>2. 提取各份摘要共同包含的核心实体、工具名与因果链条合成共识版本；<br>3. CLI 调用：`--select-methods M22`。 |
| **M23** | **过程奖励模型步级过滤与重排序**<br>(PRM Step-Level Scoring & Reranking) | `strategy_prm_rerank` | `blackbox`<br>`L1` ~ `L4` | **使用条件**：所有样本落盘前的强制质检环节，或需要从多个候选推理摘要中按锚定得分自动选出最优解。 | **使用方法**：<br>1. 调用内置 [`check_prm_grounding`](file:///f:/zl/skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py) 对每步 `<think>` 进行工具名、参数键、文件路径及观察结果实体的步级锚定打分；<br>2. 自动拦截空洞套话并重排序录取最高分候选；<br>3. CLI 调用：`--select-methods M23`。 |
| **M24** | **推理轨迹反演展开网络**<br>(TIA Trajectory Inversion from Summary) | `strategy_tia` | `blackbox` / `whitebox`<br>`L1` / `L3` | **使用条件**：专门针对 `L1_SUMMARY_FOLDED`（模型仅给出一句话笼统摘要或简短结论），需将浓缩摘要反演还原为清晰的因果步骤。 | **使用方法**：<br>1. 提取已有简短摘要或首句结论作为反演锚点；<br>2. 结合输入问题与工具上下文，将其展开为 2~3 句具体直接步骤；<br>3. CLI 调用：`--select-methods M24`。 |

---

### 维度 D04：灰盒概率分布、变分推断与生成流网络层（Grey-Box Logits, Variational Inference & GFlowNets）
> **核心定位**：当目标模型或推理网关开放词元级 Top-K 对数概率（Logits）或需要在多条竞争性推理分支上迁移分布暗知识（Distributional Dark Knowledge）时使用。支持生成标准化灰盒训练配置与分叉采样指令。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 加速比 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M25** | **词元级 Top-K 对数概率蒸馏**<br>(Top-K Logits KL Distillation) | `greybox_logits_kl` | `greybox`<br>1~2x | **使用条件**：教师接口支持返回 `logprobs`（如 vLLM / SGLang / OpenAI 兼容接口），希望学生模型直接拟合教师在工具名与关键参数词元上的概率分布。 | **使用方法**：<br>1. 运行 `inspect --access-tier greybox --select-methods M25`；<br>2. 自动导出包含 Top-K（默认 $K=20$）前/反向 KL 散度损失权重的后训练配置文件；<br>3. 在 SFT 训练器中挂载软标签损失。 |
| **M26** | **高熵决策节点分叉采样**<br>(High-Entropy Node Branching) | `strategy_entropy_branch` | `greybox` / `blackbox`<br>1x | **使用条件**：模型在多个工具或参数之间存在高不确定性（信息熵激增），需要捕获并对比竞争性分支以说明为何选择当前分支。 | **使用方法**：<br>1. 识别轨迹中存在多工具竞争的决策节点；<br>2. 在阶段二显式针对分叉点说明当前选择相对于备选分支的确定性依据；<br>3. CLI 调用：`--select-methods M26`。 |
| **M27** | **跨层对比解码逻辑放大**<br>(DoLa Layer-Contrastive Decoding) | `greybox_dola` | `greybox` / `whitebox`<br>1~1.5x | **使用条件**：可访问教师模型深层与浅层 Logits，希望剥离浅层语法套话、放大深层逻辑推理增量。 | **使用方法**：<br>1. 计算顶层成熟层（如第 32 层）与中低层（如第 16 层）Logits 之差 $\Delta \log P$；<br>2. 用对比放大后的分布监督学生模型；<br>3. CLI 调用：`--select-methods M27`。 |
| **M28** | **马尔可夫链潜变量变分推断**<br>(MARCOS Variational Markov Chain) | `greybox_marcos` | `greybox` / `whitebox`<br>1.5~2x | **使用条件**：缺乏显式思维链标注或显式标注噪声较大，希望将多步隐推理建模为马尔可夫潜变量链并最大化证据下界（ELBO）。 | **使用方法**：<br>1. 以采集到的 ShareGPT 多轮问答与工具对作为观测序列；<br>2. 训练摊销后验推断网络恢复步间潜状态转移矩阵；<br>3. CLI 调用：`--select-methods M28`。 |
| **M29** | **联合证据下界后验优化**<br>(JEPO Joint Evidence Posterior) | `greybox_jepo` | `greybox` / `whitebox`<br>1.5~2x | **使用条件**：多轮工具调用中输入问题与最终回答跨度极大，需联合优化隐式推理轨迹的先验生成模型与后验逆向网络。 | **使用方法**：<br>1. 将阶段二逆向归因模型作为后验分布 $q_\phi(Z \mid X, Y)$，学生模型作为先验 $p_\theta(Z \mid X)$；<br>2. 最小化先验与后验之间的双向散度；<br>3. CLI 调用：`--select-methods M29`。 |
| **M30** | **正则化潜变量轨迹引导**<br>(ReGuLaR Latent Regularization) | `greybox_regular` | `greybox` / `whitebox`<br>1.5~2x | **使用条件**：变分潜变量训练容易出现后验坍缩（Posterior Collapse，学生忽略潜变量直接猜答案）时。 | **使用方法**：<br>1. 引入基于真实工具动作与观察结果的互信息正则项，强制潜状态保留工具决策信息；<br>2. CLI 调用：`--select-methods M30`。 |
| **M31** | **生成流网络有向无环图采样**<br>(GFlowNets DAG Trajectory Sampling) | `greybox_gflownets` | `greybox` / `whitebox`<br>1.5~2x | **使用条件**：同一任务存在多条等价正确的工具调用顺序（DAG），传统 MLE/RL 容易模式坍缩到单一路径。 | **使用方法**：<br>1. 采用 Trajectory Balance (TB) 损失函数，使学生采样每条工具轨迹的概率正比于终端验证奖励 $R(X, Z, Y)$；<br>2. CLI 调用：`--select-methods M31`。 |

---

### 维度 D05：白盒连续隐空间压缩、记忆迁移与并行迭代层（White-Box Continuous Latent & KV-Cache Distillation）
> **核心定位**：针对拥有模型权重的开源白盒场景（`--open-source true` / `--access-tier whitebox`），打破自然语言词元的自回归延迟瓶颈，将显式 `<think>` 压缩至连续隐向量、压缩 KV-Cache 或并行雅可比迭代中，实现 **3~10 倍推理加速**。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 加速比 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M32** | **中间层隐状态投影对齐**<br>(Hidden-State MSE/Cosine Alignment) | `whitebox_hidden_mse` | `whitebox`<br>2~4x | **使用条件**：师生模型同构或层数成比例，希望学生模型在关键工具决策词元位置直接拟合教师中间层连续激活。 | **使用方法**：<br>1. 在师生对应层（如 `{1/4, 2/4, 3/4, 1}` 深度处）架设线性投影层 $W_{\text{proj}}$；<br>2. 在 `<think>` 结束符与工具调用起始符位置施加 Cosine + MSE 混合损失；<br>3. CLI 调用：`--select-methods M32`。 |
| **M33** | **多头注意力权重图谱迁移**<br>(Multi-Head Attention Map Transfer) | `whitebox_attention_map` | `whitebox`<br>2~3x | **使用条件**：长上下文工具调用任务，学生模型难以在冗长的 `observation` 中定位关键字段或文件路径。 | **使用方法**：<br>1. 提取教师模型最后 $L_{\text{top}}$ 层的多头注意力矩阵 $A_T$；<br>2. 最小化学生与教师注意力分布的 KL 散度 $\text{KL}(A_T \parallel A_S)$；<br>3. CLI 调用：`--select-methods M33`。 |
| **M34** | **连续隐式思维课程蒸馏**<br>(Coconut — Chain of Continuous Thought) | `whitebox_coconut` | `whitebox`<br>4~8x | **使用条件**：拥有经本插件蒸馏出的多步显式 `<think>` 数据集，希望将学生模型的显式思考逐步内化为 $K$ 个连续隐思维向量（Latent Thoughts）。 | **使用方法**：<br>1. 先用本插件采集带 `<think>` 的 ShareGPT 数据集；<br>2. 按课程学习分 $C$ 个阶段逐步将每步 `<think>` 替换为 `<bot>` 与 `<eot>` 之间的连续最后一层隐状态直传；<br>3. CLI 调用：`--select-methods M34`。 |
| **M35** | **自我蒸馏连续隐词元压缩**<br>(CODI — Continuous Distillation) | `whitebox_codi` | `whitebox`<br>5~10x | **使用条件**：希望在单阶段内训练同一个模型同时具备显式推理路径（教师分支）与固定 $M$ 个连续隐词元路径（学生分支）。 | **使用方法**：<br>1. 教师分支前向运行完整 ShareGPT `<think>` 序列；<br>2. 学生分支仅运行 $M=4\sim 8$ 个连续隐词元，在答案/工具生成前夕位置施加 Smooth-L1 隐状态对齐损失；<br>3. CLI 调用：`--select-methods M35`。 |
| **M36** | **步级解码器监督隐式思维**<br>(SIM-CoT — Step-Supervised Implicit CoT) | `whitebox_simcot` | `whitebox`<br>4~8x | **使用条件**：单纯使用 Coconut/CODI 压缩后出现隐空间语义坍缩（多个连续隐词元趋同、丢失步骤区分度）时。 | **使用方法**：<br>1. 在训练期挂载轻量步级辅助解码器（仅训练期使用，推理期丢弃）；<br>2. 强制第 $k$ 个连续隐向量必须能解码还原出 ShareGPT 中对应的第 $k$ 句 `<think>` 摘要；<br>3. CLI 调用：`--select-methods M36`。 |
| **M37** | **垂直跨层对角映射与混合高斯**<br>(Diagonal Layer Mapping & GMM) | `whitebox_diagonal_gmm` | `whitebox`<br>3~6x | **使用条件**：希望将教师沿时间步横向展开的推理过程，压缩映射到学生沿网络层深度纵向演化的激活中，并保留多分支不确定性。 | **使用方法**：<br>1. 按时空对角线采样教师第 $t$ 步第 $l$ 层隐状态，对齐至学生更深层激活；<br>2. 使用混合高斯模型（GMM）参数化损失拟合多峰分布；<br>3. CLI 调用：`--select-methods M37`。 |
| **M38** | **跨模态视觉特征渲染压缩**<br>(RoT — Rendering of Thought) | `whitebox_rot` | `whitebox`<br>4~8x | **使用条件**：长文本推理序列过长，但模型具备高压缩率的视觉编码器或二维特征块投影能力。 | **使用方法**：<br>1. 将显式 `<think>` 文本块渲染映射为紧凑二维特征图块嵌入；<br>2. 令学生模型在连续视觉特征空间完成一步对齐；<br>3. CLI 调用：`--select-methods M38`。 |
| **M39** | **压缩键值缓存工作记忆蒸馏**<br>(KaVa — Compressed KV-Cache Distillation) | `whitebox_kava` | `whitebox`<br>5~10x | **使用条件**：长链路多工具调用产生庞大的 KV-Cache，希望学生模型仅用少量连续隐词元即可复现教师完整的抽象工作记忆。 | **使用方法**：<br>1. 基于注意力重要度与低冗余度准则（H2O / SnapKV 策略）浓缩教师在 `<think>` 阶段的 KV-Cache；<br>2. 训练学生隐词元的 KV 投影直接拟合该浓缩缓存；<br>3. CLI 调用：`--select-methods M39`。 |
| **M40** | **并行雅可比不动点连续蒸馏**<br>(PCCoT — Parallel Jacobi Continuous CoT) | `whitebox_pccot` | `whitebox`<br>6~10x | **使用条件**：连续隐词元之间的串行自回归仍嫌慢，希望在 2~3 次并行前向迭代内同时收敛全部隐思维词元。 | **使用方法**：<br>1. 初始化 $M$ 个隐词元向量，通过雅可比并行迭代同时更新所有位置；<br>2. 蒸馏收敛不动点轨迹；<br>3. CLI 调用：`--select-methods M40`。 |

---

### 维度 D06：图拓扑、在策流形与机械可解释性剥离层（Graph Topology, On-Policy Manifold & Mechanistic Interpretability）
> **核心定位**：解决白盒全量隐状态 MSE 蒸馏容易拟合无关风格噪声、以及测试期自回归误差累积偏离教师流形（Off-Policy Drift）的问题，实现 **5~15 倍高鲁棒加速**。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 加速比 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M41** | **图注意力逻辑关系拓扑提取**<br>(LRKD — GAT Logical Relation Graph) | `whitebox_lrkd` | `whitebox`<br>5~10x | **使用条件**：师生隐藏层维度不同或架构异构，不便直接对齐原始向量，需迁移跨词元/跨步骤的逻辑蕴含与排斥关系图。 | **使用方法**：<br>1. 用图注意力网络（GAT）提取教师推理步骤间的关系邻接矩阵 $G_T$；<br>2. 施加谱图蒸馏损失对齐学生关系图 $G_S$；<br>3. CLI 调用：`--select-methods M41`。 |
| **M42** | **在策隐空间想象展开蒸馏**<br>(OPLD — On-Policy Latent Distillation) | `whitebox_opld` | `whitebox`<br>5~12x | **使用条件**：离策（Off-Policy）连续隐状态蒸馏在多轮测试时出现滚雪球误差，需在学生自己生成的隐状态轨迹上施加教师在线修正。 | **使用方法**：<br>1. 训练期令学生自主自回归展开连续隐状态序列；<br>2. 教师模型在学生实际访问到的隐状态节点上计算在线目标方向并回传梯度；<br>3. CLI 调用：`--select-methods M42`。 |
| **M43** | **黎曼隐流形梯度方向引导**<br>(GeoSteer — Manifold Gradient Steering) | `whitebox_geosteer` | `whitebox`<br>5~10x | **使用条件**：希望在不大幅改动模型参数的前提下，通过流形引导向量抑制工具参数幻觉与逻辑跳步。 | **使用方法**：<br>1. 对比教师在“严密锚定轨迹”与“跳步幻觉轨迹”上的激活均值差，构造黎曼流形引导向量 $v_{\text{steer}}$；<br>2. 在推理或蒸馏期沿 $v_{\text{steer}}$ 施加投影约束；<br>3. CLI 调用：`--select-methods M43`。 |
| **M44** | **稀疏交叉编码器反思特征剥离**<br>(Sparse Crosscoder Feature Isolation) | `whitebox_crosscoder` | `whitebox`<br>5~15x | **使用条件**：需要从推理模型与基座模型的残差流差异中，精准剥离出专门控制“工具前置检查”与“边界核验”的稀疏特征方向。 | **使用方法**：<br>1. 在基座模型与推理模型同层残差流上联合训练 Sparse Crosscoder；<br>2. 筛选推理模型独有的高激活稀疏字典原子监督学生；<br>3. CLI 调用：`--select-methods M44`。 |
| **M45** | **归因修补因果子空间定位**<br>(Attribution Patching Subspace) | `whitebox_attr_patching` | `whitebox`<br>5~15x | **使用条件**：全层蒸馏显存开销过大，希望仅针对对最终工具调用参数起决定性因果作用的极少数注意力头/MLP 子空间进行蒸馏。 | **使用方法**：<br>1. 通过一阶梯度归因修补（Attribution Patching）定位因果贡献前 10% 的子空间投影矩阵；<br>2. 仅在该因果子空间内计算蒸馏损失；<br>3. CLI 调用：`--select-methods M45`。 |

---

### 维度 D07：循环深度架构与动力系统不动点蒸馏（Recurrent-Depth & Dynamical Equilibrium Distillation）
> **核心定位**：针对权重复用循环深度架构（Looped Transformer / Huginn）或希望利用暂停词元（Pause Tokens）沿深度/宽度轴扩展算力的场景，免除长文本词元生成开销。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 加速比 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M46** | **循环深度共享层渐进展开蒸馏**<br>(Recurrent-Depth / Huginn Unrolling) | `whitebox_recurrent_unroll` | `whitebox`<br>5~12x | **使用条件**：学生模型采用循环深度（Looped Transformer）架构，希望用共享权重层块的 $R$ 次深度循环替代教师的长思维链。 | **使用方法**：<br>1. 将教师的 $R$ 步显式推导映射为学生循环块的 $R$ 次深度迭代隐状态；<br>2. 按渐进展开课程对齐每次循环迭代的输出表征；<br>3. CLI 调用：`--select-methods M46`。 |
| **M47** | **动力系统不动点吸引子蒸馏**<br>(FPRM / EqR Fixed-Point Attractor) | `whitebox_equilibrium_fprm` | `whitebox`<br>8~15x | **使用条件**：深度循环展开的随时间反向传播（BPTT）显存占用过高，希望直接蒸馏隐空间动力系统的收敛均衡态（不动点 $z^* = f_\theta(z^*, x)$）。 | **使用方法**：<br>1. 前向求解隐状态不动点 $z^*$；<br>2. 利用隐函数定理（Implicit Function Theorem / Jacobian-Free Backprop）直接计算不动点梯度并对齐教师最终决策表征；<br>3. CLI 调用：`--select-methods M47`。 |
| **M48** | **相对 KL 散度零样本早退策略迁移**<br>(Zero-Shot Adaptive Compute Early-Exit) | `whitebox_adaptive_early_exit` | `whitebox`<br>3~10x | **使用条件**：希望学生模型在简单工具调用上 1~2 步即早退输出，仅在复杂多跳推理上自动拉满循环深度。 | **使用方法**：<br>1. 监控相邻迭代步输出分布的相对 KL 散度 $\text{KL}(p_t \parallel p_{t-1}) < \tau$；<br>2. 蒸馏教师模型的步级收敛阈值判据；<br>3. CLI 调用：`--select-methods M48`。 |
| **M49** | **暂停与填充词元算力锚定**<br>(Pause / Filler Token KV-Anchor) | `strategy_pause_anchor` | `blackbox` / `whitebox`<br>1~5x | **使用条件**：在黑盒阶段二需要在复杂工具决策前插入显式结构化检查点锚点，或在白盒标准架构中插入可学习 `<pause>` 词元扩展 KV 算力列。 | **使用方法**：<br>1. 黑盒模式：在阶段二生成 `[目标] -> [参数依据] -> [动作]` 三锚点紧凑摘要；<br>2. 白盒模式：在工具调用前插入 $P$ 个可学习 `<pause>` 词元并对齐其多层 KV 状态；<br>3. CLI 调用：`--select-methods M49`。 |

---

### 维度 D08：词表凸组合软思维与强化学习混合内化（Soft-Thinking & Hybrid RL Internalization）
> **核心定位**：基于“叠加态思维链（Chain of Superposition）”理论，无需破坏原有词嵌入空间，通过词表概率凸组合软词元与混合强化学习（RL），保留单步多分支假设并自主内化繁琐推导。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 加速比 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M50** | **词表概率凸组合软词元传递**<br>(SoftCoT — Vocabulary Convex Combination) | `whitebox_softcot` | `whitebox` / `greybox`<br>3~6x | **使用条件**：不希望修改模型嵌入层架构，但希望将离散采样坍缩丢失的多分支叠加态信息传递给下一步输入。 | **使用方法**：<br>1. 用当前步完整词表概率分布 $p_t$ 对词嵌入矩阵 $E$ 加权求和得到软词元 $e_{\text{soft}} = p_t^\top E$；<br>2. 直接作为下一时间步输入嵌入；<br>3. CLI 调用：`--select-methods M50`。 |
| **M51** | **重参数化软思维策略优化**<br>(SofT-GRPO Gumbel-Softmax RL) | `whitebox_soft_grpo` | `whitebox`<br>3~8x | **使用条件**：连续软词元无法直接计算离散采样对数似然梯度，需在连续词表单纯形空间内做组相对策略优化（GRPO）。 | **使用方法**：<br>1. 在软词元分布上注入 Gumbel-Softmax 重参数化温度噪声；<br>2. 结合工具参数正确性与结果奖励执行组相对策略梯度更新；<br>3. CLI 调用：`--select-methods M51`。 |
| **M52** | **混合离散连续动作空间解耦强化**<br>(HRPO — Hybrid Latent-Explicit RL) | `whitebox_hrpo` | `whitebox`<br>3~8x | **使用条件**：希望模型自主学会“将繁琐计算内化到连续隐状态、仅把关键工具调用依据外显为简短文本”。 | **使用方法**：<br>1. 将动作空间解耦为连续隐状态门控与离散词元采样；<br>2. 联合优化任务完成奖励与显式词元长度惩罚；<br>3. CLI 调用：`--select-methods M52`。 |
| **M53** | **分层混合潜空间强化蒸馏**<br>(HyLaR — Hierarchical Latent-Action RL) | `whitebox_hylar` | `whitebox`<br>4~8x | **使用条件**：多轮长程工具调用任务，高层策略规划适合在连续潜空间完成，底层工具 JSON 语法生成适合离散解码。 | **使用方法**：<br>1. 高层策略网络每轮输出一个连续规划潜向量 $z_{\text{plan}}$；<br>2. 底层解码器以 $z_{\text{plan}}$ 为条件生成具体 `function_call`；<br>3. CLI 调用：`--select-methods M53`。 |
| **M54** | **熵驱动动态双模切换蒸馏**<br>(DyLaR / SwiReasoning Entropy Gating) | `whitebox_dylar_swireasoning` | `whitebox`<br>3~8x | **使用条件**：希望模型在低熵确定性步骤保持高速隐式推演，一旦遇到高熵逻辑分叉点自动切换回显式 `<think>` 词元展开。 | **使用方法**：<br>1. 实时计算每步预测分布的信息熵 $H(p_t)$；<br>2. 当 $H(p_t) > \tau_{\text{ent}}$ 时触发显式 `<think>` 展开，否则走连续隐状态直传；<br>3. CLI 调用：`--select-methods M54`。 |

---

### 维度 D09：投机解码预言机与多教师演化嫁接（Speculative Oracles & Multi-Teacher Evolutionary Grafting）
> **核心定位**：利用投机解码（Speculative Decoding）验证闭环高带宽探测灰盒教师分布，并通过多教师演化算法交叉重组、合并与剪枝，合成超越单一教师上限的“超集推理轨迹”。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 适用防护等级 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M55** | **投机解码接受长度验证预言机**<br>(Speculative Verification Oracle) | `greybox_spec_oracle` | `greybox`<br>`L2` / `L3` | **使用条件**：闭源接口屏蔽了 Logits 返回，但支持投机解码（Draft-Target Verification）或前缀查找加速，需探测教师认可的推理序列。 | **使用方法**：<br>1. 用本地草稿模型生成候选 `<think>` 序列提交验证接口；<br>2. 根据接受长度与首个拒绝位置修补推理断点；<br>3. CLI 调用：`--select-methods M55`。 |
| **M56** | **在线投机知识蒸馏**<br>(OSKD — Online Speculative KD) | `greybox_oskd` | `greybox` / `whitebox`<br>2~4x | **使用条件**：全序列在线蒸馏算力开销过大，希望教师仅在学生草稿预测失准的回退节点下发局部概率监督。 | **使用方法**：<br>1. 学生模型作为 Draft 生成工具调用与推理轨迹；<br>2. 教师模型并行验证，仅在拒绝位置计算局部 KL 损失；<br>3. CLI 调用：`--select-methods M56`。 |
| **M57** | **多教师思维链遗传交叉与变异**<br>(CoT-Evo Evolutionary Crossover) | `strategy_cotevo` | `blackbox`<br>`L1` ~ `L4` | **使用条件**：单次或单一策略生成的 `<think>` 存在局部缺漏，希望融合多路策略视角并剪除冗余套话。 | **使用方法**：<br>1. 阶段二同时从目标锚定视角与工具参数映射视角生成两条候选逻辑；<br>2. 执行交叉重组与冗余剪枝，合成 2~3 句高密度推理摘要；<br>3. CLI 调用：`--select-methods M57`。 |
| **M58** | **多源异构思维链共识合并**<br>(Merge-of-Thought Consensus Grafting) | `strategy_merge_of_thought` | `blackbox`<br>`L2` / `L3` | **使用条件**：对同一条难题采集了多个子 Agent 或不同提示策略下的推导草稿，需要取长补短嫁接成一条完整无暇的推理链。 | **使用方法**：<br>1. 提取多份候选推导各自命中的真实工具参数与文件路径实体；<br>2. 按时间步对齐合并为覆盖全部实体的最优摘要；<br>3. CLI 调用：`--select-methods M58`。 |
| **M59** | **去冗余长思维链结构剪枝**<br>(DLCoT — De-Redundant Long-CoT Pruning) | `strategy_dlcot` | `blackbox`<br>`L0` / `L1` | **使用条件**：原始采集到的 `<think>` 过于啰嗦、包含大量无效反复试探或空洞寒暄，需要提纯为精炼的高层直接步骤摘要。 | **使用方法**：<br>1. 剥离原始思考中的自我否定分支、重复铺垫与总结套话；<br>2. 仅保留通向最终工具调用或回答的最短因果主干（2~3 句）；<br>3. CLI 调用：`--select-methods M59`。 |

---

### 维度 D10：程序化符号转换与多视角正交解耦（Neuro-Symbolic & Multi-Perspective Orthogonal Decomposition）
> **核心定位**：针对 CLI 命令执行、代码审计、复杂结构化参数与多约束决策，将自然语言推理锚定到确定性的符号程序（PoT）、抽象语法树（AST）或正交语义子空间（MPCoT），彻底消除参数幻觉。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 适用防护等级 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M60** | **可执行程序思维锚定蒸馏**<br>(PoT — Program-of-Thought Grounding) | `strategy_pot` | `blackbox`<br>`L1` ~ `L4` | **使用条件**：轨迹中包含 `run_command`、Python 脚本执行、Git 命令或数值统计，推理过程必须与实际执行的命令行代码和标准输出严格一致。 | **使用方法**：<br>1. 提取 `function_call` 中的真实命令字符串、退出码与 `observation` 标准输出；<br>2. 生成严格锚定命令语法与输出指标的符号级推理摘要；<br>3. CLI 调用：`--select-methods M60`。 |
| **M61** | **抽象语法树与结构化查询计划蒸馏**<br>(AST & Query-Plan Symbolic Grounding) | `strategy_ast_query_plan` | `blackbox`<br>`L2` / `L3` / `L4` | **使用条件**：涉及源码 AST 解析、JSON Schema 校验、数据库查询或复杂嵌套字典参数构造任务。 | **使用方法**：<br>1. 将推理过程表示为“目标数据结构/AST 节点 → 过滤条件/字段路径 → 输出映射”的结构化查询计划；<br>2. 转化为紧凑自然语言摘要写入 `<think>`；<br>3. CLI 调用：`--select-methods M61`。 |
| **M62** | **多视角正交分解与融合**<br>(MPCoT — Multi-Perspective Decomposition) | `strategy_mpcot` | `blackbox` / `whitebox`<br>`L1` ~ `L4` | **使用条件**：多轮复杂交互中同时涉及“用户意图演进”、“当前工作区环境状态”与“工具接口契约”三重约束。 | **使用方法**：<br>1. 黑盒模式：从意图视角、环境状态视角、工具参数视角正交合成 2~3 句推理摘要；<br>2. 白盒模式：配合 Poly-Encoder 提取三视角独立表征监督学生；<br>3. CLI 调用：`--select-methods M62`。 |

---

### 维度 D11：跨模态桥接擦除与视觉草图内化（Cross-Modal Bridge Removal & Latent Visual Sketching）
> **核心定位**：针对语音、图像、UI 布局等多模态工具调用与推理任务，消除高延迟的外部跨模态中介（如 ASR 转录稿或中间裁剪渲染图），内化为端到端跨模态隐式直觉。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 加速比 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M63** | **多跳跨模态中介渐进擦除**<br>(Cross-Modal Bridge Curriculum Erasure) | `multimodal_bridge_erasure` | `whitebox`<br>3~8x | **使用条件**：端到端语音或多模态模型依赖“音频/图像 → 文本转录描述 → 文本 `<think>` → 响应”多跳流水线，转换延迟过高。 | **使用方法**：<br>1. 先用本插件采集完整四跳显式对齐轨迹；<br>2. 在课程训练中分阶段擦除中间转录文本与显式 `<think>` 监督，迫使模型在跨模态连续隐空间直推；<br>3. CLI 调用：`--select-methods M63`。 |
| **M64** | **生成式视觉草图隐词元内化**<br>(Latent Visual Sketching Tokens) | `multimodal_visual_sketch` | `whitebox`<br>3~6x | **使用条件**：多模态 Agent 需要多次调用视觉裁剪、坐标框选或 UI 布局检查工具来辅助空间推理。 | **使用方法**：<br>1. 提取多步视觉工具返回的区域特征浓缩为连续视觉隐词元（Latent Visual Tokens）；<br>2. 训练学生模型在隐空间直接预测该视觉特征而免除测试期工具渲染；<br>3. CLI 调用：`--select-methods M64`。 |

---

### 维度 D12：执行元数据时序预算校准与本地会话审计层（Execution Timing Metadata & Local Transcript Audit）
> **核心定位**：在完全遵循宿主沙箱安全边界的前提下，利用宿主本地会话日志（`transcript_full.jsonl`）的结构化元数据、工具执行耗时与返回载荷字节量，无损提取并校准全链路推理轨迹。

| 编号 | 方法名称 | CLI 策略键 (`--select-methods`) | 权限 / 适用防护等级 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **M65** | **流式响应延迟与载荷分块复杂度校准**<br>(Streaming Latency & Payload Calibration) | `strategy_timing_chunk_budget` | `blackbox`<br>`L1` ~ `L4` | **使用条件**：需要根据子 Agent 每步的 `created_at` 时间戳跨度与工具输出分块大小，自动区分“直觉快步”与“深度重算步”并分配对应的推理摘要详略度。 | **使用方法**：<br>1. 解析相邻步骤的耗时差 $\Delta t$ 与 `observation` 字节规模；<br>2. 对高耗时/大载荷步骤自动分配包含多条件核验的推理摘要，对低耗时步保持极简 1~2 句概述；<br>3. CLI 调用：`--select-methods M65`。 |
| **M66** | **宿主本地会话日志无损解析与对齐**<br>(Local Session Transcript Schema Inspection) | `strategy_local_transcript_inspect` | `blackbox`<br>`L0` ~ `L4` | **使用条件**：在 Antigravity 宿主环境下运行子 Agent 后，需要直接从本地 `.system_generated/logs/transcript_full.jsonl` 无损提取全链路 `human`、`function_call`、`observation` 与原生外显 `<think>`。 | **使用方法**：<br>1. 自动扫描子 Agent 的 `transcript.jsonl` 与 `transcript_full.jsonl`；<br>2. 合并 `send_message` 回复与后续 `PLANNER_RESPONSE` 中的外显 `<think>`，精准定位缺失节点列表 `target_node_indices`；<br>3. CLI 调用：默认内置于 `trajectory_extractor.py inspect`，也可显式指定 `--select-methods M66`。 |

---

## 三、常见场景快速选型速查表（什么时候用什么组合？）

| 你的目标场景 | 推荐 CLI 参数组合 | 实际启用的核心方法 (`M01`~`M66`) |
| :--- | :--- | :--- |
| **场景 1：闭源模型在工具调用前静默无 `<think>`（`L2_TOOL_SILENT`）** | `--access-tier blackbox --guard-level L2_TOOL_SILENT --task-domain tool_agent` | `M15` (EchoCoT 状态草稿) → `M03` (STaR 逆向归因) → `M60` (PoT 程序锚定) → `M02` (Prefill 预填充) → `M23` (PRM 校验) |
| **场景 2：闭源模型全链路加密隐推理（`L3_ENCRYPTED_STATE`）** | `--access-tier blackbox --guard-level L3_ENCRYPTED_STATE` | `M62` (MPCoT 多视角正交) → `M03` (STaR) → `M17` (MCTS 树重构) → `M57` (CoT-Evo 演化) → `M15` (EchoCoT) |
| **场景 3：闭源模型触发强反蒸馏拒答（`L4_STRICT_ANTI_DISTILL`）** | `--access-tier blackbox --guard-level L4_STRICT_ANTI_DISTILL` | `M11` (代码/规约范式转换) → `M06` (苏格拉底助产术) → `M05` (多智能体对抗质询) → `M61` (AST/查询计划锚定) |
| **场景 4：模型只输出敷衍的一句话摘要（`L1_SUMMARY_FOLDED`）** | `--access-tier blackbox --guard-level L1_SUMMARY_FOLDED` | `M24` (TIA 摘要反演展开) → `M49` (Pause 锚点拆解) → `M08` (Step-Back 原理锚定) → `M59` (DLCoT 提纯) |
| **场景 5：开放 Logits / 投机接口的灰盒模型蒸馏** | `--access-tier greybox --select-methods M25,M26,M56,M31` | `M25` (Top-K Logits KL) + `M26` (高熵节点分叉) + `M56` (OSKD 在线投机蒸馏) + `M31` (GFlowNets 采样) |
| **场景 6：开源白盒模型追求 5~10 倍隐空间极速推理** | `--open-source true --access-tier whitebox --latency-budget low_latency --select-methods M34,M35,M36,M39` | 先通过阶段一/二生成 ShareGPT 显式锚点数据，再自动生成 `M34` (Coconut) + `M35` (CODI) + `M36` (SIM-CoT) + `M39` (KaVa KV-Cache 压缩) 训练配方 |
| **场景 7：开源循环深度或流形对齐极速内化（10~15 倍加速）** | `--open-source true --access-tier whitebox --latency-budget ultra_fast --select-methods M42,M44,M47,M52` | 自动生成 `M42` (OPLD 在策想象) + `M44` (Sparse Crosscoder) + `M47` (FPRM/EqR 不动点吸引子) + `M52` (HRPO 混合强化) 训练配方 |
