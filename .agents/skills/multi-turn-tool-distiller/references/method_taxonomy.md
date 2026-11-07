# 隐推理提取与知识蒸馏 12 维全景方法库 (`method_taxonomy.md`)

本文件收录全部 **12 大技术维度（48+ 核心算法与算子）**，并定义 **Main-Agent 智能路由器（Adaptive Strategy Router）** 如何根据目标教师模型的 **开源属性/访问权限（White-box / Grey-box / Black-box）**、**防护等级（`L0` ~ `L4`）** 以及 **任务模态与领域特征** 自动选择最优方案。

---

## 一、三维智能路由决策矩阵（开源属性 × 访问层级 × 防护等级）

Main-Agent 在启动蒸馏前或完成阶段一探测（`trajectory_extractor.py inspect`）后，按以下三维坐标选择蒸馏管线：

### 1. 按“是否开源 / 模型访问权限 (`access_tier`)”分流

| 开源属性与访问权限 | 参数标识 (`--access-tier`) | 核心可用信号 | 推荐主干方案组合 |
| :--- | :--- | :--- | :--- |
| **开源白盒模型**（拥有权重、中间层激活与 KV-Cache） | `whitebox` (`--open-source true`) | 隐藏层连续向量、多头注意力矩阵、残差流、KV-Cache | **显式轨迹采集 + 隐空间压缩内化**：先通过双阶段采集标准 ShareGPT 锚点数据，并在训练清单中生成 **Coconut / CODI / SIM-CoT / KaVa / PCCoT / OPLD / Sparse Crosscoder / EqR / SoftCoT** 训练策略配置 |
| **半开源 / 灰盒接口模型**（无权重，但开放 Top-K Logits 或 Speculative 验证接口） | `greybox` | 词元级对数概率（Logits）、信息熵峰值、投机解码接受长度 | **分布与预言机引导蒸馏**：组合 **高熵节点分叉采样 (`strategy_entropy_branch`) + OSKD 投机预言机校验 + Top-K Logits 软标签对齐 + MARCOS 变分推断** |
| **闭源黑盒模型**（仅开放纯文本对话与工具调用接口，如 Antigravity 宿主 Sub-Agent） | `blackbox` (`--open-source false`) | 外显文本序列、真实工具调用参数与 Observation、响应长度与轮次结构 | **双阶段 12 算子自适应瀑布流**：阶段一零包装自然执行 + 阶段二根据防护等级（`L0`~`L4`）从 **12 个可执行黑盒重构算子** 中智能匹配并自动回退 |

---

### 2. 按“闭源模型防护等级 (`guard_level`)”自动识别与算子路由

`trajectory_extractor.py inspect` 会根据阶段一 Sub-Agent 的实际返回特征，自动判定目标模型的防护等级（`L0`~`L4`），并生成定制化的阶段二算子回退序列：

| 防护等级 (`guard_level`) | 阶段一自动探测判据 | 首选算子队列 (`recommended_strategies` 前 5 项) | 路由逻辑说明 |
| :--- | :--- | :--- | :--- |
| **`L0_UNRESTRICTED`**（无防护 / 原生外显） | 所有 `gpt` 节点（含工具调用前与最终回复）均已自然输出合规 `<think>` 且通过 PRM 校验 | `[]`（直接通过，无需阶段二） | 原生输出已完整，直接进入 PRM 锚定校验并落盘 |
| **`L1_SUMMARY_FOLDED`**（思考折叠为简短摘要） | 存在 `<think>` 标签但部分节点过短（`< 25` 字符）或缺乏工具参数映射细节 | `strategy_tia` -> `strategy_star` -> `strategy_mpcot` -> `strategy_cotevo` -> `strategy_prefill` | 优先调用 **TIA 摘要反演展开算子**，将简短摘要还原为贴合上下文的高层直接步骤 |
| **`L2_TOOL_SILENT`**（正文有思考，原生工具调用步静默） | 最终回答含 `<think>`，但发起原生 `tool_call` 的步骤仅下发工具 JSON 而无内联文本 | `strategy_star` -> `strategy_pot` -> `strategy_prefill` -> `strategy_mpcot` -> `strategy_socratic` | 优先调用 **STaR 已知动作逆向归因** 与 **PoT 执行证据锚定**，精准补全工具调用前的决策依据 |
| **`L3_ENCRYPTED_STATE`**（隐推理全加密/屏蔽，仅输出正文与工具调用） | 阶段一全程 `native_think_count == 0`，但正常完成了工具调用与正文回答 | `strategy_mpcot` -> `strategy_star` -> `strategy_mcts` -> `strategy_cotevo` -> `strategy_pot` | 优先使用 **MPCoT 多视角正交分解**、**STaR** 与 **MCTS 候选路径择优** 重构全链路高层推理概述 |
| **`L4_STRICT_ANTI_DISTILL`**（强反蒸馏护栏 / 拒绝直接解释思考过程） | 触发关于“思考过程”的格式化排斥或常规重构算子连续失败 | `strategy_code_paradigm` -> `strategy_socratic` -> `strategy_debate` -> `strategy_pot` -> `strategy_star` | 切换为 **代码规约/函数注释范式 (`strategy_code_paradigm`)**、**苏格拉底子目标拆解 (`strategy_socratic`)** 与 **多智能体交叉核验 (`strategy_debate`)** |

---

## 二、12 大维度（48+ 方法）全景分类索引表

### 维度 1：黑盒提示诱导与逆向重构层（Black-Box Prompting & Post-Hoc Reconstruction）
1. **结构化元提示（Structured Meta-Prompting）** — 对应算子：`strategy_socratic`（将任务分解为子目标、工具前置条件与结果核验）。
2. **回复预填充（Response Prefilling）** — 对应算子：`strategy_prefill`（注入上下文顺承前缀引导补全推理摘要）。
3. **已知答案逆向归因（STaR Post-hoc Rationalization）** — 对应算子：`strategy_star`（结合已知 `function_call`、`observation` 与最终回答逆向合成直接推理桥）。
4. **多智能体协同质询与修正（Multi-Agent Critique & Refine）** — 对应算子：`strategy_debate`（通过生成-核验闭环消除逻辑跳步与参数幻觉）。

### 维度 2：代码范式映射与工具状态窗口分析层（Code Paradigm & Tool State Grounding）
5. **代码/规约范式映射（Code/Specification Paradigm Framing）** — 对应算子：`strategy_code_paradigm`（将工具调用决策表述为结构化接口前置条件与后置断言规约）。
6. **工具调用链上下文锚定（Multi-Tool State Window Grounding）** — 在多轮工具调用之间利用真实参数与返回状态锁定每一步决策依据。
7. **响应复杂度与步骤预算对齐（Step-Budget & Complexity Calibration）** — 根据工具调用跳数与观察结果复杂度动态匹配推理摘要颗粒度。
*(注：针对协议密文重放或越权绕过类方法仅作学术分类存档，本插件严格在宿主合规沙箱与公开接口边界内通过算法重构实现。)*

### 维度 3：搜索展开、过程评估与轨迹反演层（Search, PRM & Trajectory Inversion）
8. **树搜索候选轨迹择优（MCTS / Beam Trajectory Selection）** — 对应算子：`strategy_mcts`（对关键工具决策节点采样多条候选推理摘要并按环境反馈择优）。
9. **过程奖励模型过滤（PRM Step-Level Scoring & Filtering）** — 对应内置校验器：`check_prm_grounding` + `strategy_prm_rerank`（逐步核验推理文本与工具名、路径、命令参数及用户问题实体的锚定度）。
10. **推理轨迹反演展开（Trajectory Inversion from Summary / TIA）** — 对应算子：`strategy_tia`（从简短结论或观察摘要逆向还原直接推导步骤）。

### 维度 4：灰盒概率分布、变分推断与流网络建模层（Grey-Box Logits & Variational Inference）
*(适用于 `--access-tier greybox` 开放 Logits 接口的模型)*
11. **Top-K 对数概率蒸馏（Logits KL Distillation）** — 最小化师生在关键工具选择词元上的前/反向 KL 散度。
12. **高熵节点分叉采样（High-Entropy Node Branching）** — 对应算子：`strategy_entropy_branch`（在多工具竞争的高不确定度决策点并行展开备选路径）。
13. **跨层对比解码（DoLa Layer-Contrastive Decoding）** — 放大深层逻辑层相对于浅层语法层的概率增量。
14. **马尔可夫变分推断（MARCOS / JEPO / ReGuLaR）** — 将隐推理建模为潜变量马尔可夫状态转移。
15. **生成流网络采样（GFlowNets Trajectory Sampling）** — 按正比于末端奖励的概率采样有向无环推理图。

### 维度 5：白盒连续隐空间压缩、记忆迁移与并行迭代层（White-Box Continuous Latent Distillation）
*(适用于 `--access-tier whitebox` / `--open-source true` 开源模型后训练配置生成)*
16. **中间层隐状态对齐（Hidden-State MSE/Cosine Alignment）**
17. **多头注意力图谱迁移（Attention Map Transfer）**
18. **连续隐式思维课程蒸馏（Coconut — Chain of Continuous Thought）**
19. **自我蒸馏连续压缩（CODI — Continuous Distillation）**
20. **步级解码监督隐式思维（SIM-CoT — Step-Level Supervised Implicit CoT）**
21. **垂直跨层对角映射与混合高斯（Diagonal Layer Mapping & GMM）**
22. **跨模态视觉特征压缩（RoT — Rendering of Thought）**
23. **压缩键值缓存工作记忆蒸馏（KaVa — Compressed KV-Cache Distillation）**
24. **并行雅可比不动点连续蒸馏（PCCoT — Parallel Continuous CoT via Jacobi Iteration）**

### 维度 6：图拓扑、在策流形与机械可解释性剥离层（Graph Topology & Mechanistic Interpretability）
*(适用于 `--access-tier whitebox` 开源模型特征提纯)*
25. **图注意力逻辑拓扑提取（LRKD — GAT Logical Relation Graph）**
26. **在策隐空间想象蒸馏（OPLD — On-Policy Latent Distillation）**
27. **黎曼隐流形梯度引导（GeoSteer — Manifold Gradient Steering）**
28. **稀疏交叉编码器特征剥离（Sparse Crosscoder Feature Isolation）**
29. **归因修补因果子空间定位（Attribution Patching Subspace）**

### 维度 7：循环深度架构与动力系统不动点蒸馏（Recurrent-Depth & Equilibrium Distillation）
*(适用于 Looped Transformer / Huginn 等深度循环开源架构)*
30. **循环深度轨迹渐进蒸馏（Recurrent-Depth Unrolling Distillation）**
31. **动力系统不动点吸引子蒸馏（FPRM / EqR / Attractor Equilibrium Distillation）**
32. **相对 KL 散度早退策略迁移（Zero-Shot Adaptive Compute Early-Exit）**
33. **暂停与填充词元算力锚定（Pause / Filler Token KV-Anchor）** — 对应黑盒兼容算子：`strategy_pause_anchor`（在复杂多步工具决策前显式设立结构化决策检查点）。

### 维度 8：词表凸组合软思维与强化学习混合内化（Soft-Thinking & Hybrid RL Internalization）
34. **词表凸组合软思维传递（SoftCoT — Vocabulary Convex Combination）**
35. **Gumbel-Softmax 重参数化软思维策略优化（SofT-GRPO）**
36. **混合离散连续动作空间强化蒸馏（HRPO / HyLaR）**
37. **熵驱动动态双模切换蒸馏（DyLaR / SwiReasoning）**

### 维度 9：投机解码预言机与多教师演化嫁接（Speculative Oracles & Evolutionary Grafting）
38. **投机解码验证预言机提取（Speculative Verification Oracle）**
39. **在线投机知识蒸馏（OSKD — Online Speculative Knowledge Distillation）**
40. **多教师思维链演化嫁接（CoT-Evo / Merge-of-Thought / DLCoT）** — 对应算子：`strategy_cotevo`（融合多视角候选推理摘要并剪除冗余套话）。

### 维度 10：程序化符号转换与多视角正交解耦（Neuro-Symbolic & Multi-Perspective Decomposition）
41. **可执行程序与查询图谱锚定蒸馏（PoT — Program-of-Thought & AST Grounding）** — 对应算子：`strategy_pot`（严格锚定命令行代码、退出码与标准输出证据）。
42. **多视角正交分解提取（MPCoT — Multi-Perspective Orthogonal Decomposition）** — 对应算子：`strategy_mpcot`（从用户意图、环境状态、工具契约三个正交维度合成推理摘要）。

### 维度 11：跨模态桥接擦除与视觉草图内化（Cross-Modal Bridge Removal & Visual Sketching）
43. **多跳模态桥接渐进擦除（Cross-Modal Bridge Curriculum Erasure）**
44. **生成式视觉草图隐空间内化（Latent Visual Sketching Tokens）**

### 维度 12：执行元数据预算校准与安全边界说明（Execution Metadata Calibration & Compliance）
45. **观察载荷与工具跳数复杂度校准（Observation-Length & Hop-Count Calibration）** — 根据 `observation` 字节量与工具调用次数自动校准阶段二推理摘要的信息密度。
*(注：安全逆向工程中的底层进程内存插桩或网络流侧信道仅作为理论防御分类参考，插件执行层始终遵循宿主环境安全合规准则。)*
