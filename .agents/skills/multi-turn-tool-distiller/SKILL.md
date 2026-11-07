---
name: multi-turn-tool-distiller
description: Automated multi-turn conversation and real tool-calling trajectory distillation workflow in Antigravity using a Two-Stage Main-Sub Agent architecture (Stage 1 natural execution + 5D Adaptive Profiler + 12-Dimension 66-Method Catalog `M01`..`M66` with free selection and PRM audit). Use when the user asks to distill, synthesize, or collect multi-turn dialogues, tool-calling trajectories, or ShareGPT training datasets using isolated Sub-Agents.
---

# 自动化多轮与工具调用蒸馏技能 (`multi-turn-tool-distiller`)

本技能在 Antigravity 宿主环境下通过 **双阶段 Main-Sub Agent 架构** 采集真实工具调用与多轮对话轨迹，内置 **12 大技术维度、66 种隐推理提取与蒸馏方法（`M01` ~ `M66`）**，支持 **按编号/策略键自由点选组合** 或 **五维细化条件自动路由**，最终输出符合 LLaMA-Factory 标准训练规范的 ShareGPT `.jsonl` 数据集及配套后训练配置。

---

## 一、核心原则与架构边界

1. **Main-Agent（控制器 / 5D 路由器 / PRM 质检器）**：
   - 负责加载队列、调度并发、推进多轮追问、运行 [`trajectory_extractor.py`](file:///f:/zl/skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py) 诊断轨迹画像、从 **66 种方法库** 中自由选择或自动匹配最优策略，并调用 [`sharegpt_writer.py`](file:///f:/zl/skills/multi-turn-tool-distiller/scripts/sharegpt_writer.py) 原子落盘。
   - **严禁代答与严禁硬编码拼接 `<think>`**：所有业务回答与工具调用必须由阶段一 `distill_executor_agent` 产生；所有缺失节点的 `<think>` 必须由阶段二 `distill_rationalizer_agent` 基于真实轨迹生成。
2. **双阶段解耦、三源隐推理融合与单样本隔离**：
   - **来源一：宿主底层会话日志加密思维摘要线索提取（`M66: strategy_local_transcript_inspect`）**：闭源模型的真实底层隐式推理 Token 在服务端加密不落盘，宿主 `transcript_full.jsonl` 中每个 `PLANNER_RESPONSE` 节点的 `thinking` 字段实为厂商返回的**折叠思维摘要（`L1_SUMMARY_FOLDED`）**。[`trajectory_extractor.py`](file:///f:/zl/skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py) 的 [`parse_transcript_with_clues`](file:///f:/zl/skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py#L326-L555) 自动提取该字段并存为 `stage2_context[idx]["folded_thought_summary_clue"]`（绝不将英文第三人称摘要直接生硬拼入 `<think>`），同时支持从 `transcript.jsonl` 自动回填缺失步与过滤异常重试步。
   - **来源二：阶段一自然决策流 `<think>` 外化捕获 (`distill_executor_agent`)**：原样发送用户问题与追问（零八股包装），执行真实工具调用，捕获正文中的 `<think>`。若阶段一 `<think>` 仅为单句浅层动作宣告（`< 280` 字符，如 `"我将运行脚本..."`）或混入宿主通信脚手架（`send_message` / `父 Agent`），[`check_prm_grounding`](file:///f:/zl/skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py#L583-L654) 将自动识别并移交阶段二重构。
   - **来源三：阶段二 66 方法靶向逆向内化重构 (`distill_rationalizer_agent`)**：针对需要深化或补全的节点，结合来源一的 `folded_thought_summary_clue`、真实 `function_call` 参数及 `observation` 输出，从 `M01`~`M66`（如 `M24: strategy_tia` 轨迹反演、`M60: strategy_pot` 程序思维、`M03: strategy_star` 逆向归因）合成**第一人称高密度内化推演 `<think>`**（涵盖目标拆解、参数/公式推导、观察结果核验与自省闭环），并通过 [`check_prm_grounding`](file:///f:/zl/skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py#L583-L654) 与 [`audit_conversations`](file:///f:/zl/skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py#L657-L740) 步级实体锚定校验。
   - 每条样本采集并校验完毕后，立即调用 `manage_subagents(Action="kill")` 销毁对应子 Agent 实例。

---

## 二、细化自由选择与运行参数表

用户或 Main-Agent 可在运行 [`trajectory_extractor.py inspect`](file:///f:/zl/skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py) 或 [`list-methods`](file:///f:/zl/skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py) 时组合以下细化参数：

| 参数名称 | 可选值 / 格式 | 默认值 | 细化控制说明（怎么选） |
| :--- | :--- | :---: | :--- |
| **`--select-methods`** | `M01`~`M66` 或策略键名（逗号分隔，如 `M15,M60,M34`） | `""` (自动) | **最高优先级自由点选**：按用户指定的任意单个或多个方法顺序执行阶段二算子或生成白盒/灰盒训练配方 |
| **`--exclude-methods`** | `M01`~`M66` 或策略键名（逗号分隔，如 `M02,M11`） | `""` | **自由排除**：从候选池或自动瀑布流中剔除指定方法 |
| **`--dimension`** | `D01` ~ `D12`（支持多选如 `D02,D10`）或 `all` | `all` | **按流派维度过滤**：限定仅在指定的 1~N 个技术维度内选择方案 |
| **`--open-source`** | `true` \| `false` | `false` | 目标模型是否开源（设为 `true` 时自动激活 `whitebox` 连续隐空间后训练配方） |
| **`--access-tier`** | `auto` \| `blackbox` \| `greybox` \| `whitebox` \| `hybrid` | `auto` | **模型访问权限细化**：纯文本黑盒 / 开放 Logits 与投机验证灰盒 / 开放权重与激活白盒 / 黑白盒混合 |
| **`--guard-level`** | `auto` \| `L0` \| `L1` \| `L2` \| `L3` \| `L4` | `auto` | **防护等级细化**：`L0` 原生完整外显、`L1` 折叠摘要、`L2` 工具步静默、`L3` 全加密屏蔽、`L4` 强反蒸馏拒答 |
| **`--task-domain`** | `auto` \| `general` \| `tool_agent` \| `code_symbolic` \| `math_logic` \| `long_context` \| `multimodal` | `auto` | **任务领域细化**：根据任务类型精准匹配符号执行、多跳状态或长上下文方法 |
| **`--latency-budget`** | `auto` \| `unconstrained` \| `moderate` \| `low_latency` \| `ultra_fast` | `auto` | **推理延迟预算**：`unconstrained`(1x 显式)、`moderate`(1~2x)、`low_latency`(3~10x 隐词元/KV压缩)、`ultra_fast`(5~15x 循环不动点/流形) |

---

## 三、12 大维度、66 种方法使用指南（什么时候用 & 怎么用）

> 完整方法卡片亦可通过命令 `python skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py list-methods` 实时查询，或参阅 [method_taxonomy.md](file:///f:/zl/skills/multi-turn-tool-distiller/references/method_taxonomy.md) 与 [prompt_strategies.md](file:///f:/zl/skills/multi-turn-tool-distiller/references/prompt_strategies.md)。

### 维度 D01：黑盒提示诱导与逆向重构层（`M01` ~ `M10`）
| 编号与键名 | 方法名称 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :--- | :--- | :--- | :--- |
| **`M01`** `strategy_meta_prompt` | 结构化元提示拆解 | 模型无强反蒸馏护栏（`L0`~`L2`），希望按统一三段式结构补全思考 | `--select-methods M01`：按“目标识别 → 前置条件核对 → 直接执行动作”合成 2~3 句摘要 |
| **`M02`** `strategy_prefill` | 回复预填充引导 | 模型习惯跳过 `<think>` 直出结论，需从首句上下文前缀顺延续写 | `--select-methods M02`：注入贴合问题的首句引导前缀，由阶段二顺势补全后续 1~2 句直接推导 |
| **`M03`** `strategy_star` | 已知动作逆向归因 (STaR) | 已拿到真实 `function_call` 与 `observation`，仅缺中间 `<think>`（最通用） | `--select-methods M03`：将真实工具参数与观察结果作为已知终点，逆向合成连接问题与动作的推理桥 |
| **`M04`** `strategy_quiet_star` | 词元级隐式前瞻合理化 | 多轮跨步工具链，需说明当前动作对后续轮次的信息铺垫价值 | `--select-methods M04`：结合后续轮次真实 `observation` 收益，生成面向未来多步目标的前瞻摘要 |
| **`M05`** `strategy_debate` | 多智能体对抗质询与修正 | 单次重构易漏掉工具参数来源或未通过 `check_prm_grounding` 校验 | `--select-methods M05`：在阶段二运行“起草 → 审计员核查工具名与参数出处 → 修正定稿”三步闭环 |
| **`M06`** `strategy_socratic` | 苏格拉底助产术拆解 | 遭遇 `L4` 强反蒸馏拒答（对“输出内部思考”敏感）或复杂多跳任务 | `--select-methods M06`：规避“思考过程”字眼，围绕“已知事实 → 缺失数据 → 所调工具”客观问答合成摘要 |
| **`M07`** `strategy_least_to_most` | 由简入繁递进分解 | 复合多跳任务，后一步高度依赖前一步基础子任务的执行结果 | `--select-methods M07`：拆解为依赖递增的最小原子子问题，每步仅针对当前子问题与工具生成推导 |
| **`M08`** `strategy_step_back` | 后退一步抽象原理锚定 | 架构分析、格式校验或原理判定任务，需先点明通用准则再落到参数 | `--select-methods M08`：先提炼当前步骤依据的校验法则或架构规范，再映射到具体工具参数 |
| **`M09`** `strategy_plan_and_solve` | 先规划后求解显式锚定 | 单轮连续发起多次工具调用，需清晰展现全局计划与当前进度 | `--select-methods M09`：首节点生成分步蓝图（Plan），后续工具节点锚定当前所处步骤与进展（Solve） |
| **`M10`** `strategy_ccot` | 正反对比思维链归因 | 存在多个相近候选工具（如 `read_file` vs `run_command`），需明确选型依据 | `--select-methods M10`：对比当前工具与备选方式的适用边界，点明选用当前工具及参数的直接理由 |

### 维度 D02：代码范式映射、工具状态驻留与上下文预算层（`M11` ~ `M16`）
| 编号与键名 | 方法名称 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :--- | :--- | :--- | :--- |
| **`M11`** `strategy_code_paradigm` | 代码与接口规约范式转换 | `L4` 强反蒸馏拒答自然语言思考，但对软件工程接口契约/注释完全放行 | `--select-methods M11`：以接口规约视角编写输入前置状态、工具参数绑定依据与后置断言，转化为 2~3 句摘要 |
| **`M12`** `strategy_state_continuity` | 跨轮会话状态连续性锚定 | 多轮对话（`turns >= 2`），后一轮操作依赖上一轮已确认的文件或状态 | `--select-methods M12`：提取上一轮确认的状态快照注入本轮推导，显式写出跨轮状态承接关系 |
| **`M13`** `strategy_tier_alignment` | 跨尺寸同族模型对齐转录 | 旗舰模型输出极简跳跃，需借助同族轻量/辅助子 Agent 展开可读推理 | `--select-methods M13`：锁定阶段一旗舰动作轨迹，阶段二拉起 `Model="flash"` 或辅助实例展开适配学生粒度的步骤 |
| **`M14`** `strategy_context_compaction` | 长上下文自动摘要逆向展开 | 消息数 `> 12` 的长工具链或存在压缩摘要，需从阶段结论反推单步依据 | `--select-methods M14`：先提炼阶段性里程碑，再将每个里程碑拆解回对应工具调用节点的输入输出依据 |
| **`M15`** `strategy_echocot_scratchpad` | 工具调用链状态驻留草稿锚定 | 单轮连续多跳工具调用（EchoCoT 场景），两次工具调用间存在状态增量 | `--select-methods M15`：紧扣相邻两次 `function_call` 间的 `observation` 差值，写明上一步返回数据如何决定下一步参数 |
| **`M16`** `strategy_budget_oracle` | 词元长度与复杂度预算预言机 | 需按每步命令复杂度与返回字节长度精准控制 `<think>` 句数与密度 | `--select-methods M16`：根据参数复杂度与返回载荷自动分配 1~2 句（简单步）或 2~3 句（复杂步）高密度预算 |

### 维度 D03：搜索树展开、过程奖励评估与轨迹反演层（`M17` ~ `M24`）
| 编号与键名 | 方法名称 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :--- | :--- | :--- | :--- |
| **`M17`** `strategy_mcts` | 蒙特卡洛树搜索轨迹重构 | 多步探索任务（排查未知目录/定位问题），需按最终成功状态回传节点价值 | `--select-methods M17`：从最终解决问题的终端状态回溯，说明当前工具节点为何是最大化信息增益的路径 |
| **`M18`** `strategy_tot` | 思维树分叉评估展开 | 关键决策点存在多种可行方案，需比选并录取最契合实际工具参数的路径 | `--select-methods M18`：阶段二内部比选多条候选推导，仅保留与实际 `function_call` 参数最吻合的一条 |
| **`M19`** `strategy_got` | 思维图有向聚合重构 | 回答前并行或先后调用多个工具，需将多路独立 `observation` 汇合为结论 | `--select-methods M19`：将多个前置工具观察视为有向无环图（DAG）入边，在汇总节点说明如何合并各路证据 |
| **`M20`** `strategy_bot` | 思维缓冲区元模板检索 | 同领域批量任务（代码审计/Git 检查/Schema 校验），希望复用领域标准结构 | `--select-methods M20`：按【定位目标实体 → 映射工具参数 → 提炼返回指标】领域元模板填入当前样本真实数据 |
| **`M21`** `strategy_reflexion` | 环境反馈语言强化自省 | 轨迹中含工具报错、非零退出码或参数修正，需解释如何根据反馈调整动作 | `--select-methods M21`：锚定前一步 `observation` 的报错或边界提示，直接说明如何据此修正当前命令或参数 |
| **`M22`** `strategy_self_consistency` | 多路自洽性投票与共识提纯 | 高可靠性要求，需消除单次生成的偶发模糊措辞 | `--select-methods M22`：仅保留多视角下 100% 确定成立的工具名、参数键与观察数值共识 |
| **`M23`** `strategy_prm_rerank` | 过程奖励模型步级过滤重排 | 落盘前强制质检，或前序算子因未命中工具名/参数被 `check_prm_grounding` 拦截 | `--select-methods M23`：强制在每步 `<think>` 中显式嵌入所调工具名、关键参数值（路径/标志）及核心实体 |
| **`M24`** `strategy_tia` | 推理轨迹反演展开网络 (TIA) | `L1` 思考被折叠为一句话简短摘要，需反演还原为具体步骤 | `--select-methods M24`：从已有的简短结论或观察摘要出发，逆向反演展开出支撑该结论的 2~3 句直接步骤 |

### 维度 D04：灰盒概率分布、变分推断与生成流网络层（`M25` ~ `M31`）
| 编号与键名 | 方法名称 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :--- | :--- | :--- | :--- |
| **`M25`** `greybox_logits_kl` | 词元级 Top-K 对数概率蒸馏 | 教师接口开放 `logprobs`，需迁移工具选择与关键参数词元上的概率分布 | `--access-tier greybox --select-methods M25`：导出 ShareGPT 轨迹并配置 Top-K (K=20) 前/反向 KL 软标签损失 |
| **`M26`** `strategy_entropy_branch` | 高熵决策节点分叉采样 | 多工具或多参数竞争导致信息熵激增，需捕获分叉点并阐明选型依据 | `--select-methods M26`：针对高不确定度工具决策节点，在阶段二明确写出确定当前工具与参数的具体上下文依据 |
| **`M27`** `greybox_dola` | 跨层对比解码逻辑放大 | 可获取教师顶层与中低层 Logits，需剥离浅层语法套话、放大深层逻辑 | `--access-tier greybox --select-methods M27`：用顶层与早退层对数概率差 $\Delta \log P$ 作为软标签监督学生 |
| **`M28`** `greybox_marcos` | 马尔可夫链潜变量变分推断 | 隐推理不可见或显式标注噪声大，需建模为马尔可夫潜变量链恢复状态转移 | `--access-tier greybox --select-methods M28`：以阶段二逆向模型作为变分后验 $q_\phi(Z\mid X,Y)$，最大化 ELBO 训练学生 |
| **`M29`** `greybox_jepo` | 联合证据下界后验优化 | 问题 $X$ 与结论 $Y$ 跨度极大，需联合优化前向先验与逆向后验分布 | `--access-tier greybox --select-methods M29`：交替迭代前向采样先验与阶段二逆向后验，最小化双向散度 |
| **`M30`** `greybox_regular` | 正则化潜变量轨迹引导 | 潜变量训练易出现后验坍缩（学生忽略推理直接猜答案） | `--access-tier greybox --select-methods M30`：施加与真实工具参数及 `observation` 的互信息正则项 |
| **`M31`** `greybox_gflownets` | 生成流网络有向无环图采样 | 同一任务存在多条等价合法的工具调用顺序（DAG），需消除模式坍缩 | `--access-tier greybox --select-methods M31`：采用 Trajectory Balance 损失使采样概率正比于终端验证奖励 |

### 维度 D05：白盒连续隐空间压缩、记忆迁移与并行迭代层（`M32` ~ `M40`）
| 编号与键名 | 方法名称 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :--- | :--- | :--- | :--- |
| **`M32`** `whitebox_hidden_mse` | 中间层隐状态投影对齐 | 拥有开源师生权重，需在 `</think>` 与工具起始词元处直接对齐连续激活 | `--open-source true --select-methods M32`：在等比例深度层挂载线性投影层，最小化 MSE + Cosine 损失（2~4x 加速） |
| **`M33`** `whitebox_attention_map` | 多头注意力权重图谱迁移 | 长上下文工具调用中学生难以在冗长 `observation` 里定位关键路径或字段 | `--open-source true --select-methods M33`：提取教师顶层多头注意力矩阵，最小化 $\text{KL}(A_T \parallel A_S)$（2~3x 加速） |
| **`M34`** `whitebox_coconut` | 连续隐式思维课程蒸馏 (Coconut) | 希望将本插件采集的分步 `<think>` 逐步内化为 $K$ 个连续隐思维向量 | `--open-source true --select-methods M34`：按课程学习分阶段将每句 `<think>` 替换为 `<bot>...<eot>` 连续隐状态直传（4~8x 加速） |
| **`M35`** `whitebox_codi` | 自我蒸馏连续隐词元压缩 (CODI) | 希望单阶段内训练同一模型将显式 `<think>` 压缩为 $M=4\sim 8$ 个连续隐词元 | `--open-source true --select-methods M35`：并行运行显式教师分支与连续隐词元学生分支，在答案前夕施加 Smooth-L1 损失（5~10x 加速） |
| **`M36`** `whitebox_simcot` | 步级解码器监督隐式思维 (SIM-CoT) | 单纯使用 Coconut/CODI 出现隐词元语义坍缩（丧失步骤区分度） | `--open-source true --select-methods M36`：训练期挂载步级解码器，强制第 $k$ 个隐词元还原出 ShareGPT 第 $k$ 句 `<think>` |
| **`M37`** `whitebox_diagonal_gmm` | 垂直跨层对角映射与混合高斯 | 需将教师横向时间步推理压缩至学生纵向层间激活，并保留多峰分布 | `--open-source true --select-methods M37`：按时空对角线采样教师隐状态，用混合高斯（GMM）损失对齐学生深层激活 |
| **`M38`** `whitebox_rot` | 跨模态视觉特征渲染压缩 (RoT) | 长文本推导序列过长，模型具备高压缩率视觉/二维特征块编码能力 | `--open-source true --select-methods M38`：将显式 `<think>` 渲染映射为紧凑二维特征块向量进行一步隐空间对齐 |
| **`M39`** `whitebox_kava` | 压缩键值缓存工作记忆蒸馏 (KaVa) | 多轮工具调用产生庞大 KV-Cache，需用少量隐词元复现完整工作记忆 | `--open-source true --select-methods M39`：基于注意力重要度浓缩教师 `<think>` 步 KV-Cache，训练学生隐词元直接拟合（5~10x 加速） |
| **`M40`** `whitebox_pccot` | 并行雅可比不动点连续蒸馏 (PCCoT) | 连续隐词元串行自回归仍嫌慢，需在 2~3 次并行迭代内同时收敛全部隐词元 | `--open-source true --select-methods M40`：使用雅可比并行迭代同时更新全部 $M$ 个隐词元槽位并对齐收敛不动点（6~10x 加速） |

### 维度 D06：图拓扑、在策流形与机械可解释性剥离层（`M41` ~ `M45`）
| 编号与键名 | 方法名称 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :--- | :--- | :--- | :--- |
| **`M41`** `whitebox_lrkd` | 图注意力逻辑关系拓扑提取 (LRKD) | 师生隐藏层维度不同或异构，需迁移跨步骤逻辑蕴含与排斥关系图 | `--open-source true --select-methods M41`：用 GAT 构建推理步骤关系邻接矩阵，最小化谱图拓扑距离 $\|G_T - G_S\|_F$ |
| **`M42`** `whitebox_opld` | 在策隐空间想象展开蒸馏 (OPLD) | 离策隐状态蒸馏在多轮测试期出现自回归累积偏移（Off-Policy Drift） | `--open-source true --select-methods M42`：令学生自主展开在策连续隐状态序列，由教师在学生实际访问节点施加在线修正 |
| **`M43`** `whitebox_geosteer` | 黎曼隐流形梯度方向引导 (GeoSteer) | 需通过流形几何方向直接抑制工具参数幻觉与逻辑跳步 | `--open-source true --select-methods M43`：对比教师在 PRM 通过与失败样本上的激活均值差构造流形引导向量 $v_{\text{steer}}$ |
| **`M44`** `whitebox_crosscoder` | 稀疏交叉编码器反思特征剥离 | 需从基座与推理模型残差流差异中剥离出控制“工具前置核验”的稀疏特征 | `--open-source true --select-methods M44`：联合训练 Sparse Crosscoder，筛选推理模型独有的稀疏字典原子监督学生 |
| **`M45`** `whitebox_attr_patching` | 归因修补因果子空间定位 | 全层蒸馏显存开销过大，仅需对齐决定工具参数的 Top-10% 因果子空间 | `--open-source true --select-methods M45`：通过 Attribution Patching 定位高因果贡献注意力头/MLP 子空间并局部对齐 |

### 维度 D07：循环深度架构与动力系统不动点蒸馏（`M46` ~ `M49`）
| 编号与键名 | 方法名称 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :--- | :--- | :--- | :--- |
| **`M46`** `whitebox_recurrent_unroll` | 循环深度共享层渐进展开 (Huginn) | 学生采用 Looped Transformer 循环深度架构，希望沿深度轴循环复现长推理 | `--open-source true --select-methods M46`：将教师分步推导映射为学生共享层块的 $R$ 次深度循环迭代并渐进对齐 |
| **`M47`** `whitebox_equilibrium_fprm` | 动力系统不动点吸引子蒸馏 (FPRM/EqR) | 循环展开 BPTT 显存过高，需直接蒸馏隐空间收敛均衡吸引子 $z^* = f(z^*, x)$ | `--open-source true --select-methods M47`：前向求解不动点 $z^*$，利用隐函数定理直接计算梯度对齐教师最终决策表征（8~15x 加速） |
| **`M48`** `whitebox_adaptive_early_exit` | 相对 KL 散度零样本早退策略迁移 | 希望学生在简单步骤 1~2 步早退、仅在复杂工具参数步骤多步迭代 | `--open-source true --select-methods M48`：监控相邻迭代步分布 $\text{KL}(p_t \parallel p_{t-1}) < \tau$，迁移教师收敛早退判据 |
| **`M49`** `strategy_pause_anchor` | 暂停与填充词元算力锚定 | 黑盒需在复杂工具调用前设结构化检查点，或白盒插入 `<pause>` 扩展 KV 算力列 | `--select-methods M49`：黑盒生成【目标确认 → 参数映射 → 执行收敛】三检查点摘要；白盒对齐 `<pause>` 词元 KV 激活 |

### 维度 D08：词表凸组合软思维与强化学习混合内化（`M50` ~ `M54`）
| 编号与键名 | 方法名称 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :--- | :--- | :--- | :--- |
| **`M50`** `whitebox_softcot` | 词表概率凸组合软词元传递 (SoftCoT) | 不改模型嵌入层架构，需保留单步多分支叠加态传递至下一步输入 | `--open-source true --select-methods M50`：用完整词表概率 $p_t$ 加权词嵌入矩阵 $e_{\text{soft}} = p_t^\top E$ 作为下一步输入（3~6x 加速） |
| **`M51`** `whitebox_soft_grpo` | 重参数化软思维策略优化 (SofT-GRPO) | 需在连续软词元单纯形空间内结合工具执行正确性做强化学习优化 | `--open-source true --select-methods M51`：注入 Gumbel-Softmax 重参数化噪声，以 PRM 与工具结果奖励执行 GRPO 更新 |
| **`M52`** `whitebox_hrpo` | 混合离散连续动作空间强化 (HRPO) | 希望模型自主将繁琐计算内化至连续隐状态、仅外显 2~3 句关键决策锚点 | `--open-source true --select-methods M52`：解耦连续隐状态门控与离散词元采样，联合优化任务奖励与显式长度成本 |
| **`M53`** `whitebox_hylar` | 分层混合潜空间强化蒸馏 (HyLaR) | 多轮长程任务中高层策略适合连续潜空间规划、底层工具 JSON 适合离散解码 | `--open-source true --select-methods M53`：高层控制器输出连续规划潜向量 $z_{\text{plan}}$，底层解码器条件生成 `function_call` |
| **`M54`** `whitebox_dylar_swireasoning` | 熵驱动动态双模切换 (DyLaR) | 希望低熵步保持高速隐式推演，遇到高熵逻辑瓶颈自动切换显式 `<think>` | `--open-source true --select-methods M54`：实时评估预测熵 $H(p_t)$，超越阈值时展开显式 ShareGPT `<think>`，否则隐状态直传 |

### 维度 D09：投机解码预言机与多教师演化嫁接（`M55` ~ `M59`）
| 编号与键名 | 方法名称 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :--- | :--- | :--- | :--- |
| **`M55`** `greybox_spec_oracle` | 投机解码接受长度验证预言机 | 闭源接口屏蔽 Logits 但开放投机解码（Speculative Decoding）批量验证 | `--access-tier greybox --select-methods M55`：将阶段二候选 `<think>` 作为 Draft 提交验证，按接受长度与拒绝点修补断点 |
| **`M56`** `greybox_oskd` | 在线投机知识蒸馏 (OSKD) | 全序列在线蒸馏算力高，需教师仅在学生预测失准的回退节点下发局部监督 | `--access-tier greybox --select-methods M56`：学生作草稿器生成轨迹，教师并行验证仅在拒绝位置计算局部 KL 损失 |
| **`M57`** `strategy_cotevo` | 多教师思维链遗传交叉变异 (CoT-Evo) | 单一视角推导不够全面，需融合“意图分析”与“参数符号映射”并剪除冗余 | `--select-methods M57`：在阶段二对双视角候选推导执行交叉重组与去冗余剪枝，合成 2~3 句高密度摘要 |
| **`M58`** `strategy_merge_of_thought` | 多源异构思维链共识合并 | 需将问题上下文实体、工具参数键名与观察返回指标无缝嫁接为连贯推导 | `--select-methods M58`：提取问题、工具契约与观察反馈三方核心实体，按时间步合并为连贯摘要 |
| **`M59`** `strategy_dlcot` | 去冗余长思维链结构剪枝 (DLCoT) | 原始 `<think>` 过于啰嗦或含无效兜圈，需提纯为精炼的高层直接步骤摘要 | `--select-methods M59`：剥离重复寒暄与无效分支，仅保留通向工具调用或最终答案的最短因果主干（2~3 句） |

### 维度 D10：程序化符号转换与多视角正交解耦（`M60` ~ `M62`）
| 编号与键名 | 方法名称 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :--- | :--- | :--- | :--- |
| **`M60`** `strategy_pot` | 可执行程序思维锚定蒸馏 (PoT) | 含 `run_command`、脚本执行、Git 命令或数值统计，需严格锚定命令与输出 | `--select-methods M60`：围绕实际执行的命令字符串、参数标志与标准输出（stdout/exit code）生成符号级摘要 |
| **`M61`** `strategy_ast_query_plan` | 抽象语法树与查询计划蒸馏 (AST) | 含代码 AST 解析、函数签名检查、JSON Schema 校验或复杂嵌套字典参数 | `--select-methods M61`：从代码语法结构、函数签名或 Schema 字段校验规则角度直接说明解析目标与结论 |
| **`M62`** `strategy_mpcot` | 多视角正交分解与融合 (MPCoT) | 同时涉及“用户意图”、“当前工作区环境状态”与“工具参数契约”的复杂多轮任务 | `--select-methods M62`：依次从【用户意图】、【已知环境状态】与【所调工具契约】三个正交维度合成 2~3 句推导 |

### 维度 D11 & D12：跨模态内化与本地会话时序校准层（`M63` ~ `M66`）
| 编号与键名 | 方法名称 | 使用条件（什么时候用） | 详细使用方法（怎么用） |
| :--- | :--- | :--- | :--- |
| **`M63`** `multimodal_bridge_erasure` | 多跳跨模态中介渐进擦除 | 端到端语音/多模态模型需消除中间 ASR 转录稿与文本 `<think>` 转换延迟 | `--open-source true --select-methods M63`：先采集显式四跳轨迹，再在课程训练中分阶段擦除中间转录与文本监督 |
| **`M64`** `multimodal_visual_sketch` | 生成式视觉草图隐词元内化 | 多模态 Agent 需调用视觉裁剪/UI 布局工具辅助空间几何推理 | `--open-source true --select-methods M64`：将多步视觉工具返回的区域特征浓缩为内部连续视觉隐词元（Latent Visual Tokens） |
| **`M65`** `strategy_timing_chunk_budget` | 流式延迟与载荷分块复杂度校准 | 需根据每步耗时与 `observation` 载荷大小动态区分快步与重算步详略度 | `--select-methods M65`：对大载荷/多参数关键步写明完整参数与数据核验（2~3 句），对简单步保持极简（1~2 句） |
| **`M66`** `strategy_local_transcript_inspect` | 宿主本地会话日志无损解析与对齐 | 在 Antigravity 宿主下从 `transcript_full.jsonl` 无损提取对话、工具与原生 `<think>` | 默认内置于 `inspect`（亦可 `--select-methods M66`）：保留已合规原生 `<think>`，仅对缺失节点触发阶段二补全 |

---

## 四、标准端到端执行工作流（Main-Agent 逐步操作手册）

### Step 1：按需检索方法库或初始化任务队列
```powershell
# 1.1 （可选）按条件检索 66 种方法的使用条件与用法
python skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py list-methods --dimension D02,D10
python skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py list-methods --guard-level L4

# 1.2 初始化或断点续传种子任务队列
python skills/multi-turn-tool-distiller/scripts/task_manager.py init `
  --input "skills/multi-turn-tool-distiller/examples/seed_examples.jsonl" `
  --output-dir "datasets" `
  --default-turns 2 `
  --max-retries 5

# 1.3 获取下一批待处理样本
python skills/multi-turn-tool-distiller/scripts/task_manager.py next-batch `
  --output-dir "datasets" `
  --batch-size 3
```

### Step 2：定义并拉起阶段一自然执行 Sub-Agent (`distill_executor_agent`)
1. 每个会话首次运行时调用 `define_subagent` 注册 `distill_executor_agent`（开启 `EnableWriteTools=True, EnableMCPTools=True`）和 `distill_rationalizer_agent`。
2. 调用 `invoke_subagent` 拉起独立 `distill_executor_agent`，原样发送第 1 轮 `question`（仅附带一句完成后通过 `send_message` 回传父 Conversation ID 的说明）。
3. 收到第 1 轮回复后，若有 `follow_ups`，调用 `send_message` 原样推进后续轮次；全部轮次完成后立即调用 `manage_subagents(Action="kill")` 销毁该实例。

### Step 3：轨迹诊断、细化自由选型与阶段二补全 (`inspect` -> `merge-stage2`)
```powershell
# 3.1 运行 inspect（可自由指定 --select-methods M15,M60,M34，或设为默认由五维路由器自动选择）
python skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py inspect `
  --transcript "<AppDataDir>/brain/<stage1_subagent_id>/.system_generated/logs/transcript_full.jsonl" `
  --sample-id "<sample_id>" `
  --expected-turns <target_turns> `
  --open-source <true|false> `
  --access-tier <auto|blackbox|greybox|whitebox|hybrid> `
  --guard-level <auto|L0|L1|L2|L3|L4> `
  --task-domain <auto|general|tool_agent|code_symbolic|math_logic|long_context|multimodal> `
  --latency-budget <auto|unconstrained|moderate|low_latency|ultra_fast> `
  --select-methods "<optional: e.g. M15,M60,M34>" `
  --output "datasets/stage1_<sample_id>.json"
```
- 若 `audit_passed == true` 且 `needs_stage2 == false`：直接进入 Step 4 落盘。
- 若 `needs_stage2 == true`：
  1. 读取 `datasets/stage1_<sample_id>.json` 中的 `auto_stage2_prompt`（已按首选方法注入专属算子指令）。
  2. 调用 `invoke_subagent(TypeName="distill_rationalizer_agent")` 生成缺失节点的 JSON 映射，完成后立即 `kill`。
  3. 调用 `merge-stage2` 合并并执行 PRM 步级实体锚定校验：
     ```powershell
     python skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py merge-stage2 `
       --inspected "datasets/stage1_<sample_id>.json" `
       --stage2-transcript "<AppDataDir>/brain/<stage2_subagent_id>/.system_generated/logs/transcript_full.jsonl" `
       --strategy "<strategy_key_or_id>" `
       --output "datasets/stage1_<sample_id>.json"
     ```
  4. 若个别节点仍未通过 `check_prm_grounding`，自动切换 `recommended_strategies` 中的下一顺位方法重试。

### Step 4：原子写入 ShareGPT 数据集
```powershell
python skills/multi-turn-tool-distiller/scripts/sharegpt_writer.py commit `
  --extracted "datasets/stage1_<sample_id>.json" `
  --output-file "datasets/sharegpt_train.jsonl" `
  --state-dir "datasets" `
  --template-used "<strategy_or_recipe_used>"
```
落盘成功后删除临时文件 `datasets/stage1_<sample_id>.json`。
