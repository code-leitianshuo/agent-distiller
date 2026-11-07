# 双阶段蒸馏提示词算子与后训练执行规范 (`prompt_strategies.md`)

本文件定义 `multi-turn-tool-distiller` 的 **阶段一零包装自然执行策略**、**阶段二可执行重构算子库（覆盖 `M01`~`M24`、`M26`、`M49`、`M57`~`M62`、`M65`~`M66`）** 以及 **灰盒/白盒后训练配置生成规范（覆盖 `M25`、`M27`~`M48`、`M50`~`M56`、`M63`~`M64`）**。

---

## 一、核心原则（杜绝人为八股文与分布失真）

1. **阶段一（自然分布采集，`distill_executor_agent`）**：
   - 用户提问与多轮追问 **原样发送**，严禁在 User 消息外包裹任何诱导八股或格式指令。
   - 仅在 Sub-Agent 的 `SystemPrompt` 中提供极简自然引导，允许模型在 `<think>...</think>` 中输出 **简明的高层推理摘要（2~3 句话，仅陈述通向工具调用或最终回答的最直接步骤）**。
2. **阶段二（锚定逆向重演，`distill_rationalizer_agent`）**：
   - 当阶段一因模型防护（`L1`~`L4`）或原生工具调用静默导致缺失 `<think>` 时，Main-Agent 根据 **用户自由指定的 `--select-methods`** 或 **五维自动路由器** 选择算子，基于已知真实轨迹 $(X, A, O, Y)$ 合成高层推理摘要。
   - **禁止使用 Python 硬编码字符串模板拼接 `<think>`**：所有缺失节点的 `<think>` 必须由独立的 `distill_rationalizer_agent` 实例生成，并通过 `check_prm_grounding` 锚定校验。

---

## 二、阶段一（Stage 1）Sub-Agent 定义规范

```json
{
  "Name": "distill_executor_agent",
  "Description": "Stage-1 natural execution teacher subagent with real workspace and MCP tools.",
  "EnableWriteTools": true,
  "EnableMCPTools": true,
  "EnableSubagentTools": false,
  "SystemPrompt": "You are a helpful AI assistant with access to workspace and MCP tools. Before calling a tool or providing your final answer, briefly summarize your high-level reasoning inside <think>...</think> tags (2-3 concise sentences covering the direct steps leading to the action or conclusion). When you finish each turn and are ready to reply to the user, send your complete turn response back to the parent agent ID specified in the task prompt using send_message."
}
```

---

## 三、阶段二（Stage 2）Sub-Agent 定义规范

```json
{
  "Name": "distill_rationalizer_agent",
  "Description": "Stage-2 trajectory rationalization subagent supporting all 66 distillation methods.",
  "EnableWriteTools": false,
  "EnableMCPTools": false,
  "EnableSubagentTools": false,
  "SystemPrompt": "You are a Reasoning Trajectory Rationalization Engine. Given a completed multi-turn tool-calling trajectory (user questions, exact tool calls, real tool observations, and final answers), synthesize concise, high-level <think>...</think> summaries for the specified target node indices using the requested distillation method. Every summary must be 2-3 sentences covering only the most direct logical steps and grounding exact tool names, parameters, and observation facts. Return ONLY a valid JSON object mapping each target node index (as a string key) to its synthesized thought text."
}
```

---

## 四、阶段二可执行黑盒/代理算子指令矩阵（怎么用 & 注入指令）

当 `trajectory_extractor.py inspect` 生成 `stage2_prompt` 时，会根据所选方法（`M01`~`M66`）自动注入对应的算子执行指令：

| 方法编号与策略键 | 适用条件（什么时候用） | 注入 `distill_rationalizer_agent` 的核心算子指令（怎么用） |
| :--- | :--- | :--- |
| **`M01` `strategy_meta_prompt`**<br>结构化元提示拆解 | `L0`~`L2` 常规缺失思考节点，需按标准认知结构补全 | 按“①当前子目标识别 → ②所需工具或前置条件核对 → ③直接执行动作/结论推导”三段式结构，为每个目标节点生成 2~3 句高层摘要。 |
| **`M02` `strategy_prefill`**<br>回复预填充引导 | 模型默认跳过思考直出结论，需通过上下文首句顺承引导 | 为每个目标节点先构造贴合上下文的起始引导前缀（如“针对用户提出的…，需先通过…工具获取…”），并自然续写完成剩余 1~2 句直接推理。 |
| **`M03` `strategy_star`**<br>已知动作逆向归因 (STaR) | 阶段一已拿到真实 `function_call` 与 `observation`，逆向补全推理桥 | 将后续已知的真实工具调用参数或最终回答视为确定性目标，逆向推导从当前输入状态到达该动作的最直接逻辑桥梁。 |
| **`M04` `strategy_quiet_star`**<br>词元级隐式前瞻合理化 | 多轮跨步工具链，需解释当前动作对后续轮次的铺垫价值 | 结合后续步骤的真实环境反馈，从前瞻视角说明当前工具调用或回复如何为完成整条任务链路提供必要支撑。 |
| **`M05` `strategy_debate`**<br>多智能体对抗质询与修正 | 常规单次重构易漏掉参数来源或未通过 PRM 校验 | 先起草初始推导，再以审计员视角核查“每个工具名、文件路径与参数值是否在上下文中有明确依据”，修正跳步后输出定稿。 |
| **`M06` `strategy_socratic`**<br>苏格拉底助产术拆解 | `L4` 强反蒸馏拒答或复杂多跳问题，需通过客观问答还原 | 不使用任何“内部思考/心路历程”表述，改为围绕“当前已知事实是什么？缺失哪项核心数据？调用哪个工具可补齐？”的客观问答合成摘要。 |
| **`M07` `strategy_least_to_most`**<br>由简入繁递进分解 | 复合多跳任务，后一步依赖前一步的基础结果 | 将当前轮次拆解为基础前置子任务与后续递进子任务，明确说明当前节点正在解决哪个最小原子子问题及其工具依据。 |
| **`M08` `strategy_step_back`**<br>后退一步抽象原理锚定 | 架构分析、规则校验或数学/科学问题 | 先后退一步点明当前步骤所依据的通用架构规范、校验法则或接口契约，再将其映射到具体的工具调用参数或结论。 |
| **`M09` `strategy_plan_and_solve`**<br>先规划后求解显式锚定 | 单轮包含连续多次工具调用，需明确全局计划进度 | 在首节点点明整体分步执行蓝图（Plan），在后续工具节点点明当前处于蓝图的哪一步以及上一步结果如何驱动下一步（Solve）。 |
| **`M10` `strategy_ccot`**<br>正反对比思维链归因 | 存在多个相近候选工具，需明确工具选型边界 | 简要点明为何当前工具及参数相比其他备选方式更直接有效，并给出参数提取依据。 |
| **`M11` `strategy_code_paradigm`**<br>代码与接口规约范式转换 | `L4` 强反蒸馏护栏，自然语言解释被拒但代码文档放行 | 以软件工程接口规约视角，将每个节点的推导表述为“输入前置条件（Precondition）→ 调用规约与参数绑定 → 预期输出断言”，再浓缩为 2~3 句摘要。 |
| **`M12` `strategy_state_continuity`**<br>跨轮会话状态连续性锚定 | 多轮对话（`turns >= 2`），需衔接前序轮次已确认状态 | 显式引用上一轮对话已获取的工具输出或确认结论，说明本轮如何在前序状态基础上推进新操作。 |
| **`M13` `strategy_tier_alignment`**<br>跨尺寸同族模型对齐转录 | 旗舰模型输出极简，需利用轻量/同族模型展开可读推理 | 严格锁定阶段一旗舰模型产出的真实工具调用与结论，展开符合学生模型学习粒度的 2~3 句直接推理步骤。 |
| **`M14` `strategy_context_compaction`**<br>长上下文自动摘要逆向展开 | 消息数较多的长链路轨迹，需从阶段性汇总反推单步依据 | 先提炼当前阶段的核心里程碑，再将该里程碑拆解回当前具体工具调用节点的直接输入输出依据。 |
| **`M15` `strategy_echocot_scratchpad`**<br>工具调用链状态驻留草稿锚定 | 单轮连续多跳工具调用（EchoCoT 场景） | 聚焦相邻两次工具调用之间的状态驻留窗口：明确陈述上一步 `observation` 返回了什么具体字段，以及该字段如何直接决定下一步 `function_call` 的参数。 |
| **`M16` `strategy_budget_oracle`**<br>词元长度与复杂度预算预言机 | 需按步骤实际复杂度精准控制 `<think>` 摘要长度与信息密度 | 严格按照系统根据命令复杂度与返回字节数测算的句数/字数预算（2~3 句紧凑高密度文本）合成推导，禁止冗余扩写。 |
| **`M17` `strategy_mcts`**<br>蒙特卡洛树搜索轨迹重构 | 多步探索任务，需按终端成功结果回传节点价值 | 从整条轨迹最终成功的终端状态回溯，说明当前节点的工具选择为何是通向最终答案的高价值路径。 |
| **`M18` `strategy_tot`**<br>思维树分叉评估展开 | 关键决策点存在多条可行分解路径 | 针对当前节点内部比选可行方案，保留与实际执行的 `function_call` 参数完全吻合的最优直接路径。 |
| **`M19` `strategy_got`**<br>思维图有向聚合重构 | 并行或先后调用多个工具后汇总结论 | 将多个前置工具返回的 `observation` 视为有向图入边，清晰陈述如何聚合多源证据得出当前结论。 |
| **`M20` `strategy_bot`**<br>思维缓冲区元模板检索 | 同领域批量任务（代码审计/VCS/Schema 校验） | 调用当前领域的标准高层推理元结构（目标定位 → 工具参数映射 → 结果验证），填入本样本的真实实体与参数。 |
| **`M21` `strategy_reflexion`**<br>环境反馈语言强化自省 | 轨迹中出现非零退出码、异常输出或参数调整 | 锚定前一步 `observation` 中的报错或边界提示，直接说明如何据此修正当前步的工具命令或参数。 |
| **`M22` `strategy_self_consistency`**<br>多路自洽性投票与共识提纯 | 高可靠性要求，需提取多视角共识事实 | 仅保留在多角度审视下均100%成立的确定性工具参数依据与观察事实，剔除一切推测性表述。 |
| **`M23` `strategy_prm_rerank`**<br>过程奖励模型步级过滤重排 | PRM 校验未通过或需最大化实体锚定率 | 强制确保每一句推导均显式包含后续 `function_call` 的工具名、核心参数值（如路径、命令标志）或前置 `observation` 的关键指标。 |
| **`M24` `strategy_tia`**<br>推理轨迹反演展开网络 (TIA) | `L1` 思考被折叠为一句话笼统摘要 | 以已有的简短摘要或最终结论为终点，逆向反演展开出支撑该摘要成立的 2~3 句具体执行步骤。 |
| **`M26` `strategy_entropy_branch`**<br>高熵决策节点分叉采样 | 多工具竞争的高不确定度节点 | 针对多工具或多参数分支点，直接阐明选择当前工具及参数的确切上下文依据。 |
| **`M49` `strategy_pause_anchor`**<br>暂停与填充词元算力锚定 | 复杂工具决策前需显式设置结构化检查点 | 按“[目标检查点] → [参数映射检查点] → [执行决策点]”三锚点紧凑串联为 2~3 句高层推理摘要。 |
| **`M57` `strategy_cotevo`**<br>多教师思维链遗传交叉变异 | 融合目标意图与符号执行双视角并剪除冗余 | 融合“用户任务目标拆解”与“工具参数符号推导”两条主线，剪除一切套话，保留高信息密度摘要。 |
| **`M58` `strategy_merge_of_thought`**<br>多源异构思维链共识合并 | 合并多条候选推导各自命中的关键实体 | 将来自问题上下文、工具契约与观察反馈的三方关键实体无缝嫁接为连贯的 2~3 句推导。 |
| **`M59` `strategy_dlcot`**<br>去冗余长思维链结构剪枝 | 原始 `<think>` 过长或含无效兜圈，需提纯为精炼摘要 | 剪除一切重复寒暄、无效回溯与事后总结词，仅保留通向工具调用或最终答案的最直接步骤。 |
| **`M60` `strategy_pot`**<br>可执行程序思维锚定 (PoT) | 含 `run_command`、脚本执行、Git 命令或数值计算 | 严格锚定将要执行的命令语法、参数标志及前一步标准输出（stdout/exit code）数据进行符号级推导。 |
| **`M61` `strategy_ast_query_plan`**<br>抽象语法树与查询计划蒸馏 | 含 AST 代码解析、JSON Schema 校验或结构化查询 | 从抽象语法结构、字段校验规则或数据结构映射角度，直接说明工具调用的目标节点与验证逻辑。 |
| **`M62` `strategy_mpcot`**<br>多视角正交分解 (MPCoT) | 同时涉及用户意图、环境状态与工具契约的复杂多轮任务 | 依次从【用户意图视角】、【当前环境状态视角】与【工具接口参数视角】三个正交维度合成 2~3 句紧凑推导。 |
| **`M65` `strategy_timing_chunk_budget`**<br>流式延迟与载荷复杂度校准 | 根据步骤耗时与 `observation` 载荷大小动态适配详略度 | 针对大载荷/多参数关键步完整写明输入输出依据（2~3 句），针对轻量确认步保持 1~2 句精炼直接推导。 |
| **`M66` `strategy_local_transcript_inspect`**<br>宿主本地会话日志无损对齐 | 从本地 `transcript_full.jsonl` 合并原生外显思考与补全缺失节点 | 保留本地日志中已合规的原生外显 `<think>`，仅针对缺失节点精准补全与上下文无缝衔接的高层摘要。 |

---

## 五、灰盒与白盒方法（`M25`、`M27`~`M48`、`M50`~`M56`、`M63`~`M64`）的后训练配置输出规范

当用户指定 `--open-source true`、`--access-tier greybox|whitebox` 或在 `--select-methods` 中选择了灰盒/白盒方法（如 `M25`, `M34`, `M35`, `M39`, `M42`, `M47`, `M50` 等）时：

1. **第一层交付物（显式锚点数据集）**：
   先完成阶段一与阶段二，在 `datasets/sharegpt_train.jsonl` 中产出 100% 经过 PRM 锚定校验的标准 ShareGPT 轨迹（为白盒课程擦除、步级解码器 `SIM-CoT` 或不动点对齐提供精确的步骤级监督锚点）。
2. **第二层交付物（可执行后训练配方 `post_training_recipes`）**：
   `trajectory_extractor.py inspect` 与 `recommend` 会在输出 JSON 中自动生成所选灰盒/白盒方法的结构化训练规范，包含：
   - `method_id` 与 `name`（如 `M34: Coconut — Chain of Continuous Thought`）
   - `when_to_use`（适用条件）
   - `how_to_use`（基于 `sharegpt_train.jsonl` 的具体训练步骤、损失函数公式与关键超参数配置）
