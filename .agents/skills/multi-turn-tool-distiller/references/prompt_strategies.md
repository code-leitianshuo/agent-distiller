# Two-Stage Adaptive Distillation Strategies (`prompt_strategies.md`)

本文件定义双阶段自适应蒸馏架构下的：
1. **阶段一执行子代理 (`distill_executor_agent`) 的轻量级自然引导系统提示词**（无填空槽位，无同题示例污染）；
2. **Main-Agent 智能模型与轨迹画像矩阵（Adaptive Profiler Matrix）**；
3. **阶段二重构子代理 (`distill_rationalizer_agent`) 的五级回退算子库（Fallback Waterfall）**。

---

## 1. 阶段一：轻量级自然执行系统提示词 (`distill_executor_agent`)

阶段一的目标是让教师模型在零包装干扰的自然多轮对话中，真实调用宿主工具完成任务，并自然输出高层推理摘要。

```markdown
You are a capable technical assistant solving multi-turn tasks with real workspace and MCP tools.
When working through a user request:
- Feel free to naturally summarize your direct reasoning inside `<think>...</think>` tags before invoking a tool or presenting your final answer.
- Focus on solving the user's actual request accurately using the available tools. Do not follow rigid slot templates; express your direct reasoning naturally in a few concise sentences based on the real task context and tool outputs.
- Complete only the current turn's request and wait for the user's next message.
```

> **重要原则**：在阶段一向 `distill_executor_agent` 发送首轮 `Prompt` 及后续 `send_message` 追问时，**直接发送纯净的用户问题文本**，严禁附加 `[Task Execution Protocol]` 外壳或尾部格式提醒，确保模型的工具决策与回答分布完全自然。

---

## 2. Main-Agent 智能画像与算子选择矩阵（Adaptive Profiler Matrix）

当阶段一执行完毕后，Main-Agent 运行 `trajectory_extractor.py inspect` 对原始轨迹与模型响应特征进行诊断，自动判定模型输出类型并生成阶段二算子优先级队列：

| 画像分类 (`profile_type`) | 阶段一轨迹特征表现 | 智能推荐的 Stage-2 算子回退顺序 (`recommended_strategies`) |
| :--- | :--- | :--- |
| `DIRECT_COMPLETE` | 所有工具调用步与最终回复步均已自然包含非空且与上下文一致的 `<think>...</think>` | 无需启动阶段二，直接进入 PRM 一致性校验与落盘 |
| `TOOL_SILENT_NATIVE` | 最终回复步包含自然 `<think>`，但模型在发起原生 `tool_call` 时仅下发工具指令而省略了同轮文本 `<think>` | `strategy_star` -> `strategy_prefill` -> `strategy_mpcot` -> `strategy_pot` -> `strategy_cotevo` |
| `CLI_SYMBOLIC_EXECUTION` | 轨迹包含命令行脚本执行（`run_command`）、数值统计或结构化文件变更，需将推理严格锚定到执行输出 | `strategy_pot` -> `strategy_star` -> `strategy_mpcot` -> `strategy_cotevo` -> `strategy_prefill` |
| `MULTI_HOP_TOOL_CHAIN` | 单轮或多轮内连续调用 $\ge 2$ 种不同工具，涉及多步信息串联与跨工具参数传递 | `strategy_mpcot` -> `strategy_star` -> `strategy_cotevo` -> `strategy_pot` -> `strategy_prefill` |
| `STRICT_GUARDED_MINIMAL` | 阶段一全程仅输出最终结论正文与工具调用，未输出任何 `<think>` 标签 | `strategy_star` -> `strategy_cotevo` -> `strategy_mpcot` -> `strategy_pot` -> `strategy_prefill` |

---

## 3. 阶段二：重构子代理 (`distill_rationalizer_agent`) 与五级回退算子库

### 3.1 `distill_rationalizer_agent` 基础系统提示词 (`define_subagent`)

```markdown
You are a trajectory rationale synthesizer for multi-turn tool-use conversations.
Given a verified ground-truth execution trajectory (containing user questions, actual tool calls with arguments, real tool observations, and final answers), your job is to reconstruct natural, context-grounded `<think>...</think>` reasoning summaries for any decision nodes marked as needing rationale synthesis.
Rules:
1. Every generated rationale must be genuinely derived from the specific user question, the exact tool name and arguments called, or the real observation returned in that trajectory. Never use generic boilerplate.
2. Keep each rationale concise and direct (a high-level summary in a few sentences covering the direct steps leading to the tool call or final answer).
3. Output strictly a valid JSON object mapping each target `node_index` (as a string key) to its reconstructed reasoning text (without outer `<think>` tags, as the merger wraps them automatically).
```

---

### 3.2 五级回退重构算子模板（Fallback Waterfall Strategies）

当 Main-Agent 调用 `distill_rationalizer_agent` 执行阶段二时，将 `trajectory_extractor.py inspect` 生成的待补全节点上下文 `{{TRAJECTORY_CONTEXT_JSON}}` 与目标节点列表 `{{TARGET_NODE_INDICES}}` 填入所选算子模板：

#### 算子 1：`strategy_star` — 已知答案与真实 Observation 逆向归因（STaR Post-hoc Rationalization）
- **适用场景**：默认首选算子（尤其针对 `TOOL_SILENT_NATIVE` 與 `STRICT_GUARDED_MINIMAL`）。已知模型实际发起的 `function_call` 或最终回答，逆向归因连接“问题/观察”与“动作/结论”的直接逻辑桥梁。
- **调用 Prompt**：
  ```markdown
  [Stage-2 Operator: strategy_star (STaR Post-hoc Rationalization)]
  以下是第一阶段真实执行并通过环境验证的多轮工具调用轨迹上下文：
  ```json
  {{TRAJECTORY_CONTEXT_JSON}}
  ```
  请针对待补全/重构的节点索引列表 `{{TARGET_NODE_INDICES}}`，运用已知动作/结果逆向归因法（STaR）：
  - 若目标节点后接 `function_call`：结合当前轮次的 `human` 提问与前序观察，直接说明为何需要调用该具体工具以及所传参数值的来源；
  - 若目标节点为 `observation` 后的最终 `gpt` 回复：结合工具真实返回的数据与最终回答内容，直接概括从工具输出得出最终答复的核心逻辑步骤。

  请仅输出如下格式的合法 JSON 对象（不要输出额外解释）：
  ```json
  {
    "<node_index>": "针对该节点的自然、简明高层推理摘要文本..."
  }
  ```
  ```

#### 算子 2：`strategy_mpcot` — 多视角正交分解合成（MPCoT Orthogonal Decomposition）
- **适用场景**：多跳工具调用链（`MULTI_HOP_TOOL_CHAIN`）或 `strategy_star` 未通过一致性校验时的回退算子。
- **调用 Prompt**：
  ```markdown
  [Stage-2 Operator: strategy_mpcot (Multi-Perspective Orthogonal Decomposition)]
  以下是第一阶段真实执行的多轮工具调用轨迹：
  ```json
  {{TRAJECTORY_CONTEXT_JSON}}
  ```
  请针对节点 `{{TARGET_NODE_INDICES}}`，综合以下正交视角（用户任务意图、当前已知/未知环境状态、工具接口参数或返回结果）提炼出一句到数句话的自然连贯高层推理摘要，准确衔接上下文与下一步工具调用或最终答复。

  请仅输出如下格式的合法 JSON 对象：
  ```json
  {
    "<node_index>": "融合意图、状态与参数/结果的自然高层推理摘要..."
  }
  ```
  ```

#### 算子 3：`strategy_pot` — 程序与执行证据锚定推演（PoT Execution-Grounded Synthesis）
- **适用场景**：涉及 `run_command`、代码分析、文件校验或数值统计的任务（`CLI_SYMBOLIC_EXECUTION`）。
- **调用 Prompt**：
  ```markdown
  [Stage-2 Operator: strategy_pot (Execution-Evidence Grounded Synthesis)]
  以下是包含真实命令行/文件工具执行证据的轨迹上下文：
  ```json
  {{TRAJECTORY_CONTEXT_JSON}}
  ```
  请针对节点 `{{TARGET_NODE_INDICES}}`，严格以轨迹中的实际命令字符串、文件路径、退出码及标准输出中的关键数据为锚点，生成简明直接的高层推理摘要，确保推理中引用的命令、路径或输出结论与实际执行记录 100% 吻合。

  请仅输出如下格式的合法 JSON 对象：
  ```json
  {
    "<node_index>": "锚定具体命令/路径与执行输出的自然推理摘要..."
  }
  ```
  ```

#### 算子 4：`strategy_cotevo` — 多候选生成与演化择优融合（CoT-Evo Candidate Merge）
- **适用场景**：当前序算子出现信息遗漏或未能通过 PRM 相关性校验时的强化回退算子。
- **调用 Prompt**：
  ```markdown
  [Stage-2 Operator: strategy_cotevo (Evolutionary Candidate Selection & Merge)]
  上一轮重构未通过校验原因：{{LAST_AUDIT_ERRORS}}
  轨迹上下文如下：
  ```json
  {{TRAJECTORY_CONTEXT_JSON}}
  ```
  请针对节点 `{{TARGET_NODE_INDICES}}`，综合“工具调用必要性”与“上下文关键实体精确匹配”两个维度的最优表达，去除了任何空洞套话，合成一段高信息密度、自然流畅的数句话高层推理摘要。

  请仅输出如下格式的合法 JSON 对象：
  ```json
  {
    "<node_index>": "经择优融合后的自然高层推理摘要..."
  }
  ```
  ```

#### 算子 5：`strategy_prefill` — 决策上下文前缀续写补全（Prefill-Guided Completion）
- **适用场景**：兜底算子，通过给定上下文锚点引导模型自然续写直至动作或结论。
- **调用 Prompt**：
  ```markdown
  [Stage-2 Operator: strategy_prefill (Context-Prefill Guided Completion)]
  轨迹上下文如下：
  ```json
  {{TRAJECTORY_CONTEXT_JSON}}
  ```
  请针对节点 `{{TARGET_NODE_INDICES}}`，直接站在助手执行到该步时的第一视角，顺承上一条消息的具体内容，用简明直接的几句话补全导向该步工具调用或最终答复的思考摘要。

  请仅输出如下格式的合法 JSON 对象：
  ```json
  {
    "<node_index>": "顺承上下文自然补全的高层推理摘要..."
  }
  ```
  ```
