---
name: multi-turn-tool-distiller
description: >-
  Automated multi-turn conversation and real tool-calling trajectory distillation
  workflow in Antigravity using Main-Sub Agent orchestration. Use when the user asks
  to distill, synthesize, or collect multi-turn dialogues, tool-calling trajectories,
  or ShareGPT training datasets using isolated Sub-Agents.
---

# Multi-Turn & Tool-Calling Trajectory Distiller (`multi-turn-tool-distiller`)

本 Skill 用于在 Antigravity 宿主环境下，通过 **Main-Sub Agent 协同架构** 自动化采集多轮对话与真实内置工具调用（文件读写、命令行执行、MCP 工具调用）轨迹，经全链路 `<think>` 与参数质量审计后，持久化为 100% 兼容 LLaMA-Factory 的标准 **ShareGPT `.jsonl`** 数据集。

---

## 角色边界与铁律

1. **Main-Agent（控制器 / 追问模拟器 / 质量审计与提示词调优器）**：
   - 负责解析输入配置、管理任务队列与断点状态（`.state.json`）、注册并拉起 Sub-Agent、扮演多轮对话中的用户（User Simulator）发起递进追问、审计 `<think>` 与工具调用质量、在未达标时切换提示词模板重试（最多 `max_retries` 次），并调用落盘脚本写入 `.jsonl`。
   - **严禁越俎代庖**：Main-Agent 绝不能直接替 Sub-Agent 生成被蒸馏的业务回答或伪造工具执行轨迹。
2. **Sub-Agent（单样本隔离教师实例）**：
   - **单样本单实例 + 工作区隔离**：每一条完整样本（含其内全部多轮对话与工具调用）必须对应一个独立的 Sub-Agent 实例。优先尝试 `Workspace="branch"` 克隆独立分支工作区；若宿主返回 `branch workspace not supported for VCS type Unknown`，则自动回退至 `Workspace="inherit"` 并为该样本分配独立沙箱子目录（如 `datasets/sandboxes/<sample_id>/`），杜绝跨样本上下文或文件系统污染。
   - **真实工具调用**：Sub-Agent 直接调用宿主内置真实工具（`read_file`、`write_file`、`edit_file`、`run_command`、`call_mcp_tool` 等）。
   - **生命周期即用即毁**：单条样本落盘（或超过最大重试归档至 `failed.jsonl`）后，Main-Agent 必须立即调用 `manage_subagents(Action="kill", ConversationIds=[subagent_id])` 销毁该实例。

---

## 可配置运行参数

当用户调用本 Skill 时，优先解析用户指定的参数；未指定项采用默认值：

| 参数名 | 可选值 / 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `input_mode` | `local_file` \| `auto_generate` | 若提供文件路径则为 `local_file`，否则 `auto_generate` | 种子来源：从本地 `.jsonl`/`.json`/`.csv`/`.txt` 读取，或由 Main-Agent 按领域自动生成 |
| `input_path` | 文件路径字符串 | `skills/multi-turn-tool-distiller/examples/seed_examples.jsonl` | 本地种子文件路径（`local_file` 模式下使用） |
| `execution_mode` | `serial` \| `parallel` | `serial` | 串行逐条执行，或按 `batch_size` 并行拉起多个独立 Sub-Agent |
| `batch_size` | 整数 (`1`~`5`) | `3` | `parallel` 模式下的单批并发 Sub-Agent 数量 |
| `turns_range` | `min-max`（如 `2-4`） | `2-4` | 每条样本的目标多轮交互轮数（含首轮提问与后续追问） |
| `max_retries` | 整数 | `3` | 单条样本审计未通过时的模板变异最大重试次数 |
| `output_file` | 文件路径字符串 | `datasets/sharegpt_train.jsonl` | 标准 ShareGPT 主输出文件路径（同目录自动维护 `.state.json` 与 `failed.jsonl`） |

---

## 标准执行工作流（Step-by-Step）

### Step 1：环境预检与队列初始化

1. **检查 Git 基础提交（支持 `Workspace="branch"`）**：
   由于 `invoke_subagent(..., Workspace="branch")` 需要当前工作区 Git 仓库至少存在一个 `HEAD` 提交，先执行预检命令：
   ```powershell
   git rev-parse --verify HEAD 2>$null; if ($LASTEXITCODE -ne 0) { git add -A; git commit --allow-empty -m "chore: initial commit for subagent branch isolation" }
   ```
2. **初始化种子队列与断点续传状态**：
   - 若为 **自动生成模式 (`auto_generate`)**：由 Main-Agent 根据用户指定的领域、难度与工具类型，先生成种子任务写入临时文件（如 `datasets/generated_seeds.jsonl`）。
   - 运行任务管理器脚本加载种子并过滤已完成的样本（断点续传）：
     ```powershell
     python skills/multi-turn-tool-distiller/scripts/task_manager.py init --input <input_path> --output-dir datasets
     ```
   - 获取当前待处理批次：
     ```powershell
     python skills/multi-turn-tool-distiller/scripts/task_manager.py next-batch --output-dir datasets --batch-size <batch_size>
     ```

---

### Step 2：注册教师 Sub-Agent (`define_subagent`)

在当前会话首次执行时，调用 `define_subagent` 注册名为 `distill_teacher_agent` 的子代理类型：
- `Name`: `"distill_teacher_agent"`
- `Description`: `"Isolated teacher agent for multi-turn conversation and real tool-calling trajectory synthesis."`
- `EnableWriteTools`: `true`
- `EnableMCPTools`: `true`
- `EnableSubagentTools`: `false`
- `SystemPrompt`: 读取 [prompt_strategies.md](./references/prompt_strategies.md) 中的基础系统提示词，要求 Sub-Agent：
  1. 在每次决定调用工具前，以及在收到工具执行结果生成最终回答前，均在正文中输出 `<think>...</think>` 标签包裹的高层推理概述；
  2. `<think>` 块内简明直接地说明当前任务目标、所选工具或结论依据，随后执行真实工具调用或给出正文回答；
  3. 仅完成当前轮次用户的请求，完成后停止等待下一轮用户追问，切勿自行模拟后续用户提问。

---

### Step 3：拉起隔离 Sub-Agent 与多轮追问交互

#### 3.1 启动首轮对话 (`invoke_subagent`)
对批次中的每条待处理样本 `sample`：
- 根据当前样本的 `retry_count`（`0` -> `Template A`，`1` -> `Template B`，`2` -> `Template C`）从 [prompt_strategies.md](./references/prompt_strategies.md) 选择对应的引导包装模板，将首轮 `human` 问题组装为 `Prompt`。
- 调用 `invoke_subagent`：
  - `TypeName`: `"distill_teacher_agent"`
  - `Role`: `"Teacher Agent - " + sample["sample_id"]`
  - `Model`: `"inherit"`
  - `Workspace`: 优先传 `"branch"`；若当前工作区 VCS 类型返回 `branch workspace not supported for VCS type Unknown`，则自动切换为 `"inherit"`（如任务涉及写入临时文件，限定在 `datasets/sandboxes/<sample_id>/` 内操作）
  - `Prompt`: 组装后的首轮任务提示词
- **串行模式 (`serial`)**：每次仅拉起 1 个 Sub-Agent，等待其返回后再推进下一轮或下一条样本。
- **并行模式 (`parallel`)**：在同一轮工具调用中同时发起 `batch_size` 个 `invoke_subagent` 调用，记录每个 `sample_id <-> conversationId` 映射表。停止工具调用以等待宿主 Reactive Wakeup 异步唤醒。

#### 3.2 推进多轮追问 (`send_message`)
当收到某个 Sub-Agent（`conversationId`）完成当前轮次的回复后：
1. 检查当前已完成的对话轮数 `current_turn` 是否小于该样本的目标轮数 `target_turns`（默认 2~4 轮）：
   - 若种子样本自带预设 `follow_ups` 列表，取下一条预设追问；
   - 若无预设追问，Main-Agent 扮演 **User Simulator**，紧密结合 Sub-Agent 上一轮的回答与真实工具执行结果，动态生成一条递进追问（例如：基于读取到的文件内容要求进一步统计/修改、要求用命令行验证某个边界条件、或针对工具输出提出深入问题），并附带当前策略模板的 `<think>` 格式提醒。
2. 调用 `send_message(Recipient=conversationId, Message=next_user_prompt)` 将下一轮追问发送给同一个 Sub-Agent 实例。
3. 重复上述过程直至达到 `target_turns` 轮。

---

### Step 4：全链路轨迹提取、质量审计与模板变异重试（Mutation Loop）

当单条样本的所有轮次交互完毕后，执行三级提取与审计流程：

1. **运行轨迹提取与审计脚本**：
   调用 [trajectory_extractor.py](./scripts/trajectory_extractor.py)，传入 Sub-Agent 的日志路径 `<appDataDir>/brain/<subagent-id>/.system_generated/logs/transcript_full.jsonl`（若无 `transcript_full.jsonl` 则回退至 `transcript.jsonl`）以及提取输出路径：
   ```powershell
   python skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py extract `
     --transcript "<appDataDir>/brain/<subagent-id>/.system_generated/logs/transcript_full.jsonl" `
     --sample-id "<sample_id>" `
     --expected-turns <target_turns> `
     --output "datasets/tmp_<sample_id>.json"
   ```
2. **检查审计结果 (`audit_passed`)**：
   脚本会自动校验以下核心指标：
   - **全节点 `<think>` 覆盖**：每一轮 `gpt` 回复及每次发起 `function_call` 前是否均包含非空、闭合的 `<think>...</think>` 推理块；
   - **工具调用合法性**：每个 `function_call` 的 `value` 是否为包含 `name` 和 `arguments` 的合法 JSON 字符串，且后接对应的 `observation` JSON；
   - **完整性与非拒答**：回复未发生异常截断、空回复或拒答，且达到预期轮数。
3. **分支处理**：
   - **审计通过 (`audit_passed == true`)**：进入 **Step 5** 持久化落盘。
   - **审计未通过 (`audit_passed == false`)**：
     1. 先调用 `manage_subagents(Action="kill", ConversationIds=[subagent_id])` 销毁当前未达标的 Sub-Agent 实例；
     2. 调用 `task_manager.py update-status` 增加重试计数并记录诊断错误码（如 `MISSING_THINK_BEFORE_TOOL`、`EMPTY_THINK_BLOCK`、`INVALID_TOOL_JSON`）；
     3. 若 `retry_count < max_retries`：查阅 [prompt_strategies.md](./references/prompt_strategies.md) 切换至下一级变异模板（`Template A` -> `Template B` -> `Template C`），并将上一轮缺失的诊断项显式写入新 Prompt 的修正提示槽位，重新拉起全新的 Sub-Agent 实例重试该样本；
     4. 若 `retry_count >= max_retries`：进入 **Step 5** 将该样本归档至 `failed.jsonl`。

---

### Step 5：纯净 ShareGPT 落盘与实例销毁

1. **写入成功样本**：
   调用 [sharegpt_writer.py](./scripts/sharegpt_writer.py) 将通过审计的轨迹原子追加写入主数据集文件（仅保留 `system`、`tools`、`conversations` 三个标准字段），并更新 `.state.json` 为 `completed`：
   ```powershell
   python skills/multi-turn-tool-distiller/scripts/sharegpt_writer.py commit `
     --extracted "datasets/tmp_<sample_id>.json" `
     --output-file "datasets/sharegpt_train.jsonl" `
     --state-dir "datasets" `
     --template-used "<template_name>"
   ```
2. **归档失败样本（超过最大重试时）**：
   ```powershell
   python skills/multi-turn-tool-distiller/scripts/sharegpt_writer.py fail `
     --sample-id "<sample_id>" `
     --reason "<audit_errors>" `
     --extracted "datasets/tmp_<sample_id>.json" `
     --state-dir "datasets"
   ```
3. **销毁 Sub-Agent 实例与临时文件**：
   - 调用 `manage_subagents(Action="kill", ConversationIds=[subagent_id])` 销毁该样本对应的 Sub-Agent 及其 `branch` 工作区；
   - 删除临时提取文件 `datasets/tmp_<sample_id>.json`。
4. **继续下一批次**：
   循环执行 `task_manager.py next-batch` 直至队列中所有样本均处于 `completed` 或 `failed` 状态，最后输出汇总统计报告（总样本数、成功数、重试成功数、失败数及输出文件路径）。
