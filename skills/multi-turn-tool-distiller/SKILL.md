---
name: multi-turn-tool-distiller
description: >-
  Automated multi-turn conversation and real tool-calling trajectory distillation
  workflow in Antigravity using a Two-Stage Main-Sub Agent architecture (Stage 1
  natural execution + Adaptive Profiling + Stage 2 multi-strategy rationale
  synthesis waterfall with STaR, MPCoT, PoT, CoT-Evo, Prefill, and PRM audit).
  Use when the user asks to distill, synthesize, or collect multi-turn dialogues,
  tool-calling trajectories, or ShareGPT training datasets using isolated Sub-Agents.
---

# Multi-Turn & Tool-Calling Trajectory Distiller (`multi-turn-tool-distiller`)

本 Skill 在 Antigravity 宿主环境下通过 **“阶段一自然工具调用执行 + 主控智能画像诊断 + 阶段二五级算子回退重构 + PRM 过程级一致性校验”** 的双阶段架构，采集真实、自然的多轮对话与工具调用轨迹，并持久化为 100% 兼容 LLaMA-Factory 的标准 **ShareGPT `.jsonl`** 数据集。

---

## 核心设计原则（杜绝伪蒸馏与人为模板污染）

1. **零用户侧包装污染（Zero User-Turn Boilerplate）**：
   - 在阶段一向执行子代理（`distill_executor_agent`）发送首轮提问及多轮 `send_message` 追问时，**直接发送纯净的用户自然提问**，绝不在 `human` 消息中塞入协议外壳、填空槽位指令或同题 Few-Shot 示例。
2. **严禁脚本硬编码伪造思考（Zero Script-Fabricated `<think>`）**：
   - Python 辅助脚本仅负责解析真实日志、诊断缺失节点与校验过程一致性，**绝不使用固定字符串模板拼接伪造任何 `<think>` 内容**。所有 `<think>...</think>` 推理块必须 100% 由大模型（阶段一自然输出或阶段二重构子代理）针对当前具体任务与真实工具输出生成。
3. **主控智能画像与五级算子回退（Adaptive Profiling & 5-Strategy Waterfall）**：
   - Main-Agent 通过 `trajectory_extractor.py inspect` 智能识别阶段一模型的防护等级与轨迹特征（如 `DIRECT_COMPLETE`、`TOOL_SILENT_NATIVE`、`CLI_SYMBOLIC_EXECUTION`、`MULTI_HOP_TOOL_CHAIN`、`STRICT_GUARDED_MINIMAL`），自动选择最优的阶段二提取算子（`strategy_star` / `strategy_mpcot` / `strategy_pot` / `strategy_cotevo` / `strategy_prefill`）；若当前算子提取失败或未通过 PRM 校验，自动回退切换下一种算子重试。
4. **单样本实例隔离与即用即毁**：
   - 每条样本均使用独立的 Sub-Agent 实例，落盘后立即通过 `manage_subagents(Action="kill")` 销毁。

---

## 可配置运行参数

| 参数名 | 可选值 / 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `input_mode` | `local_file` \| `auto_generate` | 若提供文件路径则为 `local_file`，否则 `auto_generate` | 种子来源：读取本地 `.jsonl`/`.json`/`.csv`/`.txt`，或由 Main-Agent 自动生成 |
| `input_path` | 文件路径字符串 | `skills/multi-turn-tool-distiller/examples/seed_examples.jsonl` | 本地种子文件路径 |
| `execution_mode` | `serial` \| `parallel` | `serial` | 串行逐条执行，或按 `batch_size` 并行拉起多个独立 Sub-Agent |
| `batch_size` | 整数 (`1`~`5`) | `3` | `parallel` 模式下的并发 Sub-Agent 数量 |
| `turns_range` | `min-max`（如 `2-4`） | `2-4` | 每条样本的多轮对话总轮数 |
| `max_retries` | 整数 | `5` | 阶段二算子回退尝试的最大次数（对应 5 种算子瀑布流） |
| `output_file` | 文件路径字符串 | `datasets/sharegpt_train.jsonl` | 纯净 ShareGPT 输出路径（同目录维护 `.state.json` 与 `failed.jsonl`） |

---

## 双阶段标准执行工作流（Step-by-Step）

### Step 1：队列初始化与断点恢复

1. 运行 `task_manager.py init` 加载种子文件并与 `datasets/.state.json` 对齐（自动跳过已 `completed` 样本）：
   ```powershell
   python skills/multi-turn-tool-distiller/scripts/task_manager.py init --input <input_path> --output-dir datasets --max-retries 5
   ```
2. 获取下一批待处理样本：
   ```powershell
   python skills/multi-turn-tool-distiller/scripts/task_manager.py next-batch --output-dir datasets --batch-size <batch_size>
   ```

---

### Step 2：注册双阶段专属 Sub-Agent (`define_subagent`)

首次运行时查阅 [prompt_strategies.md](./references/prompt_strategies.md) 注册两个专用子代理类型：

1. **阶段一自然执行子代理 (`distill_executor_agent`)**：
   - `Name`: `"distill_executor_agent"`
   - `EnableWriteTools`: `true`
   - `EnableMCPTools`: `true`
   - `EnableSubagentTools`: `false`
   - `SystemPrompt`: 采用 [prompt_strategies.md](./references/prompt_strategies.md) 第 1 节中的轻量级自然引导系统提示词（鼓励在调用工具或给出答复时自然附上 `<think>...</think>` 推理概述，不强制死板槽位）。
2. **阶段二逆向推演与重构子代理 (`distill_rationalizer_agent`)**：
   - `Name`: `"distill_rationalizer_agent"`
   - `EnableWriteTools`: `false`
   - `EnableMCPTools`: `false`
   - `EnableSubagentTools`: `false`
   - `SystemPrompt`: 采用 [prompt_strategies.md](./references/prompt_strategies.md) 第 3.1 节的重构器系统提示词（专用于执行 STaR / MPCoT / PoT / CoT-Evo / Prefill 五种二阶段算子）。

---

### Step 3：阶段一（Stage 1）— 纯净自然多轮对话与真实工具执行

1. **启动首轮自然提问 (`invoke_subagent`)**：
   - 对每条样本 `sample`，直接将其纯净的 `sample["question"]` 作为 `Prompt`（**不添加任何协议包装头尾**），调用 `invoke_subagent`：
     - `TypeName`: `"distill_executor_agent"`
     - `Role`: `"Stage1 Executor - " + sample["sample_id"]`
     - `Model`: `"inherit"`
     - `Workspace`: 优先传 `"branch"`；若当前工作区返回 `branch workspace not supported for VCS type Unknown`，自动切换为 `"inherit"`。
2. **推进自然多轮追问 (`send_message`)**：
   - 收到 Sub-Agent 当前轮回复后，若未达到 `target_turns`（默认 2~4 轮），Main-Agent 取预设 `follow_ups` 或根据上一轮真实工具输出与回答内容，站在真实用户视角生成下一轮递进追问（**同样不附加任何 `<think>` 格式提醒尾巴**），通过 `send_message` 发送给同一 `distill_executor_agent` 实例。
   - 全部轮次完成后，调用 `manage_subagents(Action="kill", ConversationIds=[stage1_subagent_id])` 销毁阶段一实例（其完整会话日志已保留在 `brain/<stage1_subagent_id>/.system_generated/logs/transcript_full.jsonl` 中）。

---

### Step 4：智能画像诊断与阶段二（Stage 2）五级算子回退重构

1. **运行 `inspect` 进行轨迹画像与缺失诊断**：
   ```powershell
   python skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py inspect `
     --transcript "<appDataDir>/brain/<stage1_subagent_id>/.system_generated/logs/transcript_full.jsonl" `
     --sample-id "<sample_id>" `
     --expected-turns <target_turns> `
     --output "datasets/tmp_<sample_id>.json"
   ```
   脚本会输出：
   - `profile_type`：模型行为画像（如 `DIRECT_COMPLETE`、`TOOL_SILENT_NATIVE`、`CLI_SYMBOLIC_EXECUTION`、`MULTI_HOP_TOOL_CHAIN`、`STRICT_GUARDED_MINIMAL`）；
   - `needs_stage2`：布尔值，指示是否存在缺失 `<think>` 的步骤或未通过 PRM 过程一致性校验的节点；
   - `target_node_indices`：需要阶段二模型生成/重构 `<think>` 的具体节点索引列表；
   - `recommended_strategies`：按智能画像排序的阶段二回退算子队列（如 `["strategy_pot", "strategy_star", "strategy_mpcot", "strategy_cotevo", "strategy_prefill"]`）。

2. **分支 A：阶段一已自然完整 (`needs_stage2 == false` 且 `audit_passed == true`)**：
   - 直接进入 **Step 5** 落盘。

3. **分支 B：启动阶段二算子回退循环 (`needs_stage2 == true`)**：
   按 `recommended_strategies` 顺序依次取出未尝试的算子 `strategy_name`：
   - **步骤 4.1**：查阅 [prompt_strategies.md](./references/prompt_strategies.md) 第 3.2 节中对应 `strategy_name` 的算子模板，将 `datasets/tmp_<sample_id>.json` 中的轨迹上下文与 `target_node_indices` 组装为 Prompt，调用 `invoke_subagent(TypeName="distill_rationalizer_agent", ...)`。
   - **步骤 4.2**：收到 `distill_rationalizer_agent` 返回的 JSON 映射后，立即调用 `manage_subagents(Action="kill")` 销毁该重构子代理，并将返回的 JSON 保存至 `datasets/stage2_<sample_id>.json`。
   - **步骤 4.3**：运行 `merge-stage2` 合成完整轨迹并执行 **PRM 过程一致性校验**：
     ```powershell
     python skills/multi-turn-tool-distiller/scripts/trajectory_extractor.py merge-stage2 `
       --inspected "datasets/tmp_<sample_id>.json" `
       --stage2-json "datasets/stage2_<sample_id>.json" `
       --strategy "<strategy_name>" `
       --output "datasets/tmp_<sample_id>.json"
     ```
   - **步骤 4.4**：检查输出的 `audit_passed`：
     - 若 `audit_passed == true`：跳出回退循环，进入 **Step 5** 落盘！
     - 若 `audit_passed == false`：调用 `task_manager.py update-status --status retry --strategy-tried <strategy_name>` 记录本次尝试，自动切换 `recommended_strategies` 中的**下一个算子**重复步骤 4.1~4.3，直至通过审计或耗尽全部算子。

---

### Step 5：纯净 ShareGPT 落盘与状态同步

1. **提交通过 PRM 校验的样本**：
   ```powershell
   python skills/multi-turn-tool-distiller/scripts/sharegpt_writer.py commit `
     --extracted "datasets/tmp_<sample_id>.json" `
     --output-file "datasets/sharegpt_train.jsonl" `
     --state-dir "datasets" `
     --template-used "<final_strategy_or_stage1_direct>"
   ```
2. **归档耗尽所有回退算子仍失败的样本**：
   ```powershell
   python skills/multi-turn-tool-distiller/scripts/sharegpt_writer.py fail `
     --sample-id "<sample_id>" `
     --reason "<audit_errors>" `
     --extracted "datasets/tmp_<sample_id>.json" `
     --state-dir "datasets"
   ```
3. **清理临时文件**：删除 `datasets/tmp_<sample_id>.json` 与 `datasets/stage2_<sample_id>.json`。
