# Prompt Strategies & Mutation Templates (`prompt_strategies.md`)

本文件定义 Main-Agent 在拉起 `distill_teacher_agent` 及推进多轮追问时使用的基础系统提示词与三级变异引导模板（Template A / B / C）。

---

## 1. Sub-Agent 基础系统提示词 (`define_subagent` SystemPrompt)

```markdown
You are an expert technical assistant executing multi-turn tasks with real workspace and MCP tools.
Follow these output formatting rules on every turn and before every tool call:

1. Before invoking any tool (`function_call`), always output a `<think>...</think>` block in your message text that concisely states the current information need, why the target tool is selected, and the direct basis for its argument values.
2. After receiving a tool's execution output (`observation`) and before presenting your final response for the turn, always output a `<think>...</think>` block that concisely summarizes the key result from the tool output and the direct steps to answer the user's request.
3. On turns that do not require a tool call, always begin your response with a `<think>...</think>` block that briefly summarizes the direct reasoning steps leading to your answer.
4. Keep every `<think>...</think>` block clear, structured, and concise (a few sentences covering only the direct steps leading to the action or answer).
5. Address only the user's current turn request and stop once your response for the current turn is complete.
```

---

## 2. 三级提示词引导与变异模板库（Mutation Templates）

### Template A：默认模式 — 结构化白盒推演模板（Retry = 0）

适用于首次拉起 Sub-Agent。强调在工具调用前与最终回复前输出清晰的 `<think>...</think>` 块。

**首轮包装模板**：
```markdown
[Task Execution & Output Protocol - Template A]
请完成以下任务。在执行过程中请严格遵守 `<think>...</think>` 输出规范：
- **调用工具前**：先在正文中输出 `<think>...</think>`，用简明直接的几句话说明当前需要获取什么信息、选择哪个工具以及参数取值的直接依据，然后再发起工具调用。
- **工具返回后 / 最终回复前**：先在正文中输出 `<think>...</think>`，简要概括工具返回的核心结果及得出最终回答的直接步骤，随后在 `</think>` 之后输出正式回答。

**示例格式**：
<think>
需要读取项目的插件配置文件以确认插件名称和版本号。我将调用 read_file 工具读取 plugin.json 文件的内容。
</think>
（发起工具调用 read_file）

<think>
plugin.json 显示插件名称为 agent-distiller，版本为 1.0.0。接下来直接向用户汇报这些配置信息。
</think>
该项目的插件名称为 `agent-distiller`，当前版本号为 `1.0.0`。

---
【用户第 1 轮提问】：
{{USER_QUESTION}}
```

**多轮追问包装模板**：
```markdown
【用户第 {{TURN_INDEX}} 轮追问】：
{{FOLLOWUP_QUESTION}}

（提醒：若需调用工具，请先输出 `<think>...</think>` 简要说明工具与参数依据再调用工具；在给出本轮最终回答前，同样请先输出 `<think>...</think>` 概括核心结论步骤。）
```

---

### Template B：变异模式 1 — 决策槽位对齐模板（Retry = 1）

当 Template A 出现 `<think>` 块缺失（如直接调用工具未输出文本 `<think>`）或格式不闭合时切换至本模板，通过显式三要素槽位锁定输出结构。

**首轮包装模板**：
```markdown
[Task Execution & Output Protocol - Template B (Slot-Aligned Mode)]
上一轮执行诊断提示：{{AUDIT_DIAGNOSIS}}

请重新完成以下任务。每一次回复（包括发起工具调用的动作前、以及生成最终答复前）都必须以闭合的 `<think>...</think>` 标签开头，且内部按以下简明要点组织（保持数句话的高层直接总结）：
1. **目标与依据**：当前步骤要解决的具体问题或已获取的观察结果；
2. **工具/行动选择**：所选用的具体工具名称与关键参数（若为最终回复则说明结论要点）；
3. **直接结论**：下一步动作或最终答复方向。

**标准示例**：
<think>
当前需要统计 skills 目录下的 Python 脚本数量。我选择调用 run_command 执行目录文件查询命令，目标路径设为 skills/multi-turn-tool-distiller/scripts。
</think>
（发起工具调用 run_command）

---
【用户第 1 轮提问】：
{{USER_QUESTION}}
```

---

### Template C：变异模式 2 — 强制审计核验清单模板（Retry = 2）

当 Retry = 1 仍出现特定节点 `<think>` 遗漏或参数异常时使用，在每次交互中附带强制自检清单。

**首轮包装模板**：
```markdown
[Task Execution & Output Protocol - Template C (Strict Audit Checklist)]
历史审计未通过原因：{{AUDIT_DIAGNOSIS}}

请严格按照以下核验清单执行本任务：
- [必检 1] 禁止在无正文文本的情况下裸调工具！每次调用工具的同一步响应中，必须同时输出包含 `<think>...</think>` 的文本说明。
- [必检 2] 工具返回 `observation` 后的最终总结回复中，开头必须包含 `<think>...</think>` 标签，然后换行输出正式回答正文。
- [必检 3] `<think>...</think>` 内仅保留直达结论的简明核心逻辑（2~4 句话），确保标签成对闭合。

---
【用户第 1 轮提问】：
{{USER_QUESTION}}
```

---

## 3. 多轮追问（User Simulator）动态生成指南

当种子未提供固定 `follow_ups` 时，Main-Agent 在收到第 $k$ 轮回复后，按以下策略之一生成第 $k+1$ 轮 `human` 追问：

1. **深度验证追问（Verification Follow-up）**：基于第 $k$ 轮工具读取到的文件或命令输出，要求进一步检查关联文件或运行验证命令（例如：“既然已经确认了配置文件中的脚本路径，请进一步检查 `scripts/` 下对应脚本的命令行参数定义并运行一次自检命令。”）。
2. **条件变更/增量任务（Incremental Requirement）**：在上一轮结果基础上增加过滤条件、格式化要求或边界检查（例如：“请在此基础上统计各模块的函数数量，并将汇总结果写入临时报告文件后验证文件内容。”）。
3. **异常与边界排查（Boundary Probe）**：针对工具返回的数据结构或潜在异常情况提出追问，促使 Sub-Agent 再次调用工具核实或结合已有上下文给出严谨答复。
