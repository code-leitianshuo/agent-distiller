# Agent Distiller (`agent-distiller`)

运行在 Antigravity 宿主环境下的 **自动化大模型多轮与真实工具调用轨迹蒸馏插件**。基于宿主原生 `Main-Sub Agent` 协同架构，支持单样本独立 `branch` 分支工作区隔离、多轮递进追问模拟、全链路 `<think>` 与工具调用质量审计、动态模板变异重试，以及 100% 兼容 LLaMA-Factory 的标准 ShareGPT `.jsonl` 实时持久化与断点续传。

---

## 目录结构

```text
agent-distiller/
├── plugin.json                                     # 插件清单配置
├── README.md                                       # 用户使用文档
├── assets/
│   └── logo.svg                                    # 插件图标
└── skills/
    └── multi-turn-tool-distiller/
        ├── SKILL.md                                # 核心蒸馏工作流指令
        ├── references/
        │   └── prompt_strategies.md                # 提示词引导模板库与变异策略
        ├── examples/
        │   └── seed_examples.jsonl                 # 本地种子样本示例
        └── scripts/
            ├── task_manager.py                     # 种子解析、队列与断点续传状态管理
            ├── trajectory_extractor.py             # 会话轨迹提取与全链路质量审计
            └── sharegpt_writer.py                  # 纯净 ShareGPT JSONL 写入与失败归档
```

---

## 核心特性

1. **单样本单实例分支隔离 (`Workspace="branch"`)**
   每条多轮对话样本均通过 `invoke_subagent` 拉起独立的 Sub-Agent 实例并克隆独立 Git 分支工作区，真实执行文件读写、命令行与 MCP 工具调用，采集完毕后立即通过 `manage_subagents(Action="kill")` 销毁并清理临时工作区。
2. **双输入源与串/并行灵活调度**
   - **本地文件模式**：支持流式读取 `.jsonl` / `.json` / `.csv` / `.txt` 种子任务。
   - **自动生成模式**：由 Main-Agent 按指定领域、难度、目标工具集自动合成种子任务与多轮追问脚本。
   - **串行 / 并行模式**：支持单条精细监控或按 `batch_size`（默认 3~5）并行拉起多个 Sub-Agent 异步采集。
3. **三级 `<think>` 提取与模板变异重试（Mutation Loop）**
   自动检查每一轮回复与每次 `function_call` 前的 `<think>...</think>` 推理块及参数合法性；未达标时自动在内置引导模板库（Template A/B/C）间切换并重试（默认最多 3 次）。
4. **纯净 ShareGPT 输出与独立断点续传**
   主输出文件（如 `datasets/sharegpt_train.jsonl`）仅保留 `system`、`tools`、`conversations` 三个标准字段，可直接用于 LLaMA-Factory 训练；样本 ID、重试计数与断点状态独立保存在 `datasets/.state.json` 及 `datasets/failed.jsonl` 中。

---

## 快速使用

在 Antigravity 对话中直接调用：

- **本地种子串行蒸馏**：
  > 使用 `multi-turn-tool-distiller` 读取 `skills/multi-turn-tool-distiller/examples/seed_examples.jsonl`，串行蒸馏 2 轮工具调用轨迹并输出到 `datasets/sharegpt_train.jsonl`。
- **自动生成并行蒸馏**：
  > 使用 `multi-turn-tool-distiller` 自动生成 6 条关于 Python 项目诊断与命令行分析的 3 轮工具调用任务，以 `batch_size=3` 并行蒸馏到 `datasets/sharegpt_train.jsonl`。
