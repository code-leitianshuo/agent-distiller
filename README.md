# Agent Distiller (`agent-distiller`) — v2 Adaptive Two-Stage Architecture

运行在 Antigravity 宿主环境下的 **自动化大模型多轮与真实工具调用轨迹蒸馏插件**。采用 **“阶段一自然工具调用执行 + 主控智能画像诊断 + 阶段二五级算子回退重构（STaR / MPCoT / PoT / CoT-Evo / Prefill）+ PRM 过程级一致性校验”** 的双阶段协同架构，输出 100% 兼容 LLaMA-Factory 的标准 ShareGPT `.jsonl` 数据集。

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
        ├── SKILL.md                                # 双阶段蒸馏核心工作流指令
        ├── references/
        │   └── prompt_strategies.md                # 一阶段轻量引导、智能画像矩阵与二阶段五级算子库
        ├── examples/
        │   └── seed_examples.jsonl                 # 本地种子样本示例
        └── scripts/
            ├── task_manager.py                     # 种子解析、智能画像与二阶段回退状态管理
            ├── trajectory_extractor.py             # 阶段一特征诊断(inspect)、阶段二合并(merge-stage2)与 PRM 审计
            └── sharegpt_writer.py                  # 纯净 ShareGPT JSONL 写入与失败归档
```

---

## 核心特性

1. **阶段一：零包装污染自然执行 (`distill_executor_agent`)**
   - 直接向执行子代理发送纯净的用户问题与多轮追问（无协议外壳、无死板填空槽位、无同题示例污染），真实调用宿主文件读写、命令行与 MCP 工具完成任务。
2. **主控智能画像诊断 (`Adaptive Profiler`)**
   - `trajectory_extractor.py inspect` 自动分析阶段一轨迹，识别模型输出与防护特征（`DIRECT_COMPLETE`、`TOOL_SILENT_NATIVE`、`CLI_SYMBOLIC_EXECUTION`、`MULTI_HOP_TOOL_CHAIN`、`STRICT_GUARDED_MINIMAL`），定位缺失 `<think>` 节点并生成按适配度排序的阶段二回退算子队列。
3. **阶段二：五级算子回退重构 (`distill_rationalizer_agent`) + PRM 过程校验**
   - 严禁 Python 脚本硬编码拼接伪造思考文本；当阶段一存在工具调用步未输出内联 `<think>` 时，由 Main-Agent 按画像推荐顺序调度 `distill_rationalizer_agent` 依次执行：
     1. `strategy_star`（STaR 已知答案与真实 Observation 逆向归因）
     2. `strategy_mpcot`（MPCoT 多视角正交分解合成）
     3. `strategy_pot`（PoT 程序与执行证据锚定推演）
     4. `strategy_cotevo`（CoT-Evo 多候选演化择优融合）
     5. `strategy_prefill`（Prefill 决策上下文前缀续写补全）
   - 经 `merge-stage2` 合成后自动执行 **PRM 过程一致性校验**（核验推理内容与所调工具、参数及上下文实体的真实锚定关系）；若当前算子未通过校验，自动回退切换下一算子。
4. **纯净 ShareGPT 输出与独立断点续传**
   - 主输出文件（如 `datasets/sharegpt_train.jsonl`）仅保留 `system`、`tools`、`conversations` 三个标准字段；画像分类、算子回退历史与断点状态独立保存在 `datasets/.state.json` 及 `datasets/failed.jsonl` 中。
