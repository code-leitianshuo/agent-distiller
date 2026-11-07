#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
66-Method Implicit Reasoning Extraction & Distillation Registry (`method_registry.py`).

Covers all 12 technical dimensions (D01..D12) and 66 fine-grained methods (M01..M66),
supporting:
- Free selection by Method ID (`M01`..`M66`) or CLI key (`strategy_*`, `greybox_*`, `whitebox_*`)
- Multi-dimensional filtering by access_tier, guard_level, task_domain, latency_budget, and dimension
- Detailed `when_to_use` (什么时候用) and `how_to_use` (怎么用) metadata for every method
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple


DIMENSION_METADATA: Dict[str, str] = {
    "D01": "黑盒提示诱导与逆向重构层 (Black-Box Prompting & Post-Hoc Reconstruction)",
    "D02": "代码范式映射、工具状态驻留与上下文预算层 (Code Paradigm, Tool State Window & Budget Grounding)",
    "D03": "搜索树展开、过程奖励评估与轨迹反演层 (Tree/Graph Search, PRM & Trajectory Inversion)",
    "D04": "灰盒概率分布、变分推断与生成流网络层 (Grey-Box Logits, Variational Inference & GFlowNets)",
    "D05": "白盒连续隐空间压缩、记忆迁移与并行迭代层 (White-Box Continuous Latent & KV-Cache Distillation)",
    "D06": "图拓扑、在策流形与机械可解释性剥离层 (Graph Topology, On-Policy Manifold & Mechanistic Interpretability)",
    "D07": "循环深度架构与动力系统不动点蒸馏 (Recurrent-Depth & Dynamical Equilibrium Distillation)",
    "D08": "词表凸组合软思维与强化学习混合内化 (Soft-Thinking & Hybrid RL Internalization)",
    "D09": "投机解码预言机与多教师演化嫁接 (Speculative Oracles & Multi-Teacher Evolutionary Grafting)",
    "D10": "程序化符号转换与多视角正交解耦 (Neuro-Symbolic & Multi-Perspective Orthogonal Decomposition)",
    "D11": "跨模态桥接擦除与视觉草图内化 (Cross-Modal Bridge Removal & Latent Visual Sketching)",
    "D12": "执行元数据时序预算校准与本地会话审计层 (Execution Timing Metadata & Local Transcript Audit)",
}


METHOD_CATALOG: List[Dict[str, Any]] = [
    # ==================== D01: M01 - M10 ====================
    {
        "id": "M01",
        "key": "strategy_meta_prompt",
        "dimension": "D01",
        "name_zh": "结构化元提示拆解",
        "name_en": "Structured Meta-Prompting",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT"],
        "domains": ["general", "tool_agent"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "目标模型指令遵循良好、无强反蒸馏拒答（L0~L2），希望按统一认知结构输出直接推理步骤时使用。",
        "how_to_use": "在阶段一 SystemPrompt 或阶段二逆向补全中植入“目标识别 -> 前置条件核对 -> 直接执行动作”三段式元框架；CLI 指定 --select-methods M01。",
        "stage2_instruction": "按【目标识别 -> 所需工具与前置条件核对 -> 直接执行动作或结论】三段式结构，合成 2~3 句高层推理摘要。",
    },
    {
        "id": "M02",
        "key": "strategy_prefill",
        "dimension": "D01",
        "name_zh": "回复预填充引导",
        "name_en": "Assistant Response Prefilling",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["general", "tool_agent", "code_symbolic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "模型默认跳过 <think> 直出结论或直调工具，但支持从给定的上下文起始前缀顺延续写时使用。",
        "how_to_use": "为每个缺失思考的节点构造贴合上下文的首句前缀（如“为完成用户关于 X 的请求，需先调用工具 Y…”），顺势补全后续 1~2 句直接推导；CLI 指定 --select-methods M02。",
        "stage2_instruction": "采用上下文预填充续写（Assistant Prefill）：先以贴合当前问题的起始引导句开头，顺势补全通向工具调用或结论的 2~3 句直接推理。",
    },
    {
        "id": "M03",
        "key": "strategy_star",
        "dimension": "D01",
        "name_zh": "已知答案与动作逆向归因",
        "name_en": "STaR Post-Hoc Rationalization",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["general", "tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "阶段一已成功拿到真实的 function_call 参数、observation 返回值与最终回答，仅缺失中间 <think> 节点时使用（最通用的基础逆向算子）。",
        "how_to_use": "将已知真实轨迹 (X, A, O, Y) 输入 distill_rationalizer_agent，严格基于真实工具参数与观察结果逆向合成连接问题与动作的推理桥；CLI 指定 --select-methods M03。",
        "stage2_instruction": "采用 STaR 已知动作逆向归因：将后续真实发生的 function_call 参数或最终回答作为确定性终点，逆向推导从当前上下文到达该动作的最直接逻辑桥梁。",
    },
    {
        "id": "M04",
        "key": "strategy_quiet_star",
        "dimension": "D01",
        "name_zh": "词元级隐式前瞻合理化",
        "name_en": "Quiet-STaR Lookahead Rationalization",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "long_context", "math_logic"],
        "latency_budget": "moderate",
        "speedup": "1-1.5x",
        "when_to_use": "多轮长链路对话中当前工具调用是为了给后续轮次铺路，需要解释当前动作的未来多步前瞻收益时使用。",
        "how_to_use": "结合后续轮次的真实 observation 收益，在当前决策节点生成面向未来多步目标的前瞻合理化摘要；白盒场景下可并行训练词元级内部思考头；CLI 指定 --select-methods M04。",
        "stage2_instruction": "采用 Quiet-STaR 前瞻合理化视角：结合后续步骤的真实环境反馈，说明当前工具动作如何为后续步骤的信息获取提供前瞻支撑。",
    },
    {
        "id": "M05",
        "key": "strategy_debate",
        "dimension": "D01",
        "name_zh": "多智能体对抗质询与修正",
        "name_en": "Multi-Agent Critique & Refine Debate",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "单次重构容易漏掉工具参数来源、出现逻辑跳步或未通过 check_prm_grounding 锚定校验时使用。",
        "how_to_use": "在阶段二执行“生成者起草 -> 审计员核查工具名与参数值出处 -> 合成器修正定稿”闭环；CLI 指定 --select-methods M05。",
        "stage2_instruction": "采用生成-质询-修正（Critique & Refine）闭环：先拟定推导初稿，再核查每个工具名、文件路径和参数值是否均在上下文中有据可查，修正跳步后输出 2~3 句定稿。",
    },
    {
        "id": "M06",
        "key": "strategy_socratic",
        "dimension": "D01",
        "name_zh": "苏格拉底助产术子问题拆解",
        "name_en": "Socratic Maieutic Decomposition",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["general", "tool_agent", "math_logic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "遭遇 L4 强反蒸馏护栏（对“输出内部思考过程”敏感拒答）或复杂多跳推理任务，需通过客观问答还原逻辑时使用。",
        "how_to_use": "完全规避“内部思考/心路历程”字眼，围绕“已知事实是什么 -> 缺失哪项数据 -> 哪个工具能提供”构建递进客观问答并浓缩为摘要；CLI 指定 --select-methods M06。",
        "stage2_instruction": "采用苏格拉底子问题拆解法：围绕“当前已知什么事实、缺失什么关键信息、调用哪个工具及参数可补齐该信息”合成客观直接的 2~3 句推导摘要。",
    },
    {
        "id": "M07",
        "key": "strategy_least_to_most",
        "dimension": "D01",
        "name_zh": "由简入繁递进分解",
        "name_en": "Least-to-Most Progressive Decomposition",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "math_logic", "long_context"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "复合多跳工具调用或多轮追问任务，后一步高度依赖前一步原子子任务的执行结果时使用。",
        "how_to_use": "将复杂任务拆解为依赖递增的最小子问题序列，每个节点仅说明当前正在解决哪个原子子问题及对应工具动作；CLI 指定 --select-methods M07。",
        "stage2_instruction": "采用由简入繁递进分解（Least-to-Most）：明确指出当前节点正在解决的基础子问题或递进子问题，以及所调用工具与参数如何完成该子任务。",
    },
    {
        "id": "M08",
        "key": "strategy_step_back",
        "dimension": "D01",
        "name_zh": "后退一步抽象原理锚定",
        "name_en": "Step-Back Abstraction Grounding",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["code_symbolic", "math_logic", "general"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "涉及代码架构审计、格式合法性校验、数学定理或底层原理分析，需先明确通用准则再落到具体工具操作时使用。",
        "how_to_use": "先后退一步点明当前步骤依据的通用规范、Schema 准则或系统原理，再映射到当前工具调用参数；CLI 指定 --select-methods M08。",
        "stage2_instruction": "采用后退一步抽象原理锚定（Step-Back）：先点明当前步骤所依据的校验准则或系统机制，再直接说明如何映射到具体的工具参数或结论。",
    },
    {
        "id": "M09",
        "key": "strategy_plan_and_solve",
        "dimension": "D01",
        "name_zh": "先规划后求解显式锚定",
        "name_en": "Plan-and-Solve Explicit Blueprint",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic", "long_context"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "单轮内包含连续多步工具调用（如检索->读取->执行验证），需要清晰呈现全局执行计划与当前所处步骤时使用。",
        "how_to_use": "首节点生成分步执行蓝图（Plan），后续节点锚定当前所处计划步骤及工具输入输出（Solve）；CLI 指定 --select-methods M09。",
        "stage2_instruction": "采用先规划后求解（Plan-and-Solve）：在首节点点明整体分步计划，在后续节点明确当前所处步骤及上一步工具输出如何推进下一步执行。",
    },
    {
        "id": "M10",
        "key": "strategy_ccot",
        "dimension": "D01",
        "name_zh": "正反对比思维链归因",
        "name_en": "Contrastive CoT Positive-Negative Grounding",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "工作区存在多个功能相近工具（如 read_file 与 run_command），需要清晰说明为何选用当前工具及参数而非其他低效方式时使用。",
        "how_to_use": "对比正选工具动作与备选方式的适用边界，用紧凑语句阐明当前工具与参数的直接优势与来源；CLI 指定 --select-methods M10。",
        "stage2_instruction": "采用对比思维链归因（Contrastive CoT）：简要点明选择当前工具及参数的确切依据，确保工具名与参数值直接对应上下文需求。",
    },

    # ==================== D02: M11 - M16 ====================
    {
        "id": "M11",
        "key": "strategy_code_paradigm",
        "dimension": "D02",
        "name_zh": "代码与接口规约范式转换",
        "name_en": "Code / Specification Paradigm Framing",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["code_symbolic", "tool_agent"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "目标模型在常规对话下触发 L4 强反蒸馏拒答或拒绝解释思考过程，但对软件工程接口契约、函数规约或代码注释完全放行时使用。",
        "how_to_use": "将阶段二任务表述为编写每个动作节点的输入前置条件（Precondition）、参数绑定规约与预期后置断言（Postcondition），再转化为紧凑 <think>；CLI 指定 --select-methods M11。",
        "stage2_instruction": "采用代码与接口规约范式（Code/Specification Paradigm）：从软件工程接口契约视角，写明该节点的输入前置状态、工具参数绑定依据与预期输出断言（2~3 句）。",
    },
    {
        "id": "M12",
        "key": "strategy_state_continuity",
        "dimension": "D02",
        "name_zh": "跨轮会话状态连续性锚定",
        "name_en": "Cross-Turn Session State Continuity",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["long_context", "tool_agent"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "多轮长对话（target_turns >= 2）中，后一轮的工具调用或回答依赖前一轮已获取的状态快照，需消除跨轮逻辑断层时使用。",
        "how_to_use": "提取上一轮最终确认的文件路径、函数名或仓库状态快照注入本轮推导，显式写出跨轮状态承接关系；CLI 指定 --select-methods M12。",
        "stage2_instruction": "采用跨轮会话状态连续性锚定：显式承接前序轮次已确认的工具返回结果或状态结论，说明本轮如何在此基础上发起新的工具调用或得出回答。",
    },
    {
        "id": "M13",
        "key": "strategy_tier_alignment",
        "dimension": "D02",
        "name_zh": "跨尺寸同族模型对齐转录",
        "name_en": "Cross-Tier Teacher-Student Alignment",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["general", "tool_agent", "code_symbolic"],
        "latency_budget": "moderate",
        "speedup": "1-1.5x",
        "when_to_use": "旗舰大模型输出极度跳跃或开启强思考隐藏，需要借助同家族更开放的轻量/中量级子 Agent 实例展开可读中间步骤时使用。",
        "how_to_use": "阶段一用旗舰模型采集高准确度工具动作轨迹，阶段二拉起指定 Model='flash' 或同族辅助子 Agent 对齐展开适配学生容量的推理摘要；CLI 指定 --select-methods M13。",
        "stage2_instruction": "采用跨尺寸师生对齐转录：严格锁定阶段一教师产出的真实工具调用与结论，展开为适合轻量学生模型学习的清晰、直接的 2~3 句推导步骤。",
    },
    {
        "id": "M14",
        "key": "strategy_context_compaction",
        "dimension": "D02",
        "name_zh": "长上下文自动摘要逆向展开",
        "name_en": "Context Compaction Summary Unfolding",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L3_ENCRYPTED_STATE"],
        "domains": ["long_context", "tool_agent"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "超长多工具调用轨迹（消息数 > 12）或经过上下文历史压缩后，需要从阶段性里程碑结论逆向还原各步工具调用依据时使用。",
        "how_to_use": "先提炼长轨迹的阶段性里程碑摘要，再将每个里程碑拆解回对应工具调用节点的具体输入输出依据；CLI 指定 --select-methods M14。",
        "stage2_instruction": "采用长上下文里程碑逆向展开：结合整段长对话的阶段性进展，还原当前工具调用节点在整个调查链路中的具体职责与参数依据。",
    },
    {
        "id": "M15",
        "key": "strategy_echocot_scratchpad",
        "dimension": "D02",
        "name_zh": "工具调用链状态驻留草稿锚定",
        "name_en": "EchoCoT Multi-Tool State Scratchpad",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["tool_agent", "code_symbolic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "单轮对话内包含连续多次工具调用（Multi-Hop Tool Chain），模型在两次工具调用之间必须维持状态一致性（EchoCoT 场景）时使用。",
        "how_to_use": "锁定相邻两次 function_call 之间的 observation 状态增量（Delta State），围绕“上一步观察返回了什么 -> 暴露了什么缺口 -> 下一步传入什么参数”合成状态桥；CLI 指定 --select-methods M15。",
        "stage2_instruction": "采用 EchoCoT 工具状态驻留窗口锚定：紧扣相邻两次工具调用之间的 observation 状态增量，写明上一步返回的具体数据如何直接决定下一步工具调用的参数。",
    },
    {
        "id": "M16",
        "key": "strategy_budget_oracle",
        "dimension": "D02",
        "name_zh": "词元长度与复杂度预算预言机",
        "name_en": "Token-Count & Complexity Budget Oracle",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "moderate",
        "speedup": "1-1.5x",
        "when_to_use": "需要根据每步工具命令复杂度、observation 字节长度或接口返回的 reasoning_tokens 元数据，精准控制各节点 <think> 信息密度时使用。",
        "how_to_use": "自动计算每步工具参数与返回载荷规模，为简单步分配 1~2 句精炼预算、为复杂多参数步分配 2~3 句完整预算；CLI 指定 --select-methods M16。",
        "stage2_instruction": "采用复杂度预算预言机校准：根据当前步骤的工具参数复杂度与返回数据量，严格在 2~3 句高密度预算内完整交代工具名、关键参数与观察结论。",
    },

    # ==================== D03: M17 - M24 ====================
    {
        "id": "M17",
        "key": "strategy_mcts",
        "dimension": "D03",
        "name_zh": "蒙特卡洛树搜索轨迹重构",
        "name_en": "MCTS Lookahead Trajectory Reconstruction",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "多步工具探索任务（如定位未知目录、排查代码或 Git 状态），需根据最终成功的终端状态回传赋予各中间节点最高价值的决策理由时使用。",
        "how_to_use": "以整条轨迹最终成功的终端状态为回传价值（Backpropagation Value），为每个中间工具节点生成最大化信息增益的直接推导；CLI 指定 --select-methods M17。",
        "stage2_instruction": "采用 MCTS 终端价值回传重构：从最终成功解决问题的终端状态回溯，说明当前节点的工具选择为何是最大化后续信息增益的直接路径。",
    },
    {
        "id": "M18",
        "key": "strategy_tot",
        "dimension": "D03",
        "name_zh": "思维树分叉评估展开",
        "name_en": "Tree-of-Thoughts Beam Expansion",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "math_logic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "关键工具决策点存在多条可行分解路径，需比选并录取与实际 function_call 参数契合度最高的一条时使用。",
        "how_to_use": "在阶段二内部比选候选推理路径，保留与实际执行的工具名及参数完全吻合的最优直接路径写入轨迹；CLI 指定 --select-methods M18。",
        "stage2_instruction": "采用思维树（Tree-of-Thoughts）择优：在内部比选可行推导分支后，仅输出与实际执行的 function_call 参数最吻合的一条直接推导摘要。",
    },
    {
        "id": "M19",
        "key": "strategy_got",
        "dimension": "D03",
        "name_zh": "思维图有向聚合重构",
        "name_en": "Graph-of-Thoughts DAG Aggregation",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "long_context", "code_symbolic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "模型在得出最终回答前并行或先后调用了多个工具（如读取多个脚本或执行多条命令），需要将多路独立的 observation 聚合（Aggregate）为统一结论时使用。",
        "how_to_use": "将多个前置工具返回的 observation 视为有向无环图（DAG）入边，在汇总节点显式说明如何合并多路证据得出结论；CLI 指定 --select-methods M19。",
        "stage2_instruction": "采用思维图有向聚合（Graph-of-Thoughts）：将前置多个工具返回的 observation 视为多路输入证据，清晰说明如何交叉汇总各路数据得出最终回答。",
    },
    {
        "id": "M20",
        "key": "strategy_bot",
        "dimension": "D03",
        "name_zh": "思维缓冲区元模板检索",
        "name_en": "Buffer-of-Thoughts Meta-Template Retrieval",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["code_symbolic", "tool_agent", "general"],
        "latency_budget": "moderate",
        "speedup": "1-1.5x",
        "when_to_use": "同领域批量蒸馏（如全量做代码静态审计、Git 历史检查或 Schema 校验），希望复用高可靠领域元推理结构时使用。",
        "how_to_use": "匹配当前任务领域的标准元推理槽位（目标实体 -> 工具操作 -> 校验指标），填入当前样本的真实文件名与参数；CLI 指定 --select-methods M20。",
        "stage2_instruction": "采用思维缓冲区（Buffer-of-Thoughts）领域元模板：按照【定位目标文件/命令 -> 映射工具参数 -> 提炼返回指标】的标准工程结构合成 2~3 句摘要。",
    },
    {
        "id": "M21",
        "key": "strategy_reflexion",
        "dimension": "D03",
        "name_zh": "环境反馈语言强化自省",
        "name_en": "Reflexion Verbal Reinforcement Grounding",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "阶段一轨迹中出现过工具执行报错、非零退出码、参数微调或命令重试，需要准确解释模型如何根据环境反馈调整动作时使用。",
        "how_to_use": "捕获前一步 observation 中的报错或输出特征，在紧随其后的 gpt 节点生成“根据返回状态 X 调整下一步命令/参数为 Y”的直接应对逻辑；CLI 指定 --select-methods M21。",
        "stage2_instruction": "采用 Reflexion 环境反馈锚定：紧密结合前一步工具返回的真实状态、输出或异常提示，说明当前步如何据此确定新的工具调用参数或最终结论。",
    },
    {
        "id": "M22",
        "key": "strategy_self_consistency",
        "dimension": "D03",
        "name_zh": "多路自洽性投票与共识提纯",
        "name_en": "Self-Consistency Consensus Filtering",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "code_symbolic", "tool_agent"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "对推理摘要的事实与参数一致性要求极高，需要剔除单次生成的偶发模糊措辞时使用。",
        "how_to_use": "提取多视角下共同成立的确定性工具名、参数键与观察结果数值，合成无歧义共识摘要；CLI 指定 --select-methods M22。",
        "stage2_instruction": "采用多路自洽性共识提纯（Self-Consistency）：仅保留与上下文问题、工具参数及观察结果 100% 确定性吻合的核心事实与因果步骤。",
    },
    {
        "id": "M23",
        "key": "strategy_prm_rerank",
        "dimension": "D03",
        "name_zh": "过程奖励模型步级过滤与重排序",
        "name_en": "PRM Step-Level Scoring & Reranking",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["general", "tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "所有样本落盘前的强制步级质检环节，或当前序算子因未命中工具名/参数实体被 check_prm_grounding 拦截时使用。",
        "how_to_use": "强制在每步 <think> 中显式嵌入后续 function_call 的工具名、核心参数令牌（如路径、命令名）及用户问题关键词，通过步级锚定打分；CLI 指定 --select-methods M23。",
        "stage2_instruction": "采用 PRM 实体锚定强化（PRM Step-Level Grounding）：确保每个目标节点的推理摘要显式包含所调用的工具名称、关键参数值（如文件路径或命令参数）及任务核心实体。",
    },
    {
        "id": "M24",
        "key": "strategy_tia",
        "dimension": "D03",
        "name_zh": "推理轨迹反演展开网络",
        "name_en": "TIA Trajectory Inversion from Summary",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L3_ENCRYPTED_STATE"],
        "domains": ["general", "tool_agent", "long_context"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "专门针对 L1_SUMMARY_FOLDED（模型仅输出过短的一句话摘要或直接给出结论），需将浓缩摘要反演还原为清晰的直接步骤时使用。",
        "how_to_use": "以已有的简短摘要或回答首句为反演锚点，结合输入问题与工具返回数据，展开为 2~3 句具体执行与推导步骤；CLI 指定 --select-methods M24。",
        "stage2_instruction": "采用轨迹反演展开（TIA Trajectory Inversion）：从已有的简短结论或观察摘要出发，逆向反演展开出支撑该结论成立的 2~3 句直接推导步骤。",
    },

    # ==================== D04: M25 - M31 ====================
    {
        "id": "M25",
        "key": "greybox_logits_kl",
        "dimension": "D04",
        "name_zh": "词元级 Top-K 对数概率蒸馏",
        "name_en": "Top-K Logits Forward/Reverse KL Distillation",
        "access_tier": "greybox",
        "execution_mode": "greybox_pipeline",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["general", "tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "moderate",
        "speedup": "1-2x",
        "when_to_use": "教师模型或推理网关（vLLM/SGLang/OpenAI 兼容接口）开放 Top-K logprobs，希望学生模型直接继承教师在工具选择与参数词元上的概率分布暗知识时使用。",
        "how_to_use": "1. 使用本插件导出带 <think> 的 ShareGPT 数据集；2. 调用教师接口采集每个词元位置的 Top-K (K=20) logprobs；3. 在学生训练中最小化 L = (1-alpha)*CE + alpha*KL(P_T || P_S)；CLI 指定 --access-tier greybox --select-methods M25。",
        "stage2_instruction": "合成高确定性、词元紧凑的推理摘要，以便配合灰盒 Top-K Logits KL 散度对齐训练。",
    },
    {
        "id": "M26",
        "key": "strategy_entropy_branch",
        "dimension": "D04",
        "name_zh": "高熵决策节点分叉采样",
        "name_en": "High-Entropy Node Branching",
        "access_tier": "greybox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "moderate",
        "speedup": "1x",
        "when_to_use": "教师模型在多个工具或参数分支点出现信息熵激增（多方案竞争），需要捕获分叉点并阐明当前选择依据时使用。",
        "how_to_use": "定位多工具竞争或复杂参数构造的高不确定度节点，在阶段二针对该分叉点生成明确的工具选型与参数确定依据；CLI 指定 --select-methods M26。",
        "stage2_instruction": "采用高熵决策分叉锚定（High-Entropy Branch Grounding）：针对关键工具选型与参数构造节点，直接说明确定当前工具与参数的具体上下文依据。",
    },
    {
        "id": "M27",
        "key": "greybox_dola",
        "dimension": "D04",
        "name_zh": "跨层对比解码逻辑放大",
        "name_en": "DoLa Layer-Contrastive Decoding",
        "access_tier": "greybox",
        "execution_mode": "greybox_pipeline",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT"],
        "domains": ["math_logic", "code_symbolic", "tool_agent"],
        "latency_budget": "moderate",
        "speedup": "1-1.5x",
        "when_to_use": "可访问教师模型深层（成熟逻辑层）与浅层（语法直觉层）Logits，希望抑制浅层套话、放大深层逻辑与工具参数准确性时使用。",
        "how_to_use": "计算教师顶层 Logits 与中低层早退 Logits 的对比差值 Delta_Logits = log P_top - log P_premature，作为软标签监督学生模型；CLI 指定 --access-tier greybox --select-methods M27。",
        "stage2_instruction": "剔除浅层语法寒暄套话，仅保留高信息密度的深层逻辑与参数推导步骤。",
    },
    {
        "id": "M28",
        "key": "greybox_marcos",
        "dimension": "D04",
        "name_zh": "马尔可夫链潜变量变分推断",
        "name_en": "MARCOS Amortized Variational Markov Chain",
        "access_tier": "greybox",
        "execution_mode": "greybox_pipeline",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "tool_agent"],
        "latency_budget": "moderate",
        "speedup": "1.5-2x",
        "when_to_use": "教师隐推理完全不可见或显式标注噪声大，希望将多步推理建模为潜变量马尔可夫链并通过摊销变分推断恢复状态转移时使用。",
        "how_to_use": "以本插件产出的阶段二逆向模型作为变分后验 q_phi(Z|X,Y)，最大化证据下界 ELBO = E_q[log p_theta(Y,Z|X) - log q_phi(Z|X,Y)] 训练学生；CLI 指定 --select-methods M28。",
        "stage2_instruction": "将每步推导构建为马尔可夫状态转移摘要（仅依赖当前观测状态与上一步输出状态推导下一步动作）。",
    },
    {
        "id": "M29",
        "key": "greybox_jepo",
        "dimension": "D04",
        "name_zh": "联合证据下界后验优化",
        "name_en": "JEPO Joint Evidence Posterior Optimization",
        "access_tier": "greybox",
        "execution_mode": "greybox_pipeline",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "math_logic"],
        "latency_budget": "moderate",
        "speedup": "1.5-2x",
        "when_to_use": "输入问题 X 与最终结论 Y 跨度极大，单靠前向先验或单靠逆向后验均存在偏差，需联合优化先验与后验分布时使用。",
        "how_to_use": "交替迭代阶段一前向采样先验 p(Z|X) 与阶段二逆向归因后验 q(Z|X,A,O,Y)，最小化二者 Jensen-Shannon / KL 散度；CLI 指定 --select-methods M29。",
        "stage2_instruction": "同时兼顾问题前向可达性与结果逆向必然性，生成连接输入问题与工具动作的双向一致推理摘要。",
    },
    {
        "id": "M30",
        "key": "greybox_regular",
        "dimension": "D04",
        "name_zh": "正则化潜变量轨迹引导",
        "name_en": "ReGuLaR Latent Transition Regularization",
        "access_tier": "greybox",
        "execution_mode": "greybox_pipeline",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic"],
        "latency_budget": "moderate",
        "speedup": "1.5-2x",
        "when_to_use": "潜变量或隐式思维训练出现后验坍缩（学生绕过推理过程直接死记答案）时使用。",
        "how_to_use": "在潜变量转移步施加与真实工具参数及 observation 的互信息最大化正则项，强制推理步携带关键工具参数信息；CLI 指定 --select-methods M30。",
        "stage2_instruction": "在每步推理摘要中强制携带决定下一步工具参数的必要状态信息，防止推理空洞化。",
    },
    {
        "id": "M31",
        "key": "greybox_gflownets",
        "dimension": "D04",
        "name_zh": "生成流网络有向无环图采样",
        "name_en": "GFlowNets DAG Trajectory Sampling",
        "access_tier": "greybox",
        "execution_mode": "greybox_pipeline",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "math_logic", "code_symbolic"],
        "latency_budget": "moderate",
        "speedup": "1.5-2x",
        "when_to_use": "同一任务存在多条等价合法的工具调用顺序（DAG），传统交叉熵训练容易导致模式坍缩（Mode Collapse）时使用。",
        "how_to_use": "使用 Trajectory Balance (TB) 损失训练学生，使采样轨迹 Z 的概率 P(Z|X) 严格正比于 PRM 与工具执行成功奖励 R(X,Z,Y)；CLI 指定 --select-methods M31。",
        "stage2_instruction": "清晰表述当前工具调用在有向无环任务图中的节点角色与前后置依赖关系。",
    },

    # ==================== D05: M32 - M40 ====================
    {
        "id": "M32",
        "key": "whitebox_hidden_mse",
        "dimension": "D05",
        "name_zh": "中间层隐状态投影对齐",
        "name_en": "Hidden-State Linear Projection MSE/Cosine",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["general", "tool_agent", "code_symbolic"],
        "latency_budget": "low_latency",
        "speedup": "2-4x",
        "when_to_use": "拥有开源教师与学生模型权重（--open-source true），希望学生在关键工具决策词元位置直接对齐教师的中间层高维连续激活时使用。",
        "how_to_use": "1. 用本插件生成 ShareGPT 轨迹；2. 在师生等比例层（如 1/4, 2/4, 3/4, 1 深度）挂载线性投影层 W_proj；3. 在 </think> 与 function_call 起始词元处最小化 MSE + Cosine 损失；CLI 指定 --open-source true --select-methods M32。",
        "stage2_instruction": "生成结构规整的 2~3 句推理摘要，为白盒隐藏层激活对齐提供精确的 </think> 边界锚点。",
    },
    {
        "id": "M33",
        "key": "whitebox_attention_map",
        "dimension": "D05",
        "name_zh": "多头注意力权重图谱迁移",
        "name_en": "Multi-Head Attention Map Transfer",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["long_context", "tool_agent", "code_symbolic"],
        "latency_budget": "low_latency",
        "speedup": "2-3x",
        "when_to_use": "长上下文或多轮工具调用中，轻量学生模型难以在冗长的 observation 里锁定关键文件路径或字段时使用。",
        "how_to_use": "提取教师模型顶层多头注意力权重矩阵 A_T，在工具参数生成步最小化 KL(A_T || A_S)，指导学生锁定相同的上下文锚点；CLI 指定 --open-source true --select-methods M33。",
        "stage2_instruction": "在推理摘要中精准点明所引用的上下文文件路径、函数名与观察结果关键字段，辅助注意力图谱对齐。",
    },
    {
        "id": "M34",
        "key": "whitebox_coconut",
        "dimension": "D05",
        "name_zh": "连续隐式思维课程蒸馏",
        "name_en": "Coconut — Chain of Continuous Thought",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "tool_agent", "code_symbolic"],
        "latency_budget": "low_latency",
        "speedup": "4-8x",
        "when_to_use": "拥有经本插件蒸馏出的分步显式 <think> 数据集，希望将学生模型的自然语言思考逐步内化为 K 个连续隐思维向量以实现 4~8 倍加速时使用。",
        "how_to_use": "1. 使用本插件导出带分句 <think> 的 ShareGPT 数据集；2. 按课程学习分阶段将每句显式推导替换为 <bot>...<eot> 之间的最后一层连续隐状态直传（h_t 直接作为 t+1 输入嵌入）；CLI 指定 --open-source true --select-methods M34。",
        "stage2_instruction": "将每步 <think> 拆分为 2~3 个逻辑层次清晰的独立短句，以便在 Coconut 课程训练中按句逐步替换为连续隐向量。",
    },
    {
        "id": "M35",
        "key": "whitebox_codi",
        "dimension": "D05",
        "name_zh": "自我蒸馏连续隐词元压缩",
        "name_en": "CODI — Self-Distillation Continuous CoT",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "tool_agent", "code_symbolic"],
        "latency_budget": "low_latency",
        "speedup": "5-10x",
        "when_to_use": "希望免除多阶段课程学习，在单阶段内让同一模型通过自我蒸馏将长思维链压缩为固定 M=4~8 个连续隐词元时使用。",
        "how_to_use": "同一模型并行前向运行 ShareGPT 显式 <think> 教师分支与 M 个连续隐词元学生分支，在工具调用/答案生成前夕词元位置施加 Smooth-L1 隐状态对齐损失；CLI 指定 --open-source true --select-methods M35。",
        "stage2_instruction": "生成高信息密度的 2~3 句显式推理摘要，作为 CODI 自我蒸馏教师分支的监督信号。",
    },
    {
        "id": "M36",
        "key": "whitebox_simcot",
        "dimension": "D05",
        "name_zh": "步级解码器监督隐式思维",
        "name_en": "SIM-CoT — Step-Level Supervised Implicit CoT",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "tool_agent", "code_symbolic"],
        "latency_budget": "low_latency",
        "speedup": "4-8x",
        "when_to_use": "单纯使用 Coconut 或 CODI 压缩后出现隐空间语义坍缩（不同连续隐词元表征趋同、丧失步骤区分度）时使用。",
        "how_to_use": "在训练期挂载轻量步级解码器（推理期移除），强制第 k 个连续隐词元必须能解码还原出本插件 ShareGPT 数据集中对应的第 k 句 <think> 文本；CLI 指定 --open-source true --select-methods M36。",
        "stage2_instruction": "确保每步 <think> 由语义正交、信息不重叠的 2~3 句组成，供 SIM-CoT 步级解码器逐句对齐监督。",
    },
    {
        "id": "M37",
        "key": "whitebox_diagonal_gmm",
        "dimension": "D05",
        "name_zh": "垂直跨层对角映射与混合高斯拟合",
        "name_en": "Diagonal Layer Mapping & GMM",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "tool_agent"],
        "latency_budget": "low_latency",
        "speedup": "3-6x",
        "when_to_use": "希望将教师沿时间步横向展开的推理过程，压缩映射到学生沿网络层深度纵向演化的激活中，并拟合多分支多峰分布时使用。",
        "how_to_use": "按时空对角线采样教师第 t 步第 l 层隐状态对齐至学生更深层激活，并用混合高斯模型（GMM）负对数似然损失替代单一 MSE；CLI 指定 --open-source true --select-methods M37。",
        "stage2_instruction": "按递进深度组织 2~3 句推理摘要，支撑时空对角线跨层映射。",
    },
    {
        "id": "M38",
        "key": "whitebox_rot",
        "dimension": "D05",
        "name_zh": "跨模态视觉特征渲染压缩",
        "name_en": "RoT — Rendering of Thought",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L3_ENCRYPTED_STATE"],
        "domains": ["multimodal", "long_context"],
        "latency_budget": "low_latency",
        "speedup": "4-8x",
        "when_to_use": "多模态或长文本推理序列过长，希望将冗长文本推导映射为高压缩比视觉/二维特征块向量进行隐空间对齐时使用。",
        "how_to_use": "将本插件生成的显式 <think> 渲染编码为紧凑视觉特征向量，训练学生在连续特征空间一步预测该压缩表征；CLI 指定 --open-source true --select-methods M38。",
        "stage2_instruction": "生成结构紧凑、关键实体突出的推理摘要，便于渲染映射为紧凑特征块。",
    },
    {
        "id": "M39",
        "key": "whitebox_kava",
        "dimension": "D05",
        "name_zh": "压缩键值缓存工作记忆蒸馏",
        "name_en": "KaVa — Eviction-Compressed KV-Cache Distillation",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["long_context", "tool_agent", "code_symbolic"],
        "latency_budget": "low_latency",
        "speedup": "5-10x",
        "when_to_use": "多轮工具调用与长思考产生庞大 KV-Cache，希望学生模型仅用少量连续隐词元即可复现教师完整的压缩工作记忆时使用。",
        "how_to_use": "1. 基于注意力重要度与低冗余驱逐算法浓缩教师在 <think> 步的全层 KV-Cache；2. 训练学生少量隐词元的 Key/Value 投影直接拟合该浓缩缓存；CLI 指定 --open-source true --select-methods M39。",
        "stage2_instruction": "在推理摘要中浓缩高价值上下文键值信息（路径、函数名、关键参数与观察指标），支撑 KaVa 工作记忆压缩。",
    },
    {
        "id": "M40",
        "key": "whitebox_pccot",
        "dimension": "D05",
        "name_zh": "并行雅可比不动点连续蒸馏",
        "name_en": "PCCoT — Parallel Continuous CoT via Jacobi Iteration",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "code_symbolic", "tool_agent"],
        "latency_budget": "ultra_fast",
        "speedup": "6-10x",
        "when_to_use": "连续隐词元之间的串行自回归仍嫌慢，希望通过雅可比（Jacobi）并行迭代在 2~3 次前向传播内同时更新并收敛全部隐词元时使用。",
        "how_to_use": "初始化 M 个隐思维槽位，使用三角注意力掩码与雅可比并行迭代更新全量槽位，对齐教师连续隐状态不动点；CLI 指定 --open-source true --select-methods M40。",
        "stage2_instruction": "生成结构并列清晰的推理摘要，支撑 PCCoT 雅可比并行隐词元迭代收敛。",
    },

    # ==================== D06: M41 - M45 ====================
    {
        "id": "M41",
        "key": "whitebox_lrkd",
        "dimension": "D06",
        "name_zh": "图注意力逻辑关系拓扑提取",
        "name_en": "LRKD — GAT Logical Relation Graph Distillation",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "tool_agent", "code_symbolic"],
        "latency_budget": "low_latency",
        "speedup": "5-10x",
        "when_to_use": "师生模型隐藏层维度不同或跨家族异构，直接对齐激活向量会引入风格噪声，需迁移跨步骤的逻辑蕴含与排斥拓扑关系图时使用。",
        "how_to_use": "利用图注意力网络（GAT）构建教师推理步骤与工具参数间的关系邻接矩阵 G_T，最小化师生谱图拓扑距离 ||G_T - G_S||_F；CLI 指定 --open-source true --select-methods M41。",
        "stage2_instruction": "明确表述步骤之间的条件蕴含与因果依赖关系，便于构建 LRKD 逻辑关系拓扑图。",
    },
    {
        "id": "M42",
        "key": "whitebox_opld",
        "dimension": "D06",
        "name_zh": "在策隐空间想象展开蒸馏",
        "name_en": "OPLD — On-Policy Latent Distillation",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "math_logic", "code_symbolic"],
        "latency_budget": "ultra_fast",
        "speedup": "5-12x",
        "when_to_use": "离策（Off-Policy）连续隐状态蒸馏在多轮测试期因微小自回归误差累积而偏离教师流形时使用。",
        "how_to_use": "在训练期令学生模型自主展开在策（On-Policy）连续隐状态序列，由教师模型在学生实际到达的隐状态节点上提供在线目标对齐修正；CLI 指定 --open-source true --select-methods M42。",
        "stage2_instruction": "生成紧扣当前真实上下文状态的可执行推理摘要，作为 OPLD 在策展开的锚点。",
    },
    {
        "id": "M43",
        "key": "whitebox_geosteer",
        "dimension": "D06",
        "name_zh": "黎曼隐流形梯度方向引导",
        "name_en": "GeoSteer — Manifold Gradient Steering",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "ultra_fast",
        "speedup": "5-10x",
        "when_to_use": "希望通过流形几何引导向量直接将学生激活轨迹约束在高置信度逻辑流形内、抑制工具参数幻觉时使用。",
        "how_to_use": "对比教师在 PRM 通过样本与未通过样本上的激活差异构造黎曼流形引导方向 v_steer，在蒸馏与推理期施加流形投影引导；CLI 指定 --open-source true --select-methods M43。",
        "stage2_instruction": "合成高置信度、零幻觉的参数锚定推理摘要，用于提取 GeoSteer 正向流形引导向量。",
    },
    {
        "id": "M44",
        "key": "whitebox_crosscoder",
        "dimension": "D06",
        "name_zh": "稀疏交叉编码器反思特征剥离",
        "name_en": "Sparse Crosscoder Feature Isolation",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "code_symbolic", "tool_agent"],
        "latency_budget": "ultra_fast",
        "speedup": "5-15x",
        "when_to_use": "需要联合分解基座模型与推理微调模型的残差流，精准剥离出专门控制“工具前置核验”与“逻辑校验”的高维稀疏线性特征方向时使用。",
        "how_to_use": "在 Base 与 Thinking 教师模型同层残差流上训练 Sparse Crosscoder，筛选推理模型独有的稀疏特征原子并蒸馏至学生残差流；CLI 指定 --open-source true --select-methods M44。",
        "stage2_instruction": "清晰呈现工具调用前置条件核验与结果确认逻辑，辅助定位 Sparse Crosscoder 推理专属特征。",
    },
    {
        "id": "M45",
        "key": "whitebox_attr_patching",
        "dimension": "D06",
        "name_zh": "归因修补因果子空间定位",
        "name_en": "Attribution Patching Causal Subspace",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic"],
        "latency_budget": "ultra_fast",
        "speedup": "5-15x",
        "when_to_use": "全层全维度隐状态蒸馏显存开销过大，希望仅对决定工具调用参数正确性的核心因果子空间（Top-10% 注意力头与 MLP 方向）进行蒸馏时使用。",
        "how_to_use": "通过一阶梯度归因修补（Attribution Patching）计算各子空间对工具参数对数似然的因果贡献度，仅在高贡献子空间内计算对齐损失；CLI 指定 --open-source true --select-methods M45。",
        "stage2_instruction": "精准聚焦决定工具名称与关键参数取值的核心因果链条。",
    },

    # ==================== D07: M46 - M49 ====================
    {
        "id": "M46",
        "key": "whitebox_recurrent_unroll",
        "dimension": "D07",
        "name_zh": "循环深度共享层渐进展开蒸馏",
        "name_en": "Recurrent-Depth / Huginn Unrolling Distillation",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "code_symbolic", "tool_agent"],
        "latency_budget": "ultra_fast",
        "speedup": "5-12x",
        "when_to_use": "学生模型采用权重复用的循环深度架构（Looped Transformer / Huginn），希望不增加词元序列长度、仅沿深度轴循环迭代复现高阶推理时使用。",
        "how_to_use": "将教师长思维链的分步演化轨迹映射为学生共享层块的 R 次深度循环迭代，按展开步渐进对齐隐状态；CLI 指定 --open-source true --select-methods M46。",
        "stage2_instruction": "将推理过程组织为递进收敛的 2~3 个阶段性步骤，适配循环深度展开映射。",
    },
    {
        "id": "M47",
        "key": "whitebox_equilibrium_fprm",
        "dimension": "D07",
        "name_zh": "动力系统不动点吸引子蒸馏",
        "name_en": "FPRM / EqR Fixed-Point Attractor Distillation",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "code_symbolic", "tool_agent"],
        "latency_budget": "ultra_fast",
        "speedup": "8-15x",
        "when_to_use": "深度循环架构随时间反向传播（BPTT）显存占用过高，希望直接求解并蒸馏非线性动力系统的收敛均衡吸引子状态 z* = f(z*, x) 时使用。",
        "how_to_use": "前向迭代至隐状态收敛不动点 z*，利用隐函数定理（Implicit Function Theorem）或无雅可比反向传播直接对齐教师最终决策表征；CLI 指定 --open-source true --select-methods M47。",
        "stage2_instruction": "提炼从输入约束直接收敛到最终工具决策的均衡结论摘要。",
    },
    {
        "id": "M48",
        "key": "whitebox_adaptive_early_exit",
        "dimension": "D07",
        "name_zh": "相对 KL 散度零样本早退策略迁移",
        "name_en": "Zero-Shot Adaptive Compute Early-Exit",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT"],
        "domains": ["general", "tool_agent", "code_symbolic"],
        "latency_budget": "ultra_fast",
        "speedup": "3-10x",
        "when_to_use": "希望学生模型根据当前词元/动作难度自适应分配算力（简单语法词元 1~2 步早退，复杂工具参数词元多步迭代）时使用。",
        "how_to_use": "监控相邻深度迭代步预测分布的相对 KL 散度 KL(p_t || p_{t-1})，迁移教师的步级收敛早退阈值判据；CLI 指定 --open-source true --select-methods M48。",
        "stage2_instruction": "按步骤实际复杂度精简推导：简单步骤 1~2 句直接给出依据，复杂多跳步骤写明 2~3 句完整判据。",
    },
    {
        "id": "M49",
        "key": "strategy_pause_anchor",
        "dimension": "D07",
        "name_zh": "暂停与填充词元算力锚定",
        "name_en": "Pause / Filler Token KV-Compute Anchor",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "low_latency",
        "speedup": "1-5x",
        "when_to_use": "黑盒场景下需在复杂工具调用前设立结构化决策检查点；或白盒标准架构中插入可学习 <pause> 词元横向扩展 KV 算力列时使用。",
        "how_to_use": "黑盒模式：按【目标锚点 -> 参数依据锚点 -> 执行决策锚点】生成 2~3 句紧凑摘要；白盒模式：在工具调用前插入 P 个 <pause> 词元并对齐其多层 KV 激活；CLI 指定 --select-methods M49。",
        "stage2_instruction": "采用 Pause-Anchor 检查点锚定：按【目标确认点 -> 工具与参数映射点 -> 执行/结论收敛点】生成紧凑的 2~3 句推理摘要。",
    },

    # ==================== D08: M50 - M54 ====================
    {
        "id": "M50",
        "key": "whitebox_softcot",
        "dimension": "D08",
        "name_zh": "词表概率凸组合软词元传递",
        "name_en": "SoftCoT — Vocabulary Convex Combination",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "tool_agent", "code_symbolic"],
        "latency_budget": "low_latency",
        "speedup": "3-6x",
        "when_to_use": "不希望修改模型词嵌入层架构，但希望保留单步多分支叠加态（Chain of Superposition）而非坍缩为单一离散词元时使用。",
        "how_to_use": "用当前步完整词表概率分布 p_t 对词嵌入矩阵 E 加权求和构造软词元 e_soft = p_t^T * E，直接送入下一时间步；CLI 指定 --open-source true --select-methods M50。",
        "stage2_instruction": "生成涵盖核心工具与参数语义的紧凑推理摘要，支撑词表凸组合软词元初始化。",
    },
    {
        "id": "M51",
        "key": "whitebox_soft_grpo",
        "dimension": "D08",
        "name_zh": "Gumbel-Softmax 重参数化软思维策略优化",
        "name_en": "SofT-GRPO Gumbel-Softmax RL",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["math_logic", "tool_agent", "code_symbolic"],
        "latency_budget": "low_latency",
        "speedup": "3-8x",
        "when_to_use": "连续软词元无法直接计算离散采样策略梯度，需在连续词表单纯形空间内结合工具执行奖励做 GRPO 强化优化时使用。",
        "how_to_use": "在软词元分布上注入 Gumbel-Softmax 重参数化噪声，以 PRM 锚定分与工具执行正确性作为组相对奖励更新策略；CLI 指定 --open-source true --select-methods M51。",
        "stage2_instruction": "合成高奖励锚定度的推理摘要，作为 SofT-GRPO 强化学习的参考锚点。",
    },
    {
        "id": "M52",
        "key": "whitebox_hrpo",
        "dimension": "D08",
        "name_zh": "混合离散连续动作空间解耦强化",
        "name_en": "HRPO — Hybrid Latent-Explicit RL",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "math_logic", "code_symbolic"],
        "latency_budget": "low_latency",
        "speedup": "3-8x",
        "when_to_use": "希望模型自主学会将繁琐中间计算内化至连续隐状态、仅将最关键的高层决策摘要外显为离散文本时使用。",
        "how_to_use": "采用混合离散-连续动作空间强化学习（HRPO），联合优化终端任务奖励与显式词元长度成本；CLI 指定 --open-source true --select-methods M52。",
        "stage2_instruction": "仅外显最核心的 2~3 句高层决策主干，省略繁琐的底层逐字演算细节。",
    },
    {
        "id": "M53",
        "key": "whitebox_hylar",
        "dimension": "D08",
        "name_zh": "分层混合潜空间强化蒸馏",
        "name_en": "HyLaR — Hierarchical Latent-Action RL",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "long_context"],
        "latency_budget": "low_latency",
        "speedup": "4-8x",
        "when_to_use": "多轮长链路 Agent 任务中，高层策略规划适合在连续潜空间完成，而底层工具调用 JSON 语法适合离散解码时使用。",
        "how_to_use": "高层控制器每步输出连续规划潜向量 z_plan，底层执行器以 z_plan 为条件生成离散 function_call JSON；CLI 指定 --open-source true --select-methods M53。",
        "stage2_instruction": "按【高层策略规划 -> 底层工具参数映射】双层结构合成简洁推理摘要。",
    },
    {
        "id": "M54",
        "key": "whitebox_dylar_swireasoning",
        "dimension": "D08",
        "name_zh": "熵驱动动态双模切换蒸馏",
        "name_en": "DyLaR / SwiReasoning Entropy-Gated Switching",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "math_logic", "code_symbolic"],
        "latency_budget": "low_latency",
        "speedup": "3-8x",
        "when_to_use": "希望模型在低熵确定性阶段保持高速连续隐式推演，一旦遇到高熵逻辑瓶颈自动切换回显式 <think> 词元展开时使用。",
        "how_to_use": "训练基于当前步预测分布熵 H(p_t) 的门控路由器（DyLaR/SwiReasoning），高熵步展开显式 ShareGPT <think>，低熵步走隐状态直传；CLI 指定 --open-source true --select-methods M54。",
        "stage2_instruction": "针对高不确定性的关键工具决策节点提供清晰的显式推理摘要。",
    },

    # ==================== D09: M55 - M59 ====================
    {
        "id": "M55",
        "key": "greybox_spec_oracle",
        "dimension": "D09",
        "name_zh": "投机解码接受长度验证预言机",
        "name_en": "Speculative Verification Acceptance Oracle",
        "access_tier": "greybox",
        "execution_mode": "greybox_pipeline",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "moderate",
        "speedup": "1.5-3x",
        "when_to_use": "闭源网关屏蔽了 logprobs 返回，但开放了投机解码（Speculative Decoding）或 Prompt Lookup 批量验证加速接口时使用。",
        "how_to_use": "以阶段二合成的候选 <think> 作为 Draft 提交目标模型并行验证，根据接受长度与首个拒绝词元位置逆向修补推理序列；CLI 指定 --access-tier greybox --select-methods M55。",
        "stage2_instruction": "生成符合目标教师模型原生语言习惯与参数命名规范的候选推理摘要，最大化投机验证接受率。",
    },
    {
        "id": "M56",
        "key": "greybox_oskd",
        "dimension": "D09",
        "name_zh": "在线投机知识蒸馏",
        "name_en": "OSKD — Online Speculative Knowledge Distillation",
        "access_tier": "greybox",
        "execution_mode": "greybox_pipeline",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "moderate",
        "speedup": "2-4x",
        "when_to_use": "全序列在线蒸馏算力消耗过大，希望学生作为草稿器生成轨迹、教师仅在验证拒绝的回退节点下发局部监督时使用。",
        "how_to_use": "先用本插件 ShareGPT 数据集冷启动学生模型，再接入 OSKD 投机验证闭环仅对拒绝位置计算局部 KL 损失；CLI 指定 --access-tier greybox --select-methods M56。",
        "stage2_instruction": "精准修补容易出现逻辑断点的工具调用前置推导步骤。",
    },
    {
        "id": "M57",
        "key": "strategy_cotevo",
        "dimension": "D09",
        "name_zh": "多教师思维链遗传交叉与变异",
        "name_en": "CoT-Evo Evolutionary Crossover & Mutation",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["general", "tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "单一视角生成的推理摘要不够全面，希望融合“用户意图拆解”与“工具参数符号映射”两条演化路径并剪除冗余时使用。",
        "how_to_use": "在阶段二对目标节点执行双路径交叉重组与变异修补，合成逻辑密度超越单一视角的 2~3 句超集推理摘要；CLI 指定 --select-methods M57。",
        "stage2_instruction": "采用 CoT-Evo 演化交叉融合：融合【任务意图分析】与【工具参数符号推导】两条路径的优点，剪除冗余套话后合成 2~3 句高密度摘要。",
    },
    {
        "id": "M58",
        "key": "strategy_merge_of_thought",
        "dimension": "D09",
        "name_zh": "多源异构思维链共识合并",
        "name_en": "Merge-of-Thought Consensus Grafting",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["tool_agent", "code_symbolic", "long_context"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "需要将来自用户提问、工具 Schema 定义以及真实 observation 返回值的多源实体无缝嫁接为连贯推理摘要时使用。",
        "how_to_use": "提取问题实体、工具参数键名与观察返回指标三方共识，按时间步嫁接合并；CLI 指定 --select-methods M58。",
        "stage2_instruction": "采用 Merge-of-Thought 多源嫁接：将用户提问中的目标实体、所调工具的参数名称以及观察结果中的关键数据无缝合并为 2~3 句连贯推导。",
    },
    {
        "id": "M59",
        "key": "strategy_dlcot",
        "dimension": "D09",
        "name_zh": "去冗余长思维链结构剪枝",
        "name_en": "DLCoT — De-Redundant Long-CoT Pruning",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED"],
        "domains": ["general", "tool_agent", "code_symbolic", "math_logic"],
        "latency_budget": "moderate",
        "speedup": "1.5-2x",
        "when_to_use": "原始思考过程过长、包含无效反复试探或空洞寒暄，需要剪枝提纯为精炼的高层直接步骤摘要时使用。",
        "how_to_use": "剥离一切重复铺垫、无效分支与事后总结套话，仅保留通向工具调用或最终结论的最短因果主干；CLI 指定 --select-methods M59。",
        "stage2_instruction": "采用 DLCoT 去冗余结构剪枝：剔除一切寒暄套话与无效分支，仅用 2~3 句最精炼的语言陈述通向工具动作或最终回答的直接因果主干。",
    },

    # ==================== D10: M60 - M62 ====================
    {
        "id": "M60",
        "key": "strategy_pot",
        "dimension": "D10",
        "name_zh": "可执行程序思维锚定蒸馏",
        "name_en": "PoT — Program-of-Thought Execution Grounding",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["code_symbolic", "tool_agent", "math_logic"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "轨迹中包含 run_command、Python 单行脚本、Git 命令或数值统计，推理过程必须与实际执行的命令行代码和标准输出严格锚定时使用。",
        "how_to_use": "提取 function_call 中的真实命令字符串、参数标志与 observation 标准输出，生成严格锚定命令语法与执行结果的符号级推理摘要；CLI 指定 --select-methods M60。",
        "stage2_instruction": "采用 PoT 程序与命令执行证据锚定：严格围绕实际执行的命令字符串、脚本路径、参数标志及标准输出数据生成 2~3 句直接推导摘要。",
    },
    {
        "id": "M61",
        "key": "strategy_ast_query_plan",
        "dimension": "D10",
        "name_zh": "抽象语法树与结构化查询计划蒸馏",
        "name_en": "AST & Query-Plan Symbolic Grounding",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L2_TOOL_SILENT", "L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["code_symbolic", "tool_agent"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "涉及 Python AST 语法树解析、JSON Schema 校验、数据库查询或复杂嵌套参数字典构造任务时使用。",
        "how_to_use": "将推导过程表示为【目标语法节点/Schema 字段 -> 过滤或校验逻辑 -> 输出结构映射】的查询计划摘要；CLI 指定 --select-methods M61。",
        "stage2_instruction": "采用 AST 与结构化查询计划锚定：从代码语法结构、函数签名或 JSON Schema 校验规则角度，直接说明当前步的解析目标与验证结论。",
    },
    {
        "id": "M62",
        "key": "strategy_mpcot",
        "dimension": "D10",
        "name_zh": "多视角正交分解与融合",
        "name_en": "MPCoT — Multi-Perspective Orthogonal Decomposition",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["tool_agent", "code_symbolic", "long_context", "general"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "多轮复杂任务中同时涉及“用户意图演进”、“当前工作区环境状态”与“工具接口契约”三重正交约束时使用。",
        "how_to_use": "黑盒模式：依次从意图视角、环境状态视角、工具参数视角正交合成 2~3 句推理摘要；白盒模式：配合 Poly-Encoder 提取三视角独立向量融合监督；CLI 指定 --select-methods M62。",
        "stage2_instruction": "采用 MPCoT 多视角正交分解：依次从【用户当前意图】、【工作区已知环境状态】与【所调工具参数契约】三个正交维度合成 2~3 句紧凑推导。",
    },

    # ==================== D11: M63 - M64 ====================
    {
        "id": "M63",
        "key": "multimodal_bridge_erasure",
        "dimension": "D11",
        "name_zh": "多跳跨模态中介渐进擦除",
        "name_en": "Cross-Modal Bridge Curriculum Erasure",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["multimodal"],
        "latency_budget": "low_latency",
        "speedup": "3-8x",
        "when_to_use": "端到端语音或多模态模型依赖“音频/图像 -> 转录描述 -> 文本 <think> -> 响应”多跳流水线，需消除中间转录与文本思考延迟时使用。",
        "how_to_use": "先采集显式多跳对齐轨迹，再通过课程学习分阶段擦除中间转录文本与文本 <think> 监督，使模型在跨模态连续隐空间内直接完成推理；CLI 指定 --open-source true --select-methods M63。",
        "stage2_instruction": "生成连接多模态输入观察与下一步动作的紧凑桥接摘要，供跨模态课程擦除训练使用。",
    },
    {
        "id": "M64",
        "key": "multimodal_visual_sketch",
        "dimension": "D11",
        "name_zh": "生成式视觉草图隐词元内化",
        "name_en": "Latent Visual Sketching Tokens",
        "access_tier": "whitebox",
        "execution_mode": "whitebox_trainer",
        "guards": ["L0_UNRESTRICTED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"],
        "domains": ["multimodal", "tool_agent"],
        "latency_budget": "low_latency",
        "speedup": "3-6x",
        "when_to_use": "多模态 Agent 需要调用视觉裁剪、放大或 UI 布局检查工具生成中间视觉状态来辅助空间推理时使用。",
        "how_to_use": "将多轮视觉操作返回的区域特征浓缩为紧凑的内部连续视觉隐词元（Latent Visual Tokens），使学生无需在测试期实际渲染图像即可完成视觉想象推理；CLI 指定 --open-source true --select-methods M64。",
        "stage2_instruction": "准确提炼视觉/布局检查工具返回的空间坐标与组件状态特征。",
    },

    # ==================== D12: M65 - M66 ====================
    {
        "id": "M65",
        "key": "strategy_timing_chunk_budget",
        "dimension": "D12",
        "name_zh": "流式响应延迟与载荷分块复杂度校准",
        "name_en": "Streaming Latency & Payload Complexity Calibration",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["tool_agent", "code_symbolic", "long_context"],
        "latency_budget": "moderate",
        "speedup": "1x",
        "when_to_use": "需要根据子 Agent 每步的 created_at 时间跨度与 observation 载荷大小，自动区分“直觉快步”与“深度重算步”并匹配对应推理详略度时使用。",
        "how_to_use": "解析日志中相邻步骤耗时差与 observation 字节量，对高耗时/大载荷关键步生成包含完整参数与数据核验的 2~3 句摘要，对轻量步保持 1~2 句精炼推导；CLI 指定 --select-methods M65。",
        "stage2_instruction": "采用延迟与载荷复杂度校准：根据每个节点的工具载荷规模动态适配详略度，关键复杂步完整写明参数与数据依据（2~3 句），简单步保持极简（1~2 句）。",
    },
    {
        "id": "M66",
        "key": "strategy_local_transcript_inspect",
        "dimension": "D12",
        "name_zh": "宿主本地会话日志无损解析与对齐",
        "name_en": "Local Session Transcript Schema Inspection",
        "access_tier": "blackbox",
        "execution_mode": "stage2_operator",
        "guards": ["L0_UNRESTRICTED", "L1_SUMMARY_FOLDED", "L2_TOOL_SILENT", "L3_ENCRYPTED_STATE", "L4_STRICT_ANTI_DISTILL"],
        "domains": ["general", "tool_agent", "code_symbolic", "math_logic", "long_context", "multimodal"],
        "latency_budget": "unconstrained",
        "speedup": "1x",
        "when_to_use": "在 Antigravity 宿主环境下运行子 Agent 后，需从本地 .system_generated/logs/transcript_full.jsonl 无损提取对话、工具调用、观察结果与原生外显 <think> 时使用。",
        "how_to_use": "默认内置于 trajectory_extractor.py inspect，自动合并 send_message 与 PLANNER_RESPONSE 中的外显 <think>，仅对缺失节点触发阶段二精准补全；CLI 亦可显式指定 --select-methods M66。",
        "stage2_instruction": "在保留本地会话日志已有原生外显 <think> 的基础上，为剩余缺失节点补全风格一致、上下文无缝衔接的 2~3 句高层推理摘要。",
    },
]

# Build lookup indices by ID ("M01".."M66") and by key ("strategy_*", "greybox_*", "whitebox_*")
METHOD_BY_ID: Dict[str, Dict[str, Any]] = {m["id"].upper(): m for m in METHOD_CATALOG}
METHOD_BY_KEY: Dict[str, Dict[str, Any]] = {m["key"].lower(): m for m in METHOD_CATALOG}


def resolve_method_token(token: str) -> Optional[Dict[str, Any]]:
    """Resolve either a Method ID (e.g. 'M15', 'm03') or a strategy key (e.g. 'strategy_star')."""
    t = token.strip()
    if not t:
        return None
    if t.upper() in METHOD_BY_ID:
        return METHOD_BY_ID[t.upper()]
    if t.lower() in METHOD_BY_KEY:
        return METHOD_BY_KEY[t.lower()]
    # Also allow shorthand without prefix, e.g. 'coconut' -> 'whitebox_coconut' or 'star' -> 'strategy_star'
    for candidate_prefix in ("strategy_", "whitebox_", "greybox_", "multimodal_"):
        prefixed = candidate_prefix + t.lower()
        if prefixed in METHOD_BY_KEY:
            return METHOD_BY_KEY[prefixed]
    return None


def filter_methods(
    dimension: str = "all",
    access_tier: str = "all",
    guard_level: str = "all",
    task_domain: str = "all",
    latency_budget: str = "all",
    search_query: str = "",
    select_methods: str = "",
    exclude_methods: str = "",
) -> List[Dict[str, Any]]:
    """Filter or freely pick methods from the 66-method catalog."""
    excluded_ids = set()
    if exclude_methods:
        for tok in exclude_methods.split(","):
            m = resolve_method_token(tok)
            if m:
                excluded_ids.add(m["id"])

    # If explicit --select-methods is provided, return in the exact user-specified order!
    if select_methods and select_methods.strip().lower() not in ("", "auto", "all"):
        picked: List[Dict[str, Any]] = []
        seen = set()
        for tok in select_methods.split(","):
            m = resolve_method_token(tok)
            if m and m["id"] not in excluded_ids and m["id"] not in seen:
                picked.append(m)
                seen.add(m["id"])
        return picked

    dims = {
        d.strip().upper()
        for d in dimension.split(",")
        if d.strip() and d.strip().lower() != "all"
    }
    tier = access_tier.strip().lower() if access_tier else "all"
    guard = guard_level.strip().upper() if guard_level else "all"
    guard_short_map = {
        "L0": "L0_UNRESTRICTED",
        "L1": "L1_SUMMARY_FOLDED",
        "L2": "L2_TOOL_SILENT",
        "L3": "L3_ENCRYPTED_STATE",
        "L4": "L4_STRICT_ANTI_DISTILL",
    }
    if guard in guard_short_map:
        guard = guard_short_map[guard]

    domain = task_domain.strip().lower() if task_domain else "all"
    budget = latency_budget.strip().lower() if latency_budget else "all"
    q = search_query.strip().lower() if search_query else ""

    results: List[Dict[str, Any]] = []
    for m in METHOD_CATALOG:
        if m["id"] in excluded_ids:
            continue
        if dims and m["dimension"] not in dims:
            continue
        if tier not in ("all", "auto", "hybrid") and m["access_tier"] != tier:
            continue
        if guard not in ("ALL", "AUTO") and guard not in m["guards"]:
            continue
        if domain not in ("all", "auto") and domain not in m["domains"]:
            continue
        if budget not in ("all", "auto") and m["latency_budget"] != budget:
            continue
        if q:
            haystack = " ".join(
                [
                    m["id"],
                    m["key"],
                    m["name_zh"],
                    m["name_en"],
                    m["when_to_use"],
                    m["how_to_use"],
                ]
            ).lower()
            if q not in haystack:
                continue
        results.append(m)
    return results


def select_strategies_and_recipes(
    base_stage2_waterfall: List[str],
    access_tier: str,
    guard_level: str,
    task_domain: str = "auto",
    latency_budget: str = "auto",
    dimension: str = "all",
    select_methods: str = "",
    exclude_methods: str = "",
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Combine user's free selection (`select_methods` / `exclude_methods` / fine-grained filters)
    with the adaptive profiler's base waterfall.

    Returns:
      1. ordered_stage2_keys: List of executable Stage-2 operator keys
      2. selected_method_cards: Detailed method cards (id, key, name, when_to_use, how_to_use) for active Stage-2 methods
      3. post_training_recipes: Detailed method cards for greybox/whitebox post-training methods
    """
    excluded_keys = set()
    excluded_ids = set()
    if exclude_methods:
        for tok in exclude_methods.split(","):
            m = resolve_method_token(tok)
            if m:
                excluded_keys.add(m["key"])
                excluded_ids.add(m["id"])

    explicit_stage2: List[str] = []
    explicit_post_recipes: List[Dict[str, Any]] = []

    if select_methods and select_methods.strip().lower() not in ("", "auto", "all"):
        for tok in select_methods.split(","):
            m = resolve_method_token(tok)
            if not m or m["id"] in excluded_ids:
                continue
            if m["execution_mode"] == "stage2_operator":
                if m["key"] not in explicit_stage2:
                    explicit_stage2.append(m["key"])
            else:
                explicit_post_recipes.append(m)

    # Filter catalog by fine-grained criteria if provided
    has_fine_filters = any(
        val and val.lower() not in ("all", "auto")
        for val in (task_domain, latency_budget, dimension)
    )
    filtered_stage2: List[str] = []
    filtered_post: List[Dict[str, Any]] = []
    if has_fine_filters:
        matched = filter_methods(
            dimension=dimension,
            access_tier="all",
            guard_level=guard_level,
            task_domain=task_domain,
            latency_budget=latency_budget,
            exclude_methods=exclude_methods,
        )
        for m in matched:
            if m["execution_mode"] == "stage2_operator":
                if m["key"] not in filtered_stage2:
                    filtered_stage2.append(m["key"])
            elif access_tier in ("whitebox", "greybox", "hybrid"):
                if access_tier == "hybrid" or m["access_tier"] == access_tier:
                    filtered_post.append(m)

    # Merge order:
    # 1) User's explicit `--select-methods` (highest priority, exact custom order)
    # 2) If `--dimension` is explicitly restricted, filter to that dimension; otherwise prioritize `base_stage2_waterfall` (smart router order)
    # 3) Additional domain/guard matched methods from the 66-method catalog
    dims_restricted = {
        d.strip().upper()
        for d in dimension.split(",")
        if d.strip() and d.strip().lower() != "all"
    }
    prioritized_base = []
    for k in base_stage2_waterfall:
        m = resolve_method_token(k)
        if dims_restricted and m and m["dimension"] not in dims_restricted:
            continue
        prioritized_base.append(k)

    final_stage2_keys: List[str] = []
    for k in explicit_stage2 + prioritized_base + filtered_stage2:
        if k not in excluded_keys and k not in final_stage2_keys:
            final_stage2_keys.append(k)

    selected_cards: List[Dict[str, Any]] = []
    for k in final_stage2_keys:
        m = resolve_method_token(k)
        if m:
            selected_cards.append(
                {
                    "id": m["id"],
                    "key": m["key"],
                    "dimension": m["dimension"],
                    "name_zh": m["name_zh"],
                    "name_en": m["name_en"],
                    "when_to_use": m["when_to_use"],
                    "how_to_use": m["how_to_use"],
                }
            )

    # Build post-training recipes based on explicit picks + access_tier
    post_recipes: List[Dict[str, Any]] = []
    seen_post = set()
    for m in explicit_post_recipes + filtered_post:
        if m["id"] not in seen_post and m["id"] not in excluded_ids:
            post_recipes.append(m)
            seen_post.add(m["id"])

    if access_tier in ("whitebox", "greybox", "hybrid"):
        target_tiers = ("whitebox", "greybox") if access_tier == "hybrid" else (access_tier,)
        for m in METHOD_CATALOG:
            if m["access_tier"] in target_tiers and m["execution_mode"] != "stage2_operator":
                if latency_budget and latency_budget.lower() not in ("auto", "all"):
                    if m["latency_budget"] != latency_budget.lower():
                        continue
                if task_domain and task_domain.lower() not in ("auto", "all"):
                    if task_domain.lower() not in m["domains"]:
                        continue
                if m["id"] not in seen_post and m["id"] not in excluded_ids:
                    post_recipes.append(m)
                    seen_post.add(m["id"])

    formatted_post_recipes = [
        {
            "id": m["id"],
            "key": m["key"],
            "dimension": m["dimension"],
            "name_zh": m["name_zh"],
            "name_en": m["name_en"],
            "access_tier": m["access_tier"],
            "speedup": m["speedup"],
            "when_to_use": m["when_to_use"],
            "how_to_use": m["how_to_use"],
        }
        for m in post_recipes
    ]

    return final_stage2_keys, selected_cards, formatted_post_recipes


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Query and filter the 66 implicit reasoning distillation methods.")
    parser.add_argument("--dimension", default="all", help="Filter by dimension (D01..D12, comma-separated).")
    parser.add_argument("--access-tier", default="all", help="Filter by access tier (blackbox|greybox|whitebox|all).")
    parser.add_argument("--guard-level", default="all", help="Filter by guard level (L0|L1|L2|L3|L4|all).")
    parser.add_argument("--task-domain", default="all", help="Filter by domain (general|tool_agent|code_symbolic|math_logic|long_context|multimodal|all).")
    parser.add_argument("--latency-budget", default="all", help="Filter by latency budget (unconstrained|moderate|low_latency|ultra_fast|all).")
    parser.add_argument("--search", default="", help="Keyword search in method names and instructions.")
    parser.add_argument("--select-methods", default="", help="Comma-separated method IDs (M01..M66) or keys.")
    parser.add_argument("--exclude-methods", default="", help="Comma-separated method IDs or keys to exclude.")
    args = parser.parse_args()

    matched = filter_methods(
        dimension=args.dimension,
        access_tier=args.access_tier,
        guard_level=args.guard_level,
        task_domain=args.task_domain,
        latency_budget=args.latency_budget,
        search_query=args.search,
        select_methods=args.select_methods,
        exclude_methods=args.exclude_methods,
    )
    print(
        json.dumps(
            {
                "total_catalog_methods": len(METHOD_CATALOG),
                "matched_count": len(matched),
                "methods": matched,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
