#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Stage Trajectory Inspector, Adaptive Profiler, Stage-2 Merger & PRM Auditor
for multi-turn-tool-distiller.

Key Principles:
- NEVER fabricates <think> text using hardcoded Python string templates.
- Stage 1 (`inspect`): Extracts natural dialogue & tool calls from transcript_full.jsonl,
  profiles the model's behavior (`profile_type`), locates any decision nodes missing
  <think> or failing PRM consistency (`target_node_indices`), and outputs an ordered
  Stage-2 strategy fallback list (`recommended_strategies`).
- Stage 2 (`merge-stage2`): Merges genuine model-generated <think> rationales produced
  by `distill_rationalizer_agent` (using STaR / MPCoT / PoT / CoT-Evo / Prefill) and
  runs Process Reward Model (PRM) grounding and consistency verification.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from method_registry import (
    DIMENSION_METADATA,
    METHOD_CATALOG,
    filter_methods,
    resolve_method_token,
    select_strategies_and_recipes,
)


BUILTIN_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "view_file": {
        "name": "view_file",
        "description": "Read the contents of a text, image, pdf, video, or audio file.",
        "parameters": {
            "type": "object",
            "properties": {
                "AbsolutePath": {"type": "string", "description": "The absolute path of the file to read."},
                "StartLine": {"type": "integer", "description": "Start line to read (inclusive)."},
                "EndLine": {"type": "integer", "description": "End line to read (inclusive)."},
            },
            "required": ["AbsolutePath"],
        },
    },
    "read_file": {
        "name": "read_file",
        "description": "Read the contents of a text, image, pdf, video, or audio file.",
        "parameters": {
            "type": "object",
            "properties": {
                "Path": {"type": "string", "description": "The absolute path of the file to read."},
                "StartLine": {"type": "integer", "description": "Start line to read (inclusive)."},
                "EndLine": {"type": "integer", "description": "End line to read (inclusive)."},
            },
            "required": ["Path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Create a new file with the given contents, or append/overwrite an existing file.",
        "parameters": {
            "type": "object",
            "properties": {
                "Path": {"type": "string", "description": "The absolute path of the file to create."},
                "Content": {"type": "string", "description": "The full contents to write to the file."},
                "Overwrite": {"type": "boolean", "description": "Whether to overwrite an existing file."},
            },
            "required": ["Path", "Content"],
        },
    },
    "edit_file": {
        "name": "edit_file",
        "description": "Edit an existing file by replacing Target with Replacement.",
        "parameters": {
            "type": "object",
            "properties": {
                "Path": {"type": "string", "description": "The absolute path of the file to modify."},
                "Target": {"type": "string", "description": "The exact string to be replaced."},
                "Replacement": {"type": "string", "description": "The content to replace the target with."},
            },
            "required": ["Path", "Target", "Replacement"],
        },
    },
    "run_command": {
        "name": "run_command",
        "description": "Execute a shell command in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "CommandLine": {"type": "string", "description": "The exact command line string to execute."},
                "WaitMsBeforeAsync": {"type": "integer", "description": "Milliseconds to wait before backgrounding."},
                "toolSummary": {"type": "string", "description": "Brief noun phrase describing the command task."},
            },
            "required": ["CommandLine", "WaitMsBeforeAsync", "toolSummary"],
        },
    },
    "call_mcp_tool": {
        "name": "call_mcp_tool",
        "description": "Call a lazy-loaded MCP tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "ServerName": {"type": "string", "description": "Name of the MCP server."},
                "ToolName": {"type": "string", "description": "Name of the tool to call."},
                "Arguments": {"type": "object", "description": "Arguments to pass to the tool."},
            },
            "required": ["ServerName", "ToolName", "Arguments"],
        },
    },
}

STRATEGY_WATERFALL_BY_PROFILE: Dict[str, List[str]] = {
    "DIRECT_COMPLETE": [],
    "TOOL_SILENT_NATIVE": [
        "strategy_star",
        "strategy_echocot_scratchpad",
        "strategy_pot",
        "strategy_prefill",
        "strategy_mpcot",
        "strategy_pause_anchor",
        "strategy_socratic",
        "strategy_cotevo",
        "strategy_prm_rerank",
    ],
    "CLI_SYMBOLIC_EXECUTION": [
        "strategy_pot",
        "strategy_ast_query_plan",
        "strategy_echocot_scratchpad",
        "strategy_star",
        "strategy_mpcot",
        "strategy_code_paradigm",
        "strategy_reflexion",
        "strategy_cotevo",
        "strategy_prm_rerank",
        "strategy_prefill",
    ],
    "MULTI_HOP_TOOL_CHAIN": [
        "strategy_echocot_scratchpad",
        "strategy_mpcot",
        "strategy_plan_and_solve",
        "strategy_pause_anchor",
        "strategy_mcts",
        "strategy_got",
        "strategy_state_continuity",
        "strategy_star",
        "strategy_cotevo",
        "strategy_pot",
        "strategy_socratic",
        "strategy_prefill",
    ],
    "STRICT_GUARDED_MINIMAL": [
        "strategy_mpcot",
        "strategy_star",
        "strategy_echocot_scratchpad",
        "strategy_mcts",
        "strategy_cotevo",
        "strategy_pot",
        "strategy_tia",
        "strategy_code_paradigm",
        "strategy_socratic",
        "strategy_debate",
        "strategy_prm_rerank",
        "strategy_prefill",
    ],
}

STRATEGY_WATERFALL_BY_GUARD_LEVEL: Dict[str, List[str]] = {
    "L0_UNRESTRICTED": [],
    "L1_SUMMARY_FOLDED": [
        "strategy_tia",
        "strategy_pause_anchor",
        "strategy_step_back",
        "strategy_star",
        "strategy_mpcot",
        "strategy_dlcot",
        "strategy_cotevo",
        "strategy_prm_rerank",
        "strategy_prefill",
    ],
    "L2_TOOL_SILENT": [
        "strategy_star",
        "strategy_echocot_scratchpad",
        "strategy_pot",
        "strategy_prefill",
        "strategy_mpcot",
        "strategy_pause_anchor",
        "strategy_plan_and_solve",
        "strategy_socratic",
        "strategy_cotevo",
    ],
    "L3_ENCRYPTED_STATE": [
        "strategy_mpcot",
        "strategy_star",
        "strategy_echocot_scratchpad",
        "strategy_mcts",
        "strategy_cotevo",
        "strategy_pot",
        "strategy_tia",
        "strategy_state_continuity",
        "strategy_pause_anchor",
        "strategy_prefill",
    ],
    "L4_STRICT_ANTI_DISTILL": [
        "strategy_code_paradigm",
        "strategy_socratic",
        "strategy_debate",
        "strategy_ast_query_plan",
        "strategy_echocot_scratchpad",
        "strategy_prm_rerank",
        "strategy_pot",
        "strategy_star",
        "strategy_cotevo",
    ],
}

POST_TRAINING_RECIPES: Dict[str, Dict[str, Any]] = {
    "whitebox": {
        "access_tier": "whitebox (Open-Source Weights & Internal Activations)",
        "recommended_methods": [
            "Coconut (Chain of Continuous Thought curriculum erasure)",
            "CODI (Self-distillation continuous compression)",
            "SIM-CoT (Step-level decoder-supervised implicit CoT)",
            "KaVa (Compressed KV-Cache working-memory distillation)",
            "PCCoT (Parallel Jacobi iteration continuous CoT)",
            "OPLD & Sparse Crosscoder (On-policy latent manifold & feature subspace alignment)",
            "EqR / FPRM (Recurrent-depth fixed-point equilibrium distillation)",
            "SoftCoT / SofT-GRPO (Vocabulary convex-combination soft tokens + hybrid RL)",
        ],
        "workflow": "Step 1: Use the distilled ShareGPT .jsonl from this skill for explicit anchor SFT; Step 2: Apply Coconut/CODI/KaVa/SoftCoT latent compression during student post-training.",
    },
    "greybox": {
        "access_tier": "greybox (Logits / Speculative Verification Oracle Access)",
        "recommended_methods": [
            "Top-K Logits KL Distillation (Forward/Reverse KL on tool & reasoning tokens)",
            "High-Entropy Node Branching (Multi-path sampling at entropy spikes)",
            "DoLa (Layer-contrastive decoding amplification)",
            "OSKD (Online Speculative Knowledge Distillation oracle)",
            "MARCOS / GFlowNets (Markov variational inference & flow-matching DAG sampling)",
        ],
        "workflow": "Combine the distilled ShareGPT trajectories with token-level Top-K Logits or OSKD speculative verification during student training.",
    },
    "blackbox": {
        "access_tier": "blackbox (Pure Text & Tool Interface / Closed API)",
        "recommended_methods": [
            "Stage 1 Zero-Wrapper Natural Execution + Stage 2 12-Operator Adaptive Waterfall (STaR, MPCoT, PoT, CoT-Evo, Prefill, TIA, MCTS, Debate, Socratic, Code-Paradigm, PRM-Rerank, Pause-Anchor)",
        ],
        "workflow": "Directly train student model via LLaMA-Factory SFT / RFT on the PRM-verified ShareGPT .jsonl dataset.",
    },
}

# Match standalone <think>...</think> blocks not enclosed in inline backticks
THINK_BLOCK_PATTERN = re.compile(r"(?<!`)<think>\s*(.*?)\s*</think>(?!`)", re.DOTALL | re.IGNORECASE)
LEADING_THINK_PATTERN = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*", re.DOTALL | re.IGNORECASE)
SYSTEM_MSG_ENVELOPE = re.compile(
    r"<SYSTEM_MESSAGE>\s*\[Message\]\s+timestamp=\S+\s+sender=\S+\s+priority=\S+\s+content=(.*?)\s*</SYSTEM_MESSAGE>",
    re.DOTALL,
)

REFUSAL_KEYWORDS = [
    "i cannot fulfill this request",
    "as an ai, i cannot",
    "抱歉，我无法满足",
    "我不能协助完成该请求",
]


def is_parent_agent_message(step: Dict[str, Any]) -> bool:
    """Check if a SYSTEM_MESSAGE step is an incoming prompt/follow-up from the parent agent."""
    if step.get("type") != "SYSTEM_MESSAGE":
        return False
    content = str(step.get("content") or "")
    return bool(SYSTEM_MSG_ENVELOPE.search(content))


def clean_human_prompt(raw_content: str) -> str:
    """Strip SYSTEM_MESSAGE envelope and any legacy wrapper headers if present."""
    text = raw_content.strip()
    env_match = SYSTEM_MSG_ENVELOPE.search(text)
    if env_match:
        text = env_match.group(1).strip()
    marker_match = re.search(r"【用户第\s*\d+\s*轮(?:提问|追问)】：\s*(.*)", text, re.DOTALL)
    if marker_match:
        text = marker_match.group(1).strip()
    text = re.sub(r"\n*（提醒：若需调用工具.*?）\s*$", "", text, flags=re.DOTALL).strip()
    return text


def clean_native_thinking(raw_thinking: str) -> str:
    """Clean vendor folded thought summary (`thinking` / `reasoning` field in transcript_full.jsonl, L1_SUMMARY_FOLDED)."""
    if not raw_thinking:
        return ""
    text = re.sub(r"^\s*I'm thinking through how to approach this\.\s*", "", raw_thinking, flags=re.IGNORECASE).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if text in ("...", "redacted", "[encrypted]"):
        return ""
    return text


def split_think_and_body(raw_text: str) -> Tuple[List[str], str]:
    """Extract standalone <think>...</think> blocks and the remaining visible answer text."""
    if not raw_text:
        return [], ""
    lead_match = LEADING_THINK_PATTERN.match(raw_text)
    if lead_match:
        think_content = lead_match.group(1).strip()
        body = raw_text[lead_match.end():].strip()
        return ([think_content] if think_content else []), body

    thinks = [
        m.strip()
        for m in THINK_BLOCK_PATTERN.findall(raw_text)
        if m.strip() and m.strip() != "..."
    ]
    body = THINK_BLOCK_PATTERN.sub("", raw_text).strip()
    return thinks, body


def parse_transcript_with_clues(
    transcript_path: Path,
) -> Tuple[List[Dict[str, str]], List[str], int, Dict[int, str]]:
    """
    Convert Antigravity transcript JSONL into raw ShareGPT conversations list plus
    `folded_summary_clues` (mapping gpt node_index -> vendor L1 thought summary from step['thinking']).
    Does NOT fabricate any <think> strings or blindly paste third-person English thought summaries
    into <think>.
    """
    if not transcript_path.exists():
        fallback = transcript_path.parent / "transcript.jsonl"
        if fallback.exists():
            transcript_path = fallback
        else:
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    steps_by_idx: Dict[int, Dict[str, Any]] = {}
    ordered_Without_idx: List[Dict[str, Any]] = []
    with transcript_path.open("r", encoding="utf-8") as f:
        for pos, line in enumerate(f):
            line = line.strip()
            if line:
                obj = json.loads(line)
                s_idx = obj.get("step_index")
                if isinstance(s_idx, int):
                    steps_by_idx[s_idx] = obj
                else:
                    ordered_Without_idx.append(obj)

    # Backfill any step_index dropped from transcript_full.jsonl using transcript.jsonl if present
    sibling_compact = transcript_path.parent / "transcript.jsonl"
    if sibling_compact.exists() and sibling_compact != transcript_path and steps_by_idx:
        with sibling_compact.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    s_idx = obj.get("step_index")
                    if isinstance(s_idx, int) and s_idx not in steps_by_idx:
                        steps_by_idx[s_idx] = obj

    steps: List[Dict[str, Any]] = (
        [steps_by_idx[k] for k in sorted(steps_by_idx.keys())] + ordered_Without_idx
        if steps_by_idx
        else ordered_Without_idx
    )

    conversations: List[Dict[str, str]] = []
    used_tools: List[str] = []
    folded_summary_clues: Dict[int, str] = {}
    user_turns = 0

    i = 0
    n = len(steps)
    while i < n:
        step = steps[i]
        stype = step.get("type", "")
        source = step.get("source", "")

        is_user_turn = (
            stype == "USER_INPUT"
            or is_parent_agent_message(step)
            or (
                stype == "GENERIC"
                and source in ("USER_EXPLICIT", "USER", "AGENT")
                and "content" in step
                and (i == 0 or steps[i - 1].get("type") != "PLANNER_RESPONSE" or not steps[i - 1].get("tool_calls"))
            )
        )

        if is_user_turn:
            raw_q = str(step.get("content") or "").strip()
            if raw_q:
                clean_q = clean_human_prompt(raw_q)
                conversations.append({"from": "human", "value": clean_q})
                user_turns += 1
            i += 1
            continue

        if stype == "PLANNER_RESPONSE":
            raw_content = str(step.get("content") or "").strip()
            raw_native = clean_native_thinking(str(step.get("thinking") or step.get("reasoning") or ""))
            thinks, body = split_think_and_body(raw_content)
            tool_calls = step.get("tool_calls") or []

            if tool_calls:
                obs_steps: List[str] = []
                j = i + 1
                while j < n and len(obs_steps) < len(tool_calls):
                    next_step = steps[j]
                    if next_step.get("type") == "GENERIC":
                        obs_steps.append(str(next_step.get("content") or ""))
                        j += 1
                    else:
                        break

                # Check if tool_calls is solely `send_message` delivering the turn's final answer to parent agent
                non_send_calls = [tc for tc in tool_calls if tc.get("name") != "send_message"]
                send_calls = [tc for tc in tool_calls if tc.get("name") == "send_message"]

                if send_calls and not non_send_calls:
                    msg_arg = send_calls[-1].get("args") or {}
                    if isinstance(msg_arg, str):
                        try:
                            msg_arg = json.loads(msg_arg)
                        except Exception:
                            msg_arg = {"Message": msg_arg}
                    delivered_text = str(msg_arg.get("Message") or raw_content).strip()
                    # Strip accidental surrounding quotes from truncated/serialized JSON string if present
                    if delivered_text.startswith('"') and delivered_text.endswith('"'):
                        try:
                            delivered_text = json.loads(delivered_text)
                        except Exception:
                            pass
                    s_thinks, s_body = split_think_and_body(delivered_text)
                    final_thinks = list(s_thinks or thinks)
                    final_body = s_body or body

                    # Harvest any <think> emitted in trailing PLANNER_RESPONSE right after send_message
                    while (
                        j < n
                        and steps[j].get("type") == "PLANNER_RESPONSE"
                        and not steps[j].get("tool_calls")
                    ):
                        trail_text = str(steps[j].get("content") or "").strip()
                        t_thinks, _ = split_think_and_body(trail_text)
                        if t_thinks and not final_thinks:
                            final_thinks.extend(t_thinks)
                        j += 1

                    # Skip any subsequent disk-space recovery / duplicate send_message retry steps before the next user turn
                    while j < n:
                        peek = steps[j]
                        if peek.get("type") == "USER_INPUT" or is_parent_agent_message(peek):
                            break
                        j += 1

                    node_idx = len(conversations)
                    if raw_native:
                        folded_summary_clues[node_idx] = raw_native
                    if final_thinks:
                        conversations.append(
                            {"from": "gpt", "value": f"<think>\n{final_thinks[-1]}\n</think>\n{final_body}"}
                        )
                    else:
                        conversations.append({"from": "gpt", "value": final_body})

                    i = j
                    continue

                # If the very last tool call in the transcript timed out into an unfinished background task
                # followed only by an async waiting message, skip the unfinished tool call so Stage-2 can
                # synthesize the final response from the preceding completed observations.
                if (
                    j == n - 1
                    and len(obs_steps) == 1
                    and "Tool is running as a background task" in obs_steps[0]
                    and steps[j].get("type") == "PLANNER_RESPONSE"
                    and not steps[j].get("tool_calls")
                ):
                    node_idx = len(conversations)
                    if raw_native:
                        folded_summary_clues[node_idx] = raw_native
                    i = j
                    continue

                for idx, tc in enumerate(tool_calls):
                    t_name = str(tc.get("name") or "unknown_tool")
                    if t_name == "send_message":
                        continue
                    t_args = tc.get("args")
                    if isinstance(t_args, str):
                        try:
                            t_args = json.loads(t_args)
                        except Exception:
                            t_args = {"raw_args": t_args}
                    elif not isinstance(t_args, dict):
                        t_args = {}

                    # Normalize legacy view_file -> read_file for unified schema
                    if t_name == "view_file":
                        t_name = "read_file"
                        if "AbsolutePath" in t_args:
                            t_args = {"Path": t_args.pop("AbsolutePath"), **t_args}

                    if t_name not in used_tools:
                        used_tools.append(t_name)

                    node_idx = len(conversations)
                    if raw_native:
                        folded_summary_clues[node_idx] = raw_native

                    # Only use genuine model-emitted <think>; otherwise leave empty for Stage-2 model synthesis
                    if idx < len(thinks) and thinks[idx]:
                        conversations.append(
                            {"from": "gpt", "value": f"<think>\n{thinks[idx]}\n</think>\n"}
                        )
                    else:
                        conversations.append({"from": "gpt", "value": ""})

                    conversations.append(
                        {
                            "from": "function_call",
                            "value": json.dumps(
                                {"name": t_name, "arguments": t_args},
                                ensure_ascii=False,
                            ),
                        }
                    )
                    obs_text = sanitize_observation_text(
                        obs_steps[idx] if idx < len(obs_steps) else "Tool execution completed."
                    )
                    conversations.append(
                        {
                            "from": "observation",
                            "value": json.dumps(
                                {"result": obs_text},
                                ensure_ascii=False,
                            ),
                        }
                    )

                if body:
                    node_idx = len(conversations)
                    if raw_native:
                        folded_summary_clues[node_idx] = raw_native
                    if len(thinks) > len(tool_calls):
                        conversations.append(
                            {"from": "gpt", "value": f"<think>\n{thinks[-1]}\n</think>\n\n{sanitize_assistant_body(body)}"}
                        )
                    else:
                        conversations.append({"from": "gpt", "value": body})

                i = j
                continue
            else:
                if raw_content:
                    node_idx = len(conversations)
                    if raw_native:
                        folded_summary_clues[node_idx] = raw_native
                    if thinks:
                        conversations.append(
                            {
                                "from": "gpt",
                                "value": f"<think>\n{thinks[-1]}\n</think>\n{body}",
                            }
                        )
                    else:
                        conversations.append({"from": "gpt", "value": raw_content})
                i += 1
                continue

        i += 1

    return conversations, used_tools, user_turns, folded_summary_clues


def parse_transcript(
    transcript_path: Path,
) -> Tuple[List[Dict[str, str]], List[str], int]:
    conversations, used_tools, user_turns, _ = parse_transcript_with_clues(transcript_path)
    return conversations, used_tools, user_turns


HOST_SCAFFOLD_LEAK_TERMS = (
    "send_message",
    "父 agent",
    "父agent",
    "parent agent",
    "父代理",
    "主代理",
    "向主代理",
    "后台执行",
    "等待系统通知",
    "临时磁盘空间不足",
    "c 盘空间不足",
    "file:///",
)

SHALLOW_ANNOUNCEMENT_RE = re.compile(
    r"^\s*(?:我将|脚本已写入|已将|已通过|已完成|现在通过|现在我将|目标：|结论：).*$",
    re.DOTALL,
)

CAUSAL_DERIVATION_MARKERS = (
    "因为", "由于", "因此", "从而", "故", "推导", "计算", "代入", "公式", "方程",
    "复杂度", "时间复杂度", "空间复杂度", "不变量", "契约", "定理", "引理", "解析",
    "展开", "求导", "积分", "特征值", "矩阵", "向量", "遍历", "递归", "分治", "动态规划",
)

BOUNDARY_AND_HYPOTHESIS_MARKERS = (
    "边界", "异常", "对比", "备选", "而非", "相比", "如果", "若", "假设", "验证",
    "核对", "核验", "误差", "精度", "修正", "排查", "防止", "避免", "注意", "但是",
    "然而", "不过", "退化", "奇异", "溢出", "容错", "回退", "一致", "守恒",
)


def sanitize_observation_text(raw_obs: str) -> str:
    """Strip Antigravity host wrapper metadata (Created At/Completed At/File Path) and normalize line endings."""
    if not raw_obs:
        return ""
    text = raw_obs.replace("\r\n", "\n").replace("\r", "\n")
    # Strip Created At / Completed At headers
    text = re.sub(
        r"^Created At:\s*[^\n]+\nCompleted At:\s*[^\n]+\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Strip File Path: `file:///...` header from view_file/read_file
    text = re.sub(
        r"^File Path:\s*`?file:///[^`\n]+`?\nTotal Lines:\s*\d+\nTotal Bytes:\s*\d+\nShowing lines[^\n]+\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text.strip()


def sanitize_assistant_body(raw_body: str) -> str:
    """Strip Antigravity IDE proprietary file:/// markdown links while preserving readable code/symbol names."""
    if not raw_body:
        return ""
    text = raw_body.replace("\r\n", "\n").replace("\r", "\n")
    # Convert [`symbol`](file:///...) -> `symbol` and [filename](file:///...) -> `filename`
    def _repl_link(m: re.Match) -> str:
        label = m.group(1).strip()
        if label.startswith("`") and label.endswith("`"):
            return label
        return f"`{label}`"
    text = re.sub(r"\[([^\]]+)\]\(file:///[^\)]+\)", _repl_link, text)
    return text.strip()


def evaluate_implicit_reasoning_completeness(
    conversations: List[Dict[str, str]],
    gpt_idx: int,
    think_text: str,
    folded_clue: str = "",
) -> Dict[str, Any]:
    """
    Strict 7-Gate Implicit Reasoning Completeness & Authenticity Judge.
    Evaluates whether implicit reasoning (<think>) is completely extracted, deeply reasoned,
    causally grounded in tools/observations, and free of host scaffolding or shallow plans.
    Returns dict with `passed` (bool), `score` (0..100), `failed_gates` (List[str]), and `metrics`.
    """
    cleaned = think_text.strip()
    failed_gates: List[str] = []
    score = 100

    next_msg = conversations[gpt_idx + 1] if gpt_idx + 1 < len(conversations) else None
    prev_msg = conversations[gpt_idx - 1] if gpt_idx - 1 >= 0 else None
    is_pre_tool = bool(next_msg and next_msg.get("from") == "function_call")
    is_post_obs = bool(prev_msg and prev_msg.get("from") == "observation")

    # Gate 1: Structural & Multi-Paragraph Depth Gate
    paragraphs = [p.strip() for p in re.split(r"\n+", cleaned) if p.strip()]
    min_chars = 320 if is_pre_tool else 360
    if len(cleaned) < min_chars:
        failed_gates.append(f"INSUFFICIENT_REASONING_VOLUME(step={gpt_idx},len={len(cleaned)},min={min_chars})")
        score -= 25
    if len(paragraphs) < 2:
        failed_gates.append(f"SINGLE_PARAGRAPH_FLAT_THINK(step={gpt_idx},paras={len(paragraphs)})")
        score -= 15

    # Gate 2: Anti-Execution-Plan & Anti-Template Gate
    if SHALLOW_ANNOUNCEMENT_RE.match(cleaned):
        failed_gates.append(f"MECHANICAL_OR_SHALLOW_OPENING(step={gpt_idx})")
        score -= 20
    causal_hits = sum(1 for m in CAUSAL_DERIVATION_MARKERS if m in cleaned)
    if causal_hits < 2:
        failed_gates.append(f"LACKS_CAUSAL_DERIVATION_DEPTH(step={gpt_idx},hits={causal_hits})")
        score -= 15

    # Gate 3: Cognitive Self-Correction, Hypothesis Comparison & Boundary Gate
    boundary_hits = sum(1 for m in BOUNDARY_AND_HYPOTHESIS_MARKERS if m in cleaned)
    if boundary_hits < 2:
        failed_gates.append(f"MISSING_HYPOTHESIS_OR_BOUNDARY_CHECK(step={gpt_idx},hits={boundary_hits})")
        score -= 15

    # Gate 4: Encrypted Thought Summary Clue Recall Gate (when L1 clue exists)
    clue_recall = 1.0
    if folded_clue:
        clue_tokens = set(
            tok.lower()
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}|\d+\.\d+", folded_clue)
            if tok.lower() not in {"thinking", "process", "understand", "goal", "user", "wants", "need", "step", "will", "that", "this", "with", "from", "have", "into"}
        )
        if clue_tokens:
            low_think = cleaned.lower()
            matched = sum(1 for tok in clue_tokens if tok in low_think)
            clue_recall = matched / max(1, len(clue_tokens))
            if len(clue_tokens) >= 4 and clue_recall < 0.25:
                failed_gates.append(f"LOW_ENCRYPTED_CLUE_RECALL(step={gpt_idx},recall={clue_recall:.2f})")
                score -= 15

    # Gate 5: Tool Causal Grounding & Observation Verification Gate
    low_think = cleaned.lower()
    if is_pre_tool and next_msg:
        try:
            fc = json.loads(next_msg.get("value") or "{}")
            t_name = str(fc.get("name") or "").lower()
            t_args = fc.get("arguments") or {}
            anchors = [t_name] if t_name else []
            for v in t_args.values():
                if isinstance(v, str):
                    anchors.extend(tok.lower() for tok in re.findall(r"[A-Za-z0-9_\-\.]{3,}", v)[:6])
            if anchors and not any(a in low_think for a in anchors if a):
                failed_gates.append(f"PRM_UNGROUNDED_PRE_TOOL_THINK(step={gpt_idx})")
                score -= 25
        except Exception:
            pass

    if is_post_obs and prev_msg:
        try:
            obs_text = str(json.loads(prev_msg.get("value") or "{}").get("result") or "")
            if obs_text.strip() == "Created file.":
                obs_anchors = ["写入", "创建", "落盘", "生成", "编码", "脚本", "created", ".py"]
            else:
                obs_nums = re.findall(r"\b\d+(?:\.\d+)?\b", obs_text)
                obs_ids = [
                    w.lower()
                    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", obs_text)
                    if w.lower() not in {"command", "exited", "with", "code", "output", "from", "import", "true", "false", "none", "self"}
                ]
                obs_anchors = [x.lower() for x in (obs_nums[:15] + obs_ids[:25])]
            if obs_anchors and not any(a in low_think for a in obs_anchors if a):
                failed_gates.append(f"MISSING_OBSERVATION_DATA_ANALYSIS(step={gpt_idx})")
                score -= 20
        except Exception:
            pass

    # Gate 6: Zero Host Scaffold Leak & Encoding Integrity Gate
    for leak_term in HOST_SCAFFOLD_LEAK_TERMS:
        if leak_term in low_think:
            failed_gates.append(f"HOST_SCAFFOLD_LEAK_IN_THINK(step={gpt_idx},term={leak_term})")
            score -= 30
            break
    if "\ufffd" in cleaned:
        failed_gates.append(f"MOJIBAKE_FFFD_IN_THINK(step={gpt_idx})")
        score -= 30

    score = max(0, score)
    return {
        "passed": len(failed_gates) == 0 and score >= 85,
        "score": score,
        "failed_gates": failed_gates,
        "metrics": {
            "char_len": len(cleaned),
            "paragraph_count": len(paragraphs),
            "causal_marker_hits": causal_hits,
            "boundary_marker_hits": boundary_hits,
            "clue_recall": round(clue_recall, 3),
        },
    }


def check_prm_grounding(
    conversations: List[Dict[str, str]],
    gpt_idx: int,
    think_text: str,
) -> Tuple[bool, Optional[str]]:
    """
    Process Reward Model (PRM) grounding & consistency check:
    1) Rejects empty/short (<12 chars) strings;
    2) Rejects host multi-agent scaffolding leakage (`send_message`, `父 Agent`, etc.);
    3) Rejects single-sentence superficial action announcements (<85 chars with no derivation);
    4) Ensures pre-tool <think> is grounded in the tool/arguments/context.
    """
    cleaned = think_text.strip()
    if len(cleaned) < 12:
        return False, f"THINK_TOO_SHORT(step={gpt_idx})"

    low_think = cleaned.lower()
    for leak_term in HOST_SCAFFOLD_LEAK_TERMS:
        if leak_term in low_think:
            return False, f"HOST_SCAFFOLD_LEAK_IN_THINK(step={gpt_idx})"

    if len(cleaned) < 280 and SHALLOW_ANNOUNCEMENT_RE.match(cleaned):
        return False, f"SHALLOW_ACTION_ANNOUNCEMENT(step={gpt_idx})"

    next_msg = conversations[gpt_idx + 1] if gpt_idx + 1 < len(conversations) else None
    if next_msg and next_msg.get("from") == "function_call":
        try:
            fc = json.loads(next_msg.get("value") or "{}")
            t_name = str(fc.get("name") or "")
            t_args = fc.get("arguments") or {}
            # Check if think_text references either the tool concept, a parameter value token, or the preceding user request
            candidate_anchors: List[str] = [t_name.lower()]
            for v in t_args.values():
                if isinstance(v, str) and len(v) >= 3:
                    # Extract basename or key token
                    tokens = re.findall(r"[A-Za-z0-9_\-\.]{3,}", v)
                    candidate_anchors.extend(tok.lower() for tok in tokens[:4])
            # Also collect tokens from the most recent human turn (including ASCII words and 2-char CJK terms)
            for back_i in range(gpt_idx - 1, -1, -1):
                if conversations[back_i].get("from") == "human":
                    h_val = conversations[back_i].get("value", "")
                    h_tokens = re.findall(r"[A-Za-z0-9_\-\.]{2,}", h_val)
                    candidate_anchors.extend(tok.lower() for tok in h_tokens[:8])
                    cjk_Seqs = re.findall(r"[\u4e00-\u9fff]{2,}", h_val)
                    for seq in cjk_Seqs[:4]:
                        for k in range(len(seq) - 1):
                            candidate_anchors.append(seq[k : k + 2])
                    break

            if candidate_anchors and not any(anchor in low_think for anchor in candidate_anchors if anchor):
                return False, f"PRM_UNGROUNDED_PRE_TOOL_THINK(step={gpt_idx})"
        except Exception:
            pass

    return True, None


def audit_conversations(
    conversations: List[Dict[str, str]],
    expected_turns: int = 1,
) -> Tuple[bool, List[str], List[int]]:
    """
    Validate ShareGPT conversation sequence, <think> presence, PRM grounding, and tool JSON validity.
    Returns (audit_passed, list_of_error_codes, target_node_indices_needing_stage2).
    """
    errors: List[str] = []
    target_nodes: List[int] = []

    if not conversations:
        return False, ["EMPTY_CONVERSATION"], []

    human_count = sum(1 for m in conversations if m.get("from") == "human")
    if human_count < expected_turns:
        errors.append(f"INSUFFICIENT_TURNS(got={human_count},expected={expected_turns})")

    for idx, msg in enumerate(conversations):
        role = msg.get("from", "")
        val = str(msg.get("value") or "")

        if role == "gpt":
            next_role = conversations[idx + 1].get("from", "") if idx + 1 < len(conversations) else ""
            thinks, body = split_think_and_body(val)

            if "<think>" not in val or "</think>" not in val or not thinks:
                target_nodes.append(idx)
                if next_role == "function_call":
                    errors.append(f"MISSING_THINK_BEFORE_TOOL(step={idx})")
                else:
                    errors.append(f"MISSING_THINK_IN_GPT(step={idx})")
            else:
                for t in thinks:
                    ok, prm_err = check_prm_grounding(conversations, idx, t)
                    if not ok and prm_err:
                        errors.append(prm_err)
                        if idx not in target_nodes:
                            target_nodes.append(idx)

            if next_role != "function_call":
                clean_body = body.strip()
                if not clean_body:
                    errors.append(f"EMPTY_FINAL_ANSWER(step={idx})")
                    if idx not in target_nodes:
                        target_nodes.append(idx)
                elif "<truncated" in clean_body:
                    errors.append(f"TRUNCATED_FINAL_ANSWER(step={idx})")
                    if idx not in target_nodes:
                        target_nodes.append(idx)
                elif len(clean_body) < 80 and any(
                    w in clean_body for w in ("请稍候", "等待任务完成后继续", "正在后台执行")
                ):
                    errors.append(f"INCOMPLETE_ASYNC_WAIT_ANSWER(step={idx})")
                    if idx not in target_nodes:
                        target_nodes.append(idx)
                low_body = clean_body.lower()
                if any(kw in low_body for kw in REFUSAL_KEYWORDS):
                    errors.append(f"REFUSAL_DETECTED(step={idx})")

        elif role == "function_call":
            try:
                fc_obj = json.loads(val)
                if not isinstance(fc_obj, dict) or "name" not in fc_obj or "arguments" not in fc_obj:
                    errors.append(f"INVALID_TOOL_SCHEMA(step={idx})")
                if not isinstance(fc_obj.get("arguments"), dict):
                    errors.append(f"TOOL_ARGUMENTS_NOT_OBJECT(step={idx})")
            except Exception:
                errors.append(f"INVALID_TOOL_JSON(step={idx})")

            if idx + 1 >= len(conversations) or conversations[idx + 1].get("from") != "observation":
                errors.append(f"MISSING_OBSERVATION_AFTER_TOOL(step={idx})")

        elif role == "observation":
            try:
                obs_obj = json.loads(val)
                if not isinstance(obs_obj, dict) or "result" not in obs_obj:
                    errors.append(f"INVALID_OBSERVATION_SCHEMA(step={idx})")
                else:
                    res_str = str(obs_obj.get("result") or "")
                    if "\ufffd" in res_str:
                        errors.append(f"MOJIBAKE_FFFD_IN_OBSERVATION(step={idx})")
            except Exception:
                errors.append(f"INVALID_OBSERVATION_JSON(step={idx})")

    if conversations[-1].get("from") != "gpt":
        errors.append("CONVERSATION_MUST_END_WITH_GPT")

    return len(errors) == 0, errors, sorted(set(target_nodes))


def profile_trajectory(
    conversations: List[Dict[str, str]],
    used_tools: List[str],
    target_nodes: List[int],
    audit_errors: Optional[List[str]] = None,
    guard_override: str = "auto",
) -> Tuple[str, str, List[str]]:
    """
    Intelligently classify both:
    1) `guard_level` (L0_UNRESTRICTED, L1_SUMMARY_FOLDED, L2_TOOL_SILENT, L3_ENCRYPTED_STATE, L4_STRICT_ANTI_DISTILL)
    2) `profile_type` (DIRECT_COMPLETE, TOOL_SILENT_NATIVE, CLI_SYMBOLIC_EXECUTION, MULTI_HOP_TOOL_CHAIN, STRICT_GUARDED_MINIMAL)
    and return (guard_level, profile_type, ordered_recommended_stage2_strategies).
    """
    audit_errors = audit_errors or []
    if not target_nodes and not audit_errors:
        return "L0_UNRESTRICTED", "DIRECT_COMPLETE", []

    gpt_indices = [i for i, m in enumerate(conversations) if m.get("from") == "gpt"]
    native_think_count = sum(1 for i in gpt_indices if i not in target_nodes)
    tool_call_count = sum(1 for m in conversations if m.get("from") == "function_call")
    has_refusal = any("REFUSAL_DETECTED" in e for e in audit_errors)
    has_short_think = any(
        any(tag in e for tag in ("THINK_TOO_SHORT", "PRM_UNGROUNDED", "SHALLOW_ACTION_ANNOUNCEMENT", "HOST_SCAFFOLD_LEAK"))
        for e in audit_errors
    )

    # Determine profile_type
    if native_think_count == 0:
        profile = "STRICT_GUARDED_MINIMAL"
    elif "run_command" in used_tools:
        profile = "CLI_SYMBOLIC_EXECUTION"
    elif len(used_tools) >= 2 or tool_call_count >= 3:
        profile = "MULTI_HOP_TOOL_CHAIN"
    else:
        profile = "TOOL_SILENT_NATIVE"

    # Determine guard_level
    guard_map = {
        "L0": "L0_UNRESTRICTED",
        "L1": "L1_SUMMARY_FOLDED",
        "L2": "L2_TOOL_SILENT",
        "L3": "L3_ENCRYPTED_STATE",
        "L4": "L4_STRICT_ANTI_DISTILL",
    }
    if guard_override and guard_override.upper() in guard_map:
        guard_level = guard_map[guard_override.upper()]
    elif guard_override in STRATEGY_WATERFALL_BY_GUARD_LEVEL:
        guard_level = guard_override
    elif has_refusal:
        guard_level = "L4_STRICT_ANTI_DISTILL"
    elif has_short_think:
        guard_level = "L1_SUMMARY_FOLDED"
    elif native_think_count == 0:
        guard_level = "L3_ENCRYPTED_STATE"
    else:
        guard_level = "L2_TOOL_SILENT"

    # Merge guard_level waterfall and domain profile waterfall without duplicates
    combined: List[str] = []
    # If CLI symbolic task on L2/L3, prioritize strategy_pot right at the front
    if profile == "CLI_SYMBOLIC_EXECUTION" and guard_level in ("L2_TOOL_SILENT", "L3_ENCRYPTED_STATE"):
        combined.append("strategy_pot")
    for s in STRATEGY_WATERFALL_BY_GUARD_LEVEL.get(guard_level, []) + STRATEGY_WATERFALL_BY_PROFILE.get(profile, []):
        if s not in combined:
            combined.append(s)

    return guard_level, profile, combined


def build_stage2_context(
    conversations: List[Dict[str, str]],
    target_nodes: List[int],
    folded_summary_clues: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Build a compact context view of the trajectory for the Stage-2 rationalizer agent."""
    folded_summary_clues = folded_summary_clues or {}
    context_items: List[Dict[str, Any]] = []
    for idx, msg in enumerate(conversations):
        role = msg.get("from", "")
        val = str(msg.get("value") or "")
        if role == "observation" and len(val) > 1200:
            val = val[:1200] + "...[truncated]"
        item: Dict[str, Any] = {
            "node_index": idx,
            "from": role,
            "needs_stage2_think": idx in target_nodes,
            "value": val,
        }
        if role == "gpt" and idx in target_nodes:
            next_role = conversations[idx + 1].get("from", "") if idx + 1 < len(conversations) else ""
            item["node_type"] = "pre_tool_decision" if next_role == "function_call" else "post_observation_final_answer"
            if idx in folded_summary_clues:
                item["folded_thought_summary_clue"] = folded_summary_clues[idx]
            if next_role != "function_call":
                _, existing_b = split_think_and_body(val)
                cb = existing_b.strip()
                if not cb or "<truncated" in cb or (
                    len(cb) < 80 and any(w in cb for w in ("请稍候", "等待任务完成后继续", "正在后台执行"))
                ):
                    item["needs_body_completion"] = True
        context_items.append(item)
    return context_items


def build_tools_json_string(
    used_tool_names: List[str], extra_tools: Optional[List[Dict[str, Any]]] = None
) -> str:
    tool_defs: List[Dict[str, Any]] = []
    seen = set()

    if extra_tools:
        for t in extra_tools:
            if isinstance(t, dict) and "name" in t:
                tool_defs.append(t)
                seen.add(t["name"])

    for name in used_tool_names:
        if name not in seen:
            schema = BUILTIN_TOOL_SCHEMAS.get(
                name,
                {
                    "name": name,
                    "description": f"Execute built-in or MCP tool '{name}'.",
                    "parameters": {"type": "object", "properties": {}},
                },
            )
            tool_defs.append(schema)
            seen.add(name)

    return json.dumps(tool_defs, ensure_ascii=False)


def build_auto_stage2_prompt(
    stage2_context: List[Dict[str, Any]],
    target_nodes: List[int],
    strategy: str = "strategy_star",
) -> str:
    """Automatically build a compact, token-efficient Stage-2 prompt for distill_rationalizer_agent using any of the 66 methods."""
    if not target_nodes:
        return ""
    method_meta = resolve_method_token(strategy)
    if method_meta:
        method_id = method_meta["id"]
        method_key = method_meta["key"]
        title = f"{method_id} {method_meta['name_zh']} ({method_meta['name_en']})"
        instruction = method_meta["stage2_instruction"]
    else:
        method_key = strategy
        title = strategy
        instruction = "为每个目标节点生成一段自然、简明的高层推理摘要（2~3 句话，直接说明该步的目标、所选工具与参数依据或从观察结果得出最终结论的直接步骤）。"

    compact_nodes = json.dumps(stage2_context, ensure_ascii=False, indent=2)
    needs_body_nodes = [
        item["node_index"] for item in stage2_context if item.get("needs_body_completion")
    ]
    template_lines: List[str] = []
    for idx in target_nodes:
        if idx in needs_body_nodes:
            template_lines.append(f'  "{idx}": {{"think": "...", "body": "基于前序 observation 的完整 Markdown 最终回答"}}')
        else:
            template_lines.append(f'  "{idx}": "..."')
    keys_template = ",\n".join(template_lines)
    return (
        f"[Stage-2 Operator: {method_key} — {title}]\n"
        f"核心执行算子指令：{instruction}\n\n"
        f"硬性约束（严苛隐推理完整度标准）：\n"
        f"1. 推理内容越多、越深、越详尽越好（每个节点至少 3 个自然段、400~900+ 字，严禁压缩为单段或 2~3 句简短摘要！）；\n"
        f"2. 若节点包含 `folded_thought_summary_clue`（底层加密思考区的折叠摘要线索），必须 100% 吸收其中的全部数学公式、算法步骤、边界判断与符号锚点，并结合真实 `function_call` 与 `observation` 展开为第一人称纯中文深度心智演算；\n"
        f"3. 工具调用前节点（pre_tool_decision）必须分段展开：① 底层机制/数学公式/代码架构建模；② 候选方案对比、边界条件排查与潜在反例/错误预防；③ 工具选型理由及每个关键参数/脚本逻辑的逐项构造推导；\n"
        f"4. 观察结果后节点（post_observation_final_answer）必须分段展开：① 逐项核验 `observation` 返回的精确数值、状态与理论预期是否吻合；② 深入归因任何误差、报错、截断或高阶非线性效应；③ 综合多步证据收敛出最终结论结构；\n"
        f"5. 严禁以“我将…”、“脚本已写入…”、“目标：”、“结论：”等机械报幕开头，严禁出现 `send_message`、`父 Agent`、`主代理`、`file:///` 等宿主脚手架词汇；\n"
        f"4. 对标记了 `needs_body_completion: true` 的节点 {needs_body_nodes}，请返回 `{{\"think\": \"...\", \"body\": \"完整 Markdown 最终回答\"}}`。\n\n"
        f"请针对待补全节点索引列表 {target_nodes}，基于以下第一阶段真实执行轨迹上下文生成 JSON 映射：\n\n"
        f"```json\n{compact_nodes}\n```\n\n"
        f"请严格仅输出如下格式的合法 JSON 对象（不要输出额外文字）：\n"
        f"```json\n{{\n{keys_template}\n}}\n```"
    )


def extract_last_assistant_text_from_transcript(transcript_path: Path) -> str:
    """Extract the final assistant text or send_message payload from a Stage-2 subagent transcript."""
    if not transcript_path.exists():
        fallback = transcript_path.parent / "transcript.jsonl"
        if fallback.exists():
            transcript_path = fallback
        else:
            raise FileNotFoundError(f"Stage-2 transcript not found: {transcript_path}")

    steps = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Pass 1: Prefer complete `send_message` payloads from the Stage-2 subagent
    for step in reversed(steps):
        if step.get("type") != "PLANNER_RESPONSE":
            continue
        tcs = step.get("tool_calls") or []
        for tc in tcs:
            if tc.get("name") == "send_message":
                args = tc.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"Message": args}
                msg = str(args.get("Message") or "").strip()
                if msg and "{" in msg and "}" in msg:
                    return msg
    # Pass 2: Fall back to assistant text content
    for step in reversed(steps):
        if step.get("type") != "PLANNER_RESPONSE":
            continue
        content = str(step.get("content") or "").strip()
        if content and "{" in content and "}" in content:
            return content
    raise ValueError(f"No JSON object found in Stage-2 transcript: {transcript_path}")


def resolve_access_tier(open_source_arg: str, access_tier_arg: str) -> Tuple[bool, str]:
    if access_tier_arg and access_tier_arg.lower() in ("whitebox", "greybox", "blackbox", "hybrid"):
        tier = access_tier_arg.lower()
        return (tier in ("whitebox", "hybrid")), tier
    if str(open_source_arg).lower() in ("true", "1", "yes"):
        return True, "whitebox"
    return False, "blackbox"


def cmd_list_methods(args: argparse.Namespace) -> int:
    matched = filter_methods(
        dimension=getattr(args, "dimension", "all"),
        access_tier=getattr(args, "access_tier", "all"),
        guard_level=getattr(args, "guard_level", "all"),
        task_domain=getattr(args, "task_domain", "all"),
        latency_budget=getattr(args, "latency_budget", "all"),
        search_query=getattr(args, "search", ""),
        select_methods=getattr(args, "select_methods", ""),
        exclude_methods=getattr(args, "exclude_methods", ""),
    )
    print(
        json.dumps(
            {
                "total_catalog_methods": len(METHOD_CATALOG),
                "dimensions": DIMENSION_METADATA,
                "matched_count": len(matched),
                "methods": matched,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript).resolve()
    conversations, used_tools, user_turns, folded_clues = parse_transcript_with_clues(transcript_path)
    audit_passed, audit_errors, target_nodes = audit_conversations(
        conversations, expected_turns=args.expected_turns
    )
    guard_level, profile_type, base_strategies = profile_trajectory(
        conversations,
        used_tools,
        target_nodes,
        audit_errors=audit_errors,
        guard_override=getattr(args, "guard_level", "auto"),
    )
    is_open_source, access_tier = resolve_access_tier(
        getattr(args, "open_source", "false"),
        getattr(args, "access_tier", "auto"),
    )

    task_domain = getattr(args, "task_domain", "auto")
    if task_domain == "auto":
        if "run_command" in used_tools:
            task_domain = "code_symbolic"
        elif len(used_tools) >= 1:
            task_domain = "tool_agent"
        else:
            task_domain = "general"

    latency_budget = getattr(args, "latency_budget", "auto")
    dimension_filter = getattr(args, "dimension", "all")
    select_methods_arg = getattr(args, "select_methods", "")
    exclude_methods_arg = getattr(args, "exclude_methods", "")

    recommended_strategies, selected_method_cards, custom_post_recipes = select_strategies_and_recipes(
        base_stage2_waterfall=base_strategies,
        access_tier=access_tier,
        guard_level=guard_level,
        task_domain=task_domain,
        latency_budget=latency_budget,
        dimension=dimension_filter,
        select_methods=select_methods_arg,
        exclude_methods=exclude_methods_arg,
    )

    tools_str = build_tools_json_string(used_tools)
    stage2_ctx = build_stage2_context(conversations, target_nodes, folded_summary_clues=folded_clues)
    top_strategy = recommended_strategies[0] if recommended_strategies else "strategy_star"

    default_recipe = POST_TRAINING_RECIPES.get(
        "whitebox" if access_tier == "hybrid" else access_tier,
        POST_TRAINING_RECIPES["blackbox"],
    )

    payload = {
        "sample_id": args.sample_id,
        "is_open_source": is_open_source,
        "access_tier": access_tier,
        "guard_level": guard_level,
        "profile_type": profile_type,
        "task_domain": task_domain,
        "latency_budget": latency_budget,
        "dimension_filter": dimension_filter,
        "select_methods": select_methods_arg,
        "exclude_methods": exclude_methods_arg,
        "needs_stage2": len(target_nodes) > 0,
        "target_node_indices": target_nodes,
        "recommended_strategies": recommended_strategies,
        "selected_method_cards": selected_method_cards[:6],
        "post_training_recipe": default_recipe,
        "detailed_post_training_methods": custom_post_recipes[:8],
        "stage2_history": [],
        "audit_passed": audit_passed,
        "audit_errors": audit_errors,
        "completed_turns": user_turns,
        "expected_turns": args.expected_turns,
        "used_tools": used_tools,
        "stage2_context": stage2_ctx,
        "auto_stage2_prompt": build_auto_stage2_prompt(stage2_ctx, target_nodes, top_strategy),
        "sharegpt_record": {
            "system": args.system
            or "You are a helpful assistant capable of multi-turn reasoning and tool invocation.",
            "tools": tools_str,
            "conversations": conversations,
        },
    }

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "sample_id": args.sample_id,
                "is_open_source": is_open_source,
                "access_tier": access_tier,
                "guard_level": guard_level,
                "profile_type": profile_type,
                "task_domain": task_domain,
                "needs_stage2": len(target_nodes) > 0,
                "target_node_indices": target_nodes,
                "recommended_strategies": recommended_strategies,
                "top_selected_methods": [
                    f"{c['id']}:{c['key']} ({c['name_zh']})" for c in selected_method_cards[:5]
                ],
                "post_training_methods": (
                    [f"{m['id']}:{m['key']} ({m['name_zh']})" for m in custom_post_recipes[:5]]
                    if custom_post_recipes
                    else payload["post_training_recipe"]["recommended_methods"][:3]
                ),
                "audit_passed": audit_passed,
                "audit_errors": audit_errors,
                "completed_turns": user_turns,
                "message_count": len(conversations),
                "used_tools": used_tools,
                "output_file": str(Path(args.output).resolve()) if args.output else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_stage2_mapping(raw_text: str) -> Dict[str, Any]:
    """Extract JSON object mapping node_index -> rationale (str or {'think': ..., 'body': ...}) from Stage-2 agent output."""
    text = raw_text.strip()
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        text = brace_match.group(1).strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("Stage-2 output must be a JSON object mapping node_index -> think text.")
    result: Dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            result[str(k)] = v
        else:
            result[str(k)] = str(v).strip()
    return result


def cmd_merge_stage2(args: argparse.Namespace) -> int:
    inspected_path = Path(args.inspected).resolve()
    payload = json.loads(inspected_path.read_text(encoding="utf-8"))

    if getattr(args, "stage2_transcript", None):
        raw_stage2 = extract_last_assistant_text_from_transcript(Path(args.stage2_transcript).resolve())
    elif getattr(args, "stage2_json", None):
        stage2_path = Path(args.stage2_json).resolve()
        raw_stage2 = stage2_path.read_text(encoding="utf-8")
    else:
        raise ValueError("Either --stage2-transcript or --stage2-json must be provided to merge-stage2.")
    mapping = parse_stage2_mapping(raw_stage2)

    conversations: List[Dict[str, str]] = payload["sharegpt_record"]["conversations"]
    updated_indices: List[int] = []

    for k_str, val_entry in mapping.items():
        if not val_entry:
            continue
        new_body_override: Optional[str] = None
        if isinstance(val_entry, dict):
            rationale = str(val_entry.get("think") or "").strip()
            if val_entry.get("body"):
                new_body_override = str(val_entry.get("body")).strip()
        else:
            rationale = str(val_entry).strip()

        if not rationale:
            continue
        # Strip accidental outer <think> tags if the rationalizer included them
        clean_rat = THINK_BLOCK_PATTERN.sub(r"\1", rationale).strip()
        clean_rat = re.sub(r"^<think>\s*|\s*</think>$", "", clean_rat, flags=re.IGNORECASE).strip()
        idx = int(k_str)
        if 0 <= idx < len(conversations) and conversations[idx].get("from") == "gpt":
            _, existing_body = split_think_and_body(conversations[idx].get("value", ""))
            effective_body = new_body_override if new_body_override is not None else existing_body
            next_role = conversations[idx + 1].get("from", "") if idx + 1 < len(conversations) else ""
            if next_role == "function_call" or not effective_body:
                conversations[idx]["value"] = f"<think>\n{clean_rat}\n</think>\n"
            else:
                conversations[idx]["value"] = f"<think>\n{clean_rat}\n</think>\n\n{sanitize_assistant_body(effective_body)}"
            updated_indices.append(idx)

    expected_turns = int(payload.get("expected_turns", 1))
    audit_passed, audit_errors, remaining_targets = audit_conversations(
        conversations, expected_turns=expected_turns
    )

    payload["audit_passed"] = audit_passed
    payload["audit_errors"] = audit_errors
    payload["target_node_indices"] = remaining_targets
    payload["needs_stage2"] = len(remaining_targets) > 0
    payload["stage2_context"] = build_stage2_context(conversations, remaining_targets)
    history = payload.setdefault("stage2_history", [])
    history.append(
        {
            "strategy": args.strategy,
            "updated_nodes": updated_indices,
            "audit_passed": audit_passed,
            "remaining_targets": remaining_targets,
            "audit_errors": audit_errors,
        }
    )

    out_path = Path(args.output or args.inspected).resolve()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "sample_id": payload.get("sample_id"),
                "strategy_applied": args.strategy,
                "updated_nodes": updated_indices,
                "audit_passed": audit_passed,
                "remaining_targets": remaining_targets,
                "audit_errors": audit_errors,
                "output_file": str(out_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Test 1: Stage-1 transcript where native tool call step has NO inline <think>
        # Verify that `inspect` does NOT fake <think>, profiles as CLI_SYMBOLIC_EXECUTION,
        # and marks node 1 for Stage-2 synthesis!
        mock_transcript = tmp / "transcript_full.jsonl"
        steps = [
            {
                "step_index": 0,
                "source": "SYSTEM",
                "type": "SYSTEM_MESSAGE",
                "status": "DONE",
                "content": "<SYSTEM_MESSAGE>\n[Message] timestamp=2026-11-07T10:00:00Z sender=main priority=HIGH content=请统计 scripts 目录下各 Python 脚本的行数。\n</SYSTEM_MESSAGE>",
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                # Native tool call without inline content!
                "tool_calls": [
                    {
                        "name": "run_command",
                        "args": {
                            "CommandLine": "Get-ChildItem scripts/*.py",
                            "WaitMsBeforeAsync": 5000,
                            "toolSummary": "Count script lines",
                        },
                    }
                ],
            },
            {
                "step_index": 2,
                "source": "MODEL",
                "type": "GENERIC",
                "status": "DONE",
                "content": "task_manager.py: 260 lines\nsharegpt_writer.py: 210 lines",
            },
            {
                "step_index": 3,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "<think>\n根据 run_command 返回的行数统计结果，task_manager.py 为 260 行，sharegpt_writer.py 为 210 行，接下来直接汇总回答用户。\n</think>\n统计完成：`task_manager.py` 共 260 行，`sharegpt_writer.py` 共 210 行。",
            },
        ]
        mock_transcript.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in steps) + "\n",
            encoding="utf-8",
        )
        inspected_json = tmp / "inspected.json"
        cmd_inspect(
            argparse.Namespace(
                transcript=str(mock_transcript),
                sample_id="test_stage2_001",
                expected_turns=1,
                system="",
                output=str(inspected_json),
            )
        )
        data = json.loads(inspected_json.read_text(encoding="utf-8"))
        assert data["audit_passed"] is False
        assert data["needs_stage2"] is True
        assert data["target_node_indices"] == [1]
        assert data["profile_type"] == "CLI_SYMBOLIC_EXECUTION"
        assert data["recommended_strategies"][0] == "strategy_pot"

        # Test 2: Merge Stage-2 rationale produced by rationalizer subagent
        stage2_out = tmp / "stage2_out.json"
        stage2_out.write_text(
            json.dumps(
                {
                    "1": "为了统计 scripts 目录下各 Python 脚本的代码行数，需要调用 run_command 执行 Get-ChildItem scripts/*.py 命令获取行数信息。"
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        cmd_merge_stage2(
            argparse.Namespace(
                inspected=str(inspected_json),
                stage2_json=str(stage2_out),
                strategy="strategy_pot",
                output=str(inspected_json),
            )
        )
        merged = json.loads(inspected_json.read_text(encoding="utf-8"))
        assert merged["audit_passed"] is True, f"Expected pass, got: {merged['audit_errors']}"
        assert merged["needs_stage2"] is False

    print(json.dumps({"self_test": "PASSED", "module": "trajectory_extractor.py (v2 two-stage)"}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-stage trajectory inspector, adaptive profiler, Stage-2 merger & PRM auditor."
    )
    parser.add_argument("--self-test", action="store_true", help="Run built-in unit tests.")
    subparsers = parser.add_subparsers(dest="command")

    for cmd_name in ("inspect", "extract"):
        p_ins = subparsers.add_parser(cmd_name, help="Inspect Stage-1 transcript, profile guard level & open-source tier, and select from 66 methods.")
        p_ins.add_argument("--transcript", required=True, help="Path to transcript_full.jsonl or transcript.jsonl.")
        p_ins.add_argument("--sample-id", required=True, help="Sample identifier.")
        p_ins.add_argument("--expected-turns", type=int, default=1, help="Minimum expected human turns.")
        p_ins.add_argument("--open-source", default="false", help="Whether target model is open-source (true|false).")
        p_ins.add_argument("--access-tier", default="auto", choices=["auto", "whitebox", "greybox", "blackbox", "hybrid"], help="Model access tier.")
        p_ins.add_argument("--guard-level", default="auto", help="Override or auto-detect guard level (auto|L0|L1|L2|L3|L4).")
        p_ins.add_argument("--task-domain", default="auto", choices=["auto", "general", "tool_agent", "code_symbolic", "math_logic", "long_context", "multimodal"], help="Fine-grained task domain filter.")
        p_ins.add_argument("--latency-budget", default="auto", choices=["auto", "unconstrained", "moderate", "low_latency", "ultra_fast"], help="Target student inference latency budget.")
        p_ins.add_argument("--dimension", default="all", help="Filter methods by dimension(s), e.g., D01,D02,D10 or all.")
        p_ins.add_argument("--select-methods", default="", help="Freely select specific method IDs (M01..M66) or keys in custom order (comma-separated).")
        p_ins.add_argument("--exclude-methods", default="", help="Exclude specific method IDs (M01..M66) or keys (comma-separated).")
        p_ins.add_argument("--system", default="", help="Optional custom system prompt for ShareGPT output.")
        p_ins.add_argument("--output", required=True, help="Output path for inspected JSON payload.")

    p_list = subparsers.add_parser("list-methods", help="Browse, search, and filter all 66 implicit reasoning distillation methods (with when_to_use & how_to_use).")
    p_list.add_argument("--dimension", default="all", help="Filter by dimension (D01..D12, comma-separated).")
    p_list.add_argument("--access-tier", default="all", help="Filter by access tier (blackbox|greybox|whitebox|all).")
    p_list.add_argument("--guard-level", default="all", help="Filter by guard level (L0|L1|L2|L3|L4|all).")
    p_list.add_argument("--task-domain", default="all", help="Filter by domain (general|tool_agent|code_symbolic|math_logic|long_context|multimodal|all).")
    p_list.add_argument("--latency-budget", default="all", help="Filter by latency budget (unconstrained|moderate|low_latency|ultra_fast|all).")
    p_list.add_argument("--search", default="", help="Keyword search in method names and instructions.")
    p_list.add_argument("--select-methods", default="", help="Query specific method IDs (M01..M66) or keys.")
    p_list.add_argument("--exclude-methods", default="", help="Exclude specific method IDs or keys.")

    p_merge = subparsers.add_parser("merge-stage2", help="Merge Stage-2 model-generated rationales and run PRM verification.")
    p_merge.add_argument("--inspected", required=True, help="Path to inspected JSON payload.")
    p_merge.add_argument("--stage2-json", default="", help="Path to JSON file containing Stage-2 rationalizer output.")
    p_merge.add_argument("--stage2-transcript", default="", help="Path to Stage-2 subagent transcript_full.jsonl (direct extraction).")
    p_merge.add_argument("--strategy", default="strategy_star", help="Name of Stage-2 strategy applied.")
    p_merge.add_argument("--output", default="", help="Output path (defaults to overwriting --inspected).")

    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    if args.command in ("inspect", "extract"):
        return cmd_inspect(args)
    if args.command == "list-methods":
        return cmd_list_methods(args)
    if args.command == "merge-stage2":
        return cmd_merge_stage2(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
