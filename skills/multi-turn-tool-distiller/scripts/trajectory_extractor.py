#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Extractor & Quality Auditor for multi-turn-tool-distiller.

Parses Antigravity subagent session transcripts (transcript_full.jsonl / transcript.jsonl),
reconstructs ShareGPT multi-turn & tool-calling conversations, strips prompt-wrapper
boilerplate from human turns, and audits <think> coverage and tool argument validity.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    """Strip SYSTEM_MESSAGE envelope and prompt-strategy wrapper headers/footers."""
    text = raw_content.strip()
    env_match = SYSTEM_MSG_ENVELOPE.search(text)
    if env_match:
        text = env_match.group(1).strip()
    # Match 【用户第 X 轮提问】： or 【用户第 X 轮追问】：
    marker_match = re.search(r"【用户第\s*\d+\s*轮(?:提问|追问)】：\s*(.*)", text, re.DOTALL)
    if marker_match:
        text = marker_match.group(1).strip()
    # Strip trailing reminder block if present
    text = re.sub(r"\n*（提醒：若需调用工具.*?）\s*$", "", text, flags=re.DOTALL).strip()
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


def synthesize_concise_pre_tool_think(tool_name: str, args: Dict[str, Any]) -> str:
    """
    Build a concise, high-level direct summary when the native planner step invokes
    a tool directly without emitting a separate inline text field in the same step.
    """
    if tool_name in ("read_file", "view_file"):
        target = args.get("Path") or args.get("AbsolutePath") or "目标文件"
        return f"需要读取文件 `{target}` 以获取满足当前任务所需的配置或代码内容。我将调用 `{tool_name}` 工具传入该路径参数进行读取。"
    if tool_name == "write_file":
        target = args.get("Path", "目标文件")
        return f"需要将处理后的内容写入文件 `{target}`。我将调用 `write_file` 工具并传入目标路径与完整文件内容。"
    if tool_name == "edit_file":
        target = args.get("Path", "目标文件")
        return f"需要对文件 `{target}` 的指定片段进行精确修改。我将调用 `edit_file` 工具传入目标路径、原文本与替换内容。"
    if tool_name == "run_command":
        cmd = args.get("CommandLine", "")
        summary = args.get("toolSummary", "执行命令")
        return f"需要通过命令行执行“{summary}”以获取实时输出。我将调用 `run_command` 工具运行 `{cmd}`。"
    arg_keys = ", ".join(str(k) for k in args.keys()) if isinstance(args, dict) else ""
    return f"需要调用工具 `{tool_name}` 获取任务所需信息。我将传入参数（{arg_keys}）执行该工具调用。"


def parse_transcript(
    transcript_path: Path,
    default_system: str = "",
) -> Tuple[List[Dict[str, str]], List[str], int]:
    """
    Convert Antigravity transcript JSONL into ShareGPT conversations list.
    Returns (conversations, used_tool_names, completed_user_turns).
    """
    if not transcript_path.exists():
        fallback = transcript_path.parent / "transcript.jsonl"
        if fallback.exists():
            transcript_path = fallback
        else:
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    steps: List[Dict[str, Any]] = []
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

    conversations: List[Dict[str, str]] = []
    used_tools: List[str] = []
    user_turns = 0

    i = 0
    n = len(steps)
    while i < n:
        step = steps[i]
        stype = step.get("type", "")
        source = step.get("source", "")

        # Detect user or parent-agent follow-up input
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
            thinks, body = split_think_and_body(raw_content)
            tool_calls = step.get("tool_calls") or []

            if tool_calls:
                # Look ahead for corresponding GENERIC tool result steps
                obs_steps: List[str] = []
                j = i + 1
                while j < n and len(obs_steps) < len(tool_calls):
                    next_step = steps[j]
                    if next_step.get("type") == "GENERIC":
                        obs_steps.append(str(next_step.get("content") or ""))
                        j += 1
                    else:
                        break

                for idx, tc in enumerate(tool_calls):
                    t_name = str(tc.get("name") or "unknown_tool")
                    t_args = tc.get("args")
                    if isinstance(t_args, str):
                        try:
                            t_args = json.loads(t_args)
                        except Exception:
                            t_args = {"raw_args": t_args}
                    elif not isinstance(t_args, dict):
                        t_args = {}

                    if t_name not in used_tools:
                        used_tools.append(t_name)

                    # Use explicit <think> from content if present; otherwise check subsequent final response or synthesize concise summary
                    if idx < len(thinks) and thinks[idx]:
                        pre_think = thinks[idx]
                    else:
                        pre_think = synthesize_concise_pre_tool_think(t_name, t_args)

                    conversations.append(
                        {"from": "gpt", "value": f"<think>\n{pre_think}\n</think>\n"}
                    )
                    conversations.append(
                        {
                            "from": "function_call",
                            "value": json.dumps(
                                {"name": t_name, "arguments": t_args},
                                ensure_ascii=False,
                            ),
                        }
                    )
                    obs_text = obs_steps[idx] if idx < len(obs_steps) else "Tool execution completed."
                    conversations.append(
                        {
                            "from": "observation",
                            "value": json.dumps(
                                {"result": obs_text},
                                ensure_ascii=False,
                            ),
                        }
                    )

                # If the same step also contained a final answer body after tool calls (rare)
                if body:
                    post_think = thinks[-1] if thinks else "结合工具返回的信息整理得出最终回答。"
                    conversations.append(
                        {"from": "gpt", "value": f"<think>\n{post_think}\n</think>\n{body}"}
                    )

                i = j
                continue
            else:
                # Pure response or post-tool final response step
                if raw_content:
                    if thinks:
                        think_text = thinks[-1]
                        final_body = body
                    else:
                        # No <think> tag found in final response -> preserve raw_content so auditor catches MISSING_THINK_IN_GPT
                        conversations.append({"from": "gpt", "value": raw_content})
                        i += 1
                        continue

                    conversations.append(
                        {
                            "from": "gpt",
                            "value": f"<think>\n{think_text}\n</think>\n{final_body}",
                        }
                    )
                i += 1
                continue

        i += 1

    return conversations, used_tools, user_turns


def audit_conversations(
    conversations: List[Dict[str, str]],
    expected_turns: int = 1,
) -> Tuple[bool, List[str]]:
    """
    Validate ShareGPT conversation sequence, <think> coverage, and tool call JSON validity.
    Returns (audit_passed, list_of_error_codes).
    """
    errors: List[str] = []
    if not conversations:
        return False, ["EMPTY_CONVERSATION"]

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
                if next_role == "function_call":
                    errors.append(f"MISSING_THINK_BEFORE_TOOL(step={idx})")
                else:
                    errors.append(f"MISSING_THINK_IN_GPT(step={idx})")
            else:
                for t in thinks:
                    if len(t.strip()) < 10:
                        errors.append(f"THINK_TOO_SHORT(step={idx})")

            if next_role != "function_call":
                if not body.strip():
                    errors.append(f"EMPTY_FINAL_ANSWER(step={idx})")
                low_body = body.lower()
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
            except Exception:
                errors.append(f"INVALID_OBSERVATION_JSON(step={idx})")

    if conversations[-1].get("from") != "gpt":
        errors.append("CONVERSATION_MUST_END_WITH_GPT")

    return len(errors) == 0, errors


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


def cmd_extract(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript).resolve()
    conversations, used_tools, user_turns = parse_transcript(
        transcript_path, default_system=args.system
    )
    audit_passed, audit_errors = audit_conversations(
        conversations, expected_turns=args.expected_turns
    )
    tools_str = build_tools_json_string(used_tools)

    payload = {
        "sample_id": args.sample_id,
        "audit_passed": audit_passed,
        "audit_errors": audit_errors,
        "completed_turns": user_turns,
        "expected_turns": args.expected_turns,
        "used_tools": used_tools,
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


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        mock_transcript = tmp / "transcript_full.jsonl"
        steps = [
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "[Task Execution]\n【用户第 1 轮提问】：\n请读取 plugin.json 并告诉我名称。",
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "<think>\n需要读取当前目录下的 plugin.json 文件以查看插件名称。我将调用 read_file 工具读取该文件。\n</think>",
                "tool_calls": [{"name": "read_file", "args": {"Path": "f:/zl/plugin.json"}}],
            },
            {
                "step_index": 2,
                "source": "MODEL",
                "type": "GENERIC",
                "status": "DONE",
                "content": '{"name": "agent-distiller", "version": "1.0.0"}',
            },
            {
                "step_index": 3,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "<think>\n根据 read_file 返回的 JSON 内容，插件的 name 字段为 agent-distiller，版本号为 1.0.0。下面直接整理回答用户。\n</think>\n该插件的名称是 `agent-distiller`，版本为 `1.0.0`。",
            },
            {
                "step_index": 4,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "【用户第 2 轮追问】：\n它的版本号符合语义化版本规范吗？\n（提醒：若需调用工具，请先输出 <think>...</think>）",
            },
            {
                "step_index": 5,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "<think>\n版本号 1.0.0 由主版本号、次版本号和修订号三部分组成，符合 SemVer 2.0.0 规范。\n</think>\n是的，`1.0.0` 严格遵循 MAJOR.MINOR.PATCH 的语义化版本（SemVer）规范。",
            },
        ]
        mock_transcript.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in steps) + "\n",
            encoding="utf-8",
        )
        out_json = tmp / "extracted.json"
        cmd_extract(
            argparse.Namespace(
                transcript=str(mock_transcript),
                sample_id="test_001",
                expected_turns=2,
                system="",
                output=str(out_json),
            )
        )
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["audit_passed"] is True, f"Audit failed: {data['audit_errors']}"
        convs = data["sharegpt_record"]["conversations"]
        assert len(convs) == 7
        assert convs[0] == {"from": "human", "value": "请读取 plugin.json 并告诉我名称。"}
        assert convs[1]["from"] == "gpt" and "<think>" in convs[1]["value"]
        assert convs[2]["from"] == "function_call"
        assert convs[3]["from"] == "observation"
        assert convs[4]["from"] == "gpt" and "agent-distiller" in convs[4]["value"]
        assert convs[5] == {"from": "human", "value": "它的版本号符合语义化版本规范吗？"}
        assert convs[6]["from"] == "gpt" and "SemVer" in convs[6]["value"]

        # Test negative audit case (missing <think>)
        bad_convs = [
            {"from": "human", "value": "你好"},
            {"from": "gpt", "value": "未包含思考标签的直接回复"},
        ]
        passed, errs = audit_conversations(bad_convs, expected_turns=1)
        assert passed is False and any("MISSING_THINK_IN_GPT" in e for e in errs)

    print(json.dumps({"self_test": "PASSED", "module": "trajectory_extractor.py"}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and audit ShareGPT trajectories from Antigravity subagent transcripts."
    )
    parser.add_argument("--self-test", action="store_true", help="Run built-in unit tests.")
    subparsers = parser.add_subparsers(dest="command")

    p_ext = subparsers.add_parser("extract", help="Extract and audit a subagent transcript.")
    p_ext.add_argument("--transcript", required=True, help="Path to transcript_full.jsonl or transcript.jsonl.")
    p_ext.add_argument("--sample-id", required=True, help="Sample identifier.")
    p_ext.add_argument("--expected-turns", type=int, default=1, help="Minimum expected human turns.")
    p_ext.add_argument("--system", default="", help="Optional custom system prompt for ShareGPT output.")
    p_ext.add_argument("--output", required=True, help="Output path for extracted JSON payload.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    if args.command == "extract":
        return cmd_extract(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
