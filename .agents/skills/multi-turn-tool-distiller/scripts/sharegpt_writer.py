#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure ShareGPT JSONL Writer & Failure Archiver for multi-turn-tool-distiller.

Guarantees that the main training dataset (.jsonl) strictly contains only the three
standard LLaMA-Factory ShareGPT keys ("system", "tools", "conversations"), while
syncing sample completion and failure metadata into .state.json and failed.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


ALLOWED_SHAREGPT_KEYS = ("system", "tools", "conversations")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(state_path: Path) -> Dict[str, Any]:
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"created_at": now_iso(), "updated_at": now_iso(), "max_retries": 3, "samples": {}}


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    tmp_path = state_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, state_path)


def validate_pure_sharegpt(record: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure strictly ("system", "tools", "conversations") keys and valid types."""
    system_val = str(record.get("system") or "")
    tools_val = record.get("tools", "[]")
    if not isinstance(tools_val, str):
        if not isinstance(tools_val, list):
            raise ValueError("'tools' field must be a list or a JSON string representing a list.")
        tools_val = json.dumps(tools_val, ensure_ascii=False)
    else:
        # Ensure it parses as JSON array
        parsed_tools = json.loads(tools_val)
        if not isinstance(parsed_tools, list):
            raise ValueError("'tools' field must be a JSON string representing a list.")

    convs = record.get("conversations")
    if not isinstance(convs, list) or not convs:
        raise ValueError("'conversations' must be a non-empty list.")

    for idx, turn in enumerate(convs):
        if not isinstance(turn, dict):
            raise ValueError(f"Conversation turn {idx} must be an object.")
        if turn.get("from") not in ("human", "gpt", "function_call", "observation"):
            raise ValueError(f"Invalid role '{turn.get('from')}' at turn {idx}.")
        if not isinstance(turn.get("value"), str):
            raise ValueError(f"Turn {idx} 'value' must be a string.")

    return {
        "system": system_val,
        "tools": tools_val,
        "conversations": convs,
    }


def cmd_commit(args: argparse.Namespace) -> int:
    extracted_path = Path(args.extracted).resolve()
    if not extracted_path.exists():
        print(json.dumps({"error": f"Extracted file not found: {extracted_path}"}), file=sys.stderr)
        return 1

    with extracted_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not payload.get("audit_passed", False) and not args.force:
        print(
            json.dumps(
                {
                    "error": "Cannot commit trajectory that failed quality audit.",
                    "audit_errors": payload.get("audit_errors", []),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1

    raw_record = payload.get("sharegpt_record") or payload
    pure_record = validate_pure_sharegpt(raw_record)

    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(pure_record, ensure_ascii=False) + "\n")

    sample_id = str(payload.get("sample_id") or args.sample_id or "unknown")
    state_dir = Path(args.state_dir).resolve()
    state_path = state_dir / ".state.json"
    state = load_state(state_path)
    samples = state.setdefault("samples", {})
    entry = samples.setdefault(sample_id, {"sample_id": sample_id})
    entry["status"] = "completed"
    entry["template_used"] = args.template_used
    entry["completed_at"] = now_iso()
    entry["output_file"] = str(output_file)
    entry["message_count"] = len(pure_record["conversations"])
    save_state(state_path, state)

    print(
        json.dumps(
            {
                "status": "committed",
                "sample_id": sample_id,
                "output_file": str(output_file),
                "keys_written": list(pure_record.keys()),
                "message_count": len(pure_record["conversations"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_fail(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    failed_file = state_dir / "failed.jsonl"
    state_path = state_dir / ".state.json"

    partial_data: Dict[str, Any] = {}
    if args.extracted:
        ext_path = Path(args.extracted).resolve()
        if ext_path.exists():
            try:
                partial_data = json.loads(ext_path.read_text(encoding="utf-8"))
            except Exception:
                partial_data = {}

    state = load_state(state_path)
    samples = state.setdefault("samples", {})
    entry = samples.setdefault(args.sample_id, {"sample_id": args.sample_id})
    entry["status"] = "failed"
    entry["last_diagnosis"] = args.reason
    entry["failed_at"] = now_iso()
    save_state(state_path, state)

    fail_record = {
        "sample_id": args.sample_id,
        "failed_at": entry["failed_at"],
        "reason": args.reason,
        "retry_count": entry.get("retry_count", 0),
        "audit_errors": partial_data.get("audit_errors", []),
        "partial_sharegpt": partial_data.get("sharegpt_record"),
    }
    with failed_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(fail_record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "status": "archived_to_failed",
                "sample_id": args.sample_id,
                "failed_file": str(failed_file),
                "reason": args.reason,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        extracted_file = tmp / "tmp_s1.json"
        extracted_payload = {
            "sample_id": "s1",
            "audit_passed": True,
            "audit_errors": [],
            "sharegpt_record": {
                "system": "System prompt",
                "tools": '[{"name": "read_file", "description": "Read file", "parameters": {"type": "object"}}]',
                "conversations": [
                    {"from": "human", "value": "读取配置"},
                    {"from": "gpt", "value": "<think>\n需要调用 read_file 读取配置。\n</think>\n"},
                    {"from": "function_call", "value": '{"name": "read_file", "arguments": {"Path": "a.json"}}'},
                    {"from": "observation", "value": '{"result": "{\\"ok\\": true}"}'},
                    {"from": "gpt", "value": "<think>\n读取成功，返回配置结果。\n</think>\n配置正常。"},
                ],
                "extra_unwanted_field": "should_be_stripped",
            },
        }
        extracted_file.write_text(json.dumps(extracted_payload, ensure_ascii=False), encoding="utf-8")

        out_jsonl = tmp / "sharegpt_train.jsonl"
        rc = cmd_commit(
            argparse.Namespace(
                extracted=str(extracted_file),
                output_file=str(out_jsonl),
                state_dir=str(tmp),
                sample_id="s1",
                template_used="Template A",
                force=False,
            )
        )
        assert rc == 0
        written_line = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
        assert tuple(written_line.keys()) == ALLOWED_SHAREGPT_KEYS

        rc_fail = cmd_fail(
            argparse.Namespace(
                sample_id="s2",
                reason="MAX_RETRIES_EXCEEDED: MISSING_THINK_IN_GPT",
                extracted=None,
                state_dir=str(tmp),
            )
        )
        assert rc_fail == 0
        assert (tmp / "failed.jsonl").exists()

    print(json.dumps({"self_test": "PASSED", "module": "sharegpt_writer.py"}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write validated ShareGPT records and manage failed.jsonl archives."
    )
    parser.add_argument("--self-test", action="store_true", help="Run built-in unit tests.")
    subparsers = parser.add_subparsers(dest="command")

    p_commit = subparsers.add_parser("commit", help="Append validated trajectory to ShareGPT JSONL.")
    p_commit.add_argument("--extracted", required=True, help="Path to extracted JSON payload.")
    p_commit.add_argument("--output-file", default="datasets/sharegpt_train.jsonl", help="Target ShareGPT JSONL.")
    p_commit.add_argument("--state-dir", default="datasets", help="Directory containing .state.json.")
    p_commit.add_argument("--sample-id", default="", help="Optional override for sample ID.")
    p_commit.add_argument("--template-used", default="Template A", help="Prompt template name used.")
    p_commit.add_argument("--force", action="store_true", help="Force commit even if audit_passed is false.")

    p_fail = subparsers.add_parser("fail", help="Archive failed sample into failed.jsonl.")
    p_fail.add_argument("--sample-id", required=True, help="Failed sample ID.")
    p_fail.add_argument("--reason", required=True, help="Failure diagnosis reason.")
    p_fail.add_argument("--extracted", default=None, help="Optional path to last extracted JSON payload.")
    p_fail.add_argument("--state-dir", default="datasets", help="Directory for failed.jsonl and .state.json.")

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
    if args.command == "commit":
        return cmd_commit(args)
    if args.command == "fail":
        return cmd_fail(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
