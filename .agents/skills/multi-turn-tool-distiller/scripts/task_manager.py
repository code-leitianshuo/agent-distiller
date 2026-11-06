#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Queue & Checkpoint State Manager for multi-turn-tool-distiller.

Supports:
- Streaming seed tasks from .jsonl, .json, .csv, or .txt files.
- Maintaining checkpoint state in <output_dir>/.state.json for resume support.
- Selecting prompt templates (Template A / B / C) according to retry_count.
- Batch scheduling for both serial (batch_size=1) and parallel (batch_size=N) modes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


TEMPLATE_MAP = {
    0: "Template A",
    1: "Template B",
    2: "Template C",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_sample_id(raw_text: str, index: int) -> str:
    digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:10]
    return f"sample_{index:04d}_{digest}"


def parse_seed_file(input_path: Path, default_turns: int = 2) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Seed input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    records: List[Dict[str, Any]] = []

    if suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(_normalize_seed_item(obj, idx, default_turns))
    elif suffix == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "samples" in data:
                data = data["samples"]
            if not isinstance(data, list):
                raise ValueError("JSON seed file must contain a list of objects or {'samples': [...]}")
            for idx, obj in enumerate(data, start=1):
                records.append(_normalize_seed_item(obj, idx, default_turns))
    elif suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                records.append(_normalize_seed_item(dict(row), idx, default_turns))
    elif suffix == ".txt":
        with input_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                records.append(
                    _normalize_seed_item({"question": text}, idx, default_turns)
                )
    else:
        raise ValueError(f"Unsupported seed file extension: {suffix}")

    return records


def _normalize_seed_item(
    item: Any, index: int, default_turns: int
) -> Dict[str, Any]:
    if isinstance(item, str):
        item = {"question": item}
    if not isinstance(item, dict):
        raise ValueError(f"Seed item at index {index} must be dict or str.")

    question = (
        item.get("question")
        or item.get("prompt")
        or item.get("instruction")
        or ""
    ).strip()
    follow_ups = item.get("follow_ups") or item.get("turns") or []
    if isinstance(follow_ups, str):
        try:
            parsed = json.loads(follow_ups)
            follow_ups = parsed if isinstance(parsed, list) else [follow_ups]
        except Exception:
            follow_ups = [s.strip() for s in follow_ups.split("||") if s.strip()]

    if not question and follow_ups:
        question = str(follow_ups[0])
        follow_ups = list(follow_ups[1:])

    if not question:
        raise ValueError(f"Seed item at index {index} is missing 'question'/'prompt'.")

    raw_key = json.dumps(
        {"question": question, "follow_ups": follow_ups},
        ensure_ascii=False,
        sort_keys=True,
    )
    sample_id = str(item.get("id") or item.get("sample_id") or compute_sample_id(raw_key, index))
    target_turns = int(item.get("target_turns") or (1 + len(follow_ups) if follow_ups else default_turns))

    return {
        "sample_id": sample_id,
        "domain": str(item.get("domain") or "general"),
        "system": str(item.get("system") or ""),
        "tools": item.get("tools") or [],
        "question": question,
        "follow_ups": [str(x) for x in follow_ups],
        "target_turns": max(1, target_turns),
    }


def load_state(state_path: Path) -> Dict[str, Any]:
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "max_retries": 3,
        "samples": {},
    }


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    tmp_path = state_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, state_path)


def cmd_init(args: argparse.Namespace) -> int:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    state_path = output_dir / ".state.json"

    seeds = parse_seed_file(input_path, default_turns=args.default_turns)
    state = load_state(state_path)
    state["max_retries"] = args.max_retries
    state["input_source"] = str(input_path)

    existing_samples: Dict[str, Any] = state.setdefault("samples", {})
    added = 0
    skipped_completed = 0

    for seed in seeds:
        sid = seed["sample_id"]
        if sid in existing_samples:
            if existing_samples[sid].get("status") == "completed":
                skipped_completed += 1
            continue
        existing_samples[sid] = {
            **seed,
            "status": "pending",
            "retry_count": 0,
            "recommended_template": TEMPLATE_MAP[0],
            "last_diagnosis": "",
            "subagent_history": [],
            "updated_at": now_iso(),
        }
        added += 1

    save_state(state_path, state)
    summary = _build_summary(state)
    summary["newly_added"] = added
    summary["resumed_completed"] = skipped_completed
    summary["state_file"] = str(state_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_next_batch(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    state_path = output_dir / ".state.json"
    state = load_state(state_path)
    max_retries = int(state.get("max_retries", 3))

    batch: List[Dict[str, Any]] = []
    for sid, entry in state.get("samples", {}).items():
        status = entry.get("status", "pending")
        retry_count = int(entry.get("retry_count", 0))
        if status in ("pending", "retry") and retry_count < max_retries:
            template = TEMPLATE_MAP.get(retry_count, "Template C")
            entry["recommended_template"] = template
            batch.append(entry)
            if len(batch) >= args.batch_size:
                break

    result = {
        "batch_size": len(batch),
        "remaining_actionable": sum(
            1
            for e in state.get("samples", {}).values()
            if e.get("status") in ("pending", "retry")
            and int(e.get("retry_count", 0)) < max_retries
        ),
        "batch": batch,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_update_status(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    state_path = output_dir / ".state.json"
    state = load_state(state_path)
    samples = state.get("samples", {})

    if args.sample_id not in samples:
        print(
            json.dumps({"error": f"Sample ID '{args.sample_id}' not found in .state.json"}),
            file=sys.stderr,
        )
        return 1

    entry = samples[args.sample_id]
    max_retries = int(state.get("max_retries", 3))

    if args.subagent_id:
        history = entry.setdefault("subagent_history", [])
        if args.subagent_id not in history:
            history.append(args.subagent_id)

    if args.diagnosis is not None:
        entry["last_diagnosis"] = args.diagnosis

    if args.status == "retry":
        entry["retry_count"] = int(entry.get("retry_count", 0)) + 1
        if entry["retry_count"] >= max_retries:
            entry["status"] = "failed"
        else:
            entry["status"] = "retry"
            entry["recommended_template"] = TEMPLATE_MAP.get(
                entry["retry_count"], "Template C"
            )
    else:
        entry["status"] = args.status

    entry["updated_at"] = now_iso()
    save_state(state_path, state)
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    state_path = output_dir / ".state.json"
    state = load_state(state_path)
    print(json.dumps(_build_summary(state), ensure_ascii=False, indent=2))
    return 0


def _build_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    samples = state.get("samples", {})
    counts = {"total": len(samples), "pending": 0, "in_progress": 0, "retry": 0, "completed": 0, "failed": 0}
    for item in samples.values():
        st = item.get("status", "pending")
        counts[st] = counts.get(st, 0) + 1
    return counts


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        seed_file = tmp / "seeds.jsonl"
        seed_file.write_text(
            '{"id": "t1", "question": "Q1", "follow_ups": ["F1"]}\n'
            '{"id": "t2", "question": "Q2", "target_turns": 3}\n',
            encoding="utf-8",
        )
        out_dir = tmp / "out"

        # 1. Init
        cmd_init(
            argparse.Namespace(
                input=str(seed_file),
                output_dir=str(out_dir),
                default_turns=2,
                max_retries=3,
            )
        )
        state = load_state(out_dir / ".state.json")
        assert len(state["samples"]) == 2
        assert state["samples"]["t1"]["target_turns"] == 2
        assert state["samples"]["t2"]["target_turns"] == 3

        # 2. Retry escalation
        cmd_update_status(
            argparse.Namespace(
                output_dir=str(out_dir),
                sample_id="t1",
                status="retry",
                diagnosis="MISSING_THINK_BEFORE_TOOL",
                subagent_id="sub_001",
            )
        )
        state = load_state(out_dir / ".state.json")
        assert state["samples"]["t1"]["retry_count"] == 1
        assert state["samples"]["t1"]["recommended_template"] == "Template B"

        # 3. Complete t2 and re-init to verify resume protection
        cmd_update_status(
            argparse.Namespace(
                output_dir=str(out_dir),
                sample_id="t2",
                status="completed",
                diagnosis="",
                subagent_id="sub_002",
            )
        )
        cmd_init(
            argparse.Namespace(
                input=str(seed_file),
                output_dir=str(out_dir),
                default_turns=2,
                max_retries=3,
            )
        )
        state = load_state(out_dir / ".state.json")
        assert state["samples"]["t2"]["status"] == "completed"

    print(json.dumps({"self_test": "PASSED", "module": "task_manager.py"}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage seed queues and checkpoint state for multi-turn-tool-distiller."
    )
    parser.add_argument("--self-test", action="store_true", help="Run built-in unit tests.")
    subparsers = parser.add_subparsers(dest="command")

    p_init = subparsers.add_parser("init", help="Initialize or resume queue from seed file.")
    p_init.add_argument("--input", required=True, help="Path to seed file (.jsonl/.json/.csv/.txt).")
    p_init.add_argument("--output-dir", default="datasets", help="Output directory for .state.json.")
    p_init.add_argument("--default-turns", type=int, default=2, help="Default target turns per sample.")
    p_init.add_argument("--max-retries", type=int, default=3, help="Max mutation retries per sample.")

    p_next = subparsers.add_parser("next-batch", help="Fetch next actionable batch of samples.")
    p_next.add_argument("--output-dir", default="datasets", help="Output directory containing .state.json.")
    p_next.add_argument("--batch-size", type=int, default=3, help="Number of samples to fetch.")

    p_upd = subparsers.add_parser("update-status", help="Update sample status and retry counter.")
    p_upd.add_argument("--output-dir", default="datasets", help="Output directory containing .state.json.")
    p_upd.add_argument("--sample-id", required=True, help="Target sample ID.")
    p_upd.add_argument(
        "--status",
        required=True,
        choices=["pending", "in_progress", "retry", "completed", "failed"],
        help="New status.",
    )
    p_upd.add_argument("--diagnosis", default=None, help="Audit diagnosis or error reason.")
    p_upd.add_argument("--subagent-id", default=None, help="Sub-agent conversation ID used.")

    p_stat = subparsers.add_parser("status", help="Show queue statistics.")
    p_stat.add_argument("--output-dir", default="datasets", help="Output directory containing .state.json.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    if args.command == "init":
        return cmd_init(args)
    if args.command == "next-batch":
        return cmd_next_batch(args)
    if args.command == "update-status":
        return cmd_update_status(args)
    if args.command == "status":
        return cmd_status(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
