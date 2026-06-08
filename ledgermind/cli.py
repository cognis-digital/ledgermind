"""Command-line interface for LEDGERMIND.

Usage:
    python -m ledgermind audit demos/01-basic/requests.jsonl
    python -m ledgermind audit logs.jsonl --format json
    python -m ledgermind audit logs.jsonl --pricing custom_pricing.json
    python -m ledgermind --version
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import TOOL_NAME, TOOL_VERSION, PRICING
from .core import load_events, build_report


def _load_pricing(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("pricing file must be a JSON object")
    merged = dict(PRICING)
    merged.update(data)
    if "_default" not in merged:
        merged["_default"] = PRICING["_default"]
    return merged


def _fmt_usd(v: float) -> str:
    return f"${v:,.4f}"


def _render_table(report) -> str:
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"LEDGERMIND audit  (v{TOOL_VERSION})")
    lines.append("=" * 60)
    lines.append(f"Total calls   : {report.total_calls}")
    lines.append(f"Total tokens  : {report.total_tokens:,}")
    lines.append(f"Total cost    : {_fmt_usd(report.total_cost_usd)}")
    lines.append(f"Anomalies     : {report.anomaly_count}")
    lines.append("")

    lines.append("-- Cost by model " + "-" * 42)
    lines.append(f"{'model':<24}{'calls':>7}{'tokens':>12}{'cost':>14}")
    for model, b in sorted(
        report.by_model.items(), key=lambda kv: kv[1]["cost_usd"], reverse=True
    ):
        lines.append(
            f"{model[:24]:<24}{int(b['calls']):>7}{int(b['total_tokens']):>12,}"
            f"{_fmt_usd(b['cost_usd']):>14}"
        )
    lines.append("")

    lines.append("-- Cost by API key " + "-" * 40)
    lines.append(f"{'key':<24}{'calls':>7}{'tokens':>12}{'cost':>14}")
    for key, b in sorted(
        report.by_key.items(), key=lambda kv: kv[1]["cost_usd"], reverse=True
    ):
        lines.append(
            f"{key[:24]:<24}{int(b['calls']):>7}{int(b['total_tokens']):>12,}"
            f"{_fmt_usd(b['cost_usd']):>14}"
        )

    if report.anomalies:
        lines.append("")
        lines.append("-- Anomalies " + "!" * 46)
        for a in report.anomalies:
            lines.append(
                f"  #{a['index']} {a['model']} key={a['api_key']} "
                f"{_fmt_usd(a['cost_usd'])}"
            )
            for reason in a["anomaly_reasons"]:
                lines.append(f"      - {reason}")

    if report.warnings:
        lines.append("")
        lines.append("-- Warnings " + "." * 47)
        for w in report.warnings:
            lines.append(f"  ! {w}")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Local LLM cost & token forensics with anomaly detection.",
    )
    parser.add_argument(
        "--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}"
    )
    sub = parser.add_subparsers(dest="command")

    audit = sub.add_parser(
        "audit", help="Audit an LLM request log (JSONL or JSON array)."
    )
    audit.add_argument("logfile", help="Path to JSONL/JSON request log.")
    audit.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )
    audit.add_argument(
        "--pricing",
        default=None,
        help="Optional pricing override JSON (merged over defaults).",
    )
    audit.add_argument(
        "--mad-threshold",
        type=float,
        default=3.5,
        help="Modified z-score threshold for anomaly flagging (default 3.5).",
    )
    audit.add_argument(
        "--fail-on-anomaly",
        action="store_true",
        help="Exit non-zero (2) if any anomaly is detected.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "audit":
        parser.print_help(sys.stderr)
        return 1

    pricing = None
    try:
        if args.pricing:
            pricing = _load_pricing(args.pricing)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: failed to load pricing file: {exc}", file=sys.stderr)
        return 1

    try:
        events, warnings = load_events(args.logfile)
    except OSError as exc:
        print(f"error: cannot read log file: {exc}", file=sys.stderr)
        return 1

    if not events:
        print("error: no valid events found in log", file=sys.stderr)
        for w in warnings:
            print(f"  ! {w}", file=sys.stderr)
        return 1

    report = build_report(
        events,
        pricing=pricing,
        mad_threshold=args.mad_threshold,
        extra_warnings=warnings,
    )

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(_render_table(report))

    if args.fail_on_anomaly and report.anomaly_count > 0:
        return 2
    return 0
