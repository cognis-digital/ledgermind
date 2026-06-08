"""LEDGERMIND - Local LLM cost & token forensics with anomaly detection.

A zero-dependency, standard-library-only engine that ingests LLM request
logs (JSONL), computes per-model / per-key cost & token accounting, and
flags anomalous spend using robust statistics (median absolute deviation).

Spiritual cousin of litellm's cost tracking, but offline-first and forensic:
it answers "who spent what, and which calls look suspicious?"
"""
from .core import (
    PRICING,
    PricedCall,
    LedgerReport,
    price_call,
    load_events,
    build_report,
    detect_anomalies,
)

TOOL_NAME = "ledgermind"
TOOL_VERSION = "1.0.0"

__all__ = [
    "TOOL_NAME",
    "TOOL_VERSION",
    "PRICING",
    "PricedCall",
    "LedgerReport",
    "price_call",
    "load_events",
    "build_report",
    "detect_anomalies",
]
