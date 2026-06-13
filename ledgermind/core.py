"""Core forensics engine for LEDGERMIND.

No third-party dependencies. All pricing is per-1K-token USD and can be
overridden via a pricing JSON file. Unknown models fall back to a default
rate so accounting is never silently dropped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Per-1K-token USD pricing (input, output). Representative public rates.
# Override at runtime with --pricing pricing.json (same shape).
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.0050, "output": 0.0150},
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4-turbo": {"input": 0.0100, "output": 0.0300},
    "claude-3-5-sonnet": {"input": 0.0030, "output": 0.0150},
    "claude-3-opus": {"input": 0.0150, "output": 0.0750},
    "claude-3-haiku": {"input": 0.000250, "output": 0.00125},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.0050},
    "llama-3-70b": {"input": 0.000590, "output": 0.000790},
    "_default": {"input": 0.0010, "output": 0.0020},
}


@dataclass
class PricedCall:
    """A single LLM request after pricing has been applied."""

    index: int
    timestamp: Optional[str]
    model: str
    api_key: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    priced_model: str  # the pricing key actually used (may be _default)
    is_fallback_price: bool
    anomaly: bool = False
    anomaly_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cost_usd"] = round(self.cost_usd, 6)
        return d


@dataclass
class LedgerReport:
    total_calls: int
    total_cost_usd: float
    total_tokens: int
    by_model: Dict[str, Dict[str, float]]
    by_key: Dict[str, Dict[str, float]]
    anomaly_count: int
    anomalies: List[Dict[str, Any]]
    fallback_priced_calls: int
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "by_model": self.by_model,
            "by_key": self.by_key,
            "anomaly_count": self.anomaly_count,
            "anomalies": self.anomalies,
            "fallback_priced_calls": self.fallback_priced_calls,
            "warnings": self.warnings,
        }


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def price_call(
    event: Dict[str, Any],
    index: int,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> PricedCall:
    """Price a single raw event dict into a PricedCall.

    Accepts flexible token field names: prompt_tokens/input_tokens and
    completion_tokens/output_tokens, optionally nested under "usage".
    """
    pricing = pricing or PRICING
    usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}

    def pick(*keys: str) -> Any:
        for k in keys:
            if k in event and event[k] is not None:
                return event[k]
            if k in usage and usage[k] is not None:
                return usage[k]
        return None

    prompt = _coerce_int(pick("prompt_tokens", "input_tokens", "prompt"))
    completion = _coerce_int(pick("completion_tokens", "output_tokens", "completion"))
    total_field = pick("total_tokens")
    total = _coerce_int(total_field) if total_field is not None else prompt + completion
    if total < prompt + completion:
        total = prompt + completion

    model = str(event.get("model") or "unknown")
    api_key = str(event.get("api_key") or event.get("key") or event.get("user") or "unknown")

    priced_model = model if model in pricing else "_default"
    rates = pricing.get(priced_model, pricing.get("_default", {"input": 0.0, "output": 0.0}))
    cost = (prompt / 1000.0) * rates["input"] + (completion / 1000.0) * rates["output"]

    return PricedCall(
        index=index,
        timestamp=event.get("timestamp") or event.get("ts"),
        model=model,
        api_key=api_key,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cost_usd=cost,
        priced_model=priced_model,
        is_fallback_price=(priced_model == "_default"),
    )


def load_events(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load events from a JSONL file (one JSON object per line) or a JSON array.

    Returns (events, warnings). Malformed lines are skipped with a warning
    rather than aborting the whole run.
    """
    warnings: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()

    stripped = raw.lstrip()
    if stripped.startswith("["):
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("top-level JSON is not a list")
            return [e for e in data if isinstance(e, dict)], warnings
        except (json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"failed to parse JSON array: {exc}")
            return [], warnings

    events: List[Dict[str, Any]] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"line {lineno}: malformed JSON skipped ({exc.msg})")
            continue
        if isinstance(obj, dict):
            events.append(obj)
        else:
            warnings.append(f"line {lineno}: not a JSON object, skipped")
    return events, warnings


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def detect_anomalies(
    calls: List[PricedCall],
    mad_threshold: float = 3.5,
    min_samples: int = 5,
) -> int:
    """Flag anomalous calls using the robust modified z-score (MAD).

    A call is anomalous if its cost's modified z-score exceeds mad_threshold,
    or its completion-to-prompt token ratio is a severe outlier (runaway
    generation). Mutates the PricedCall objects in place. Returns the count.
    """
    if len(calls) < min_samples:
        return 0

    costs = [c.cost_usd for c in calls]
    med = _median(costs)
    abs_dev = [abs(c - med) for c in costs]
    mad = _median(abs_dev)

    # Token ratio outliers (completion >> prompt suggests runaway output).
    ratios = [
        c.completion_tokens / c.prompt_tokens
        for c in calls
        if c.prompt_tokens > 0
    ]
    ratio_med = _median(ratios)
    ratio_abs_dev = [abs(r - ratio_med) for r in ratios]
    ratio_mad = _median(ratio_abs_dev)

    count = 0
    for c in calls:
        reasons: List[str] = []
        if mad > 0:
            mz = 0.6745 * (c.cost_usd - med) / mad
            if mz > mad_threshold:
                reasons.append(f"cost modified-z {mz:.2f} > {mad_threshold}")
        elif med > 0 and c.cost_usd > med * 4:
            # Degenerate spread: fall back to multiplicative rule.
            reasons.append(f"cost {c.cost_usd:.4f} > 4x median {med:.4f}")

        if c.prompt_tokens > 0 and ratio_mad > 0:
            r = c.completion_tokens / c.prompt_tokens
            rmz = 0.6745 * (r - ratio_med) / ratio_mad
            if rmz > mad_threshold and c.completion_tokens > 256:
                reasons.append(
                    f"runaway output ratio {r:.1f} (z {rmz:.2f})"
                )

        if reasons:
            c.anomaly = True
            c.anomaly_reasons = reasons
            count += 1
    return count


def _accumulate(target: Dict[str, Dict[str, float]], key: str, c: PricedCall) -> None:
    bucket = target.setdefault(
        key, {"calls": 0, "cost_usd": 0.0, "total_tokens": 0}
    )
    bucket["calls"] += 1
    bucket["cost_usd"] += c.cost_usd
    bucket["total_tokens"] += c.total_tokens


def build_report(
    events: Iterable[Dict[str, Any]],
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
    mad_threshold: float = 3.5,
    extra_warnings: Optional[List[str]] = None,
) -> LedgerReport:
    """Price all events, run anomaly detection, and aggregate a LedgerReport."""
    pricing = pricing or PRICING
    warnings = list(extra_warnings or [])

    calls = [price_call(e, i, pricing) for i, e in enumerate(events)]
    anomaly_count = detect_anomalies(calls, mad_threshold=mad_threshold)

    by_model: Dict[str, Dict[str, float]] = {}
    by_key: Dict[str, Dict[str, float]] = {}
    total_cost = 0.0
    total_tokens = 0
    fallback = 0

    for c in calls:
        _accumulate(by_model, c.model, c)
        _accumulate(by_key, c.api_key, c)
        total_cost += c.cost_usd
        total_tokens += c.total_tokens
        if c.is_fallback_price:
            fallback += 1

    for bucket in list(by_model.values()) + list(by_key.values()):
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)

    if fallback:
        warnings.append(
            f"{fallback} call(s) priced at fallback _default rate (unknown model)"
        )

    anomalies = [c.to_dict() for c in calls if c.anomaly]

    return LedgerReport(
        total_calls=len(calls),
        total_cost_usd=total_cost,
        total_tokens=total_tokens,
        by_model=by_model,
        by_key=by_key,
        anomaly_count=anomaly_count,
        anomalies=anomalies,
        fallback_priced_calls=fallback,
        warnings=warnings,
    )
