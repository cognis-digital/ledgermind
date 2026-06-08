"""LEDGERMIND — Local LLM cost & token forensics."""
from __future__ import annotations
import json, sqlite3, time, statistics
from pathlib import Path
from cognis_core import Finding, ScanResult, score

TOOL_NAME = "LEDGERMIND"
TOOL_VERSION = "0.1.0"

# Price-per-1k-tokens snapshot (input/output). Update as needed.
PRICING = {
    "gpt-4o":          (0.0025, 0.01),
    "gpt-4o-mini":     (0.00015, 0.0006),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-haiku":   (0.00025, 0.00125),
    "llama-3-70b":     (0.0007, 0.0009),
    "gemini-1.5-pro":  (0.00125, 0.005),
}

def _cost(model: str, in_tok: int, out_tok: int) -> float:
    p_in, p_out = PRICING.get(model, (0.001, 0.002))
    return (in_tok / 1000) * p_in + (out_tok / 1000) * p_out

def _load_log(path: Path) -> list[dict]:
    if path.is_file() and path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if path.is_file() and path.suffix == ".db":
        with sqlite3.connect(path) as c:
            try:
                rows = c.execute("SELECT model, input_tokens, output_tokens, feature, user_id, ts FROM calls").fetchall()
                return [dict(model=r[0], input_tokens=r[1], output_tokens=r[2], feature=r[3], user_id=r[4], ts=r[5]) for r in rows]
            except Exception:
                return []
    return []

def scan(target: str, **opts) -> ScanResult:
    t0 = time.time()
    result = ScanResult(tool_name=TOOL_NAME, tool_version=TOOL_VERSION, target=str(target))
    rows = _load_log(Path(target))
    result.items_scanned = len(rows)
    if not rows:
        return result

    total_cost = 0.0
    by_user: dict[str, float] = {}
    by_feature: dict[str, float] = {}
    sizes: list[int] = []
    for r in rows:
        c = _cost(r.get("model","gpt-4o-mini"), int(r.get("input_tokens",0)), int(r.get("output_tokens",0)))
        total_cost += c
        by_user[r.get("user_id","?")] = by_user.get(r.get("user_id","?"),0) + c
        by_feature[r.get("feature","?")] = by_feature.get(r.get("feature","?"),0) + c
        sizes.append(int(r.get("input_tokens",0)) + int(r.get("output_tokens",0)))

    result.metadata.update(total_cost_usd=round(total_cost,4),
                           top_user=max(by_user, key=by_user.get) if by_user else None,
                           top_feature=max(by_feature, key=by_feature.get) if by_feature else None,
                           call_count=len(rows))

    # Anomaly: runaway loop / repeated identical large requests
    if sizes:
        mean = statistics.mean(sizes); stdev = statistics.stdev(sizes) if len(sizes)>1 else 0
        outliers = [s for s in sizes if stdev and s > mean + 3*stdev]
        if outliers:
            result.add(Finding(
                id="LM-ANOM-001", severity="high", weight=2.5,
                title="TOKEN_SPIKE_OUTLIERS",
                description=f"{len(outliers)} requests >3σ above mean ({mean:.0f}±{stdev:.0f} tokens). Possible runaway loop.",
                location=str(target), category="cost-anomaly",
                remediation="Add max_tokens cap and per-conversation token budget enforcement.",
            ))
    # Anomaly: one user > 40% of spend
    if by_user:
        top_user, top_spend = max(by_user.items(), key=lambda kv: kv[1])
        if total_cost and top_spend/total_cost > 0.4:
            result.add(Finding(
                id="LM-ANOM-002", severity="medium", weight=2.0,
                title="USER_COST_CONCENTRATION",
                description=f"User {top_user} accounts for {top_spend/total_cost:.0%} of total spend ${total_cost:.2f}.",
                location=str(target), category="cost-anomaly",
                remediation="Verify legitimate usage; consider per-user rate limit.",
            ))
    # Anomaly: feature spend > 60%
    if by_feature:
        top_feat, top_fspend = max(by_feature.items(), key=lambda kv: kv[1])
        if total_cost and top_fspend/total_cost > 0.6:
            result.add(Finding(
                id="LM-ANOM-003", severity="low", weight=1.5,
                title="FEATURE_COST_CONCENTRATION",
                description=f"Feature `{top_feat}` accounts for {top_fspend/total_cost:.0%} of spend.",
                location=str(target), category="cost-anomaly",
                remediation="Consider cheaper model or prompt caching for this feature.",
            ))

    result.composite_score, result.risk_level = score(result.findings)
    result.scan_duration_ms = int((time.time()-t0)*1000)
    return result
