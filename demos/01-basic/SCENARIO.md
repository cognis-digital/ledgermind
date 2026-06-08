# Demo 01 - Basic cost audit & anomaly detection

## Scenario

You run a small internal LLM gateway that logs every request as a line of
JSON (the shape most proxies, including litellm, can emit). At month-end you
want to answer three questions without uploading anything to a vendor
dashboard:

1. **How much did we spend, by model and by API key?**
2. **Which keys are the cost drivers?**
3. **Are there any anomalous calls** (a single runaway request, a key being
   abused, a model mispriced)?

`requests.jsonl` contains 12 real-looking gateway events. Eleven are normal
chat/RAG traffic across `gpt-4o-mini`, `claude-3-5-sonnet`, and
`gpt-4o`. One call (`key=svc-batch-77`) is an outlier: a single
`claude-3-opus` request that consumed ~120K prompt tokens and generated
~40K completion tokens — the kind of runaway/abuse event that quietly
blows a budget.

## Run it

```bash
python -m ledgermind audit demos/01-basic/requests.jsonl
```

JSON output (for piping into other tooling / CI):

```bash
python -m ledgermind audit demos/01-basic/requests.jsonl --format json
```

Fail a CI job if any anomaly is detected:

```bash
python -m ledgermind audit demos/01-basic/requests.jsonl --fail-on-anomaly
```

## What you should see

- A by-model and by-key cost breakdown, sorted by spend.
- The `claude-3-opus` call flagged as an anomaly with a high modified
  z-score on cost (it dominates total spend despite being 1 of 12 calls).
- Total spend reconciling to the sum of per-call costs.

## Why it matters

Anomaly detection uses the **median absolute deviation (MAD)** modified
z-score, which is robust to exactly the kind of heavy-tailed spend
distribution you get from LLM traffic — a couple of huge calls won't poison
the baseline the way a mean/stddev detector would.
