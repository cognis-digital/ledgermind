# Scenario: 30-day SaaS startup LLM bill

800 LLM calls over 30 days across 5 features and 50 users. Realistic distribution: most users light, ~5 power users, mix of models.

## Expected findings

- LM-ANOM-002 (top user > 40% spend) — depending on draw
- LM-ANOM-003 (top feature > 60% spend) — depending on draw

## Why this matters

First thing any CFO asks: which feature is eating the OpenAI bill?
