# Scenario: Production runaway loop incident

A bug shipped at 2am: an infinite-retry handler around an OpenAI call. 80 requests at 24K tokens each before the on-call killed it. $30 in 4 minutes.

## Expected findings

- LM-ANOM-001 (TOKEN_SPIKE_OUTLIERS)
- LM-ANOM-002 (user cost concentration on 'alice')

## Why this matters

This is the exact pattern Prompt-Layer and LangSmith customers describe in postmortems. LEDGERMIND catches it locally with no cloud dependency.
