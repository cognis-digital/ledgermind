# Scenario: Multi-tenant SaaS where one customer is abusing the API

30 customers normally; one customer ('user-013') is shovelling 6K-token chats into GPT-4o on a $20/mo plan, putting the unit economics underwater.

## Expected findings

- LM-ANOM-002 (user cost concentration)
- LM-ANOM-001 (token-size spikes from user-013)

## Why this matters

Without LEDGERMIND, you discover this when the OpenAI bill arrives. With it, you see it the same hour.
