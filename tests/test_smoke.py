"""Smoke + unit tests for LEDGERMIND. No network. Standard library only."""
import json
import os
import tempfile
import unittest

from ledgermind import (
    TOOL_NAME,
    TOOL_VERSION,
    PRICING,
    price_call,
    load_events,
    build_report,
    detect_anomalies,
)
from ledgermind.cli import main

DEMO = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "demos", "01-basic", "requests.jsonl"
)


class TestMetadata(unittest.TestCase):
    def test_tool_identity(self):
        self.assertEqual(TOOL_NAME, "ledgermind")
        self.assertTrue(TOOL_VERSION)


class TestPricing(unittest.TestCase):
    def test_known_model_cost(self):
        # gpt-4o: 1000 in @ 0.005, 1000 out @ 0.015 = 0.005 + 0.015 = 0.02
        c = price_call(
            {"model": "gpt-4o", "prompt_tokens": 1000, "completion_tokens": 1000}, 0
        )
        self.assertAlmostEqual(c.cost_usd, 0.02, places=6)
        self.assertFalse(c.is_fallback_price)
        self.assertEqual(c.total_tokens, 2000)

    def test_unknown_model_uses_fallback(self):
        c = price_call(
            {"model": "some-random-model", "prompt_tokens": 1000, "completion_tokens": 0},
            0,
        )
        self.assertTrue(c.is_fallback_price)
        self.assertEqual(c.priced_model, "_default")
        self.assertAlmostEqual(c.cost_usd, PRICING["_default"]["input"], places=6)

    def test_usage_nested_fields(self):
        c = price_call(
            {"model": "claude-3-5-sonnet", "usage": {"input_tokens": 2000, "output_tokens": 1000}},
            0,
        )
        self.assertEqual(c.prompt_tokens, 2000)
        self.assertEqual(c.completion_tokens, 1000)

    def test_garbage_tokens_coerced(self):
        c = price_call(
            {"model": "gpt-4o", "prompt_tokens": "oops", "completion_tokens": -5}, 0
        )
        self.assertEqual(c.prompt_tokens, 0)
        self.assertEqual(c.completion_tokens, 0)


class TestLoad(unittest.TestCase):
    def test_load_jsonl_skips_malformed(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write('{"model": "gpt-4o", "prompt_tokens": 10}\n')
            fh.write('not json at all\n')
            fh.write('{"model": "gpt-4o-mini", "prompt_tokens": 5}\n')
            path = fh.name
        try:
            events, warnings = load_events(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(len(warnings), 1)
        finally:
            os.unlink(path)

    def test_load_json_array(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump([{"model": "gpt-4o", "prompt_tokens": 1}], fh)
            path = fh.name
        try:
            events, warnings = load_events(path)
            self.assertEqual(len(events), 1)
        finally:
            os.unlink(path)


class TestAnomaly(unittest.TestCase):
    def test_detects_cost_outlier(self):
        events = [
            {"model": "gpt-4o-mini", "prompt_tokens": 100, "completion_tokens": 50}
            for _ in range(10)
        ]
        events.append(
            {"model": "claude-3-opus", "prompt_tokens": 120000, "completion_tokens": 40000}
        )
        report = build_report(events)
        self.assertGreaterEqual(report.anomaly_count, 1)
        self.assertTrue(any("opus" in a["model"] for a in report.anomalies))

    def test_no_anomaly_when_too_few_samples(self):
        calls = [
            price_call({"model": "gpt-4o", "prompt_tokens": 10, "completion_tokens": 5}, i)
            for i in range(3)
        ]
        self.assertEqual(detect_anomalies(calls), 0)


class TestReportAndCLI(unittest.TestCase):
    def test_report_totals_reconcile(self):
        events, warnings = load_events(DEMO)
        report = build_report(events, extra_warnings=warnings)
        self.assertEqual(report.total_calls, 12)
        summed = sum(b["cost_usd"] for b in report.by_model.values())
        self.assertAlmostEqual(report.total_cost_usd, summed, places=4)
        self.assertGreaterEqual(report.anomaly_count, 1)

    def test_cli_table_ok(self):
        self.assertEqual(main(["audit", DEMO]), 0)

    def test_cli_json_ok(self):
        self.assertEqual(main(["audit", DEMO, "--format", "json"]), 0)

    def test_cli_fail_on_anomaly(self):
        self.assertEqual(main(["audit", DEMO, "--fail-on-anomaly"]), 2)

    def test_cli_missing_file(self):
        self.assertEqual(main(["audit", "/nonexistent/path/xyz.jsonl"]), 1)

    def test_cli_no_command(self):
        self.assertEqual(main([]), 1)


if __name__ == "__main__":
    unittest.main()
