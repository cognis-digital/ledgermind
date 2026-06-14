"""Hardening tests: bad input, edge cases, and error-path coverage."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from ledgermind.core import (
    load_events,
    price_call,
    detect_anomalies,
    PRICING,
)
from ledgermind.cli import main, _load_pricing


# ---------------------------------------------------------------------------
# load_events edge cases
# ---------------------------------------------------------------------------

class TestLoadEventsEdgeCases(unittest.TestCase):
    def test_empty_path_raises_valueerror(self):
        with self.assertRaises(ValueError):
            load_events("")

    def test_whitespace_only_path_raises_valueerror(self):
        with self.assertRaises(ValueError):
            load_events("   ")

    def test_nonstring_path_raises_valueerror(self):
        with self.assertRaises(ValueError):
            load_events(None)  # type: ignore[arg-type]

    def test_empty_jsonl_file_returns_no_events(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            path = fh.name  # write nothing
        try:
            events, warnings = load_events(path)
            self.assertEqual(events, [])
            self.assertEqual(warnings, [])
        finally:
            os.unlink(path)

    def test_empty_json_array_returns_no_events(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("[]")
            path = fh.name
        try:
            events, warnings = load_events(path)
            self.assertEqual(events, [])
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises_oserror(self):
        with self.assertRaises(OSError):
            load_events("/nonexistent/path/does_not_exist_xyz.jsonl")


# ---------------------------------------------------------------------------
# price_call with malformed pricing dict
# ---------------------------------------------------------------------------

class TestPriceCallMalformedPricing(unittest.TestCase):
    def test_missing_output_key_does_not_raise(self):
        """A pricing entry missing 'output' should fall back to 0.0, not KeyError."""
        custom_pricing = dict(PRICING)
        custom_pricing["my-model"] = {"input": 0.005}  # no "output"
        c = price_call(
            {"model": "my-model", "prompt_tokens": 1000, "completion_tokens": 500},
            0,
            pricing=custom_pricing,
        )
        # Cost should be (1000/1000)*0.005 + (500/1000)*0.0 = 0.005
        self.assertAlmostEqual(c.cost_usd, 0.005, places=6)

    def test_missing_input_key_does_not_raise(self):
        """A pricing entry missing 'input' should fall back to 0.0, not KeyError."""
        custom_pricing = dict(PRICING)
        custom_pricing["my-model"] = {"output": 0.010}  # no "input"
        c = price_call(
            {"model": "my-model", "prompt_tokens": 1000, "completion_tokens": 500},
            0,
            pricing=custom_pricing,
        )
        # Cost = 0 + (500/1000)*0.010 = 0.005
        self.assertAlmostEqual(c.cost_usd, 0.005, places=6)

    def test_rates_not_a_dict_does_not_raise(self):
        """If a pricing entry is not a dict at all, cost should default to 0."""
        custom_pricing = dict(PRICING)
        custom_pricing["weird-model"] = "not-a-dict"  # type: ignore[assignment]
        c = price_call(
            {"model": "weird-model", "prompt_tokens": 500, "completion_tokens": 200},
            0,
            pricing=custom_pricing,
        )
        # rates is a non-dict so in_rate and out_rate both default to 0.0
        self.assertEqual(c.cost_usd, 0.0)


# ---------------------------------------------------------------------------
# detect_anomalies validation
# ---------------------------------------------------------------------------

class TestDetectAnomaliesValidation(unittest.TestCase):
    def _make_calls(self, n: int = 6):
        return [
            price_call(
                {"model": "gpt-4o", "prompt_tokens": 100, "completion_tokens": 50}, i
            )
            for i in range(n)
        ]

    def test_zero_threshold_raises(self):
        with self.assertRaises(ValueError):
            detect_anomalies(self._make_calls(), mad_threshold=0)

    def test_negative_threshold_raises(self):
        with self.assertRaises(ValueError):
            detect_anomalies(self._make_calls(), mad_threshold=-1.0)

    def test_nan_threshold_raises(self):
        with self.assertRaises(ValueError):
            detect_anomalies(self._make_calls(), mad_threshold=float("nan"))

    def test_empty_calls_returns_zero(self):
        self.assertEqual(detect_anomalies([]), 0)


# ---------------------------------------------------------------------------
# _load_pricing validation
# ---------------------------------------------------------------------------

class TestLoadPricingValidation(unittest.TestCase):
    def _write_pricing(self, data) -> str:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(data, fh)
            return fh.name

    def test_valid_pricing_loads(self):
        path = self._write_pricing({"my-llm": {"input": 0.001, "output": 0.002}})
        try:
            merged = _load_pricing(path)
            self.assertIn("my-llm", merged)
        finally:
            os.unlink(path)

    def test_entry_missing_output_raises(self):
        path = self._write_pricing({"bad-model": {"input": 0.001}})
        try:
            with self.assertRaises(ValueError):
                _load_pricing(path)
        finally:
            os.unlink(path)

    def test_entry_missing_input_raises(self):
        path = self._write_pricing({"bad-model": {"output": 0.002}})
        try:
            with self.assertRaises(ValueError):
                _load_pricing(path)
        finally:
            os.unlink(path)

    def test_entry_non_numeric_raises(self):
        path = self._write_pricing({"bad-model": {"input": "free", "output": 0.002}})
        try:
            with self.assertRaises(ValueError):
                _load_pricing(path)
        finally:
            os.unlink(path)

    def test_top_level_not_object_raises(self):
        path = self._write_pricing([1, 2, 3])
        try:
            with self.assertRaises(ValueError):
                _load_pricing(path)
        finally:
            os.unlink(path)

    def test_entry_not_dict_raises(self):
        path = self._write_pricing({"bad-model": "flat-string"})
        try:
            with self.assertRaises(ValueError):
                _load_pricing(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------

class TestCLIValidation(unittest.TestCase):
    def _make_log(self, entries) -> str:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
            return fh.name

    def test_negative_mad_threshold_exits_1(self):
        path = self._make_log(
            [{"model": "gpt-4o", "prompt_tokens": 100, "completion_tokens": 50}]
        )
        try:
            self.assertEqual(main(["audit", path, "--mad-threshold", "-1.0"]), 1)
        finally:
            os.unlink(path)

    def test_zero_mad_threshold_exits_1(self):
        path = self._make_log(
            [{"model": "gpt-4o", "prompt_tokens": 100, "completion_tokens": 50}]
        )
        try:
            self.assertEqual(main(["audit", path, "--mad-threshold", "0"]), 1)
        finally:
            os.unlink(path)

    def test_bad_pricing_file_exits_1(self):
        path = self._make_log(
            [{"model": "gpt-4o", "prompt_tokens": 100, "completion_tokens": 50}]
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as pfh:
            json.dump({"bad-model": {"input": "free", "output": 0.002}}, pfh)
            pricing_path = pfh.name
        try:
            self.assertEqual(main(["audit", path, "--pricing", pricing_path]), 1)
        finally:
            os.unlink(path)
            os.unlink(pricing_path)

    def test_missing_pricing_file_exits_1(self):
        path = self._make_log(
            [{"model": "gpt-4o", "prompt_tokens": 100, "completion_tokens": 50}]
        )
        nonexistent = path + "_no_such_pricing.json"
        try:
            self.assertEqual(
                main(["audit", path, "--pricing", nonexistent]), 1
            )
        finally:
            os.unlink(path)

    def test_empty_log_file_exits_1(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            path = fh.name
        try:
            self.assertEqual(main(["audit", path]), 1)
        finally:
            os.unlink(path)

    def test_empty_json_array_exits_1(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("[]")
            path = fh.name
        try:
            self.assertEqual(main(["audit", path]), 1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
