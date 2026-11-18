"""Week 5: Unit tests for TokenBudgetManager and token economics helpers."""

import unittest

from budget import BudgetExceededError, TokenBudgetManager, estimate_tokens


class TestTokenBudgetManager(unittest.TestCase):
    """Verify cost calculations, threshold warnings, and budget guardrails."""

    def test_invalid_initialization_raises(self) -> None:
        with self.assertRaises(ValueError):
            TokenBudgetManager(max_budget_usd=0)
        with self.assertRaises(ValueError):
            TokenBudgetManager(max_budget_usd=1.0, max_total_tokens=-10)
        with self.assertRaises(ValueError):
            TokenBudgetManager(max_budget_usd=1.0, warning_threshold=1.5)

    def test_estimate_cost_known_and_unknown_model(self) -> None:
        manager = TokenBudgetManager(max_budget_usd=10.0)
        cost = manager.estimate_cost("gemini-2.5-flash", 1_000_000, 500_000)
        self.assertAlmostEqual(cost, 0.45)

        with self.assertRaises(KeyError):
            manager.estimate_cost("non-existent-model", 100, 100)

        with self.assertRaises(ValueError):
            manager.estimate_cost("gemini-2.5-flash", -1, 100)

    def test_record_usage_updates_summary_and_warning_status(self) -> None:
        manager = TokenBudgetManager(max_budget_usd=1.00, warning_threshold=0.80)
        manager.record_usage("gemini-2.5-flash", 1_000_000, 500_000, tag="batch-1")
        self.assertEqual(manager.status(), "OK")
        self.assertAlmostEqual(manager.spent_usd, 0.45)

        manager.record_usage("gemini-2.5-flash", 1_000_000, 500_000, tag="batch-2")
        self.assertEqual(manager.status(), "WARNING")
        summary = manager.summary()
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["total_tokens"], 3_000_000)
        self.assertAlmostEqual(summary["spent_usd"], 0.90)
        self.assertAlmostEqual(summary["remaining_usd"], 0.10)

    def test_usd_budget_enforcement_blocks_overspend(self) -> None:
        manager = TokenBudgetManager(max_budget_usd=0.01)
        self.assertFalse(manager.can_afford("gemini-2.5-pro", 10_000, 5_000))
        with self.assertRaises(BudgetExceededError):
            manager.record_usage("gemini-2.5-pro", 10_000, 5_000)
        self.assertEqual(len(manager.records), 0)

    def test_token_limit_enforcement_blocks_excess_tokens(self) -> None:
        manager = TokenBudgetManager(max_budget_usd=100.0, max_total_tokens=1_000)
        manager.record_usage("gemini-2.5-flash", 400, 300)
        self.assertTrue(manager.can_afford("gemini-2.5-flash", 200, 100))
        self.assertFalse(manager.can_afford("gemini-2.5-flash", 200, 200))
        with self.assertRaises(BudgetExceededError):
            manager.record_usage("gemini-2.5-flash", 200, 200)

    def test_reset_clears_usage_records(self) -> None:
        manager = TokenBudgetManager(max_budget_usd=5.0)
        manager.record_usage("gemini-2.5-flash", 50_000, 25_000)
        self.assertGreater(manager.spent_usd, 0)
        manager.reset()
        self.assertEqual(manager.spent_usd, 0.0)
        self.assertEqual(manager.total_tokens, 0)
        self.assertEqual(manager.status(), "OK")

    def test_estimate_tokens_returns_positive_for_text(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreater(estimate_tokens("Token economics helps control LLM spend."), 0)


if __name__ == "__main__":
    unittest.main()
