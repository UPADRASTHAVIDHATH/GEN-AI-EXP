"""Week 3: Unit tests for cost calculator, MongoDB logger, and tracked LLMClient."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cost_calculator import calculate_cost
from llm_client import LLMClient
from mongo_logger import MongoLogger


class TestWeek3Components(unittest.TestCase):
    """Test token cost estimation, MongoDB persistence, and client integration."""

    def test_calculate_cost_known_model(self) -> None:
        breakdown = calculate_cost("gemini-2.5-flash", input_tokens=1_000_000, output_tokens=500_000)
        self.assertAlmostEqual(breakdown.input_cost_usd, 0.15)
        self.assertAlmostEqual(breakdown.output_cost_usd, 0.30)
        self.assertAlmostEqual(breakdown.total_cost_usd, 0.45)

    def test_calculate_cost_negative_tokens_raises(self) -> None:
        with self.assertRaises(ValueError):
            calculate_cost("gemini-2.5-flash", input_tokens=-5, output_tokens=10)

    def test_mongo_logger_inserts_document(self) -> None:
        fake_collection = MagicMock()
        fake_collection.insert_one.return_value = SimpleNamespace(inserted_id="507f1f77bcf86cd799439011")
        fake_db = MagicMock()
        fake_db.__getitem__.return_value = fake_collection
        fake_client = MagicMock()
        fake_client.__getitem__.return_value = fake_db

        logger = MongoLogger(
            db_name="test_db",
            collection_name="test_logs",
            client=fake_client,
        )
        doc_id = logger.log_interaction(
            prompt="Hello",
            response_text="Hi!",
            model="gemini-2.5-flash",
            input_tokens=4,
            output_tokens=2,
            total_tokens=6,
            total_cost_usd=0.0000018,
            latency_ms=125.4,
            metadata={"roll_number": "2520030597"},
        )

        self.assertEqual(doc_id, "507f1f77bcf86cd799439011")
        fake_collection.insert_one.assert_called_once()
        inserted_doc = fake_collection.insert_one.call_args[0][0]
        self.assertEqual(inserted_doc["model"], "gemini-2.5-flash")
        self.assertEqual(inserted_doc["tokens"]["total"], 6)
        self.assertEqual(inserted_doc["metadata"]["roll_number"], "2520030597")

    @patch("llm_client.genai.Client")
    def test_tracked_client_calculates_cost_and_logs(self, mock_genai: MagicMock) -> None:
        fake_raw = SimpleNamespace(
            text="Cost tracking prevents budget surprises.",
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
            ),
        )
        mock_genai.return_value.models.generate_content.return_value = fake_raw

        mock_logger = MagicMock()
        mock_logger.log_interaction.return_value = "mock-object-id"

        client = LLMClient(
            api_key="test-key",
            model="gemini-2.5-flash",
            logger=mock_logger,
        )
        result = client.generate("Why track LLM costs?")

        self.assertEqual(result.text, "Cost tracking prevents budget surprises.")
        self.assertEqual(result.total_tokens, 150)
        self.assertEqual(result.log_id, "mock-object-id")
        self.assertGreater(result.cost.total_cost_usd, 0.0)
        mock_logger.log_interaction.assert_called_once()


if __name__ == "__main__":
    unittest.main()
