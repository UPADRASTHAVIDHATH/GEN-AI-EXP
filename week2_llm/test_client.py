"""Week 2: Unit tests for LLMClient using mocked Google GenAI responses."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_client import LLMClient, LLMResponse


class TestLLMClient(unittest.TestCase):
    """Verify LLMClient validation, configuration, and response parsing."""

    @patch.dict(os.environ, {}, clear=True)
    @patch("llm_client.load_dotenv")
    def test_missing_api_key_raises_error(self, _mock_dotenv: MagicMock) -> None:
        with self.assertRaises(ValueError):
            LLMClient(api_key=None)

    @patch("llm_client.genai.Client")
    def test_empty_prompt_raises_error(self, _mock_genai: MagicMock) -> None:
        client = LLMClient(api_key="test-key", model="gemini-2.5-flash")
        with self.assertRaises(ValueError):
            client.generate("   ")

    @patch("llm_client.genai.Client")
    def test_generate_returns_structured_response(self, mock_genai: MagicMock) -> None:
        fake_raw = SimpleNamespace(
            text="Modular code improves testability and reuse.",
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=8,
                total_token_count=20,
            ),
        )
        mock_genai.return_value.models.generate_content.return_value = fake_raw

        client = LLMClient(
            api_key="test-key",
            model="gemini-2.5-flash",
            system_instruction="Be brief.",
        )
        result = client.generate("Why use modules?", temperature=0.3, max_output_tokens=64)

        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.text, "Modular code improves testability and reuse.")
        self.assertEqual(result.model, "gemini-2.5-flash")
        self.assertEqual(result.prompt_tokens, 12)
        self.assertEqual(result.completion_tokens, 8)
        self.assertEqual(result.total_tokens, 20)
        mock_genai.return_value.models.generate_content.assert_called_once()


if __name__ == "__main__":
    unittest.main()
