import os
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from rag import get_openai_client, split_text


class SplitTextTest(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(split_text(""), [])

    def test_short_text_returns_one_chunk(self):
        self.assertEqual(split_text("hello world", chunk_size=100), ["hello world"])

    def test_long_text_is_split_with_overlap(self):
        chunks = split_text("abcdefghij", chunk_size=4, overlap=1)
        self.assertEqual(chunks, ["abcd", "defg", "ghij"])


class OpenAIConfigTest(unittest.TestCase):
    def test_missing_api_key_has_clear_error(self):
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                get_openai_client()
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
