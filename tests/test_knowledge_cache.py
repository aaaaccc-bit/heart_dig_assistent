from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ai.bm25 import BM25Document
from ai.vector_retriever import VectorIndex


class CountingEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, text: str) -> list[float]:
        self.calls += 1
        return [float(len(text)), 1.0]


class KnowledgeVectorCacheTest(unittest.TestCase):
    def test_cached_vectors_are_reused_without_reencoding(self) -> None:
        documents = [
            BM25Document("one", "主题一", "正文一"),
            BM25Document("two", "主题二", "正文二"),
        ]
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "vectors.npz"
            first_encoder = CountingEncoder()
            first = VectorIndex(documents, encoder=first_encoder, cache_path=cache_path)
            second_encoder = CountingEncoder()
            second = VectorIndex(documents, encoder=second_encoder, cache_path=cache_path)

        self.assertEqual(first_encoder.calls, 2)
        self.assertEqual(second_encoder.calls, 0)
        self.assertEqual(second.cache_status, "hit")
        self.assertEqual(first.vectors, second.vectors)

    def test_changed_documents_invalidate_cached_vectors(self) -> None:
        original = [BM25Document("one", "主题", "旧正文")]
        changed = [BM25Document("one", "主题", "新正文")]
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "vectors.npz"
            VectorIndex(original, encoder=CountingEncoder(), cache_path=cache_path)
            encoder = CountingEncoder()
            rebuilt = VectorIndex(changed, encoder=encoder, cache_path=cache_path)

        self.assertEqual(encoder.calls, 1)
        self.assertEqual(rebuilt.cache_status, "written")


if __name__ == "__main__":
    unittest.main()
