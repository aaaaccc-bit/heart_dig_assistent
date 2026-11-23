"""Chinese vector retrieval with BGE embeddings and an offline fallback."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .bm25 import BM25Document


class ChineseHashEncoder:
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        compact = re.sub(r"\s+", "", (text or "").lower())
        tokens: list[str] = []
        for char in compact:
            if "\u4e00" <= char <= "\u9fff":
                tokens.append(char)
        tokens.extend(compact[i : i + 2] for i in range(max(0, len(compact) - 1)))
        tokens.extend(compact[i : i + 3] for i in range(max(0, len(compact) - 2)))
        for token in re.findall(r"[a-z0-9_]+", compact):
            tokens.append(token)
        vector = [0.0] * self.dimensions
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class BGEChineseEncoder:
    """BAAI BGE Chinese encoder loaded lazily from Hugging Face."""

    _shared_models = {}

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", *, allow_download: bool | None = None):
        self.model_name = model_name
        self.allow_download = (
            os.getenv("CARDIO_BGE_ALLOW_DOWNLOAD", "0") == "1"
            if allow_download is None
            else allow_download
        )
        self._model = None

    def _load(self):
        if self._model is None:
            if self.model_name in self._shared_models:
                self._model = self._shared_models[self.model_name]
                return self._model
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                if not self.allow_download:
                    raise
                self._model = SentenceTransformer(self.model_name)
            self._shared_models[self.model_name] = self._model
        return self._model

    def encode(self, text: str) -> list[float]:
        model = self._load()
        query = text or ""
        vector = model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vector, dtype=np.float32).tolist()


def create_default_encoder():
    """Prefer a locally cached BGE model and fall back without network I/O."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return ChineseHashEncoder()
    # 模型下载必须显式开启，否则首次点击报告按钮可能长时间阻塞。
    allow_download = os.getenv("CARDIO_BGE_ALLOW_DOWNLOAD", "0") == "1"
    return BGEChineseEncoder(allow_download=allow_download)


class VectorIndex:
    def __init__(
        self,
        documents: Iterable[BM25Document] = (),
        *,
        encoder=None,
        dimensions: int = 384,
        cache_path: str | Path | None = None,
    ):
        self.encoder = encoder or create_default_encoder()
        self.backend = "bge" if isinstance(self.encoder, BGEChineseEncoder) else "hash"
        self.fallback_reason: str | None = None
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.cache_status = "disabled" if self.cache_path is None else "miss"
        self.documents: list[BM25Document] = []
        self.vectors: list[list[float]] = []
        self.add_documents(documents)

    def add_documents(self, documents: Iterable[BM25Document]) -> None:
        pending = list(documents)
        if not pending:
            return
        initial_load = not self.documents
        if initial_load and self._load_cached_vectors(pending):
            return
        try:
            vectors = [self.encoder.encode(f"{document.title} {document.text}") for document in pending]
        except Exception as exc:
            # Model download/runtime failure should not disable local retrieval.
            self.encoder = ChineseHashEncoder()
            self.backend = "hash_fallback"
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            vectors = [self.encoder.encode(f"{document.title} {document.text}") for document in pending]
        self.documents.extend(pending)
        self.vectors.extend(vectors)
        if initial_load:
            self._write_cache(pending, vectors)

    def _load_cached_vectors(self, documents: list[BM25Document]) -> bool:
        if self.cache_path is None or not self.cache_path.exists():
            return False
        try:
            with np.load(self.cache_path, allow_pickle=False) as cached:
                fingerprint = str(cached["fingerprint"].item())
                if fingerprint != self._documents_fingerprint(documents):
                    return False
                cached_encoder = str(cached["encoder"].item())
                cached_backend = str(cached["backend"].item())
                if cached_encoder == "hash":
                    if isinstance(self.encoder, BGEChineseEncoder):
                        self.encoder = ChineseHashEncoder()
                    elif not isinstance(self.encoder, ChineseHashEncoder):
                        return False
                elif cached_encoder.startswith("bge:"):
                    if cached_encoder != self._encoder_key():
                        return False
                    try:
                        self.encoder._load()
                    except Exception:
                        return False
                elif cached_encoder != self._encoder_key():
                    return False
                vectors = np.asarray(cached["vectors"], dtype=np.float32)
                if vectors.ndim != 2 or vectors.shape[0] != len(documents):
                    return False
                if not np.isfinite(vectors).all():
                    return False
            self.backend = cached_backend
            self.documents.extend(documents)
            self.vectors.extend(vectors.tolist())
            self.cache_status = "hit"
            return True
        except (OSError, EOFError, KeyError, ValueError):
            # A stale or corrupt optimization cache must never block retrieval.
            self.cache_status = "invalid"
            return False

    def _write_cache(
        self, documents: list[BM25Document], vectors: list[list[float]]
    ) -> None:
        if self.cache_path is None:
            return
        temporary_path = self.cache_path.with_suffix(
            f"{self.cache_path.suffix}.{os.getpid()}.tmp"
        )
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with temporary_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    fingerprint=self._documents_fingerprint(documents),
                    encoder=self._encoder_key(),
                    backend=self.backend,
                    vectors=np.asarray(vectors, dtype=np.float32),
                )
            os.replace(temporary_path, self.cache_path)
            self.cache_status = "written"
        except (OSError, ValueError):
            self.cache_status = "write_failed"
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _encoder_key(self) -> str:
        if isinstance(self.encoder, BGEChineseEncoder):
            return f"bge:{self.encoder.model_name}"
        if isinstance(self.encoder, ChineseHashEncoder):
            return "hash"
        return f"custom:{type(self.encoder).__module__}.{type(self.encoder).__qualname__}"

    @staticmethod
    def _documents_fingerprint(documents: Iterable[BM25Document]) -> str:
        digest = hashlib.sha256()
        for document in documents:
            for value in (document.doc_id, document.title, document.text):
                digest.update(str(value).encode("utf-8"))
                digest.update(b"\0")
        return digest.hexdigest()

    def search(self, query: str, *, top_k: int = 4) -> list[tuple[BM25Document, float]]:
        query_vector = self.encoder.encode(query)
        scored = []
        for document, vector in zip(self.documents, self.vectors):
            score = sum(left * right for left, right in zip(query_vector, vector))
            if score > 0:
                scored.append((document, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(0, top_k)]

