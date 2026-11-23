"""心脏风险分析主题的本地 Markdown 混合知识库。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from threading import RLock

from .bm25 import BM25Document, BM25Index
from .reranker import LocalReranker, create_default_reranker
from .vector_retriever import VectorIndex


_SHARED_KNOWLEDGE_BASES: dict[tuple[str, str], "ChineseKnowledgeBase"] = {}
_SHARED_KNOWLEDGE_BASES_LOCK = RLock()


class ChineseKnowledgeBase:
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        vector_encoder=None,
        cache_dir: str | Path | None = None,
    ):
        self.root = Path(root) if root is not None else Path(__file__).with_name("knowledge_base")
        self.quality_issues: list[str] = []
        documents = self._load_documents()
        self.index = BM25Index(documents)
        cache_path = None
        if vector_encoder is None:
            # Keep generated vectors outside the repository so packaged apps and
            # source checkouts share the same cache without modifying project files.
            cache_path = self._vector_cache_path(cache_dir)
        self.vector_index = VectorIndex(
            documents,
            encoder=vector_encoder,
            cache_path=cache_path,
        )
        self.reranker = create_default_reranker()
        self.reranker_backend = type(self.reranker).__name__

    def _vector_cache_path(self, cache_dir: str | Path | None) -> Path:
        configured_dir = (
            str(cache_dir).strip()
            if cache_dir is not None
            else os.getenv("CARDIO_KNOWLEDGE_CACHE", "").strip()
        )
        if configured_dir:
            directory = Path(configured_dir).expanduser()
        else:
            local_app_data = os.getenv("LOCALAPPDATA", "").strip()
            directory = (
                Path(local_app_data) / "CardioScope" / "knowledge_cache"
                if local_app_data
                else Path.home() / ".cache" / "cardioscope" / "knowledge_cache"
            )
        root_key = str(self.root.resolve()).encode("utf-8")
        root_digest = hashlib.sha256(root_key).hexdigest()[:16]
        return directory / f"vectors-{root_digest}.npz"

    def _load_documents(self) -> list[BM25Document]:
        documents: list[BM25Document] = []
        paths = list(sorted(self.root.glob("*.md"))) if self.root.exists() else []
        paths += list(sorted(self.root.glob("*.json"))) if self.root.exists() else []
        corpus_root = self.root / "语料库_v1"
        if corpus_root.exists():
            paths.extend(sorted(corpus_root.glob("*.md")))
            paths.extend(sorted(corpus_root.glob("*.json")))
        external_root = os.getenv("CARDIO_KNOWLEDGE_CORPUS", "").strip()
        if external_root:
            external_path = Path(external_root).expanduser()
            if external_path.exists():
                paths.extend(sorted(external_path.glob("*.md")))
                paths.extend(sorted(external_path.glob("*.json")))
                self._external_corpus_path = str(external_path)
            else:
                self.quality_issues.append(f"外部语料目录不存在：{external_root}")
        for path in paths:
            if path.suffix.lower() == ".json":
                documents.extend(self._load_json_documents(path))
                continue
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            chunks = self._chunk_markdown(content)
            seen: set[str] = set()
            title_text: dict[str, str] = {}
            for number, chunk in enumerate(chunks, start=1):
                lines = chunk.splitlines()
                title = lines[0].lstrip("# ").strip() or path.stem
                normalized = re.sub(r"\W+", "", chunk).lower()
                if not normalized or normalized in seen:
                    self.quality_issues.append(f"重复片段已跳过：{path.name}#{number}")
                    continue
                seen.add(normalized)
                if title in title_text and title_text[title] != normalized:
                    self.quality_issues.append(f"同标题内容可能冲突：{path.name}#{title}")
                title_text[title] = normalized
                documents.append(BM25Document(f"{path.stem}-{number}", title, chunk, metadata=self._metadata_for(path)))
        return documents

    def _load_json_documents(self, path: Path) -> list[BM25Document]:
        """Load imported literature records, retaining source metadata."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.quality_issues.append(f"JSON 语料无法读取：{path.name} ({exc})")
            return []
        records = payload.get("documents", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            self.quality_issues.append(f"JSON 语料格式错误：{path.name}，需要 documents 数组")
            return []
        documents: list[BM25Document] = []
        for number, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            title = str(record.get("title", "")).strip()
            text = str(record.get("text", "")).strip()
            if not title or not text:
                self.quality_issues.append(f"JSON 语料缺少标题或正文：{path.name}#{number}")
                continue
            metadata = dict(record.get("metadata") or {})
            metadata.setdefault("source", f"导入语料/{path.name}")
            metadata.setdefault("source_group", "corpus")
            metadata.setdefault("evidence_level", "literature_catalog")
            metadata.setdefault("version", "1.0")
            doc_id = str(record.get("doc_id") or f"{path.stem}-{number}")
            chunks = self._chunk_markdown(text)
            for chunk_number, chunk in enumerate(chunks, start=1):
                chunk_title = title if len(chunks) == 1 else f"{title}（片段 {chunk_number}/{len(chunks)}）"
                documents.append(BM25Document(f"{doc_id}-{chunk_number}", chunk_title, chunk, metadata=metadata))
        return documents

    @staticmethod
    def _chunk_markdown(content: str, *, max_chars: int = 520, overlap: int = 80) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 2 <= max_chars:
                current = f"{current}\n\n{paragraph}".strip()
                continue
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{paragraph}".strip()
            while len(current) > max_chars:
                chunks.append(current[:max_chars].strip())
                current = current[max_chars - overlap :].strip()
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _metadata_for(path: Path) -> dict[str, object]:
        domains = {
            "heart_risk_indicators": "心脏风险指标",
            "heart_positive_rate": "阳性率与分层解释",
            "heart_outlier_handling": "心脏数据异常值处理",
            "statistical_limits": "统计解释边界",
            "medical_data_safety": "医疗数据安全",
        }
        return {"domain": domains.get(path.stem, "心脏风险分析"), "source": f"本地知识库/{path.name}", "source_group": "builtin", "evidence_level": "internal_summary", "jurisdiction": "通用科研说明，临床和法规要求以机构规定为准", "approved": "pending_review", "version": "1.0"}

    def retrieve(self, query: str, *, top_k: int = 4, bm25_weight: float = 0.55, min_score: float = 0.15):
        candidates = max(top_k * 3, 8)
        sparse = self.index.search(query, top_k=candidates)
        dense = self.vector_index.search(query, top_k=candidates)
        scores: dict[str, tuple[BM25Document, float, float]] = {}
        sparse_max = max((score for _, score in sparse), default=1.0)
        dense_max = max((score for _, score in dense), default=1.0)
        for document, score in sparse:
            scores[document.doc_id] = (document, score / sparse_max, 0.0)
        for document, score in dense:
            old = scores.get(document.doc_id, (document, 0.0, 0.0))
            scores[document.doc_id] = (document, old[1], score / dense_max)
        ranked = [(document, bm25_weight * sparse_score + (1 - bm25_weight) * dense_score) for document, sparse_score, dense_score in scores.values()]
        ranked.sort(key=lambda item: item[1], reverse=True)
        try:
            reranked = self.reranker.rerank(query, ranked[:candidates], top_k=top_k)
        except Exception as exc:
            self.reranker = LocalReranker()
            self.reranker_backend = "LocalReranker_fallback"
            self.reranker_error = f"{type(exc).__name__}: {exc}"
            reranked = self.reranker.rerank(query, ranked[:candidates], top_k=top_k)
        return [item for item in reranked if item[1] >= min_score]

    @property
    def retrieval_status(self) -> dict[str, object]:
        source_counts: dict[str, int] = {}
        for document in self.index.documents:
            group = str((document.metadata or {}).get("source_group", "unknown"))
            source_counts[group] = source_counts.get(group, 0) + 1
        return {"vector_backend": self.vector_index.backend, "vector_cache": self.vector_index.cache_status, "fallback_reason": self.vector_index.fallback_reason, "reranker_backend": self.reranker_backend, "reranker_error": getattr(self, "reranker_error", None), "document_count": len(self.index.documents), "source_counts": source_counts, "quality_issues": list(self.quality_issues)}

    def format_context(self, query: str, *, top_k: int = 4) -> str:
        hits = self.retrieve(query, top_k=top_k)
        return "\n\n".join(f"[KB-{i}] 来源：{doc.metadata.get('source', doc.doc_id) if doc.metadata else doc.doc_id} · 主题：{doc.metadata.get('domain', '未标注') if doc.metadata else '未标注'} · 证据级别：{doc.metadata.get('evidence_level', '未标注') if doc.metadata else '未标注'} · 版本：{doc.metadata.get('version', '未标注') if doc.metadata else '未标注'} · 重排分数 {score:.3f}\n{doc.text}" for i, (doc, score) in enumerate(hits, start=1))


def _shared_cache_key() -> tuple[str, str]:
    root = Path(__file__).with_name("knowledge_base").resolve()
    external_corpus = os.getenv("CARDIO_KNOWLEDGE_CORPUS", "").strip()
    return str(root), external_corpus


def get_shared_knowledge_base() -> ChineseKnowledgeBase:
    """Return the process-wide default knowledge base, initializing it once."""

    key = _shared_cache_key()
    with _SHARED_KNOWLEDGE_BASES_LOCK:
        knowledge_base = _SHARED_KNOWLEDGE_BASES.get(key)
        if knowledge_base is None:
            knowledge_base = ChineseKnowledgeBase()
            _SHARED_KNOWLEDGE_BASES[key] = knowledge_base
        return knowledge_base
