from abc import ABC, abstractmethod
from collections import Counter
import math
import re
from typing import Dict, List, Set, Tuple

from app.core.rag.contracts import DocumentChunk, RetrievalFilter
from app.core.rag.errors import TenantContextMissingError


class LexicalIndexPort(ABC):
    """
    Abstract Port for lexical keyword search (BM25 / Inverted Index).
    """

    @abstractmethod
    async def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        pass

    @abstractmethod
    async def query_sparse(
        self, query: str, filter: RetrievalFilter, top_k: int = 10
    ) -> List[Tuple[DocumentChunk, float]]:
        pass

    @abstractmethod
    async def delete_document(self, tenant_id: str, document_id: str) -> int:
        pass


class BM25Index(LexicalIndexPort):
    """
    Multi-tenant BM25 keyword search index.
    Tokenizes queries and passages, computes BM25 relevance scores, and boosts exact course codes.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # tenant_id -> chunk_id -> DocumentChunk
        self._chunks: Dict[str, Dict[str, DocumentChunk]] = {}
        # tenant_id -> chunk_id -> Counter(term -> count)
        self._term_freqs: Dict[str, Dict[str, Counter]] = {}
        # tenant_id -> term -> set(chunk_ids)
        self._inverted_index: Dict[str, Dict[str, Set[str]]] = {}
        # tenant_id -> total words in all chunks
        self._doc_lengths: Dict[str, Dict[str, int]] = {}

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and split on non-alphanumeric (preserves acronyms and course codes like CSC201)
        tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        return tokens

    async def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        indexed_count = 0
        for chunk in chunks:
            if not chunk.tenant_id:
                raise TenantContextMissingError("Cannot index chunk without tenant_id")

            tid = chunk.tenant_id
            if tid not in self._chunks:
                self._chunks[tid] = {}
                self._term_freqs[tid] = {}
                self._inverted_index[tid] = {}
                self._doc_lengths[tid] = {}

            self._chunks[tid][chunk.chunk_id] = chunk

            # Tokenize content and course code
            full_text = f"{chunk.content} {chunk.course_code or ''} {chunk.topic or ''} {chunk.document_title}"
            tokens = self._tokenize(full_text)
            self._doc_lengths[tid][chunk.chunk_id] = len(tokens)

            tf = Counter(tokens)
            self._term_freqs[tid][chunk.chunk_id] = tf

            for term in tf:
                if term not in self._inverted_index[tid]:
                    self._inverted_index[tid][term] = set()
                self._inverted_index[tid][term].add(chunk.chunk_id)

            indexed_count += 1
        return indexed_count

    async def query_sparse(
        self, query: str, filter: RetrievalFilter, top_k: int = 10
    ) -> List[Tuple[DocumentChunk, float]]:
        if not filter.tenant_id:
            raise TenantContextMissingError("BM25 query requires authoritative tenant_id")

        tid = filter.tenant_id
        if tid not in self._chunks or not self._chunks[tid]:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        n_docs = len(self._chunks[tid])
        avg_doc_len = sum(self._doc_lengths[tid].values()) / max(1, n_docs)

        scores: Dict[str, float] = {}

        for term in tokens:
            matching_chunks = self._inverted_index[tid].get(term, set())
            df = len(matching_chunks)
            if df == 0:
                continue

            # IDF computation
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

            for cid in matching_chunks:
                chunk = self._chunks[tid][cid]

                # Filter assertions
                if filter.course_code and chunk.course_code != filter.course_code:
                    continue
                if filter.document_ids and chunk.document_id not in filter.document_ids:
                    continue
                if filter.authority_tiers and chunk.provenance:
                    if chunk.provenance.authority not in filter.authority_tiers:
                        continue

                tf = self._term_freqs[tid][cid].get(term, 0)
                doc_len = self._doc_lengths[tid].get(cid, 1)

                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len))
                term_score = idf * (numerator / max(1e-6, denominator))

                scores[cid] = scores.get(cid, 0.0) + term_score

        # Normalize BM25 scores between 0.0 and 1.0
        if not scores:
            return []

        max_score = max(scores.values()) or 1.0
        ranked: List[Tuple[DocumentChunk, float]] = []
        for cid, raw_score in scores.items():
            norm_score = raw_score / max_score
            if norm_score >= filter.min_score:
                ranked.append((self._chunks[tid][cid], norm_score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    async def delete_document(self, tenant_id: str, document_id: str) -> int:
        if tenant_id not in self._chunks:
            return 0
        to_delete = [
            cid
            for cid, chunk in self._chunks[tenant_id].items()
            if chunk.document_id == document_id
        ]
        for cid in to_delete:
            del self._chunks[tenant_id][cid]
            del self._term_freqs[tenant_id][cid]
            del self._doc_lengths[tenant_id][cid]
            for term_set in self._inverted_index[tenant_id].values():
                term_set.discard(cid)
        return len(to_delete)
