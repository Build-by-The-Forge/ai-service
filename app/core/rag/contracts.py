from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional
import uuid


class ContentAuthority(str, Enum):
    REGULATORY = "REGULATORY"
    INSTITUTIONAL = "INSTITUTIONAL"
    APPROVED_REFERENCE = "APPROVED_REFERENCE"
    USER_PROVIDED = "USER_PROVIDED"
    AI_DERIVED = "AI_DERIVED"


@dataclass(frozen=True)
class ChunkProvenance:
    """
    Cryptographic and structural citation provenance tracing a chunk back to its source document.
    """

    source_uri: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    authority: ContentAuthority = ContentAuthority.INSTITUTIONAL
    content_sha256: Optional[str] = None

    def to_citation_dict(self) -> Dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "authority": self.authority.value,
        }


@dataclass
class DocumentChunk:
    """
    Unit of semantic context stored in vector and lexical indexes.
    """

    chunk_id: str
    tenant_id: str
    document_id: str
    document_title: str
    content: str
    course_code: Optional[str] = None
    topic: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[ChunkProvenance] = None
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        tenant_id: str,
        document_id: str,
        document_title: str,
        content: str,
        course_code: Optional[str] = None,
        topic: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        provenance: Optional[ChunkProvenance] = None,
    ) -> "DocumentChunk":
        return cls(
            chunk_id=f"chk_{uuid.uuid4().hex[:12]}",
            tenant_id=tenant_id,
            document_id=document_id,
            document_title=document_title,
            content=content,
            course_code=course_code,
            topic=topic,
            embedding=embedding,
            metadata=metadata or {},
            provenance=provenance,
        )


@dataclass(frozen=True)
class RetrievalFilter:
    """
    Mandatory authoritatively-scoped query filter.
    Enforces strict multi-tenant boundary isolation.
    """

    tenant_id: str
    course_code: Optional[str] = None
    document_ids: Optional[List[str]] = None
    authority_tiers: Optional[List[ContentAuthority]] = None
    min_score: float = 0.0


@dataclass(frozen=True)
class RetrievalPolicy:
    """
    Tunable hyper-parameters governing hybrid search fusion and rank truncation.
    """

    top_k: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60
    max_context_bytes: int = 65536  # 64 KB
    rerank_enabled: bool = True


@dataclass(frozen=True)
class RetrievedChunk:
    """
    Scored and ranked chunk result with citation metadata.
    """

    chunk: DocumentChunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    rank: int = 1


@dataclass(frozen=True)
class RetrievedContext:
    """
    Authoritative outcome envelope returned by RAG engine to Agents/Models.
    """

    query: str
    tenant_id: str
    chunks: List[RetrievedChunk]
    retrieval_method: str = "hybrid_rrf"
    latency_ms: float = 0.0
    total_candidates: int = 0

    def to_citation_summary(self) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for rc in self.chunks:
            item = {
                "chunk_id": rc.chunk.chunk_id,
                "document_title": rc.chunk.document_title,
                "course_code": rc.chunk.course_code,
                "snippet": rc.chunk.content[:200] + ("..." if len(rc.chunk.content) > 200 else ""),
                "hybrid_score": round(rc.hybrid_score, 4),
                "rank": rc.rank,
            }
            if rc.chunk.provenance:
                item["provenance"] = rc.chunk.provenance.to_citation_dict()
            citations.append(item)
        return citations
