"""Knowledge scope (M3-D): which documents retrieval may return.

Scope is held once per document in ``asset_scopes``, and only the backend
sets it, through a signed, versioned assignment computed from its own
ownership and rights records. A document with no row is ``unknown``, so
every document indexed before the table existed starts out of retrieval.

Until M4's allow-list, only ``institutional`` documents are retrieved, and
only within their tenant. ``personal`` and ``unknown`` never are.
"""

from enum import Enum


class KnowledgeScope(str, Enum):
    # Verified by the backend for tenant-wide use.
    INSTITUTIONAL = "institutional"
    # A student's own document; M4 opens it to its owner.
    PERSONAL = "personal"
    # Ownership or eligibility not established: retrieved for no one.
    UNKNOWN = "unknown"


RETRIEVABLE_SCOPES: tuple[str, ...] = (KnowledgeScope.INSTITUTIONAL.value,)
