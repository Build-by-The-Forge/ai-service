"""Column types the AI-owned tables need beyond SQLAlchemy's built-ins."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import Text, cast, type_coerce
from sqlalchemy.types import UserDefinedType


class Vector(UserDefinedType[list[float]]):
    """pgvector's ``vector(d)``.

    Values travel as pgvector's text form (``[0.1,0.2]``) with explicit casts
    in both directions, so asyncpg needs no codec for the extension type and
    no client-side dependency is added.
    """

    cache_ok = True

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_col_spec(self, **kw: Any) -> str:
        return f"vector({self.dimension})"

    def bind_expression(self, bindvalue: Any) -> Any:
        return cast(cast(bindvalue, Text), self)

    def column_expression(self, col: Any) -> Any:
        # Read as text, but keep this type so result_processor parses it.
        return type_coerce(cast(col, Text), self)

    def bind_processor(self, dialect: Any) -> Any:
        dimension = self.dimension

        def process(value: Sequence[float] | None) -> str | None:
            if value is None:
                return None
            if len(value) != dimension:
                raise ValueError(f"expected {dimension} dimensions, got {len(value)}")
            return "[" + ",".join(repr(float(x)) for x in value) + "]"

        return process

    def result_processor(self, dialect: Any, coltype: Any) -> Any:
        def process(value: str | None) -> list[float] | None:
            if value is None:
                return None
            return [float(x) for x in value.strip("[]").split(",")] if value != "[]" else []

        return process
