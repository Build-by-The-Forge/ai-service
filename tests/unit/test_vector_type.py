import pytest
from sqlalchemy.dialects import postgresql

from app.db.types import Vector


def test_column_spec_names_the_dimension():
    assert Vector(384).get_col_spec() == "vector(384)"


def test_values_travel_in_pgvector_text_form():
    vector = Vector(3)
    bind = vector.bind_processor(postgresql.dialect())
    result = vector.result_processor(postgresql.dialect(), None)
    assert bind([1, 0.5, -2]) == "[1.0,0.5,-2.0]"
    assert result("[1,0.5,-2]") == [1.0, 0.5, -2.0]
    assert bind(None) is None and result(None) is None


def test_wrong_dimension_is_refused():
    with pytest.raises(ValueError, match="expected 3 dimensions, got 2"):
        Vector(3).bind_processor(postgresql.dialect())([1.0, 2.0])
