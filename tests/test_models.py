# File: tests/test_models.py
"""
Tests for Pydantic models in LLMHandler.
"""

import pytest
from datetime import datetime
from LLMHandler.models import (
    BatchMetadata,
    SimpleResponse,
    MathResponse,
    PersonResponse,
    BatchResult,
    UnifiedResponse
)


def test_simple_response_model() -> None:
    """Test that SimpleResponse can be instantiated correctly."""
    sr = SimpleResponse(content="Hello", confidence=0.9)
    assert sr.content == "Hello"
    assert sr.confidence == 0.9

    # Test boundary
    sr2 = SimpleResponse(confidence=0)
    assert sr2.confidence == 0

    # Pydantic should raise an error for out-of-bounds confidence
    with pytest.raises(ValueError):
        _ = SimpleResponse(confidence=1.1)


def test_math_response_model() -> None:
    """Test that MathResponse can handle numeric fields."""
    mr = MathResponse(answer=42, reasoning="Because reasons", confidence=0.5)
    assert mr.answer == 42
    assert mr.reasoning == "Because reasons"
    assert mr.confidence == 0.5


def test_person_response_model() -> None:
    """Test PersonResponse fields."""
    pr = PersonResponse(name="Alice", age=30, occupation="Engineer", skills=["Python", "C++"])
    assert pr.name == "Alice"
    assert pr.age == 30
    assert "Python" in pr.skills


def test_batch_metadata_model() -> None:
    """Test creating a BatchMetadata instance."""
    bm = BatchMetadata(
        batch_id="batch123",
        input_file_id="file456",
        status="in_progress",
        created_at=datetime.now(),
        last_updated=datetime.now(),
        num_requests=10,
    )
    assert bm.batch_id == "batch123"
    assert bm.num_requests == 10


def test_unified_response_model() -> None:
    """Check that UnifiedResponse can hold success or error."""
    ur_success = UnifiedResponse[SimpleResponse](
        success=True,
        data=SimpleResponse(content="Success!"),
    )
    assert ur_success.success is True
    assert ur_success.data
    assert isinstance(ur_success.data, SimpleResponse)

    ur_error = UnifiedResponse[SimpleResponse](
        success=False,
        error="Something went wrong",
    )
    assert ur_error.success is False
    assert ur_error.error == "Something went wrong"
    assert ur_error.data is None
