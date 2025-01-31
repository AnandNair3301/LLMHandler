"""
Tests for the Pydantic models in llmhandler.
"""

import pytest
from datetime import datetime, timezone
from llmhandler._internal_models import (
    BatchMetadata,
    BatchResult,
    SimpleResponse,
    MathResponse,
    PersonResponse,
    UnifiedResponse,
)



def test_simple_response_model():
    """Test that SimpleResponse can be instantiated correctly and bounds check."""
    sr = SimpleResponse(content="Hello", confidence=0.9)
    assert sr.content == "Hello"
    assert sr.confidence == 0.9

    # Confidence must be <= 1
    with pytest.raises(ValueError):
        _ = SimpleResponse(confidence=1.1)


def test_math_response_model():
    """Test the MathResponse fields."""
    mr = MathResponse(answer=42, reasoning="Because 42 is the ultimate answer", confidence=0.7)
    assert mr.answer == 42
    assert mr.reasoning == "Because 42 is the ultimate answer"
    assert mr.confidence == 0.7


def test_person_response_model():
    """Test the PersonResponse fields."""
    pr = PersonResponse(name="Bob", age=35, occupation="Chef", skills=["Cooking", "Baking"])
    assert pr.name == "Bob"
    assert pr.age == 35
    assert "Cooking" in pr.skills


def test_batch_metadata_model():
    """Test creating a BatchMetadata instance."""
    now = datetime.now(timezone.utc)
    bm = BatchMetadata(
        batch_id="batch123",
        input_file_id="fileXYZ",
        status="in_progress",
        created_at=now,
        last_updated=now,
        num_requests=5
    )
    assert bm.batch_id == "batch123"
    assert bm.num_requests == 5
    assert bm.status == "in_progress"


def test_unified_response_model():
    """Check that UnifiedResponse can hold success, error, or data."""
    ur_success = UnifiedResponse[SimpleResponse](
        success=True,
        data=SimpleResponse(content="Test content"),
    )
    assert ur_success.success is True
    assert ur_success.data is not None
    assert ur_success.error is None

    ur_error = UnifiedResponse[SimpleResponse](
        success=False,
        error="An error occurred",
    )
    assert ur_error.success is False
    assert ur_error.error == "An error occurred"
    assert ur_error.data is None
