"""
Data models for LLM Handler.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Union, Generic

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class BatchMetadata(BaseModel):
    """
    Metadata for batch processing jobs.

    Attributes:
        batch_id: Unique identifier for the batch.
        input_file_id: The file ID associated with the batch input.
        status: The current batch status.
        created_at: Datetime when the batch was created.
        last_updated: Datetime when the batch was last updated.
        num_requests: Number of requests/prompts in this batch.
        error: Any error message if the batch failed.
        output_file_path: Path to the output file containing results.
    """
    batch_id: str
    input_file_id: str
    status: str
    created_at: datetime
    last_updated: datetime
    num_requests: int
    error: Optional[str] = None
    output_file_path: Optional[str] = None


class BatchResult(BaseModel):
    """
    Results from batch processing.

    Attributes:
        metadata: Information about the batch job itself.
        results: A list of dict objects containing 'prompt' and 'response' or 'error'.
    """
    metadata: BatchMetadata
    results: List[Dict[str, Any]]


class SimpleResponse(BaseModel):
    """
    Simple response model for testing generic text answers.

    Attributes:
        content: The textual response content.
        confidence: Confidence level (0 <= confidence <= 1).
    """
    content: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)


class MathResponse(BaseModel):
    """
    Response model for math problems.

    Attributes:
        answer: The numeric answer or solution.
        reasoning: Explanation for the solution.
        confidence: Confidence level (0 <= confidence <= 1).
    """
    answer: Optional[float] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)


class PersonResponse(BaseModel):
    """
    Response model for describing a person.

    Attributes:
        name: Person's name.
        age: Person's age (0 <= age <= 150).
        occupation: The person's occupation.
        skills: A list of skill strings.
    """
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=150)
    occupation: Optional[str] = None
    skills: Optional[List[str]] = None


class UnifiedResponse(BaseModel, Generic[T]):
    """
    A unified response envelope for all LLM responses.

    Attributes:
        success: Whether the request was successful.
        data: The typed data, a list of typed data, or a BatchResult object.
        error: Error message if not successful.
        original_prompt: The original prompt that triggered this response.
    """
    success: bool
    data: Optional[Union[T, List[T], BatchResult]] = None
    error: Optional[str] = None
    original_prompt: Optional[str] = None
