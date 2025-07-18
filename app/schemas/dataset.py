"""Pydantic schemas for dataset validation."""

from pydantic import BaseModel, Field
from typing import List, Optional


class Question(BaseModel):
    """A single question in the dataset."""
    id: str
    prompt: str
    expected: str
    category: Optional[str] = None
    difficulty: Optional[str] = None


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    name: str
    description: Optional[str] = None
    version: str = "1.0"
    questions: List[Question]


class DatasetSchema(BaseModel):
    """Main dataset schema."""
    dataset: DatasetConfig
