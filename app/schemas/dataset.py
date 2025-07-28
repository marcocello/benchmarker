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


class ImageTask(BaseModel):
    """A single image-based task in the dataset."""
    id: str
    prompt: str
    image_path: str
    expected: str
    category: Optional[str] = None
    difficulty: Optional[str] = None
    task_type: Optional[str] = None  # e.g., "text_extraction", "handwriting_recognition", "ocr"
    analysis_criteria: Optional[List[str]] = None


class PdfTask(BaseModel):
    """A single PDF-based task in the dataset."""
    id: str
    prompt: str
    pdf_path: str
    expected: str
    category: Optional[str] = None
    difficulty: Optional[str] = None
    task_type: Optional[str] = None  # e.g., "text_extraction", "form_extraction", "analysis"
    analysis_criteria: Optional[List[str]] = None


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    name: str
    description: Optional[str] = None
    version: str = "1.0"
    questions: Optional[List[Question]] = None
    image_tasks: Optional[List[ImageTask]] = None
    pdf_tasks: Optional[List[PdfTask]] = None


class DatasetSchema(BaseModel):
    """Main dataset schema."""
    dataset: DatasetConfig
