"""Core suite and dataset loading and environment setup."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from app.schemas.suite import SuiteSchema, ProviderConfig
from app.schemas.dataset import DatasetSchema
from app.core.settings import resolve_provider_credentials

load_dotenv()
logger = logging.getLogger(__name__)


def expand_environment_variables(text: str) -> str:
    """
    Expand environment variables in text using ${VAR} syntax.
    
    Args:
        text: Text containing environment variable placeholders
        
    Returns:
        Text with environment variables expanded
    """
    import re
    
    def replace_var(match):
        var_name = match.group(1)
        
        # Check for default value syntax: ${VAR:-default}
        if ':-' in var_name:
            var_name, default_value = var_name.split(':-', 1)
            env_value = os.getenv(var_name, default_value)
        else:
            env_value = os.getenv(var_name)
            if env_value is None:
                logger.warning(f"Environment variable '{var_name}' not found, keeping original placeholder")
                return match.group(0)
        
        return env_value
    
    # Match ${VAR} or ${VAR:-default}
    return re.sub(r'\$\{([^}]+)\}', replace_var, text)


def load_suite(config_path: str) -> SuiteSchema:
    """
    Load and validate suite configuration from YAML file with environment variable expansion.
    
    Args:
        config_path: Path to YAML suite file
        
    Returns:
        Validated suite schema
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Suite file not found: {config_path}")
        
    with open(config_file, "r", encoding="utf-8") as f:
        raw_content = f.read()
    
    # Expand environment variables
    expanded_content = expand_environment_variables(raw_content)
    logger.debug(f"Expanded environment variables in suite config: {config_path}")
    
    # Parse the expanded YAML
    config_data = yaml.safe_load(expanded_content) or {}
    
    try:
        suite = SuiteSchema(**config_data)
        return suite
    except ValidationError as e:
        logger.error(f"Suite validation failed: {e}")
        raise


def load_dataset(dataset_path: str, strategy: str = None) -> DatasetSchema:
    """
    Load and validate dataset from YAML file or folder structure.
    
    Args:
        dataset_path: Path to YAML dataset file or folder containing images
        strategy: Optional strategy name to validate dataset type compatibility
        
    Returns:
        Validated dataset schema
    """
    dataset_path_obj = Path(dataset_path)
    
    # For advanced_pdf strategy, only allow folder datasets
    if strategy == "advanced_pdf":
        if not dataset_path_obj.is_dir():
            raise ValueError(
                f"For strategy 'advanced_pdf', dataset must be a folder containing PDF files, "
                f"not a YAML file. Please use a folder path like 'data/datasets/corello' instead of '{dataset_path}'"
            )
        logger.info(f"Using folder dataset for advanced_pdf strategy: {dataset_path}")
        return load_folder_dataset(dataset_path_obj)
    
    # Check if it's a directory (folder-based dataset)
    if dataset_path_obj.is_dir():
        return load_folder_dataset(dataset_path_obj)
    
    # Otherwise, treat as YAML file
    if not dataset_path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    with open(dataset_path_obj, "r", encoding="utf-8") as f:
        raw_content = f.read()
    
    # Expand environment variables
    expanded_content = expand_environment_variables(raw_content)
    logger.debug(f"Expanded environment variables in dataset: {dataset_path}")
    
    # Parse the expanded YAML
    dataset_data = yaml.safe_load(expanded_content) or {}
    
    try:
        dataset = DatasetSchema(**dataset_data)
        return dataset
    except ValidationError as e:
        logger.error(f"Dataset validation failed: {e}")
        raise


def load_folder_dataset(folder_path: Path) -> DatasetSchema:
    """
    Load dataset from a folder structure containing images and PDFs.
    
    Args:
        folder_path: Path to folder containing images and PDF files
        
    Returns:
        Validated dataset schema with auto-generated tasks for images and PDFs
    """
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
    
    # Supported file extensions (images and PDFs)
    supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.pdf'}
    
    # Find all supported files in the folder
    supported_files = []
    for ext in supported_extensions:
        supported_files.extend(folder_path.glob(f'*{ext}'))
        supported_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    if not supported_files:
        raise ValueError(f"No supported files (images/PDFs) found in folder: {folder_path}")
    
    # Sort files for consistent ordering
    supported_files.sort()
    
    # Generate tasks from files
    image_tasks = []
    pdf_tasks = []
    for i, file_path in enumerate(supported_files, 1):
        # Generate task ID from filename
        task_id = f"file_{i:03d}_{file_path.stem}"
        
        # Determine task type from filename patterns and file extension
        filename_lower = file_path.name.lower()
        file_ext = file_path.suffix.lower()
        
        # Set base properties based on file type
        if file_ext == '.pdf':
            base_prompt_prefix = "Analyze this PDF document and "
            file_type = "pdf"
        else:
            base_prompt_prefix = "Analyze this image and "
            file_type = "image"
        
        if 'handwriting' in filename_lower or 'handwritten' in filename_lower:
            task_type = "handwriting_recognition"
            category = "handwriting"
            prompt = f"{base_prompt_prefix}read the handwritten text, transcribing it exactly as written."
        elif 'form' in filename_lower:
            task_type = "structured_extraction"
            category = "form_extraction"
            prompt = f"{base_prompt_prefix}extract the form data, including field names and values."
        elif 'table' in filename_lower:
            task_type = "table_recognition"
            category = "table_extraction"
            prompt = f"{base_prompt_prefix}extract the table data and present it in a structured format."
        elif 'invoice' in filename_lower or 'receipt' in filename_lower:
            task_type = "numerical_extraction"
            category = "invoice_processing"
            prompt = f"{base_prompt_prefix}identify and extract all numerical data and amounts."
        else:
            task_type = "text_extraction"
            category = "document_ocr"
            prompt = f"{base_prompt_prefix}extract all text. Focus on accuracy and completeness."
        
        # Create task data with appropriate path field
        if file_type == "pdf":
            task_data = {
                "id": task_id,
                "prompt": prompt,
                "pdf_path": str(file_path),
                "expected": f"[Expected output for {file_path.name}]",  # Placeholder
                "category": category,
                "difficulty": "medium",
                "task_type": task_type
            }
            pdf_tasks.append(task_data)
        else:
            task_data = {
                "id": task_id,
                "prompt": prompt,
                "image_path": str(file_path),
                "expected": f"[Expected output for {file_path.name}]",  # Placeholder
                "category": category,
                "difficulty": "medium",
                "task_type": task_type
            }
            image_tasks.append(task_data)
    
    # Create dataset structure
    dataset_data = {
        "dataset": {
            "name": f"Folder Dataset: {folder_path.name}",
            "description": f"Auto-generated dataset from folder: {folder_path}",
            "version": "1.0"
        }
    }
    
    # Add tasks if they exist
    if image_tasks:
        dataset_data["dataset"]["image_tasks"] = image_tasks
    if pdf_tasks:
        dataset_data["dataset"]["pdf_tasks"] = pdf_tasks
    
    total_tasks = len(image_tasks) + len(pdf_tasks)
    
    try:
        dataset = DatasetSchema(**dataset_data)
        logger.info(f"Loaded folder dataset with {total_tasks} tasks ({len(image_tasks)} images, {len(pdf_tasks)} PDFs) from: {folder_path}")
        return dataset
    except ValidationError as e:
        logger.error(f"Folder dataset validation failed: {e}")
        raise


def resolve_all_provider_credentials(providers: Dict[str, ProviderConfig]) -> Dict[str, Any]:
    """Resolve credentials for all providers in the configuration."""
    resolved_providers = {}
    
    for provider_name, provider_config in providers.items():
        try:
            resolved_providers[provider_name] = resolve_provider_credentials(provider_config)
            logger.debug(f"Resolved credentials for provider: {provider_name}")
        except ValueError as e:
            logger.error(f"Failed to resolve credentials for {provider_name}: {e}")
            raise
    
    return resolved_providers

def get_api_key(service: str) -> str:
    """Get API key for a service from environment variables."""
    env_var = f"{service.upper()}_API_KEY"
    api_key = os.getenv(env_var)
    
    if not api_key:
        raise ValueError(f"API key not found. Please set {env_var} environment variable.")
    
    return api_key