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


def load_dataset(dataset_path: str) -> DatasetSchema:
    """
    Load and validate dataset from YAML file with environment variable expansion.
    
    Args:
        dataset_path: Path to YAML dataset file
        
    Returns:
        Validated dataset schema
    """
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    with open(dataset_file, "r", encoding="utf-8") as f:
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