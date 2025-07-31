"""Settings management for environment variables and provider credentials."""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from app.schemas.suite import ProviderConfig


class AppSettings(BaseModel):
    """Application-wide settings."""
    LOG_LEVEL: str = Field(default="INFO", description="Default logging level")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override with environment variable if set
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)


# Global settings instance
settings = AppSettings()


class ProviderSettings(BaseModel):
    """Runtime settings for a provider with resolved credentials."""
    api_key: str
    api_type: str
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    deployment: Optional[str] = None
    default_model: str
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    # OpenRouter specific fields
    site_url: Optional[str] = None
    site_name: Optional[str] = None


def resolve_provider_credentials(provider_config: ProviderConfig) -> ProviderSettings:
    """Resolve provider configuration by loading API key from environment or config."""
    api_key = provider_config.api_key
    if not api_key and getattr(provider_config, 'api_key_env', None):
        api_key = os.getenv(provider_config.api_key_env)
    
    if not api_key:
        env_name = getattr(provider_config, 'api_key_env', None) or f"{provider_config.type.upper()}_API_KEY"
        raise ValueError(f"API key not found. Set {env_name} environment variable or add api_key to config.")
    
    # Handle different endpoint field names
    api_base = getattr(provider_config, 'endpoint', None)
    
    # For OpenRouter, set the default base URL
    if provider_config.type == 'openrouter':
        api_base = "https://openrouter.ai/api/v1"
    
    # Determine default model
    default_model = (
        provider_config.model or 
        getattr(provider_config, 'deployment', None) or 
        "gpt-3.5-turbo"
    )
    
    # Extract OpenRouter specific fields
    site_url = getattr(provider_config, 'site_url', None)
    site_name = getattr(provider_config, 'site_name', None)
    
    return ProviderSettings(
        api_key=api_key,
        api_type=provider_config.type,
        api_base=api_base,
        api_version=getattr(provider_config, 'api_version', None),
        deployment=getattr(provider_config, 'deployment', None),
        default_model=default_model,
        default_parameters=provider_config.defaults,
        site_url=site_url,
        site_name=site_name,
    )


def get_effective_model_config(
    scenario_provider: str,
    scenario_model: Optional[str],
    scenario_parameters: Dict[str, Any],
    provider_settings: ProviderSettings,
) -> Dict[str, Any]:
    """Get effective model configuration by merging provider defaults with scenario overrides."""
    effective_config = {
        "provider": scenario_provider,
        "model": scenario_model or provider_settings.default_model,
        "api_key": provider_settings.api_key,
        "api_type": provider_settings.api_type,
        "api_base": provider_settings.api_base,
        "api_version": provider_settings.api_version,
        "deployment": provider_settings.deployment,
        **provider_settings.default_parameters,
        **scenario_parameters,
    }
    
    return {k: v for k, v in effective_config.items() if v is not None}
