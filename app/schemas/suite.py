"""Pydantic schemas for suite and dataset validation."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal, Annotated


class AzureOpenAIProviderConfig(BaseModel):
    """Configuration for Azure OpenAI provider."""
    type: Literal["azure_openai"]
    api_key: Optional[str] = None
    endpoint: str
    api_version: str
    deployment: str
    model: str
    defaults: Dict[str, Any] = Field(default_factory=dict)

class AnthropicProviderConfig(BaseModel):
    """Configuration for Anthropic provider."""
    type: Literal["anthropic"]
    api_key: str
    model: str
    defaults: Dict[str, Any] = Field(default_factory=dict)


class HuggingFaceProviderConfig(BaseModel):
    """Configuration for HuggingFace provider."""
    type: Literal["huggingface"]
    api_key: str
    model: str
    endpoint: Optional[str] = None
    defaults: Dict[str, Any] = Field(default_factory=dict)


class OpenRouterProviderConfig(BaseModel):
    """Configuration for OpenRouter provider."""
    type: Literal["openrouter"]
    api_key: str
    model: str
    site_url: Optional[str] = None
    site_name: Optional[str] = None
    defaults: Dict[str, Any] = Field(default_factory=dict)


# Discriminated union for provider configurations
ProviderConfig = Annotated[
    Union[
        AzureOpenAIProviderConfig,
        AnthropicProviderConfig,
        HuggingFaceProviderConfig,
        OpenRouterProviderConfig
    ],
    Field(discriminator="type")
]


class AgentConfig(BaseModel):
    """Configuration for an agent in agentic strategy."""
    name: str
    role: str
    goal: str
    backstory: str
    instructions: Optional[str] = None


class ScenarioConfig(BaseModel):
    """Configuration for a test scenario."""
    name: str
    strategy: str
    provider: str
    prompt_path: Optional[str] = None
    agents: Optional[List[AgentConfig]] = None
    task_template: Optional[str] = None


class ScoringConfig(BaseModel):
    """Scoring configuration."""
    name: str
    strategy: str
    provider: str
    max_score: int = 10
    prompt: str
    scoring: List[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    """Output configuration."""
    name: str


class SuiteConfig(BaseModel):
    """Suite configuration."""
    name: str
    description: Optional[str] = None
    dataset: str
    scenarios: List[ScenarioConfig]
    scoring: Optional[ScoringConfig] = None
    output: OutputConfig
    providers: Dict[str, ProviderConfig]


class SuiteSchema(BaseModel):
    """Main suite schema for benchmarker."""
    suite: SuiteConfig
