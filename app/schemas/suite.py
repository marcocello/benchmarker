"""Pydantic schemas for suite and dataset validation."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ProviderConfig(BaseModel):
    """Configuration for a model provider."""
    type: str
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    deployment: Optional[str] = None
    model: Optional[str] = None
    defaults: Dict[str, Any] = Field(default_factory=dict)


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
