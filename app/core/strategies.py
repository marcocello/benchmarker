"""Strategy implementations for different execution methods."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import openai
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from crewai import Agent, Task, Crew
from crewai.llm import LLM

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Abstract base class for all strategies."""
    
    def __init__(self, provider_settings: Dict[str, Any]):
        self.provider_settings = provider_settings
    
    @abstractmethod
    async def execute(self, prompt: str, **kwargs) -> str:
        """Execute the strategy with the given prompt."""
        pass


class DirectPromptStrategy(Strategy):
    """Strategy for direct prompting using LangChain."""
    
    async def execute(self, prompt: str, **kwargs) -> str:
        """Execute direct prompt strategy using LangChain."""
        try:
            # Set up LangChain Azure OpenAI client
            if self.provider_settings.api_type == 'azure_openai':
                llm = AzureChatOpenAI(
                    azure_endpoint=self.provider_settings.api_base,
                    api_key=self.provider_settings.api_key,
                    api_version=self.provider_settings.api_version,
                    deployment_name=self.provider_settings.deployment,
                    model_name=self.provider_settings.deployment,
                    temperature=self.provider_settings.default_parameters.get('temperature', 0.1),
                    max_tokens=self.provider_settings.default_parameters.get('max_tokens', 100)
                )
                
                # Create message and get response
                messages = [HumanMessage(content=prompt)]
                response = await llm.ainvoke(messages)
                
                return response.content
            else:
                # Fallback for unsupported provider types
                return f"Error: Provider type '{self.provider_settings.api_type}' not supported"
                
        except Exception as e:
            logger.error(f"Error in direct prompt strategy: {e}")
            return f"Error: {str(e)}"


class LLMJudgeStrategy(Strategy):
    """Strategy for LLM-based scoring using LangChain."""
    
    async def execute(self, prompt: str, **kwargs) -> str:
        """Execute LLM judge strategy using LangChain."""
        try:
            # Set up LangChain Azure OpenAI client
            if self.provider_settings.api_type == 'azure_openai':
                llm = AzureChatOpenAI(
                    azure_endpoint=self.provider_settings.api_base,
                    api_key=self.provider_settings.api_key,
                    api_version=self.provider_settings.api_version,
                    deployment_name=self.provider_settings.deployment,
                    model_name=self.provider_settings.deployment,
                    temperature=self.provider_settings.default_parameters.get('temperature', 0.1),
                    max_tokens=self.provider_settings.default_parameters.get('max_tokens', 100)
                )
                
                # Create message and get response
                messages = [HumanMessage(content=prompt)]
                response = await llm.ainvoke(messages)
                
                return response.content
            else:
                # Fallback for unsupported provider types
                return f'{{"score": 0, "explanation": "Error: Provider type \'{self.provider_settings.api_type}\' not supported for scoring"}}'
                
        except Exception as e:
            logger.error(f"Error in LLM judge strategy: {e}")
            return '{"score": 0, "explanation": "Error in scoring"}'


class AgenticStrategy(Strategy):
    """Strategy for agentic execution using CrewAI."""
    
    async def execute(self, prompt: str, **kwargs) -> str:
        """Execute agentic strategy using CrewAI."""
        try:
            # Get scenario configuration from kwargs
            scenario_config = kwargs.get('scenario_config')
            if not scenario_config:
                return "Error: No scenario configuration provided for agentic strategy"
                
            # Set up CrewAI LLM with Azure OpenAI
            if self.provider_settings.api_type == 'azure_openai':
                # Configure LLM for Azure OpenAI
                llm = LLM(
                    model="azure/" + self.provider_settings.deployment,
                    api_key=self.provider_settings.api_key,
                    base_url=self.provider_settings.api_base,
                    api_version=self.provider_settings.api_version,
                    temperature=self.provider_settings.default_parameters.get('temperature', 0.1),
                    max_tokens=self.provider_settings.default_parameters.get('max_tokens', 100)
                )
                
                # Run CrewAI synchronously and return the result
                return self._run_crew_sync(llm, prompt, scenario_config)
                    
            else:
                # Fallback for unsupported provider types
                return f"Error: Provider type '{self.provider_settings.api_type}' not supported"
                
        except Exception as e:
            logger.error(f"Error in agentic strategy: {e}")
            return f"Error: {str(e)}"
    
    def _run_crew_sync(self, llm, prompt: str, scenario_config) -> str:
        """Run CrewAI crew synchronously."""
        try:
            # Check if agents are configured
            if not scenario_config.agents:
                return "Error: No agents configured for agentic strategy"
                
            # Create agents from configuration
            agents = []
            for agent_config in scenario_config.agents:
                agent = Agent(
                    role=agent_config.role,
                    goal=agent_config.goal,
                    backstory=agent_config.backstory,
                    llm=llm,
                    verbose=False
                )
                agents.append(agent)
            
            # Create tasks - for simplicity, we'll create one task per agent
            # The last agent will be responsible for the final synthesis
            tasks = []
            task_template = scenario_config.task_template or "Answer the following question: {question}"
            
            for i, (agent, agent_config) in enumerate(zip(agents, scenario_config.agents)):
                if i == 0:
                    # First agent gets the original question
                    task_description = task_template.format(question=prompt)
                elif i == len(agents) - 1:
                    # Last agent synthesizes all previous work
                    task_description = f"Based on all previous research and analysis, provide a final, comprehensive answer to: {prompt}"
                else:
                    # Middle agents analyze previous work
                    task_description = f"Analyze and verify the previous research for the question: {prompt}"
                
                # Add instructions if available
                if agent_config.instructions:
                    task_description += f"\n\nSpecific instructions: {agent_config.instructions}"
                
                task = Task(
                    description=task_description,
                    agent=agent,
                    expected_output=f"Clear and accurate response following the {agent_config.role} role."
                )
                tasks.append(task)
            
            # Create and execute the crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                verbose=False
            )
            
            # Execute the crew and return the result
            result = crew.kickoff()
            
            # Extract the final answer from the crew result
            if hasattr(result, 'raw'):
                return str(result.raw)
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Error in crew execution: {e}")
            return f"Error: {str(e)}"


class StrategyFactory:
    """Factory for creating strategy instances."""
    
    _strategies = {
        'direct_prompt': DirectPromptStrategy,
        'llm_judge': LLMJudgeStrategy,
        'agentic': AgenticStrategy,
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, provider_settings: Dict[str, Any]) -> Strategy:
        """Create a strategy instance."""
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(provider_settings)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all available strategies."""
        return list(cls._strategies.keys())
