"""Strategy implementations for different execution methods."""

import json
import logging
import base64
import time
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from datetime import datetime
from pathlib import Path
from PIL import Image
import io

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
    
    def get_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage from LangChain response"""
        usage_metadata = getattr(response, 'response_metadata', {})
        token_usage = usage_metadata.get('token_usage', {})
        
        return {
            "input_tokens": token_usage.get('prompt_tokens', 0),
            "output_tokens": token_usage.get('completion_tokens', 0),
            "total_tokens": token_usage.get('total_tokens', 0)
        }
    
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
                
                # Log token usage
                token_usage = self.get_token_usage(response)
                logger.info(f"Direct prompt token usage: {token_usage}")
                
                return response.content
            else:
                # Fallback for unsupported provider types
                return f"Error: Provider type '{self.provider_settings.api_type}' not supported"
                
        except Exception as e:
            logger.error(f"Error in direct prompt strategy: {e}")
            return f"Error: {str(e)}"


class LLMJudgeStrategy(Strategy):
    """Strategy for LLM-based scoring using LangChain."""
    
    def get_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage from LangChain response"""
        usage_metadata = getattr(response, 'response_metadata', {})
        token_usage = usage_metadata.get('token_usage', {})
        
        return {
            "input_tokens": token_usage.get('prompt_tokens', 0),
            "output_tokens": token_usage.get('completion_tokens', 0),
            "total_tokens": token_usage.get('total_tokens', 0)
        }
    
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
                
                # Log token usage
                token_usage = self.get_token_usage(response)
                logger.info(f"LLM judge token usage: {token_usage}")
                
                return response.content
            else:
                # Fallback for unsupported provider types
                return f'{{"score": 0, "explanation": "Error: Provider type \'{self.provider_settings.api_type}\' not supported for scoring"}}'
                
        except Exception as e:
            logger.error(f"Error in LLM judge strategy: {e}")
            return '{"score": 0, "explanation": "Error in scoring"}'


class AdvancedPDFStrategy(Strategy):
    """Strategy for advanced PDF analysis with structured data extraction, similar to Kimi LLM processing."""
    
    def __init__(self, provider_settings: Dict[str, Any]):
        super().__init__(provider_settings)
    
    def get_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage from LangChain response"""
        usage_metadata = getattr(response, 'response_metadata', {})
        token_usage = usage_metadata.get('token_usage', {})
        
        return {
            "input_tokens": token_usage.get('prompt_tokens', 0),
            "output_tokens": token_usage.get('completion_tokens', 0),
            "total_tokens": token_usage.get('total_tokens', 0)
        }
    
    def preprocess_image_for_llm(self, image: Image.Image) -> Image.Image:
        """Preprocess image for optimal LLM understanding"""
        # Resize if too large
        max_dimension = 2048
        width, height = image.size
        
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * max_dimension / width)
            else:
                new_height = max_dimension
                new_width = int(width * max_dimension / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    async def process_pdf_with_vision(self, pdf_path: str, prompt: str, task_type: str = None, max_retries: int = 3) -> Dict[str, Any]:
        """Process PDF using vision model with retry logic and token tracking"""
        
        for attempt in range(max_retries):
            try:
                # Convert PDF to images using pdf2image
                try:
                    from pdf2image import convert_from_path
                except ImportError:
                    logger.error("pdf2image not available. Install with: pip install pdf2image")
                    return {"error": "pdf2image library not available"}
                
                # Convert PDF to images
                images = convert_from_path(pdf_path, dpi=300, fmt='png')
                logger.info(f"Converted PDF to {len(images)} images")
                
                if not images:
                    return {"error": "No pages found in PDF"}
                
                # Process each page
                all_pages_data = []
                total_input_tokens = 0
                total_output_tokens = 0
                
                for page_num, image in enumerate(images, 1):
                    logger.debug(f"Processing page {page_num}/{len(images)}")
                    
                    # Preprocess image
                    processed_image = self.preprocess_image_for_llm(image)
                    image_base64 = self.encode_image_to_base64(processed_image)
                    
                    # Set up Azure OpenAI client using LangChain
                    if self.provider_settings.api_type == 'azure_openai':
                        llm = AzureChatOpenAI(
                            azure_endpoint=self.provider_settings.api_base,
                            api_key=self.provider_settings.api_key,
                            api_version=self.provider_settings.api_version,
                            deployment_name=self.provider_settings.deployment,
                            model_name=self.provider_settings.deployment,
                            temperature=self.provider_settings.default_parameters.get('temperature', 0.1),
                            max_tokens=self.provider_settings.default_parameters.get('max_tokens', 2000)
                        )
                        
                        # Create message for vision processing
                        # Convert the message content to HumanMessage format
                        text_content = prompt
                        image_url = f"data:image/png;base64,{image_base64}"
                        
                        # Create HumanMessage with vision content
                        message = HumanMessage(
                            content=[
                                {"type": "text", "text": text_content},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                        "detail": "high"
                                    }
                                }
                            ]
                        )
                        
                        # Make API request using LangChain
                        response = await llm.ainvoke([message])
                        content = response.content
                        
                        # Get token usage information
                        token_usage = self.get_token_usage(response)
                        input_tokens = token_usage["input_tokens"]
                        output_tokens = token_usage["output_tokens"]
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                        
                        # Log token usage for this page
                        logger.info(f"Page {page_num} token usage: {token_usage}")
                        
                        # Parse JSON response
                        try:
                            extracted_data = json.loads(content)
                        except json.JSONDecodeError:
                            # Try to extract JSON from the content
                            import re
                            json_match = re.search(r'\{.*\}', content, re.DOTALL)
                            if json_match:
                                try:
                                    extracted_data = json.loads(json_match.group())
                                except json.JSONDecodeError:
                                    extracted_data = {"error": "Unable to parse JSON from response", "raw_content": content}
                            else:
                                extracted_data = {"error": "No JSON found in response", "raw_content": content}
                        
                        page_result = {
                            "page_number": page_num,
                            "extracted_data": extracted_data,
                            "token_usage": token_usage,
                            "processing_time": time.time()
                        }
                        
                        all_pages_data.append(page_result)
                    
                    else:
                        return {"error": f"Provider type '{self.provider_settings.api_type}' not supported for advanced PDF processing"}
                
                # Combine data from all pages
                combined_result = self._combine_page_data(all_pages_data, pdf_path, total_input_tokens, total_output_tokens)
                return combined_result
                
            except Exception as e:
                logger.error(f"Error in PDF processing attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return {"error": f"PDF processing failed after {max_retries} attempts: {str(e)}"}
        
        return {"error": "Max retries exceeded"}
    
    def _combine_page_data(self, pages_data: List[Dict], pdf_path: str, total_input_tokens: int, total_output_tokens: int) -> Dict[str, Any]:
        """Combine extracted data from all PDF pages into a unified structure"""
        
        combined_data = {
            "pdf_file": Path(pdf_path).name,
            "total_pages": len(pages_data),
            "total_token_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            },
            "average_tokens_per_page": {
                "input_tokens": total_input_tokens // len(pages_data) if len(pages_data) > 0 else 0,
                "output_tokens": total_output_tokens // len(pages_data) if len(pages_data) > 0 else 0
            },
            "pages": pages_data,
            "summary": {
                "document_info": {},
                "process_specs": {},
                "quality_specs": {},
                "production_data": {},
                "process_steps": [],
                "test_results": [],
                "inspection_data": [],
                "quality_defects": [],
                "approvals": [],
                "tables": [],
                "numerical_data": [],
                "form_fields": {}
            }
        }
        
        # Aggregate data from all pages
        for page_data in pages_data:
            if "extracted_data" in page_data and not page_data["extracted_data"].get("error"):
                extracted = page_data["extracted_data"]
                
                # Merge document info (first non-empty values win)
                if "document_info" in extracted:
                    for key, value in extracted["document_info"].items():
                        if value and value != "" and key not in combined_data["summary"]["document_info"]:
                            combined_data["summary"]["document_info"][key] = value
                
                # Merge other sections, extending lists and updating dicts
                for section in ["process_specs", "quality_specs", "production_data", "form_fields"]:
                    if section in extracted:
                        if isinstance(extracted[section], dict):
                            combined_data["summary"][section].update(extracted[section])
                
                # Extend list sections
                for section in ["process_steps", "test_results", "inspection_data", "quality_defects", "approvals", "numerical_data"]:
                    if section in extracted and isinstance(extracted[section], list):
                        combined_data["summary"][section].extend(extracted[section])
                
                # Handle tables specially
                if "tables" in extracted and isinstance(extracted["tables"], list):
                    for table in extracted["tables"]:
                        table["source_page"] = page_data["page_number"]
                        combined_data["summary"]["tables"].append(table)
        
        return combined_data
    
    async def execute(self, prompt: str, **kwargs) -> str:
        """Execute advanced PDF strategy with structured data extraction"""
        try:
            # Get PDF path from kwargs
            pdf_path = kwargs.get('pdf_path')
            task_type = kwargs.get('task_type', 'comprehensive')
            
            if not pdf_path:
                return json.dumps({"error": "No PDF path provided for advanced PDF strategy"})
            
            # Process PDF with advanced extraction
            result = await self.process_pdf_with_vision(pdf_path, prompt, task_type)
            
            # Return as JSON string for consistency with other strategies
            return json.dumps(result, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error in advanced PDF strategy: {e}")
            return json.dumps({"error": str(e)})


class AgenticStrategy(Strategy):
    """Strategy for agentic execution using CrewAI."""
    
    def get_token_usage(self, response=None) -> Dict[str, int]:
        """Extract token usage - CrewAI may not provide detailed token usage"""
        # CrewAI doesn't always provide token usage in the same way
        # This is a placeholder for potential future implementation
        logger.info("Token usage tracking for CrewAI not fully implemented")
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    
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
                result = self._run_crew_sync(llm, prompt, scenario_config)
                
                # Log token usage (placeholder)
                token_usage = self.get_token_usage()
                logger.info(f"Agentic strategy token usage: {token_usage}")
                
                return result
                    
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
        'advanced_pdf': AdvancedPDFStrategy,
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
