"""Suite runner service."""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
import openai

from app.schemas.suite import SuiteSchema
from app.schemas.dataset import DatasetSchema
from app.core.suite import load_dataset, resolve_all_provider_credentials
from app.core.strategies import StrategyFactory
from app.core.logging import get_logger

logger = get_logger(__name__)


class SuiteRunner:
    """Service for running benchmark suites."""
    
    def __init__(
        self,
        suite: SuiteSchema,
        verbose: bool = False,
    ) -> None:
        self.suite = suite
        self.verbose = verbose
        self.console = Console()
        self.provider_settings = resolve_all_provider_credentials(suite.suite.providers)
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    async def execute(self) -> None:
        """Execute all scenarios in the suite."""
        suite_config = self.suite.suite
        logger.info(f"Starting suite execution: {suite_config.name}")
        logger.debug(f"Providers configured: {list(suite_config.providers.keys())}")
        logger.debug(f"Number of scenarios: {len(suite_config.scenarios)}")
        
        if self.verbose:
            self.console.print(f"[green]Suite: {suite_config.name}[/green]")
            self.console.print(f"[blue]Providers: {list(suite_config.providers.keys())}[/blue]")
            self.console.print(f"[yellow]Scenarios: {len(suite_config.scenarios)}[/yellow]")
        
        # Load dataset
        logger.info(f"Loading dataset: {suite_config.dataset}")
        dataset = load_dataset(suite_config.dataset)
        logger.debug(f"Dataset loaded: {len(dataset.dataset.questions)} questions")
        
        results: List[Dict[str, Any]] = []
        
        # Execute scenarios
        for i, scenario in enumerate(suite_config.scenarios, 1):
            logger.info(f"Executing scenario {i}/{len(suite_config.scenarios)}: {scenario.name} (strategy: {scenario.strategy})")
            await self._execute_scenario(scenario, dataset, results)
        
        # Generate report
        logger.info("Generating final report")
        self._report(results)
        logger.info("Suite execution completed successfully")
    
    async def _execute_scenario(self, scenario, dataset: DatasetSchema, results: List[Dict[str, Any]]) -> None:
        """Execute a single scenario against the dataset."""
        try:
            logger.debug(f"Starting scenario: {scenario.name} with provider: {scenario.provider}")
            start_time = time.time()
            
            # Execute all questions in the dataset
            question_results = []
            total_questions = len(dataset.dataset.questions)
            
            for i, question in enumerate(dataset.dataset.questions, 1):
                logger.debug(f"Processing question {i}/{total_questions} (ID: {question.id})")
                question_start = time.time()
                
                # Generate response using the provider
                response = await self._generate_response(question.prompt, scenario)
                
                question_end = time.time()
                
                question_results.append({
                    "question_id": question.id,
                    "question": question.prompt,
                    "expected": question.expected,
                    "response": response,
                    "category": question.category,
                    "difficulty": question.difficulty,
                    "latency": question_end - question_start
                })
                
                logger.debug(f"Question {question.id} completed in {question_end - question_start:.2f}s")
                if self.verbose:
                    self.console.print(f"[dim]Question {question.id}: {response[:50]}...[/dim]")
            
            end_time = time.time()
            total_latency = end_time - start_time
            logger.info(f"Scenario {scenario.name} completed: {total_questions} questions in {total_latency:.2f}s")
            
            # Run scoring if configured
            scoring_result = None
            if hasattr(self.suite.suite, 'scoring') and self.suite.suite.scoring:
                logger.debug("Running scoring evaluation")
                scoring_result = await self._run_scoring(question_results, self.suite.suite.scoring)
                if scoring_result and 'average_score' in scoring_result:
                    logger.info(f"Scoring completed: average score {scoring_result['average_score']}")
            
            result = {
                "scenario": scenario.name,
                "provider": scenario.provider,
                "strategy": scenario.strategy,
                "dataset": dataset.dataset.name,
                "questions_count": len(dataset.dataset.questions),
                "questions": question_results,
                "scoring": scoring_result,
                "latency": end_time - start_time,
                "timestamp": start_time,
                "status": "success"
            }
            
            results.append(result)
            
            if self.verbose:
                self.console.print(f"[green]✓ Scenario {scenario.name} completed[/green]")
                
        except Exception as e:
            logger.error(f"Error executing scenario {scenario.name}: {e}")
            results.append({
                "scenario": scenario.name,
                "provider": scenario.provider,
                "error": str(e),
                "latency": 0.0,
                "timestamp": time.time(),
                "status": "error"
            })
    
    async def _generate_response(self, prompt: str, scenario) -> str:
        """Generate a response using the configured provider and strategy."""
        try:
            if scenario.provider not in self.provider_settings:
                raise ValueError(f"Provider {scenario.provider} not found in settings")
            
            provider_settings = self.provider_settings[scenario.provider]
            
            # Create strategy instance
            strategy = StrategyFactory.create_strategy(scenario.strategy, provider_settings)
            
            # Execute the strategy with scenario configuration
            return await strategy.execute(prompt, scenario_config=scenario)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    async def _run_scoring(self, question_results: List[Dict[str, Any]], scoring_config) -> Dict[str, Any]:
        """Run scoring evaluation on the results."""
        try:
            scores = []
            
            for result in question_results:
                # Format the scoring prompt
                scoring_prompt = scoring_config.prompt.format(
                    question=result["question"],
                    expected_answer=result["expected"],
                    response=result["response"]
                )
                
                # Generate score using the scoring provider
                score_response = await self._generate_scoring_response(scoring_prompt, scoring_config)
                
                # Clean up the response (remove markdown formatting if present)
                cleaned_response = score_response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()
                
                # Parse the JSON response
                try:
                    score_data = json.loads(cleaned_response)
                    scores.append({
                        "question_id": result["question_id"],
                        "score": score_data.get("score", 0),
                        "explanation": score_data.get("explanation", "No explanation provided"),
                        "max_score": scoring_config.max_score
                    })
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse scoring response: {cleaned_response}")
                    scores.append({
                        "question_id": result["question_id"],
                        "score": 0,
                        "explanation": f"Failed to parse scoring response: {e}",
                        "max_score": scoring_config.max_score
                    })
            
            # Calculate summary statistics
            total_score = sum(s["score"] for s in scores)
            max_possible = len(scores) * scoring_config.max_score
            average_score = total_score / len(scores) if scores else 0
            
            return {
                "scores": scores,
                "total_score": total_score,
                "max_possible": max_possible,
                "average_score": average_score,
                "percentage": (total_score / max_possible * 100) if max_possible > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return {"error": str(e)}
    
    async def _generate_scoring_response(self, prompt: str, scoring_config) -> str:
        """Generate a scoring response using the configured provider and strategy."""
        try:
            if scoring_config.provider not in self.provider_settings:
                raise ValueError(f"Scoring provider {scoring_config.provider} not found in settings")
            
            provider_settings = self.provider_settings[scoring_config.provider]
            
            # Create strategy instance
            strategy = StrategyFactory.create_strategy(scoring_config.strategy, provider_settings)
            
            # Execute the strategy
            return await strategy.execute(prompt)
                
        except Exception as e:
            logger.error(f"Error generating scoring response: {e}")
            return '{"score": 0, "explanation": "Error in scoring"}'
    
    def _get_suite_folder_name(self, suite_name: str) -> str:
        """Convert suite name to a folder-friendly format (lowercase, spaces/special chars to underscores) with short UUID."""
        import re
        # Convert to lowercase and replace spaces and special characters with underscores
        folder_name = re.sub(r'[^\w\s-]', '', suite_name.lower())  # Remove special chars except spaces and hyphens
        folder_name = re.sub(r'[-\s]+', '_', folder_name)  # Replace spaces and hyphens with underscores
        folder_name = folder_name.strip('_')  # Remove leading/trailing underscores
        
        # Add short UUID (first 8 characters)
        short_uuid = str(uuid.uuid4())[:8]
        return f"{folder_name}_{short_uuid}"
    
    def _report(self, results: List[Dict[str, Any]]) -> None:
        """Generate and display the suite results report."""
        if not results:
            self.console.print("[red]No results to report[/red]")
            return
        
        table = Table(title="Suite Results")
        table.add_column("Scenario", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Questions", style="yellow")
        table.add_column("Avg Score", style="blue")
        table.add_column("Status", style="yellow")
        table.add_column("Latency", style="blue", justify="right")
        
        for result in results:
            status = "✓ Success" if result["status"] == "success" else "✗ Error"
            
            # Get average score if available
            avg_score = "N/A"
            if result.get("scoring") and "average_score" in result["scoring"]:
                avg_score = f"{result['scoring']['average_score']:.1f}"
            
            table.add_row(
                result["scenario"],
                result.get("provider", "N/A"),
                str(result.get("questions_count", 0)),
                avg_score,
                status,
                f"{result['latency']:.2f}s"
            )
        
        self.console.print(table)
        
        # Save results to file
        suite_config = self.suite.suite
        output_filename = suite_config.output.name
        
        # Create suite-specific folder structure with UUID
        suite_folder_name = self._get_suite_folder_name(suite_config.name)
        
        # Construct the full path: data/results/suite_folder/filename
        base_dir = Path("data/results")
        new_path = base_dir / suite_folder_name / output_filename
        
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        
        self.console.print(f"\n[dim]Results saved to {new_path}[/dim]")
