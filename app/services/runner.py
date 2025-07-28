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
        
        # Check if any scenario uses advanced_pdf strategy
        advanced_pdf_strategies = [scenario.strategy for scenario in suite_config.scenarios if scenario.strategy == "advanced_pdf"]
        strategy_for_validation = advanced_pdf_strategies[0] if advanced_pdf_strategies else None
        
        dataset = load_dataset(suite_config.dataset, strategy_for_validation)
        
        # Count total tasks (questions + image tasks + PDF tasks)
        question_count = len(dataset.dataset.questions) if dataset.dataset.questions else 0
        image_task_count = len(dataset.dataset.image_tasks) if dataset.dataset.image_tasks else 0
        pdf_task_count = len(dataset.dataset.pdf_tasks) if dataset.dataset.pdf_tasks else 0
        total_tasks = question_count + image_task_count + pdf_task_count
        
        logger.debug(f"Dataset loaded: {question_count} questions, {image_task_count} image tasks, {pdf_task_count} PDF tasks")
        
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
            
            # Execute all tasks in the dataset (questions + image tasks)
            task_results = []
            
            # Process text questions if they exist
            if dataset.dataset.questions:
                total_questions = len(dataset.dataset.questions)
                logger.debug(f"Processing {total_questions} text questions")
                
                for i, question in enumerate(dataset.dataset.questions, 1):
                    logger.debug(f"Processing question {i}/{total_questions} (ID: {question.id})")
                    question_start = time.time()
                    
                    # Generate response using the provider
                    response = await self._generate_response(question.prompt, scenario)
                    
                    question_end = time.time()
                    
                    task_results.append({
                        "task_id": question.id,
                        "task_type": "question",
                        "prompt": question.prompt,
                        "expected": question.expected,
                        "response": response,
                        "category": question.category,
                        "difficulty": question.difficulty,
                        "latency": question_end - question_start
                    })
                    
                    logger.debug(f"Question {question.id} completed in {question_end - question_start:.2f}s")
                    if self.verbose:
                        self.console.print(f"[dim]Question {question.id}: {response[:50]}...[/dim]")
            
            # Process image tasks if they exist
            if dataset.dataset.image_tasks:
                total_image_tasks = len(dataset.dataset.image_tasks)
                logger.debug(f"Processing {total_image_tasks} image tasks")
                
                for i, image_task in enumerate(dataset.dataset.image_tasks, 1):
                    logger.debug(f"Processing image task {i}/{total_image_tasks} (ID: {image_task.id})")
                    task_start = time.time()
                    
                    # Generate response using the provider with image data
                    response = await self._generate_image_response(image_task, scenario)
                    
                    task_end = time.time()
                    
                    task_results.append({
                        "task_id": image_task.id,
                        "task_type": "image_task",
                        "prompt": image_task.prompt,
                        "image_path": image_task.image_path,
                        "expected": image_task.expected,
                        "response": response,
                        "category": image_task.category,
                        "difficulty": image_task.difficulty,
                        "ocr_task_type": image_task.task_type,
                        "analysis_criteria": image_task.analysis_criteria,
                        "latency": task_end - task_start
                    })
                    
                    logger.debug(f"Image task {image_task.id} completed in {task_end - task_start:.2f}s")
                    if self.verbose:
                        self.console.print(f"[dim]Image task {image_task.id}: {response[:50]}...[/dim]")

            # Process PDF tasks if they exist
            if dataset.dataset.pdf_tasks:
                total_pdf_tasks = len(dataset.dataset.pdf_tasks)
                logger.debug(f"Processing {total_pdf_tasks} PDF tasks")
                
                for i, pdf_task in enumerate(dataset.dataset.pdf_tasks, 1):
                    logger.debug(f"Processing PDF task {i}/{total_pdf_tasks} (ID: {pdf_task.id})")
                    task_start = time.time()
                    
                    # Generate response using the provider with PDF data
                    response = await self._generate_pdf_response(pdf_task, scenario)
                    
                    task_end = time.time()
                    
                    task_results.append({
                        "task_id": pdf_task.id,
                        "task_type": "pdf_task",
                        "prompt": pdf_task.prompt,
                        "pdf_path": pdf_task.pdf_path,
                        "expected": pdf_task.expected,
                        "response": response,
                        "category": pdf_task.category,
                        "difficulty": pdf_task.difficulty,
                        "ocr_task_type": pdf_task.task_type,
                        "analysis_criteria": pdf_task.analysis_criteria,
                        "latency": task_end - task_start
                    })
                    
                    logger.debug(f"PDF task {pdf_task.id} completed in {task_end - task_start:.2f}s")
                    if self.verbose:
                        self.console.print(f"[dim]PDF task {pdf_task.id}: {response[:50]}...[/dim]")
            
            end_time = time.time()
            total_latency = end_time - start_time
            total_tasks = len(task_results)
            logger.info(f"Scenario {scenario.name} completed: {total_tasks} tasks in {total_latency:.2f}s")
            
            # Run scoring if configured
            scoring_result = None
            if hasattr(self.suite.suite, 'scoring') and self.suite.suite.scoring:
                logger.debug("Running scoring evaluation")
                scoring_result = await self._run_scoring(task_results, self.suite.suite.scoring)
                if scoring_result and 'average_score' in scoring_result:
                    logger.info(f"Scoring completed: average score {scoring_result['average_score']}")
            
            result = {
                "scenario": scenario.name,
                "provider": scenario.provider,
                "strategy": scenario.strategy,
                "dataset": dataset.dataset.name,
                "tasks_count": len(task_results),
                "tasks": task_results,
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
            
            # Use prompt from file if prompt_path is specified, otherwise use provided prompt
            prompt_to_use = prompt
            if hasattr(scenario, 'prompt_path') and scenario.prompt_path:
                try:
                    prompt_file_path = Path(scenario.prompt_path)
                    if not prompt_file_path.is_absolute():
                        # Make relative to the project root
                        prompt_file_path = Path("data") / prompt_file_path
                    
                    with open(prompt_file_path, 'r', encoding='utf-8') as f:
                        prompt_to_use = f.read().strip()
                    logger.debug(f"Loaded prompt from {prompt_file_path}")
                except Exception as e:
                    logger.warning(f"Could not load prompt from {scenario.prompt_path}: {e}, using provided prompt")
                    prompt_to_use = prompt
            
            # Create strategy instance
            strategy = StrategyFactory.create_strategy(scenario.strategy, provider_settings)
            
            # Execute the strategy with scenario configuration
            return await strategy.execute(prompt_to_use, scenario_config=scenario)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    async def _generate_image_response(self, image_task, scenario) -> str:
        """Generate a response for an image task using the configured provider and strategy."""
        try:
            if scenario.provider not in self.provider_settings:
                raise ValueError(f"Provider {scenario.provider} not found in settings")
            
            provider_settings = self.provider_settings[scenario.provider]
            
            # Use prompt from file if prompt_path is specified, otherwise use task prompt
            prompt_to_use = image_task.prompt
            if hasattr(scenario, 'prompt_path') and scenario.prompt_path:
                try:
                    prompt_file_path = Path(scenario.prompt_path)
                    if not prompt_file_path.is_absolute():
                        # Make relative to the project root
                        prompt_file_path = Path("data") / prompt_file_path
                    
                    with open(prompt_file_path, 'r', encoding='utf-8') as f:
                        prompt_to_use = f.read().strip()
                    logger.debug(f"Loaded prompt from {prompt_file_path}")
                except Exception as e:
                    logger.warning(f"Could not load prompt from {scenario.prompt_path}: {e}, using task prompt")
                    prompt_to_use = image_task.prompt
            
            # Create strategy instance
            strategy = StrategyFactory.create_strategy(scenario.strategy, provider_settings)
            
            # Execute the strategy with image data and additional context
            return await strategy.execute(
                prompt_to_use, 
                scenario_config=scenario,
                image_path=image_task.image_path,
                analysis_criteria=image_task.analysis_criteria,
                task_type=image_task.task_type
            )
                
        except Exception as e:
            logger.error(f"Error generating image response: {e}")
            return f"Error: {str(e)}"

    async def _generate_pdf_response(self, pdf_task, scenario) -> str:
        """Generate a response for a PDF task using the configured provider and strategy."""
        try:
            if scenario.provider not in self.provider_settings:
                raise ValueError(f"Provider {scenario.provider} not found in settings")
            
            provider_settings = self.provider_settings[scenario.provider]
            
            # Create strategy instance
            strategy = StrategyFactory.create_strategy(scenario.strategy, provider_settings)
            
            # Use prompt from file if prompt_path is specified, otherwise use task prompt
            prompt_to_use = pdf_task.prompt
            if hasattr(scenario, 'prompt_path') and scenario.prompt_path:
                try:
                    prompt_file_path = Path(scenario.prompt_path)
                    if not prompt_file_path.is_absolute():
                        # Make relative to the project root
                        prompt_file_path = Path("data") / prompt_file_path
                    
                    with open(prompt_file_path, 'r', encoding='utf-8') as f:
                        prompt_to_use = f.read().strip()
                    logger.debug(f"Loaded prompt from {prompt_file_path}")
                except Exception as e:
                    logger.warning(f"Could not load prompt from {scenario.prompt_path}: {e}, using task prompt")
                    prompt_to_use = pdf_task.prompt
            
            # Execute the strategy with PDF data and additional context
            return await strategy.execute(
                prompt_to_use, 
                scenario_config=scenario,
                pdf_path=pdf_task.pdf_path,
                analysis_criteria=pdf_task.analysis_criteria,
                task_type=pdf_task.task_type
            )
                
        except Exception as e:
            logger.error(f"Error generating PDF response: {e}")
            return f"Error: {str(e)}"
    
    async def _run_scoring(self, task_results: List[Dict[str, Any]], scoring_config) -> Dict[str, Any]:
        """Run scoring evaluation on the results."""
        try:
            scores = []
            
            for result in task_results:
                # Format the scoring prompt based on task type
                if result.get("task_type") == "image_task":
                    # For image tasks, include additional context
                    scoring_prompt = scoring_config.prompt.format(
                        prompt=result["prompt"],
                        expected=result["expected"],
                        response=result["response"],
                        task_type=result.get("ocr_task_type", "image_analysis")
                    )
                else:
                    # For regular questions
                    scoring_prompt = scoring_config.prompt.format(
                        question=result["prompt"],
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
                        "task_id": result["task_id"],
                        "task_type": result.get("task_type", "question"),
                        "score": score_data.get("score", 0),
                        "explanation": score_data.get("explanation", "No explanation provided"),
                        "max_score": scoring_config.max_score
                    })
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse scoring response: {cleaned_response}")
                    scores.append({
                        "task_id": result["task_id"],
                        "task_type": result.get("task_type", "question"),
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
        table.add_column("Tasks", style="yellow")
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
                str(result.get("tasks_count", result.get("questions_count", 0))),
                avg_score,
                status,
                f"{result['latency']:.2f}s"
            )
        
        self.console.print(table)
        
        # Save results with new structure: scenario subfolders with individual JSON files for each PDF
        suite_config = self.suite.suite
        output_base_name = suite_config.output.name
        
        # Create suite-specific folder structure with UUID
        suite_folder_name = self._get_suite_folder_name(suite_config.name)
        base_dir = Path("data/results") / suite_folder_name
        
        # Create base directory
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each scenario result
        for result in results:
            if result["status"] != "success":
                # For failed scenarios, still create a folder with error info
                scenario_folder = base_dir / result["scenario"]
                scenario_folder.mkdir(parents=True, exist_ok=True)
                
                error_file = scenario_folder / "error.json"
                with open(error_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "scenario": result["scenario"],
                        "error": result.get("error", "Unknown error"),
                        "timestamp": result.get("timestamp", time.time())
                    }, f, indent=2, default=str)
                continue
            
            # Create scenario subfolder
            scenario_folder = base_dir / result["scenario"]
            scenario_folder.mkdir(parents=True, exist_ok=True)
            
            # Separate PDF tasks from other tasks
            pdf_tasks = [task for task in result.get("tasks", []) if task.get("task_type") == "pdf_task"]
            other_tasks = [task for task in result.get("tasks", []) if task.get("task_type") != "pdf_task"]
            
            # Create individual JSON files for each PDF task
            for pdf_task in pdf_tasks:
                # Extract filename from PDF path for the output filename
                pdf_filename = Path(pdf_task["pdf_path"]).stem
                output_filename = f"{pdf_filename}.json"
                task_file = scenario_folder / output_filename
                
                # Create individual task result
                task_result = {
                    "scenario": result["scenario"],
                    "provider": result["provider"],
                    "strategy": result["strategy"],
                    "task_id": pdf_task["task_id"],
                    "pdf_path": pdf_task["pdf_path"],
                    "prompt": pdf_task["prompt"],
                    "response": pdf_task["response"],
                    "expected": pdf_task.get("expected"),
                    "category": pdf_task.get("category"),
                    "difficulty": pdf_task.get("difficulty"),
                    "task_type": pdf_task.get("ocr_task_type"),
                    "analysis_criteria": pdf_task.get("analysis_criteria"),
                    "latency": pdf_task["latency"],
                    "timestamp": result["timestamp"]
                }
                
                # Add scoring info if available
                if result.get("scoring") and result["scoring"].get("scores"):
                    matching_scores = [s for s in result["scoring"]["scores"] if s.get("task_id") == pdf_task["task_id"]]
                    if matching_scores:
                        task_result["scoring"] = matching_scores[0]
                
                with open(task_file, "w", encoding="utf-8") as f:
                    json.dump(task_result, f, indent=2, default=str)
            
            # If there are other (non-PDF) tasks, save them in a consolidated file
            if other_tasks:
                other_tasks_file = scenario_folder / f"{output_base_name}_other_tasks.json"
                other_result = {
                    "scenario": result["scenario"],
                    "provider": result["provider"],
                    "strategy": result["strategy"],
                    "dataset": result["dataset"],
                    "tasks_count": len(other_tasks),
                    "tasks": other_tasks,
                    "scoring": result.get("scoring"),
                    "latency": result["latency"],
                    "timestamp": result["timestamp"],
                    "status": result["status"]
                }
                
                with open(other_tasks_file, "w", encoding="utf-8") as f:
                    json.dump(other_result, f, indent=2, default=str)
            
            # Create a summary file for the scenario
            scenario_summary = {
                "scenario": result["scenario"],
                "provider": result["provider"],
                "strategy": result["strategy"],
                "dataset": result["dataset"],
                "total_tasks": result.get("tasks_count", 0),
                "pdf_tasks_count": len(pdf_tasks),
                "other_tasks_count": len(other_tasks),
                "scoring_summary": result.get("scoring"),
                "latency": result["latency"],
                "timestamp": result["timestamp"],
                "status": result["status"]
            }
            
            summary_file = scenario_folder / "scenario_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(scenario_summary, f, indent=2, default=str)
        
        self.console.print(f"\n[dim]Results saved to {base_dir}[/dim]")
        self.console.print(f"[dim]Structure: Suite folder -> Scenario subfolders -> Individual PDF JSON files[/dim]")
