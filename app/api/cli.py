"""Simple CLI interface for the benchmarker tool."""

import asyncio
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console

from app.core.suite import load_suite
from app.core.logging import setup_logging, get_logger
from app.services.runner import SuiteRunner

app = typer.Typer(
    name="benchmarker",
    help="🚀 Simple benchmarking CLI tool",
    rich_markup_mode="rich",
    add_completion=False,
)

console = Console()


@app.command()
def run(
    suite_file: str = typer.Argument(..., help="Path to YAML suite configuration file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """🏃 Run benchmark suite from configuration file."""
    try:
        # Setup logging based on verbose flag
        setup_logging(verbose)
        logger = get_logger(__name__)
        
        # Check if suite file exists
        if not Path(suite_file).exists():
            console.print(f"[red]Suite file not found: {suite_file}[/red]")
            raise typer.Exit(1)
        
        # Load suite configuration and run
        logger.info(f"Loading suite configuration from: {suite_file}")
        suite = load_suite(suite_file)
        
        logger.info(f"Loaded suite: {suite.suite.name}")
        if verbose:
            logger.debug(f"Suite details: {len(suite.suite.scenarios)} scenarios")
        
        runner = SuiteRunner(suite, verbose=verbose)
        
        console.print(f"[green]Running suite: {suite.suite.name}[/green]")
        
        asyncio.run(runner.execute())
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Execution failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            logger.exception("Full error details:")
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """📦 Show version information."""
    from app import __version__
    console.print(f"benchmarker version: [bold blue]{__version__}[/bold blue]")


def main() -> None:
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
