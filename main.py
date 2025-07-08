# -*- coding: utf-8 -*-
# Author: Yuxiang Lin

import os
import typer
from pathlib import Path
from rich.console import Console
import asyncio

# Set gRPC verbosity to ERROR before other imports
os.environ["GRPC_VERBOSITY"] = "ERROR"

# Import from local packages
from agents.graph import create_graph
from agents.models import LLMModels
from utils.config import AppConfig, ProcessingType
from utils.file_handler import find_files_to_process, load_labels_from_file
from utils.processing_manager import (
    run_feature_extraction,
    run_main_processing,
    build_initial_state,
)
import functools

# Initialize Typer app and Rich console
app = typer.Typer(
    name="merr-cli",
    help="A modular CLI tool to construct the MERR dataset from video and image files.",
    add_completion=False,
)
console = Console(stderr=True)


def main_orchestrator(config: AppConfig):
    """The main function that orchestrates the entire processing pipeline."""
    console.rule(
        f"[bold magenta]MERR CLI - Mode: {config.processing_type.value}[/bold magenta]"
    )

    if error := config.get_model_choice_error():
        console.print(f"[bold red]Error: {error}[/bold red]")
        raise typer.Exit(1)

    if warning := config.get_openface_path_error():
        console.print(f"[bold yellow]{warning}[/bold yellow]")
        if "Error" in warning:
            raise typer.Exit(1)

    if config.label_file:
        config.labels = load_labels_from_file(config.label_file, config.verbose)

    try:
        models = LLMModels(
            api_key=config.api_key,
            ollama_text_model_name=config.ollama_text_model,
            ollama_vision_model_name=config.ollama_vision_model,
            chatgpt_model_name=config.chatgpt_model,
            huggingface_model_id=config.huggingface_model_id,
            verbose=config.verbose,
        )
    except (ValueError, ImportError) as e:
        console.print(f"[bold red]Failed to initialize models: {e}[/bold red]")
        raise typer.Exit(1)

    # --- File Discovery ---
    files_to_process = find_files_to_process(config.input_path, config.verbose)
    total_files = len(files_to_process)

    # --- Phase 1: Feature Extraction ---
    console.rule("[bold yellow]Phase 1: Feature Extraction[/bold yellow]")
    asyncio.run(run_feature_extraction(files_to_process, config))
    console.rule("[bold yellow]Phase 1 Complete[/bold yellow]")

    # --- Phase 2: Main Processing ---
    console.rule("[bold blue]Phase 2: Main Processing[/bold blue]")
    is_sync_model = models.model_type == "huggingface"
    graph_app = create_graph(use_sync_nodes=is_sync_model)

    initial_state_builder = functools.partial(
        build_initial_state,
        config=config,
        models=models,
    )

    results = asyncio.run(
        run_main_processing(
            files_to_process, graph_app, initial_state_builder, config, is_sync_model
        )
    )

    # --- Completion ---
    console.rule("[bold green]Processing Complete[/bold green]")
    console.print(f"Total files attempted: {total_files}")
    console.print(f"✅ [green]Successful[/green]: {results['success']}")
    if results.get("skipped", 0) > 0:
        console.print(f"⏭️  [blue]Skipped (Cached)[/blue]: {results['skipped']}")
    if results["failure"] > 0:
        console.print(f"❌ [red]Failed[/red]: {results['failure']}")
        console.print(f"Error logs saved in: [cyan]{config.error_logs_dir}[/cyan]")


@app.command()
def process(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Path to a single file or a directory."
    ),
    output_dir: Path = typer.Argument(
        ..., file_okay=False, help="Directory to save all outputs."
    ),
    processing_type: ProcessingType = typer.Option(
        ProcessingType.MER, "--type", "-t", case_sensitive=False
    ),
    label_file: Path = typer.Option(
        None,
        "--label-file",
        "-l",
        exists=True,
        help="Path to a CSV file with 'name' and 'label' columns. Optional, for ground truth labels.",
    ),
    threshold: float = typer.Option(
        0.8, "--threshold", "-th", min=0.0, max=5.0, help="Emotion detection threshold."
    ),
    peak_distance_frames: int = typer.Option(
        15, "--peak_dis", "-pd", min=8, help="The steps between peak frame detection."
    ),
    concurrency: int = typer.Option(
        4, "--concurrency", "-c", min=1, help="Concurrent files for async processing."
    ),
    ollama_vision_model: str = typer.Option(
        None, "--ollama-vision-model", "-ovm", help="Ollama vision model name."
    ),
    ollama_text_model: str = typer.Option(
        None, "--ollama-text-model", "-otm", help="Ollama text model name."
    ),
    chatgpt_model: str = typer.Option(
        None, "--chatgpt-model", "-cgm", help="ChatGPT model name (e.g., gpt-4o)."
    ),
    huggingface_model_id: str = typer.Option(
        None, "--huggingface-model", "-hfm", help="Hugging Face model ID."
    ),
    silent: bool = typer.Option(
        False, "--silent", "-s", help="Run with minimal output."
    ),
    cache: bool = typer.Option(
        False,
        "--cache",
        "-ca",
        help="Reuse existing audio/video/AU results from previous pipeline runs.",
    ),
):
    """Processes media files for Multimodal Emotion Recognition and Reasoning (MERR)."""
    try:
        config = AppConfig(
            input_path=input_path,
            output_dir=output_dir,
            processing_type=processing_type,
            label_file=label_file,
            threshold=threshold,
            peak_distance_frames=peak_distance_frames,
            silent=silent,
            cache=cache,
            concurrency=concurrency,
            ollama_vision_model=ollama_vision_model,
            ollama_text_model=ollama_text_model,
            chatgpt_model=chatgpt_model,
            huggingface_model_id=huggingface_model_id,
        )
        main_orchestrator(config)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
