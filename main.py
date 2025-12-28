# -*- coding: utf-8 -*-
# Author: Yuxiang Lin

import os
import typer
from pathlib import Path
from rich.console import Console
import asyncio
import diskcache
import functools
import json

# Set gRPC verbosity to ERROR before other imports
os.environ["GRPC_VERBOSITY"] = "ERROR"

# Import from local packages
from mer_factory.graph import create_graph
from mer_factory.models import LLMModels
from mer_factory.prompts import PromptTemplates
from utils.config import AppConfig, ProcessingType, TaskType
from utils.file_handler import find_files_to_process, load_labels_from_file, check_json_completeness
from utils.processing_manager import (
    run_feature_extraction,
    run_main_processing,
    build_initial_state,
)

# Initialize Typer app and Rich console
app = typer.Typer(
    name="mer-factory",
    help="A modular CLI tool to construct the MERR dataset from video and image files.",
    add_completion=False,
)
console = Console(stderr=True)

def load_filter_list(filter_file: Path) -> set:
    """Load video names from filter file."""
    if not filter_file.exists():
        console.print(f"[bold red]Error: Filter file not found at '{filter_file}'[/bold red]")
        raise typer.Exit(1)
    
    video_names = set()
    with open(filter_file, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                video_names.add(name)
    
    console.log(f"Loaded {len(video_names)} video names from filter file: [cyan]{filter_file.name}[/cyan]")
    return video_names

def filter_incomplete_files(files_to_process: list, output_dir: Path, verbose: bool = True, filter_list: set = None) -> tuple[list, dict]:
    """
    Filter files to only process those with missing or incomplete JSON files.
    
    Returns:
        (filtered_files, stats): Tuple with filtered file list and statistics
    """
    filtered_files = []
    stats = {
        'total': len(files_to_process),
        'missing_json': 0,
        'incomplete_json': 0,
        'complete': 0,
        'to_process': 0,
        'filtered_out': 0
    }
    
    console.log("[yellow]Checking existing JSON files for completeness...[/yellow]")
    if filter_list:
        console.log(f"[yellow]Filtering by {len(filter_list)} video names from filter file[/yellow]")
    
    for file_info in files_to_process:
        if isinstance(file_info, dict):
            video_name = file_info['name']
        else:
            video_name = file_info.stem

        # If filter list is provided, check if video is in the list
        if filter_list is not None and video_name not in filter_list:
            stats['filtered_out'] += 1
            if verbose:
                console.log(f"  ⏭️  [dim]{video_name}[/dim]: Not in filter list (skipping)")
            continue

        # Construct expected JSON path
        json_subdir = output_dir / video_name
        json_path = json_subdir / f"{video_name}_merr_data.json"
        
        is_complete, missing_fields = check_json_completeness(json_path)
        
        if not json_path.exists():
            stats['missing_json'] += 1
            filtered_files.append(file_info)
            if verbose:
                console.log(f"  📝 [yellow]{video_name}[/yellow]: JSON missing")
        elif not is_complete:
            stats['incomplete_json'] += 1
            filtered_files.append(file_info)
            if verbose:
                console.log(f"  ⚠️  [orange]{video_name}[/orange]: Missing fields: {', '.join(missing_fields[:3])}{'...' if len(missing_fields) > 3 else ''}")
        else:
            stats['complete'] += 1
            if verbose:
                console.log(f"  ✅ [green]{video_name}[/green]: Complete (skipping)")
    
    stats['to_process'] = len(filtered_files)
    
    return filtered_files, stats

def main_orchestrator(config: AppConfig, skip_complete: bool = True, filter_file: Path = None):
    """The main function that orchestrates the entire processing pipeline."""
    console.rule(
        f"[bold magenta]MERR CLI - Mode: {config.processing_type.value} | Task: {config.task.value}[/bold magenta]"
    )

    # Load filter list if provided
    filter_list = None
    if filter_file:
        filter_list = load_filter_list(filter_file)

    llm_cache = None
    try:
        if config.cache:
            cache_dir = config.output_dir / ".llm_cache"
            cache_dir.mkdir(exist_ok=True)
            llm_cache = diskcache.Cache(str(cache_dir))
            console.log(
                f"LLM response caching is enabled. Cache dir: [cyan]{cache_dir}[/cyan]"
            )

        if error := config.get_model_choice_error():
            console.print(f"[bold red]Error: {error}[/bold red]")
            raise typer.Exit(1)

        if config.processing_type not in [ProcessingType.AUDIO, ProcessingType.VIDEO]:
            if not config.cache:  # Only check OpenFace if cache is disabled
                if error := config.get_openface_path_error():
                    console.print(f"[bold red]{error}[/bold red]")
                    raise typer.Exit(1)
            else:
                console.log("[yellow]Cache enabled - skipping OpenFace validation (using pre-extracted AU files)[/yellow]")

        if config.label_file:
            config.labels = load_labels_from_file(config.label_file, config.verbose)

        try:
            models = LLMModels(
                api_key=config.api_key,
                ollama_text_model_name=config.ollama_text_model,
                ollama_vision_model_name=config.ollama_vision_model,
                chatgpt_model_name=config.chatgpt_model,
                huggingface_model_id=config.huggingface_model_id,
                cache=llm_cache,
                verbose=config.verbose,
            )
            prompts = PromptTemplates(prompts_file=config.prompts_file)
        except (ValueError, ImportError, FileNotFoundError) as e:
            console.print(f"[bold red]Failed to initialize components: {e}[/bold red]")
            raise typer.Exit(1)

        # --- File Discovery ---
        all_files = find_files_to_process(config.input_path, config.verbose)

        if skip_complete:
            console.rule("[bold cyan]Filtering Files[/bold cyan]")
            files_to_process, filter_stats = filter_incomplete_files(
                all_files, 
                config.output_dir, 
                verbose=config.verbose
            )
            
            console.log(f"\n📊 Filtering Results:")
            console.log(f"  Total files found: {filter_stats['total']}")
            console.log(f"  ✅ Complete JSON files (skipping): {filter_stats['complete']}")
            console.log(f"  📝 Missing JSON files: {filter_stats['missing_json']}")
            console.log(f"  ⚠️  Incomplete JSON files: {filter_stats['incomplete_json']}")
            console.log(f"  🔄 Files to process: {filter_stats['to_process']}\n")
            
            if filter_stats['to_process'] == 0:
                console.print("[bold green]✅ All files have complete JSON data. Nothing to process![/bold green]")
                return
        else:
            files_to_process = all_files
            console.log(f"Processing all {len(files_to_process)} files (skip-complete disabled)")

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
            prompts=prompts,
        )

        results = asyncio.run(
            run_main_processing(
                files_to_process,
                graph_app,
                initial_state_builder,
                config,
                is_sync_model,
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

    finally:
        if llm_cache:
            llm_cache.close()
            console.log("LLM cache closed.")


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
    task: TaskType = typer.Option(
        TaskType.EMOTION_RECOGNITION,
        "--task",
        "-tk",
        case_sensitive=False,
        help="The analysis task to perform.",
    ),
    prompts_file: Path = typer.Option(
        "utils/prompts/prompts.json",
        "--prompts-file",
        "-pf",
        exists=True,
        help="Path to the prompts JSON file.",
    ),
    label_file: Path = typer.Option(
        None,
        "--label-file",
        "-l",
        exists=True,
        help="Path to a CSV file with 'name' and 'label' columns. Optional, for ground truth labels.",
    ),
    filter_file: Path = typer.Option(
        None,
        "--filter-file",
        "-ff",
        exists=True,
        help="Path to a text file containing video names to process (one per line).",
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
        help="Reuse existing audio/video/AU results from previous pipeline runs & cache LLM calls.",
    ),
    skip_complete: bool = typer.Option(
        True,
        "--skip-complete",
        "-sc",
        help="Skip files that already have complete JSON output. Set to False to reprocess all files.",
    ),
):
    """Processes media files for Multimodal Emotion Recognition and Reasoning (MERR)."""
    try:
        config = AppConfig(
            input_path=input_path,
            output_dir=output_dir,
            processing_type=processing_type,
            task=task,
            prompts_file=prompts_file,
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
        main_orchestrator(config, skip_complete=skip_complete, filter_file=filter_file)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
