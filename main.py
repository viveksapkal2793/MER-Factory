import os

os.environ["GRPC_VERBOSITY"] = "ERROR"

import typer
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from enum import Enum
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TaskID,
)
import asyncio
import traceback
import functools

from agents.graph import create_graph, MERRState
from agents.models import LLMModels
from tools.ffmpeg_adapter import FFMpegAdapter
from tools.openface_adapter import OpenFaceAdapter


class ProcessingType(str, Enum):
    au = "AU"
    audio = "audio"
    video = "video"
    mer = "MER"
    image = "image"


app = typer.Typer(
    name="merr-cli",
    help="A modular CLI tool to construct the MERR dataset from video and image files.",
    add_completion=False,
)
console = Console(stderr=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _setup_and_preprocess(input_path: Path, verbose: bool) -> list[Path]:
    """Shared setup logic to find files to process."""
    files_to_process = []
    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS:
            files_to_process.append(input_path)
        else:
            console.print(
                f"[bold red]Error: '{input_path}' is not a recognized video/image file.[/bold red]"
            )
            raise typer.Exit(code=1)
    elif input_path.is_dir():
        if verbose:
            console.print(f"Searching for files in [cyan]{input_path}[/cyan]...")
        all_extensions = VIDEO_EXTENSIONS.union(IMAGE_EXTENSIONS)
        for ext in all_extensions:
            files_to_process.extend(input_path.rglob(f"*{ext}"))
        if not files_to_process:
            console.print(
                f"[bold red]Error: No video or image files found in '{input_path}'.[/bold red]"
            )
            raise typer.Exit(code=1)
        if verbose:
            console.print(f"Found {len(files_to_process)} file(s).")
    return files_to_process


def _run_processing_loop_sync(
    files_to_process: list[Path], graph_app, initial_state_builder, verbose: bool
):
    """Synchronous processing loop for Hugging Face."""
    results = {"success": 0, "failure": 0}
    total_files = len(files_to_process)

    if not verbose:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("({task.completed} of {task.total})"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Processing files...", total=total_files)
            for file_path in files_to_process:
                try:
                    initial_state = initial_state_builder(file_path)
                    final_state = graph_app.invoke(initial_state)
                    if final_state and final_state.get("error"):
                        results["failure"] += 1
                    else:
                        results["success"] += 1
                except Exception as e:
                    results["failure"] += 1
                    console.print(
                        f"[bold red]FATAL ERROR processing {file_path.name}: {e}[/bold red]"
                    )
                    traceback.print_exc()
                progress.update(
                    task,
                    advance=1,
                    description=f"Finished [cyan]{file_path.name}[/cyan]",
                )
    else:  # Verbose mode, no progress bar
        for file_path in files_to_process:
            try:
                initial_state = initial_state_builder(file_path)
                final_state = graph_app.invoke(initial_state)
                if final_state and final_state.get("error"):
                    results["failure"] += 1
                else:
                    results["success"] += 1
            except Exception as e:
                results["failure"] += 1
                console.print(
                    f"[bold red]FATAL ERROR processing {file_path.name}: {e}[/bold red]"
                )
                traceback.print_exc()

    return results


async def _run_processing_loop_async(
    files_to_process: list[Path],
    graph_app,
    initial_state_builder,
    concurrency: int,
    verbose: bool,
):
    """Asynchronous processing loop for Gemini/Ollama with real-time progress."""
    results = {"success": 0, "failure": 0}
    total_files = len(files_to_process)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_file(
        file_path: Path, task_id: TaskID, progress: Progress
    ) -> Path:
        """Processes a single file and returns its path upon completion."""
        nonlocal results
        async with semaphore:
            try:
                initial_state = initial_state_builder(file_path)
                final_state = await graph_app.ainvoke(initial_state)
                if final_state and final_state.get("error"):
                    results["failure"] += 1
                else:
                    results["success"] += 1
                if progress and task_id is not None:
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Finished [cyan]{file_path.name}[/cyan]",
                    )
            except Exception as e:
                results["failure"] += 1
                console.print(
                    f"[bold red]FATAL ERROR processing {file_path.name}: {e}[/bold red]"
                )
                traceback.print_exc()
        return file_path

    if not verbose:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("({task.completed} of {task.total})"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Processing files...", total=total_files)
            tasks = [process_file(fp, task, progress) for fp in files_to_process]
            await asyncio.gather(*tasks)
    else:  # Verbose mode, no progress bar
        tasks = [process_file(fp, None, None) for fp in files_to_process]
        await asyncio.gather(*tasks)

    return results


def _process_wrapper(
    input_path: Path,
    output_dir: Path,
    processing_type: ProcessingType,
    threshold: float,
    peak_distance_frames: int,
    silent: bool,
    concurrency: int,
    ollama_text_model_name: str,
    ollama_vision_model_name: str,
    huggingface_model_id: str,
):
    """
    Main logic wrapper that decides whether to run sync or async processing.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if processing_type in [ProcessingType.mer, ProcessingType.au]:
        openface_executable = os.getenv("OPENFACE_EXECUTABLE")
        if not openface_executable:
            console.print(
                "[bold yellow]Warning: OPENFACE_EXECUTABLE not set in .env file. Using default path.[/bold yellow]"
            )
        elif not os.path.exists(openface_executable):
            console.print(
                f"[bold red]Error: OpenFace executable not found at '{openface_executable}'[/bold red]"
            )
            console.print(
                "Please check your OPENFACE_EXECUTABLE setting in the .env file."
            )
            raise typer.Exit(code=1)

    if not any(
        [
            huggingface_model_id,
            ollama_text_model_name,
            ollama_vision_model_name,
            api_key,
        ]
    ):
        console.print(
            "[bold red]Error: A model must be provided via --huggingface-model, --ollama-..., or GOOGLE_API_KEY.[/bold red]"
        )
        raise typer.Exit(code=1)

    verbose = not silent
    console.rule(
        f"[bold magenta]MERR CLI - Mode: {processing_type.value}[/bold magenta]"
    )

    try:
        models = LLMModels(
            api_key=api_key,
            ollama_text_model_name=ollama_text_model_name,
            ollama_vision_model_name=ollama_vision_model_name,
            huggingface_model_id=huggingface_model_id,
            verbose=verbose,
        )
    except (ValueError, ImportError) as e:
        console.print(f"[bold red]Failed to initialize models: {e}[/bold red]")
        raise typer.Exit(code=1)

    files_to_process = _setup_and_preprocess(input_path, verbose)
    total_files = len(files_to_process)

    output_dir.mkdir(exist_ok=True)
    error_logs_dir = output_dir / "error_logs"
    error_logs_dir.mkdir(exist_ok=True)

    # --- Pre-processing (Feature Extraction) ---
    console.rule("[bold yellow]Phase 1: Feature Extraction[/bold yellow]")
    asyncio.run(
        _run_feature_extraction(files_to_process, output_dir, processing_type, verbose)
    )
    console.rule("[bold yellow]Phase 1 Complete[/bold yellow]")

    # --- Main Processing (Graph Invocation) ---
    console.rule("[bold blue]Phase 2: Main Processing[/bold blue]")

    initial_state_builder = functools.partial(
        build_initial_state,
        output_dir=output_dir,
        processing_type=processing_type,
        models=models,
        threshold=threshold,
        peak_distance_frames=peak_distance_frames,
        verbose=verbose,
        error_logs_dir=error_logs_dir,
    )

    if huggingface_model_id:
        if concurrency > 1 and not silent:
            console.log(
                "[yellow]Concurrency set to 1 for synchronous Hugging Face model.[/yellow]"
            )
        graph_app = create_graph(use_sync_nodes=True)
        results = _run_processing_loop_sync(
            files_to_process, graph_app, initial_state_builder, verbose
        )
    else:
        graph_app = create_graph(use_sync_nodes=False)
        results = asyncio.run(
            _run_processing_loop_async(
                files_to_process, graph_app, initial_state_builder, concurrency, verbose
            )
        )

    console.rule("[bold green]Processing Complete[/bold green]")
    console.print(f"Total files attempted: {total_files}")
    console.print(f"✅ [green]Successful[/green]: {results['success']}")
    if results["failure"] > 0:
        console.print(f"❌ [red]Failed[/red]: {results['failure']}")
        console.print(f"Error logs saved in: [cyan]{error_logs_dir}[/cyan]")


def build_initial_state(
    file_path: Path,
    output_dir: Path,
    processing_type: ProcessingType,
    models: LLMModels,
    threshold: float,
    peak_distance_frames: int,
    verbose: bool,
    error_logs_dir: Path,
) -> MERRState:
    """Builds the initial state dictionary for a file."""
    file_id = file_path.stem
    file_output_dir = output_dir / file_id
    is_image = file_path.suffix.lower() in IMAGE_EXTENSIONS
    current_processing_type = "image" if is_image else processing_type.value

    return {
        "video_path": file_path,
        "output_dir": output_dir,
        "processing_type": current_processing_type,
        "models": models,
        "threshold": threshold,
        "peak_distance_frames": peak_distance_frames,
        "verbose": verbose,
        "error_logs_dir": error_logs_dir,
        "video_id": file_id,
        "video_output_dir": file_output_dir,
        "audio_path": file_output_dir / f"{file_id}.wav",
        "au_data_path": file_output_dir / f"{file_id}.csv",
    }


async def _run_feature_extraction(
    files_to_process, output_dir, processing_type, verbose
):
    """Asynchronous feature extraction for OpenFace and FFmpeg with progress bars."""
    extraction_semaphore = asyncio.Semaphore(8)

    video_files = [f for f in files_to_process if f.suffix.lower() in VIDEO_EXTENSIONS]
    image_files = [f for f in files_to_process if f.suffix.lower() in IMAGE_EXTENSIONS]

    total_openface = len(image_files) + (
        len(video_files)
        if processing_type in [ProcessingType.mer, ProcessingType.au]
        else 0
    )
    total_ffmpeg = (
        len(video_files)
        if processing_type in [ProcessingType.mer, ProcessingType.audio]
        else 0
    )

    async def run_job(
        file_path: Path,
        openface_task_id: TaskID,
        ffmpeg_task_id: TaskID,
        progress: Progress,
    ):
        async with extraction_semaphore:
            file_output_dir = output_dir / file_path.stem
            file_output_dir.mkdir(exist_ok=True)
            is_video = file_path.suffix.lower() in VIDEO_EXTENSIONS
            is_image = file_path.suffix.lower() in IMAGE_EXTENSIONS
            needs_openface = is_image or (
                is_video and processing_type in [ProcessingType.mer, ProcessingType.au]
            )
            needs_ffmpeg = is_video and processing_type in [
                ProcessingType.mer,
                ProcessingType.audio,
            ]

            openface_output_path = file_output_dir / f"{file_path.stem}.csv"
            ffmpeg_output_path = file_output_dir / f"{file_path.stem}.wav"

            try:
                if needs_openface:
                    if openface_output_path.exists():
                        if verbose:
                            console.log(
                                f"[yellow]Skipping OpenFace extraction for {file_path.name} (already exists)[/yellow]"
                            )
                    else:
                        await OpenFaceAdapter.run_feature_extraction(
                            file_path, file_output_dir, verbose
                        )
                    if progress and openface_task_id is not None:
                        progress.update(openface_task_id, advance=1)
                if needs_ffmpeg:
                    if ffmpeg_output_path.exists():
                        if verbose:
                            console.log(
                                f"[yellow]Skipping FFmpeg extraction for {file_path.name} (already exists)[/yellow]"
                            )
                    else:
                        await FFMpegAdapter.extract_audio(
                            file_path,
                            file_output_dir / f"{file_path.stem}.wav",
                            verbose,
                        )
                    if progress and ffmpeg_task_id is not None:
                        progress.update(ffmpeg_task_id, advance=1)
            except Exception as e:
                console.log(
                    f"[bold red]Extraction failed for {file_path.name}: {e}[/bold red]"
                )
                return False
            return True

    if not verbose:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("({task.completed} of {task.total})"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            openface_task_id = (
                progress.add_task("[bold blue]OpenFace", total=total_openface)
                if total_openface > 0
                else None
            )
            ffmpeg_task_id = (
                progress.add_task("[bold green]FFmpeg", total=total_ffmpeg)
                if total_ffmpeg > 0
                else None
            )

            tasks = [
                run_job(fp, openface_task_id, ffmpeg_task_id, progress)
                for fp in files_to_process
            ]
            await asyncio.gather(*tasks)
    else:  # Verbose mode, no progress bars
        tasks = [run_job(fp, None, None, None) for fp in files_to_process]
        await asyncio.gather(*tasks)


@app.command()
def process(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Path to a single file or a directory."
    ),
    output_dir: Path = typer.Argument(
        ..., file_okay=False, help="Directory to save all outputs."
    ),
    processing_type: ProcessingType = typer.Option(
        ProcessingType.mer, "--type", "-t", case_sensitive=False
    ),
    threshold: float = typer.Option(
        0.8, "--threshold", "-th", min=0.0, max=5.0, help="Emotion detection threshold."
    ),
    peak_distance_frames: int = typer.Option(
        15, "--peak_dis", "-pd", min=8, help="The steps between peak frame detection."
    ),
    silent: bool = typer.Option(
        False, "--silent", "-s", help="Run with minimal output."
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
    # TODO: Add support for Hugging Face models
    # Currently only supports multimodal models like google/gemma-3n-E4B-it
    # and google/gemma-3n-E2B-it.
    huggingface_model_id: str = typer.Option(
        None, "--huggingface-model", "-hfm", help="Hugging Face model ID."
    ),
):
    """Processes media files for Multimodal Emotion Recognition and Reasoning (MERR)."""
    _process_wrapper(
        input_path=input_path,
        output_dir=output_dir,
        processing_type=processing_type,
        threshold=threshold,
        peak_distance_frames=peak_distance_frames,
        silent=silent,
        concurrency=concurrency,
        ollama_text_model_name=ollama_text_model,
        ollama_vision_model_name=ollama_vision_model,
        huggingface_model_id=huggingface_model_id,
    )


if __name__ == "__main__":
    app()
