import os

os.environ["GRPC_VERBOSITY"] = "ERROR"

import typer
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from typing import Optional
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

from agents.graph import create_graph, MERRState
from agents.models import LLMModels
from tools.ffmpeg_adapter import FFMpegAdapter
from tools.openface_adapter import OpenFaceAdapter


class ProcessingType(str, Enum):
    au = "AU"
    audio = "audio"
    video = "video"
    mer = "MER"


app = typer.Typer(
    name="merr-cli",
    help="A modular CLI tool to construct the MERR dataset from video files.",
    add_completion=False,
)
console = Console(stderr=True)  # Log errors to stderr
graph_app = create_graph()

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


async def _process(
    video_path: Path,
    output_dir: Path,
    processing_type: ProcessingType,
    threshold: float,
    silent: bool,
    concurrency: int,
    ollama_text_model_name: str = None,  # Added ollama_text_model_name parameter
    ollama_vision_model_name: str = None,  # Added ollama_vision_model_name parameter
):
    """
    Internal async implementation for processing video files.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not ollama_text_model_name and not ollama_vision_model_name and not api_key:
        console.print(
            "[bold red]Error: Either GOOGLE_API_KEY must be set in .env or at least one Ollama model (--ollama-text-model or --ollama-vision-model) must be provided.[/bold red]"
        )
        raise typer.Exit(code=1)

    verbose = not silent
    if verbose:
        console.rule(
            f"[bold magenta]MERR CLI - Mode: {processing_type.value} (Verbose)[/bold magenta]"
        )
    else:
        console.print(
            f"[bold magenta]MERR CLI - Mode: {processing_type.value}[/bold magenta]"
        )

    # Initialize LLMModels with either api_key or ollama_model_name
    models = LLMModels(
        api_key=api_key,
        ollama_text_model_name=ollama_text_model_name,
        ollama_vision_model_name=ollama_vision_model_name,
        verbose=verbose,
    )

    video_files_to_process = []
    if video_path.is_file():
        if video_path.suffix.lower() in VIDEO_EXTENSIONS:
            video_files_to_process.append(video_path)
        else:
            console.print(
                f"[bold red]Error: '{video_path}' is not a recognized video file type.[/bold red]"
            )
            raise typer.Exit(code=1)
    elif video_path.is_dir():
        if verbose:
            console.print(
                f"Searching for video files in directory: [cyan]{video_path}[/cyan]"
            )
        for ext in VIDEO_EXTENSIONS:
            video_files_to_process.extend(video_path.rglob(f"*{ext}"))
        if not video_files_to_process:
            console.print(
                f"[bold red]Error: No video files found in '{video_path}' or its subdirectories with extensions: {', '.join(VIDEO_EXTENSIONS)}[/bold red]"
            )
            raise typer.Exit(code=1)
        if verbose:
            console.print(f"Found {len(video_files_to_process)} video(s) to process.")

    results = {"success": 0, "failure": 0}
    total_files = len(video_files_to_process)
    output_dir.mkdir(exist_ok=True)
    error_logs_dir = output_dir / "error_logs"
    error_logs_dir.mkdir(exist_ok=True)

    extraction_semaphore = asyncio.Semaphore(8)  # Limit concurrent extractions

    async def run_extraction_job(
        video_file: Path,
        video_output_dir: Path,
        progress: Progress,
        openface_task_id: Optional[TaskID],
        ffmpeg_task_id: Optional[TaskID],
    ):
        """
        Wrapper to run a single video's extraction tasks under semaphore control
        and update the consolidated progress bars.
        """
        async with extraction_semaphore:
            if verbose:
                console.log(
                    f"[yellow]Starting extraction for {video_file.name}...[/yellow]"
                )

            needs_openface = processing_type in [ProcessingType.mer, ProcessingType.au]
            needs_ffmpeg = processing_type in [ProcessingType.mer, ProcessingType.audio]

            try:
                if needs_openface:
                    res = await OpenFaceAdapter.run_feature_extraction(
                        video_file, video_output_dir, verbose
                    )
                    if openface_task_id is not None:
                        progress.update(openface_task_id, advance=1)
                    if isinstance(res, Exception) or res is False:
                        console.log(
                            f"[bold red]Error during OpenFace for {video_file.name}: {res}[/bold red]"
                        )
                        return False

                if needs_ffmpeg:
                    audio_path = video_output_dir / f"{video_file.stem}.wav"
                    res = await FFMpegAdapter.extract_audio(
                        video_file, audio_path, verbose
                    )
                    if ffmpeg_task_id is not None:
                        progress.update(ffmpeg_task_id, advance=1)
                    if isinstance(res, Exception) or res is False:
                        console.log(
                            f"[bold red]Error during FFmpeg for {video_file.name}: {res}[/bold red]"
                        )
                        return False

            except Exception as e:
                console.log(
                    f"[bold red]Fatal error during extraction for {video_file.name}: {e}[/bold red]"
                )
                return False

            return True

    if verbose:
        console.rule(
            "[bold yellow]Phase 1: Kicking off background feature extractions[/bold yellow]"
        )

    needs_openface = processing_type in [ProcessingType.mer, ProcessingType.au]
    needs_ffmpeg = processing_type in [ProcessingType.mer, ProcessingType.audio]
    num_openface_tasks = len(video_files_to_process) if needs_openface else 0
    num_ffmpeg_tasks = len(video_files_to_process) if needs_ffmpeg else 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed} of {task.total})"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        openface_task_id = (
            progress.add_task(
                "[bold blue]OpenFace Extraction", total=num_openface_tasks
            )
            if num_openface_tasks > 0
            else None
        )
        ffmpeg_task_id = (
            progress.add_task("[bold green]FFmpeg Extraction", total=num_ffmpeg_tasks)
            if num_ffmpeg_tasks > 0
            else None
        )

        all_extraction_tasks = []
        for video_file in video_files_to_process:
            video_output_dir = output_dir / video_file.stem
            video_output_dir.mkdir(exist_ok=True)
            all_extraction_tasks.append(
                run_extraction_job(
                    video_file,
                    video_output_dir,
                    progress,
                    openface_task_id,
                    ffmpeg_task_id,
                )
            )

        await asyncio.gather(*all_extraction_tasks)

    if verbose:
        console.rule(
            "[bold yellow]Phase 1 Complete: All extractions finished or failed[/bold yellow]"
        )

    main_processing_semaphore = asyncio.Semaphore(concurrency)

    async def run_processing_graph(video_file: Path):
        nonlocal results
        async with main_processing_semaphore:
            video_id = video_file.stem
            video_output_dir = output_dir / video_id

            if verbose:
                console.rule(
                    f"[bold blue]Phase 2: Processing Graph for: {video_file.name}[/bold blue]"
                )

            try:
                # Check if required files exist before proceeding
                au_data_path = video_output_dir / f"{video_id}.csv"
                audio_path = video_output_dir / f"{video_id}.wav"

                # Conditional checks based on selected model type for audio/video
                if models.model_type == "ollama":
                    # For Ollama, audio/video analysis is not supported directly by the LLM
                    # The LLMModels class will return an error message for these specific calls.
                    # No need to skip the entire pipeline based on this, as other parts
                    # of the pipeline (like AU) might still be relevant.
                    pass
                elif models.model_type == "gemini":
                    if (
                        processing_type in [ProcessingType.mer, ProcessingType.audio]
                        and not audio_path.exists()
                    ):
                        raise FileNotFoundError(
                            f"Required FFMpeg audio output not found for {video_id}. Skipping."
                        )

                if (
                    processing_type in [ProcessingType.mer, ProcessingType.au]
                    and not au_data_path.exists()
                ):
                    raise FileNotFoundError(
                        f"Required OpenFace output not found for {video_id}. Skipping."
                    )

                initial_state: MERRState = {
                    "video_path": video_file,
                    "output_dir": output_dir,
                    "processing_type": processing_type.value,
                    "models": models,
                    "threshold": threshold,
                    "verbose": verbose,
                    "error_logs_dir": error_logs_dir,
                    "video_id": video_id,
                    "video_output_dir": video_output_dir,
                    "audio_path": audio_path,
                    "au_data_path": au_data_path,
                }

                final_state = await graph_app.ainvoke(initial_state)
                if final_state and final_state.get("error"):
                    results["failure"] += 1
                else:
                    results["success"] += 1

            except (Exception, FileNotFoundError) as e:
                results["failure"] += 1
                error_log_path = error_logs_dir / f"{video_file.stem}_fatal_error.log"
                with open(error_log_path, "w") as f:
                    f.write(f"Error: {e}\n")
                    f.write(traceback.format_exc())
                console.print(
                    f"[bold red]FATAL ERROR processing {video_file.name}. Log saved to {error_log_path}.[/bold red]"
                )
            return video_file

    tasks = [run_processing_graph(vf) for vf in video_files_to_process]

    if not verbose and total_files > 0:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed} of {task.total})"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            prog_task = progress.add_task("Processing videos...", total=total_files)
            for future in asyncio.as_completed(tasks):
                completed_file = await future
                progress.update(
                    prog_task,
                    advance=1,
                    description=f"Finished [cyan]{completed_file.name}[/cyan]",
                )
    elif total_files > 0:
        await asyncio.gather(*tasks, return_exceptions=True)

    console.rule("[bold green]Processing Complete[/bold green]")
    console.print(f"Total videos attempted: {total_files}")
    console.print(f"✅ [green]Successful[/green]: {results['success']}")

    if results["failure"] > 0:
        console.print(f"❌ [red]Failed[/red]: {results['failure']}")
        console.print(f"Error logs have been saved in: [cyan]{error_logs_dir}[/cyan]")


@app.command()
def process(
    video_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Path to a single video file or a directory containing video files to process.",
    ),
    output_dir: Path = typer.Argument(
        ...,
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory to save all outputs.",
    ),
    processing_type: ProcessingType = typer.Option(
        ProcessingType.mer,
        "--type",
        "-t",
        case_sensitive=False,
        help="The type of processing to perform.",
    ),
    threshold: float = typer.Option(
        0.45,
        "--threshold",
        "-th",
        min=0.0,
        max=1.0,
        help="Threshold for Action Unit (AU) presence in emotion filtering (0.0 to 1.0).",
    ),
    silent: bool = typer.Option(
        False,
        "--silent",
        "-s",
        help="Run silently, showing only a progress bar and final summary.",
    ),
    concurrency: int = typer.Option(
        2,
        "--concurrency",
        "-c",
        min=1,
        help="Number of videos to process concurrently.",
    ),
    ollama_vision_model: str = typer.Option(  # Renamed and adjusted help
        None,
        "--ollama-vision-model",
        "-ovm",
        help="Name of the Ollama vision model to use (e.g., 'bakllava').",
    ),
    ollama_text_model: str = typer.Option(  # Added new option
        None,
        "--ollama-text-model",
        "-otm",
        help="Name of the Ollama text model to use. If not provided, --ollama-vision-model will be used for text tasks.",
    ),
):
    """
    Processes a single video file or multiple video files from a directory
    based on the selected processing type.
    """
    asyncio.run(
        _process(
            video_path=video_path,
            output_dir=output_dir,
            processing_type=processing_type,
            threshold=threshold,
            silent=silent,
            concurrency=concurrency,
            ollama_text_model_name=ollama_text_model,
            ollama_vision_model_name=ollama_vision_model,
        )
    )


if __name__ == "__main__":
    app()
