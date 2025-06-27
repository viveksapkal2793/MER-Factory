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
    image = "image"


app = typer.Typer(
    name="merr-cli",
    help="A modular CLI tool to construct the MERR dataset from video and image files.",
    add_completion=False,
)
console = Console(stderr=True)  # Log errors to stderr
graph_app = create_graph()

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


async def _process(
    input_path: Path,
    output_dir: Path,
    processing_type: ProcessingType,
    threshold: float,
    silent: bool,
    concurrency: int,
    ollama_text_model_name: str = None,
    ollama_vision_model_name: str = None,
    huggingface_model_id: str = None,
):
    """
    Internal async implementation for processing video and image files.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if (
        not huggingface_model_id
        and not ollama_text_model_name
        and not ollama_vision_model_name
        and not api_key
    ):
        console.print(
            "[bold red]Error: One of --huggingface-model, GOOGLE_API_KEY, or --ollama-... must be provided.[/bold red]"
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

    files_to_process = []
    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS:
            files_to_process.append(input_path)
        else:
            console.print(
                f"[bold red]Error: '{input_path}' is not a recognized video or image file type.[/bold red]"
            )
            raise typer.Exit(code=1)
    elif input_path.is_dir():
        if verbose:
            console.print(
                f"Searching for video and image files in directory: [cyan]{input_path}[/cyan]"
            )
        all_extensions = VIDEO_EXTENSIONS.union(IMAGE_EXTENSIONS)
        for ext in all_extensions:
            files_to_process.extend(input_path.rglob(f"*{ext}"))
        if not files_to_process:
            console.print(
                f"[bold red]Error: No video or image files found in '{input_path}' or its subdirectories with extensions: {', '.join(all_extensions)}[/bold red]"
            )
            raise typer.Exit(code=1)
        if verbose:
            console.print(f"Found {len(files_to_process)} file(s) to process.")

    results = {"success": 0, "failure": 0}
    total_files = len(files_to_process)
    output_dir.mkdir(exist_ok=True)
    error_logs_dir = output_dir / "error_logs"
    error_logs_dir.mkdir(exist_ok=True)

    extraction_semaphore = asyncio.Semaphore(8)  # Limit concurrent extractions

    async def run_extraction_job(
        file_path: Path,
        file_output_dir: Path,
        progress: Optional[Progress],
        openface_task_id: Optional[TaskID],
        ffmpeg_task_id: Optional[TaskID],
    ):
        """
        Wrapper to run a single image/video's extraction tasks under semaphore control
        and update the consolidated progress bars.
        """
        async with extraction_semaphore:
            if verbose:
                console.log(
                    f"[yellow]Starting extraction for {file_path.name}...[/yellow]"
                )

            is_video = file_path.suffix.lower() in VIDEO_EXTENSIONS
            is_image = file_path.suffix.lower() in IMAGE_EXTENSIONS

            # For videos, OpenFace need depends on processing type. For images, it's always needed.
            needs_openface = is_image or (
                is_video and processing_type in [ProcessingType.mer, ProcessingType.au]
            )
            # FFmpeg is only needed for videos.
            needs_ffmpeg = is_video and processing_type in [
                ProcessingType.mer,
                ProcessingType.audio,
            ]

            try:
                if needs_openface:
                    res = await OpenFaceAdapter.run_feature_extraction(
                        file_path, file_output_dir, verbose
                    )
                    if progress and openface_task_id is not None:
                        progress.update(openface_task_id, advance=1)
                    if isinstance(res, Exception) or res is False:
                        console.log(
                            f"[bold red]Error during OpenFace for {file_path.name}: {res}[/bold red]"
                        )
                        return False

                if needs_ffmpeg:
                    audio_path = file_output_dir / f"{file_path.stem}.wav"
                    res = await FFMpegAdapter.extract_audio(
                        file_path, audio_path, verbose
                    )
                    if progress and ffmpeg_task_id is not None:
                        progress.update(ffmpeg_task_id, advance=1)
                    if isinstance(res, Exception) or res is False:
                        console.log(
                            f"[bold red]Error during FFmpeg for {file_path.name}: {res}[/bold red]"
                        )
                        return False

            except Exception as e:
                console.log(
                    f"[bold red]Fatal error during extraction for {file_path.name}: {e}[/bold red]"
                )
                return False

            return True

    if verbose:
        console.rule(
            "[bold yellow]Phase 1: Kicking off background feature extractions[/bold yellow]"
        )

    video_files = [f for f in files_to_process if f.suffix.lower() in VIDEO_EXTENSIONS]
    image_files = [f for f in files_to_process if f.suffix.lower() in IMAGE_EXTENSIONS]

    # Calculate required extractions
    openface_for_videos = (
        len(video_files)
        if processing_type in [ProcessingType.mer, ProcessingType.au]
        else 0
    )
    openface_for_images = len(image_files)
    total_openface_tasks = openface_for_videos + openface_for_images

    total_ffmpeg_tasks = (
        len(video_files)
        if processing_type in [ProcessingType.mer, ProcessingType.audio]
        else 0
    )

    if files_to_process and (total_openface_tasks > 0 or total_ffmpeg_tasks > 0):
        if not verbose:
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
                        "[bold blue]OpenFace Extraction", total=total_openface_tasks
                    )
                    if total_openface_tasks > 0
                    else None
                )
                ffmpeg_task_id = (
                    progress.add_task(
                        "[bold green]FFmpeg Extraction", total=total_ffmpeg_tasks
                    )
                    if total_ffmpeg_tasks > 0
                    else None
                )

                extraction_tasks = []
                for file in files_to_process:
                    file_output_dir = output_dir / file.stem
                    file_output_dir.mkdir(exist_ok=True)
                    extraction_tasks.append(
                        run_extraction_job(
                            file,
                            file_output_dir,
                            progress,
                            openface_task_id,
                            ffmpeg_task_id,
                        )
                    )
                await asyncio.gather(*extraction_tasks)
        else:
            extraction_tasks = []
            for file in files_to_process:
                file_output_dir = output_dir / file.stem
                file_output_dir.mkdir(exist_ok=True)
                extraction_tasks.append(
                    run_extraction_job(file, file_output_dir, None, None, None)
                )
            await asyncio.gather(*extraction_tasks)

    if verbose:
        console.rule(
            "[bold yellow]Phase 1 Complete: All extractions finished or failed[/bold yellow]"
        )

    # if using huggingface, the concurrency is set to 1
    concurrency = 1 if huggingface_model_id else concurrency
    if verbose:
        console.log(
            f"[bold yellow]Setting ocnurrency to {concurrency} for using huggingface models.[/bold yellow]"
        )

    main_processing_semaphore = asyncio.Semaphore(concurrency)

    async def run_processing_graph(file_path: Path):
        nonlocal results
        async with main_processing_semaphore:
            file_id = file_path.stem
            file_output_dir = output_dir / file_id
            is_image = file_path.suffix.lower() in IMAGE_EXTENSIONS
            current_processing_type = (
                ProcessingType.image if is_image else processing_type
            )

            if verbose:
                console.rule(
                    f"[bold blue]Phase 2: Processing Graph for: {file_path.name} (Type: {current_processing_type.value})[/bold blue]"
                )

            try:
                initial_state: MERRState = {
                    "video_path": file_path,
                    "output_dir": output_dir,
                    "processing_type": current_processing_type.value,
                    "models": models,
                    "threshold": threshold,
                    "verbose": verbose,
                    "error_logs_dir": error_logs_dir,
                    "video_id": file_id,
                    "video_output_dir": file_output_dir,
                    "audio_path": file_output_dir / f"{file_id}.wav",
                    "au_data_path": file_output_dir / f"{file_id}.csv",
                }

                final_state = await graph_app.ainvoke(initial_state)
                if final_state and final_state.get("error"):
                    results["failure"] += 1
                else:
                    results["success"] += 1

            except Exception as e:
                results["failure"] += 1
                error_log_path = error_logs_dir / f"{file_path.stem}_fatal_error.log"
                with open(error_log_path, "w") as f:
                    f.write(f"Error: {e}\n")
                    f.write(traceback.format_exc())
                console.print(
                    f"[bold red]FATAL ERROR processing {file_path.name}. Log saved to {error_log_path}.[/bold red]"
                )
            return file_path

    tasks = [run_processing_graph(vf) for vf in files_to_process]

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
            prog_task = progress.add_task("Processing files...", total=total_files)
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
    console.print(f"Total files attempted: {total_files}")
    console.print(f"✅ [green]Successful[/green]: {results['success']}")

    if results["failure"] > 0:
        console.print(f"❌ [red]Failed[/red]: {results['failure']}")
        console.print(f"Error logs have been saved in: [cyan]{error_logs_dir}[/cyan]")


@app.command()
def process(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Path to a single video/image file or a directory containing files to process.",
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
        help="The type of processing to perform for videos. For images, this is ignored and 'image' is used.",
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
        help="Number of files to process concurrently.",
    ),
    ollama_vision_model: str = typer.Option(
        None,
        "--ollama-vision-model",
        "-ovm",
        help="Name of the Ollama vision model to use (e.g., 'llava-llama3:latest').",
    ),
    ollama_text_model: str = typer.Option(
        None,
        "--ollama-text-model",
        "-otm",
        help="Name of the Ollama text model to use (e.g. llama3.2). If not provided, --ollama-vision-model will be used for text tasks.",
    ),
    huggingface_model_id: str = typer.Option(
        None,
        "--huggingface-model",
        "-hfm",
        # TODO: only gemma multimodal model is supported now
        help="ID of the Hugging Face model to use (e.g., 'google/gemma-3n-E4B-it', 'google/gemma-3n-E2B-it'). Takes precedence over other model providers.",
    ),
):
    """
    Processes a single file or multiple files from a directory
    based on the selected processing type. For images, the type is
    automatically set to 'image'.
    """
    asyncio.run(
        _process(
            input_path=input_path,
            output_dir=output_dir,
            processing_type=processing_type,
            threshold=threshold,
            silent=silent,
            concurrency=concurrency,
            ollama_text_model_name=ollama_text_model,
            ollama_vision_model_name=ollama_vision_model,
            huggingface_model_id=huggingface_model_id,
        )
    )


if __name__ == "__main__":
    app()
