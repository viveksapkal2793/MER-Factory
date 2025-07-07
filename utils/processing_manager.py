import asyncio
import functools
import traceback
from pathlib import Path
from typing import List, Dict, Coroutine, Any

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TaskID,
)

from agents.graph import MERRState
from agents.models import LLMModels
from .config import AppConfig, VIDEO_EXTENSIONS, IMAGE_EXTENSIONS, ProcessingType
from tools.ffmpeg_adapter import FFMpegAdapter
from tools.openface_adapter import OpenFaceAdapter

console = Console(stderr=True)


def build_initial_state(
    file_path: Path, config: AppConfig, models: LLMModels
) -> MERRState:
    """Builds the initial state dictionary for a file to be processed by the graph."""
    file_id = file_path.stem
    file_output_dir = config.output_dir / file_id
    is_image = file_path.suffix.lower() in IMAGE_EXTENSIONS
    current_processing_type = "image" if is_image else config.processing_type.value

    state = {
        "video_path": file_path,
        "output_dir": config.output_dir,
        "processing_type": current_processing_type,
        "models": models,
        "threshold": config.threshold,
        "peak_distance_frames": config.peak_distance_frames,
        "verbose": config.verbose,
        "error_logs_dir": config.error_logs_dir,
        "video_id": file_id,
        "video_output_dir": file_output_dir,
        "audio_path": file_output_dir / f"{file_id}.wav",
        "au_data_path": file_output_dir / f"{file_id}.csv",
    }

    ground_truth_label = config.labels.get(file_id)
    if ground_truth_label:
        state["ground_truth_label"] = ground_truth_label
        if config.verbose:
            console.log(
                f"Found ground truth label for {file_id}: [bold yellow]{ground_truth_label}[/bold yellow]"
            )

    return state


async def _run_async_job(
    coro: Coroutine,
    file_path: Path,
    task_id: TaskID,
    progress: Progress,
    results: Dict[str, int],
    semaphore: asyncio.Semaphore,
):
    """A helper to run a single asynchronous processing job with a semaphore and progress updates."""
    async with semaphore:
        try:
            final_state = await coro
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
        finally:
            if progress and task_id is not None:
                progress.update(
                    task_id,
                    advance=1,
                    description=f"Finished [cyan]{file_path.name}[/cyan]",
                )


def _run_sync_job(
    invoke_func: Any,
    file_path: Path,
    task_id: TaskID,
    progress: Progress,
    results: Dict[str, int],
):
    """A helper to run a single synchronous processing job with progress updates."""
    try:
        final_state = invoke_func()
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
    finally:
        if progress and task_id is not None:
            progress.update(
                task_id,
                advance=1,
                description=f"Finished [cyan]{file_path.name}[/cyan]",
            )


async def run_main_processing(
    files_to_process: List[Path],
    graph_app: Any,
    initial_state_builder: functools.partial,
    config: AppConfig,
    is_sync: bool,
) -> Dict[str, int]:
    """
    Manages the main processing loop, dispatching to sync or async helpers.
    """
    results = {"success": 0, "failure": 0}
    total_files = len(files_to_process)
    semaphore = asyncio.Semaphore(config.concurrency)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed} of {task.total})"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        disable=config.verbose,
    ) as progress:
        task = progress.add_task("Processing files...", total=total_files)

        if is_sync:
            # Synchronous execution (for Hugging Face)
            for file_path in files_to_process:
                initial_state = initial_state_builder(file_path=file_path)
                invoke_func = functools.partial(graph_app.invoke, initial_state)
                _run_sync_job(invoke_func, file_path, task, progress, results)
        else:
            # Asynchronous execution (for Gemini/Ollama)
            tasks = []
            for file_path in files_to_process:
                initial_state = initial_state_builder(file_path=file_path)
                coro = graph_app.ainvoke(initial_state)
                tasks.append(
                    _run_async_job(coro, file_path, task, progress, results, semaphore)
                )
            await asyncio.gather(*tasks)

    return results


async def run_feature_extraction(files_to_process: List[Path], config: AppConfig):
    """
    Asynchronously runs FFmpeg and OpenFace feature extraction on the files.
    """
    extraction_semaphore = asyncio.Semaphore(4)  # Limit concurrent external processes

    async def run_job(
        file_path: Path,
        openface_task_id: TaskID,
        ffmpeg_task_id: TaskID,
        progress: Progress,
    ):
        async with extraction_semaphore:
            file_output_dir = config.output_dir / file_path.stem
            file_output_dir.mkdir(exist_ok=True)
            is_video = file_path.suffix.lower() in VIDEO_EXTENSIONS
            is_image = file_path.suffix.lower() in IMAGE_EXTENSIONS
            needs_openface = is_image or (
                is_video
                and config.processing_type in [ProcessingType.MER, ProcessingType.AU]
            )
            needs_ffmpeg = is_video and config.processing_type in [
                ProcessingType.MER,
                ProcessingType.AUDIO,
            ]

            try:
                if (
                    needs_openface
                    and not (file_output_dir / f"{file_path.stem}.csv").exists()
                ):
                    await OpenFaceAdapter.run_feature_extraction(
                        file_path, file_output_dir, config.verbose
                    )
                if (
                    needs_ffmpeg
                    and not (file_output_dir / f"{file_path.stem}.wav").exists()
                ):
                    await FFMpegAdapter.extract_audio(
                        file_path,
                        file_output_dir / f"{file_path.stem}.wav",
                        config.verbose,
                    )
            except Exception as e:
                console.log(
                    f"[bold red]Extraction failed for {file_path.name}: {e}[/bold red]"
                )
            finally:
                if needs_openface and progress and openface_task_id is not None:
                    progress.update(openface_task_id, advance=1)
                if needs_ffmpeg and progress and ffmpeg_task_id is not None:
                    progress.update(ffmpeg_task_id, advance=1)

    video_files = [f for f in files_to_process if f.suffix.lower() in VIDEO_EXTENSIONS]
    image_files = [f for f in files_to_process if f.suffix.lower() in IMAGE_EXTENSIONS]
    total_openface = len(image_files) + (
        len(video_files)
        if config.processing_type in [ProcessingType.MER, ProcessingType.AU]
        else 0
    )
    total_ffmpeg = (
        len(video_files)
        if config.processing_type in [ProcessingType.MER, ProcessingType.AUDIO]
        else 0
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed} of {task.total})"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        disable=config.verbose,
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
