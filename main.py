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
)
import asyncio
import traceback

from agents.graph import create_graph, MERRState
from agents.models import GeminiModels


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
):
    """
    Internal async implementation for processing video files.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error: GOOGLE_API_KEY not found in .env file.[/bold red]"
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

    models = GeminiModels(api_key=api_key, verbose=verbose)

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

    semaphore = asyncio.Semaphore(concurrency)

    async def run_processing(video_file: Path):
        async with semaphore:
            if verbose:
                console.rule(f"[bold blue]Processing: {video_file.name}[/bold blue]")

            initial_state: MERRState = {
                "video_path": video_file,
                "output_dir": output_dir,
                "processing_type": processing_type.value,
                "models": models,
                "threshold": threshold,
                "verbose": verbose,
                "error_logs_dir": error_logs_dir,
            }

            try:
                final_state = await graph_app.ainvoke(initial_state)
                if final_state and final_state.get("error"):
                    results["failure"] += 1
                else:
                    results["success"] += 1
            except Exception:
                results["failure"] += 1
                error_log_path = error_logs_dir / f"{video_file.stem}_fatal_error.log"
                with open(error_log_path, "w") as f:
                    f.write(traceback.format_exc())
                console.print(
                    f"[bold red]FATAL ERROR processing {video_file.name}. Log saved to {error_log_path}.[/bold red]"
                )
            return video_file

    tasks = [run_processing(vf) for vf in video_files_to_process]

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
        await asyncio.gather(*tasks)

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
        )
    )


if __name__ == "__main__":
    app()
