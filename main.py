import typer
from pathlib import Path
from dotenv import load_dotenv
import os
from rich.console import Console
from enum import Enum


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
console = Console()
graph_app = create_graph()

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


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
        ProcessingType.mer,  # Default to the full pipeline
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
):
    """
    Processes a single video file or multiple video files from a directory
    based on the selected processing type.

    - AU: Extracts Action Units and generates a facial expression description.
    - audio: Extracts audio and generates a transcript and tone analysis.
    - video: Generates a general content description of the video.
    - MER: Runs the full, end-to-end multimodal emotion recognition pipeline.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error: GOOGLE_API_KEY not found in .env file. It is required for all processing types.[/bold red]"
        )
        raise typer.Exit(code=1)

    console.rule(
        f"[bold magenta]MERR CLI - Mode: {processing_type.value}[/bold magenta]"
    )

    models = GeminiModels(api_key=api_key)

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
        console.print(
            f"Searching for video files in directory: [cyan]{video_path}[/cyan]"
        )
        for ext in VIDEO_EXTENSIONS:
            video_files_to_process.extend(
                video_path.rglob(f"*{ext}")
            )  # rglob for recursive search
        if not video_files_to_process:
            console.print(
                f"[bold red]Error: No video files found in '{video_path}' or its subdirectories with extensions: {', '.join(VIDEO_EXTENSIONS)}[/bold red]"
            )
            raise typer.Exit(code=1)
        console.print(f"Found {len(video_files_to_process)} video(s) to process.")
    else:
        console.print(
            f"[bold red]Error: '{video_path}' is neither a file nor a directory.[/bold red]"
        )
        raise typer.Exit(code=1)

    for i, current_video_file in enumerate(video_files_to_process):
        console.rule(
            f"[bold blue]Processing video {i+1}/{len(video_files_to_process)}: {current_video_file.name}[/bold blue]"
        )
        initial_state: MERRState = {
            "video_path": current_video_file,  # Use current_video_file
            "output_dir": output_dir,
            "processing_type": processing_type.value,
            "models": models,
            "threshold": threshold,
        }

        try:
            graph_app.invoke(initial_state)
        except Exception as e:
            console.rule(
                f"[bold red]FATAL ERROR processing {current_video_file.name}[/bold red]"
            )
            console.print_exception(show_locals=True)
            console.print(f"[bold red]Skipping to next video due to error.[/bold red]")


if __name__ == "__main__":
    app()
