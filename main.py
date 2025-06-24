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


@app.command()
def process(
    video_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the single video file to process.",
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
):
    """
    Processes a single video file based on the selected processing type.

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

    initial_state: MERRState = {
        "video_path": video_file,
        "output_dir": output_dir,
        "processing_type": processing_type.value,
        "models": models,
    }

    try:
        graph_app.invoke(initial_state)
    except Exception as e:
        console.rule(f"[bold red]FATAL ERROR processing {video_file.name}[/bold red]")
        console.print_exception(show_locals=True)


if __name__ == "__main__":
    app()
