import typer
import subprocess
from pathlib import Path
from rich.console import Console


app = typer.Typer()
console = Console()


@app.command()
def run(
    openface_executable: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Full path to the OpenFace 'FeatureExtraction' executable.",
    ),
    video_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to the video file you want to analyze.",
    ),
    output_dir: Path = typer.Argument(
        ...,
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory to save the OpenFace output CSV file.",
    ),
):
    """
    Tests the integration with the OpenFace FeatureExtraction tool.
    """
    console.rule("[bold magenta]OpenFace Integration Test[/bold magenta]")
    console.print(f"OpenFace Path: [cyan]{openface_executable}[/cyan]")
    console.print(f"Video File: [cyan]{video_file}[/cyan]")
    console.print(f"Output Directory: [cyan]{output_dir}[/cyan]")

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(openface_executable),
        "-f",
        str(video_file),
        "-out_dir",
        str(output_dir),
        "-aus",
    ]

    console.print("\n[bold]Running command:[/bold]")
    console.print(f"[yellow]{' '.join(command)}[/yellow]\n")

    try:

        console.log("🚀 Starting OpenFace analysis... (This might take a while)")
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        console.rule("[bold green]✅ Success![/bold green]")
        console.log("OpenFace completed the analysis successfully.")
        console.log(f"Check for a '.csv' file in your output directory: {output_dir}")

    except FileNotFoundError:
        console.rule("[bold red]❌ Error: Command not found[/bold red]")
        console.log(
            f"The script could not find the executable at the path you provided:"
        )
        console.log(f"[cyan]{openface_executable}[/cyan]")
        console.log(
            "Please double-check that the path is correct and the file is executable."
        )

    except subprocess.CalledProcessError as e:
        console.rule(f"[bold red]❌ Error: OpenFace Failed[/bold red]")
        console.log("OpenFace ran but encountered an error. See details below.")
        console.print("\n[bold]--- Stderr ---[/bold]")
        console.print(f"[red]{e.stderr}[/red]")
        console.print("\n[bold]--- Stdout ---[/bold]")
        console.print(f"{e.stdout}")

    except Exception as e:
        console.rule(f"[bold red]❌ An unexpected error occurred[/bold red]")
        console.print_exception()


if __name__ == "__main__":
    app()
