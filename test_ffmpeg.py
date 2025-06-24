import typer
import subprocess
from pathlib import Path
from rich.console import Console


app = typer.Typer(
    name="ffmpeg-tester",
    help="A simple tool to test ffmpeg video and audio processing.",
    add_completion=False,
)
console = Console()


def get_video_duration(video_path: Path) -> float:
    """Uses ffprobe to get the duration of a video in seconds."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        console.log("Probing video for duration...")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        console.log(f"Video duration: [yellow]{duration:.2f} seconds[/yellow]")
        return duration
    except FileNotFoundError:
        console.rule("[bold red]❌ ERROR: 'ffprobe' not found[/bold red]")
        console.log(
            "Please ensure FFmpeg is installed and its 'bin' directory is in your system's PATH."
        )
        raise typer.Exit(code=1)
    except (ValueError, subprocess.CalledProcessError) as e:
        console.log(
            f"[bold red]Could not determine video duration. Error: {e}[/bold red]"
        )
        return 0.0


@app.command()
def run(
    video_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to the video file you want to test.",
    ),
    output_dir: Path = typer.Argument(
        ...,
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory to save the output files.",
    ),
):
    """
    Tests FFmpeg by extracting audio and a video frame.
    """
    console.rule(f"[bold magenta]FFmpeg Test for: {video_file.name}[/bold magenta]")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = video_file.stem

    console.print("\n[bold]-- Test 1: Extracting Audio --[/bold]")
    audio_output_path = output_dir / f"{video_id}_audio.wav"
    audio_command = [
        "ffmpeg",
        "-i",
        str(video_file),
        "-vn",  # No video
        "-q:a",
        "0",
        "-y",  # Overwrite output file if it exists
        str(audio_output_path),
    ]
    try:
        console.log(f"Running: {' '.join(audio_command)}")
        subprocess.run(audio_command, check=True, capture_output=True, text=True)
        console.print(
            f"[green]✅ Success![/green] Audio extracted to: [cyan]{audio_output_path}[/cyan]"
        )
    except FileNotFoundError:
        console.rule("[bold red]❌ ERROR: 'ffmpeg' not found[/bold red]")
        console.log(
            "Please ensure FFmpeg is installed and its 'bin' directory is in your system's PATH."
        )
        raise typer.Exit(code=1)
    except subprocess.CalledProcessError as e:
        console.print("[bold red]❌ Audio extraction failed.[/bold red]")
        console.print(f"FFmpeg stderr:\n{e.stderr}")

    console.print("\n[bold]-- Test 2: Extracting Middle Frame --[/bold]")
    duration = get_video_duration(video_file)
    if duration > 0:
        middle_point = duration / 2
        frame_output_path = output_dir / f"{video_id}_middle_frame.png"
        frame_command = [
            "ffmpeg",
            "-i",
            str(video_file),
            "-ss",
            str(middle_point),
            "-vframes",
            "1",
            "-q:v",
            "2",
            "-y",
            str(frame_output_path),
        ]
        try:
            console.log(f"Running: {' '.join(frame_command)}")
            subprocess.run(frame_command, check=True, capture_output=True, text=True)
            console.print(
                f"[green]✅ Success![/green] Frame extracted to: [cyan]{frame_output_path}[/cyan]"
            )
        except subprocess.CalledProcessError as e:
            console.print("[bold red]❌ Frame extraction failed.[/bold red]")
            console.print(f"FFmpeg stderr:\n{e.stderr}")


if __name__ == "__main__":
    app()
