import subprocess
from pathlib import Path
from rich.console import Console

console = Console()


class FFMpegAdapter:
    """A wrapper for common ffmpeg commands needed for the MERR pipeline."""

    @staticmethod
    def extract_audio(video_path: Path, output_path: Path) -> bool:
        """
        Extracts the audio from a video file into a WAV file.

        Args:
            video_path: Path to the input video.
            output_path: Path to save the output WAV audio file.

        Returns:
            True if successful, False otherwise.
        """
        if output_path.exists():
            console.log(
                f"Audio file already exists: [cyan]{output_path}[/cyan]. Skipping extraction."
            )
            return True

        command = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-q:a",
            "0",  # Best quality
            "-map",
            "a",  # Map audio stream
            "-ac",
            "1",  # Mono channel
            "-ar",
            "16000",  # 16kHz sample rate
            str(output_path),
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            console.log(f"✅ Extracted audio to [green]{output_path}[/green]")
            return True
        except subprocess.CalledProcessError as e:
            console.log(f"❌ Failed to extract audio from {video_path}.")
            console.log(f"FFmpeg Error: {e.stderr}")
            return False

    @staticmethod
    def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
        """
        Extracts a single frame from a video at a specific timestamp.

        Args:
            video_path: Path to the input video.
            timestamp: The time in seconds to extract the frame from.
            output_path: Path to save the output PNG image file.

        Returns:
            True if successful, False otherwise.
        """
        if output_path.exists():
            console.log(
                f"Frame file already exists: [cyan]{output_path}[/cyan]. Skipping extraction."
            )
            return True

        command = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-ss",
            str(timestamp),  # Seek to timestamp
            "-vframes",
            "1",  # Extract one frame
            "-q:v",
            "2",  # High quality
            str(output_path),
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            console.log(
                f"✅ Extracted frame at {timestamp:.2f}s to [green]{output_path}[/green]"
            )
            return True
        except subprocess.CalledProcessError as e:
            console.log(
                f"❌ Failed to extract frame from {video_path} at {timestamp}s."
            )
            console.log(f"FFmpeg Error: {e.stderr}")
            return False
