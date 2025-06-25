import asyncio
from pathlib import Path
from rich.console import Console

console = Console(stderr=True)


class FFMpegAdapter:
    """A wrapper for common ffmpeg commands using asyncio."""

    @staticmethod
    async def extract_audio(
        video_path: Path, output_path: Path, verbose: bool = True
    ) -> bool:
        """Extracts audio from a video into a WAV file asynchronously."""
        if output_path.exists():
            if verbose:
                console.log(f"Audio file exists: [cyan]{output_path}[/cyan]. Skipping.")
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

        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            console.log(f"❌ Failed to extract audio from {video_path.name}.")
            console.log(f"FFmpeg Error: {stderr.decode().strip()}")
            return False

        if verbose:
            console.log(f"✅ Extracted audio to [green]{output_path}[/green]")
        return True

    @staticmethod
    async def extract_frame(
        video_path: Path, timestamp: float, output_path: Path, verbose: bool = True
    ) -> bool:
        """Extracts a single frame from a video at a specific timestamp asynchronously."""
        if output_path.exists():
            if verbose:
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

        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            console.log(
                f"❌ Failed to extract frame from {video_path.name} at {timestamp}s."
            )
            console.log(f"FFmpeg Error: {stderr.decode().strip()}")
            return False

        if verbose:
            console.log(
                f"✅ Extracted frame at {timestamp:.2f}s to [green]{output_path}[/green]"
            )
        return True
