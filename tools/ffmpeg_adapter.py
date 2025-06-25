import asyncio
from pathlib import Path
from rich.console import Console
import shutil

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
        # This is the core worker function that attempts a single extraction.
        # It uses the -update flag which is the most modern approach.
        command = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-update",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output_path),
        ]

        if verbose:
            console.log(f"Attempting to extract frame at {timestamp:.3f}s...")

        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

        if (
            proc.returncode == 0
            and output_path.exists()
            and output_path.stat().st_size > 0
        ):
            if verbose:
                console.log(
                    f"✅ Successfully extracted frame at {timestamp:.3f}s to [green]{output_path}[/green]"
                )
            return True

        if verbose:
            console.log(f"Failed to extract frame at {timestamp:.3f}s.")
        return False

    @staticmethod
    async def extract_nearby_frame(
        video_path: Path,
        timestamp: float,
        output_path: Path,
        verbose: bool = True,
        attempts: int = 2,
    ) -> bool:
        """
        Tries to extract a frame at the given timestamp. If it fails, it will
        try to extract nearby frames as a fallback.
        """
        if output_path.exists():
            if verbose:
                console.log(f"Frame file exists: [cyan]{output_path}[/cyan]. Skipping.")
            return True

        # 1. First attempt at the exact timestamp
        if await FFMpegAdapter.extract_frame(
            video_path, timestamp, output_path, verbose
        ):
            return True

        console.log(
            f"[yellow]Initial frame extraction failed at {timestamp:.3f}s. Trying nearby frames as a fallback.[/yellow]"
        )

        # 2. Get framerate to calculate the time offset for one frame
        fps = await FFMpegAdapter._get_video_framerate(video_path)
        if not fps:
            console.log(
                "[bold red]Could not determine video framerate to attempt fallback seeks.[/bold red]"
            )
            return False

        frame_duration = 1 / fps

        # 3. Loop for fallback attempts (e.g., frame-1, frame+1, frame-2, frame+2)
        for i in range(1, attempts + 1):
            offset = i * frame_duration

            # Try seeking backward
            backward_ts = max(0, timestamp - offset)  # Ensure timestamp isn't negative
            if await FFMpegAdapter.extract_frame(
                video_path, backward_ts, output_path, verbose
            ):
                return True

            # Try seeking forward
            forward_ts = timestamp + offset
            if await FFMpegAdapter.extract_frame(
                video_path, forward_ts, output_path, verbose
            ):
                return True

        console.log(
            f"[bold red]All fallback attempts failed for {video_path.name}. Could not extract a valid frame.[/bold red]"
        )
        return False

    @staticmethod
    async def _get_video_duration(video_path: Path) -> float | None:
        """Gets the video duration in seconds using ffprobe."""
        if not shutil.which("ffprobe"):
            console.log(
                "[bold red]Warning: ffprobe not found. Cannot get video duration to clamp timestamp.[/bold red]"
            )
            return None

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
        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            console.log(f"❌ Failed to get duration for {video_path.name}.")
            console.log(f"ffprobe Error: {stderr.decode().strip()}")
            return None

        try:
            return float(stdout.decode().strip())
        except (ValueError, TypeError):
            console.log(
                f"❌ Could not parse duration from ffprobe output: {stdout.decode().strip()}"
            )
            return None

    @staticmethod
    async def _get_video_framerate(video_path: Path) -> float | None:
        """Gets the video framerate in seconds using ffprobe."""
        if not shutil.which("ffprobe"):
            console.log(
                "[bold red]Warning: ffprobe not found. Cannot get video framerate.[/bold red]"
            )
            return None

        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            console.log(f"❌ Failed to get framerate for {video_path.name}.")
            console.log(f"ffprobe Error: {stderr.decode().strip()}")
            return None

        try:
            rate_str = stdout.decode().strip()
            if "/" in rate_str:
                num, den = map(int, rate_str.split("/"))
                return num / den
            else:
                return float(rate_str)
        except (ValueError, TypeError, ZeroDivisionError) as e:
            console.log(
                f"❌ Could not parse framerate from ffprobe output: '{stdout.decode().strip()}'. Error: {e}"
            )
            return None
