import asyncio
from pathlib import Path
from rich.console import Console
import cv2
import numpy as np

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

console = Console(stderr=True)


class OpenCVAdapter:
    """A replacement for FFMpegAdapter using OpenCV and MoviePy."""

    @staticmethod
    async def extract_audio(
        video_path: Path, output_path: Path, verbose: bool = True
    ) -> bool:
        """Extracts audio from a video into a WAV file using MoviePy."""
        if output_path.exists():
            if verbose:
                console.log(f"Audio file exists: [cyan]{output_path}[/cyan]. Skipping.")
            return True

        if not MOVIEPY_AVAILABLE:
            console.log("[red]MoviePy not installed. Install with: pip install moviepy[/red]")
            return False

        try:
            if verbose:
                console.log(f"Extracting audio from [cyan]{video_path.name}[/cyan] using MoviePy...")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, OpenCVAdapter._extract_audio_sync, video_path, output_path, verbose
            )
            return success

        except Exception as e:
            console.log(f"❌ Failed to extract audio from {video_path.name}: {e}")
            return False

    @staticmethod
    def _extract_audio_sync(video_path: Path, output_path: Path, verbose: bool) -> bool:
        """Synchronous audio extraction using MoviePy."""
        try:
            video = VideoFileClip(str(video_path))
            
            if video.audio is None:
                console.log(f"[yellow]No audio track found in {video_path.name}[/yellow]")
                video.close()
                # Create minimal silent audio to avoid pipeline errors
                return OpenCVAdapter._create_silent_audio(output_path, verbose, duration=1.0)
            
            video.audio.write_audiofile(
                str(output_path),
                codec='pcm_s16le',  # WAV format
                fps=16000,  # 16kHz sample rate
                nbytes=2,
                verbose=False,
                logger=None
            )
            video.close()
            
            if verbose:
                console.log(f"✅ Extracted audio to [green]{output_path}[/green] using MoviePy")
            return True
            
        except Exception as e:
            console.log(f"❌ MoviePy extraction failed: {e}")
            # Fallback to silent audio
            return False
            return OpenCVAdapter._create_silent_audio(output_path, verbose, duration=1.0)

    @staticmethod
    def _create_silent_audio(output_path: Path, verbose: bool, duration: float = 1.0) -> bool:
        """Create a silent audio file with proper format."""
        try:
            import wave
            
            sample_rate = 16000
            samples = int(sample_rate * duration)
            
            with wave.open(str(output_path), 'w') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Create silent audio data
                silence = np.zeros(samples, dtype=np.int16)
                wav_file.writeframes(silence.tobytes())
            
            if verbose:
                console.log(f"[yellow]Created {duration:.1f}s silent audio: {output_path}[/yellow]")
            return True
            
        except Exception as e:
            console.log(f"❌ Failed to create silent audio: {e}")
            return False

    @staticmethod
    async def extract_frame(
        video_path: Path, timestamp: float, output_path: Path, verbose: bool = True
    ) -> bool:
        """Extracts a single frame from a video at a specific timestamp using OpenCV."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, OpenCVAdapter._extract_frame_sync, video_path, timestamp, output_path, verbose
            )
            return success

        except Exception as e:
            console.log(f"❌ Failed to extract frame from {video_path.name}: {e}")
            return False

    @staticmethod
    def _extract_frame_sync(
        video_path: Path, timestamp: float, output_path: Path, verbose: bool
    ) -> bool:
        """Synchronous frame extraction using OpenCV."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                console.log(f"❌ Could not open video: {video_path}")
                return False
            
            # Get FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                console.log(f"❌ Invalid FPS: {fps}")
                cap.release()
                return False
            
            # Calculate frame number
            frame_number = int(timestamp * fps)
            
            # Set position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            
            if ret and frame is not None:
                success = cv2.imwrite(str(output_path), frame)
                cap.release()
                
                if success and verbose:
                    console.log(f"✅ Successfully extracted frame at {timestamp:.3f}s to [green]{output_path}[/green]")
                return success
            else:
                console.log(f"❌ Failed to read frame at {timestamp:.3f}s")
                cap.release()
                return False
            
        except Exception as e:
            console.log(f"❌ Error extracting frame: {e}")
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
        Tries to extract a frame at the given timestamp. If it fails, tries nearby frames.
        """
        if output_path.exists():
            if verbose:
                console.log(f"Frame file exists: [cyan]{output_path}[/cyan]. Skipping.")
            return True

        # First attempt at exact timestamp
        if await OpenCVAdapter.extract_frame(video_path, timestamp, output_path, verbose):
            return True

        console.log(f"[yellow]Initial frame extraction failed at {timestamp:.3f}s. Trying nearby frames.[/yellow]")

        # Get FPS for offset calculation
        fps = await OpenCVAdapter._get_video_framerate(video_path)
        if not fps:
            return False

        frame_duration = 1 / fps

        # Try nearby frames
        for i in range(1, attempts + 1):
            offset = i * frame_duration

            # Try backward
            backward_ts = max(0, timestamp - offset)
            if await OpenCVAdapter.extract_frame(video_path, backward_ts, output_path, verbose):
                return True

            # Try forward
            forward_ts = timestamp + offset
            if await OpenCVAdapter.extract_frame(video_path, forward_ts, output_path, verbose):
                return True

        console.log(f"[bold red]All fallback attempts failed for {video_path.name}[/bold red]")
        return False

    @staticmethod
    async def _get_video_framerate(video_path: Path) -> float | None:
        """Gets video framerate using OpenCV."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            return fps if fps > 0 else None
            
        except Exception:
            return None

    @staticmethod
    async def _get_video_duration(video_path: Path) -> float | None:
        """Gets video duration using OpenCV."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            
            if fps > 0 and frame_count > 0:
                return frame_count / fps
            return None
            
        except Exception:
            return None

    # Synchronous versions for compatibility
    @staticmethod
    def extract_frame_sync(
        video_path: Path, timestamp: float, output_path: Path, verbose: bool = True
    ) -> bool:
        """Synchronous frame extraction."""
        return OpenCVAdapter._extract_frame_sync(video_path, timestamp, output_path, verbose)

    @staticmethod
    def extract_nearby_frame_sync(
        video_path: Path,
        timestamp: float,
        output_path: Path,
        verbose: bool = True,
        attempts: int = 2,
    ) -> bool:
        """Synchronous nearby frame extraction."""
        if output_path.exists():
            if verbose:
                console.log(f"Frame file exists: [cyan]{output_path}[/cyan]. Skipping.")
            return True

        # First attempt
        if OpenCVAdapter.extract_frame_sync(video_path, timestamp, output_path, verbose):
            return True

        # Get FPS
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        except Exception:
            return False

        if fps <= 0:
            return False

        frame_duration = 1 / fps

        # Try nearby frames
        for i in range(1, attempts + 1):
            offset = i * frame_duration

            # Backward
            backward_ts = max(0, timestamp - offset)
            if OpenCVAdapter.extract_frame_sync(video_path, backward_ts, output_path, verbose):
                return True

            # Forward
            forward_ts = timestamp + offset
            if OpenCVAdapter.extract_frame_sync(video_path, forward_ts, output_path, verbose):
                return True

        return False

    @staticmethod
    def _get_video_framerate_sync(video_path: Path) -> float | None:
        """Synchronous framerate detection."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            return fps if fps > 0 else None
            
        except Exception:
            return None