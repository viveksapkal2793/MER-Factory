import asyncio
from pathlib import Path
from rich.console import Console
import os

console = Console(stderr=True)


class OpenFaceAdapter:
    """A wrapper for the OpenFace FeatureExtraction tool using asyncio."""

    @staticmethod
    async def run_feature_extraction(
        video_path: Path, output_dir: Path, verbose: bool = True
    ) -> bool:
        """Runs OpenFace feature extraction asynchronously."""
        output_csv = output_dir / f"{video_path.stem}.csv"
        if output_csv.exists():
            if verbose:
                console.log(
                    f"OpenFace output exists: [cyan]{output_csv}[/cyan]. Skipping."
                )
            return True

        # --- IMPORTANT: Update this path to your OpenFace installation ---
        openface_executable = (
            "/Users/linyuxiang/Desktop/openFace/OpenFace/build/bin/FeatureExtraction"
        )
        if os.name == "nt":
            openface_executable += ".exe"

        command = [
            openface_executable,
            "-f",
            str(video_path),
            "-out_dir",
            str(output_dir),
            "-aus",  # Extract Action Units
        ]

        if verbose:
            console.log(f"Running OpenFace on [magenta]{video_path.name}[/magenta]...")

        try:
            proc = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                console.log(f"❌ OpenFace failed to process {video_path.name}.")
                console.log(f"OpenFace Error: {stderr.decode().strip()}")
                return False

            if verbose:
                console.log(
                    f"✅ OpenFace analysis complete. Output in [green]{output_dir}[/green]"
                )
            return True

        except FileNotFoundError:
            console.log(
                f"[bold red]❌ Error: '{openface_executable}' not found.[/bold red]"
            )
            console.log("Please ensure the path in `openface_adapter.py` is correct.")
            return False
