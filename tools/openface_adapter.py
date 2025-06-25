import subprocess
from pathlib import Path
from rich.console import Console
import os

console = Console(stderr=True)


class OpenFaceAdapter:
    """A wrapper for the OpenFace FeatureExtraction command-line tool."""

    @staticmethod
    def run_feature_extraction(
        video_path: Path, output_dir: Path, verbose: bool = True
    ) -> bool:
        """
        Runs OpenFace feature extraction on a video file.
        This will generate a .csv file in the output directory.
        """
        output_csv = output_dir / f"{video_path.stem}.csv"
        if output_csv.exists():
            if verbose:
                console.log(
                    f"OpenFace output already exists: [cyan]{output_csv}[/cyan]. Skipping analysis."
                )
            return True

        command = [
            "/Users/linyuxiang/Desktop/openFace/OpenFace/build/bin/FeatureExtraction",
            "-f",
            str(video_path),
            "-out_dir",
            str(output_dir),
            "-aus",  # Extract Action Units
        ]

        try:
            if verbose:
                console.log(
                    f"Running OpenFace on [magenta]{video_path.name}[/magenta]..."
                )
            # On Windows, the executable might have a .exe extension
            if os.name == "nt":
                command[0] = "FeatureExtraction.exe"

            subprocess.run(command, check=True, capture_output=True, text=True)
            if verbose:
                console.log(
                    f"✅ OpenFace analysis complete. Output in [green]{output_dir}[/green]"
                )
            return True
        except FileNotFoundError:
            console.log(
                "[bold red]❌ Error: 'FeatureExtraction' command not found.[/bold red]"
            )
            console.log(
                "Please ensure OpenFace is installed and its build/bin (or Release) directory is in your system's PATH."
            )
            return False
        except subprocess.CalledProcessError as e:
            console.log(f"❌ OpenFace failed to process {video_path}.")
            console.log(f"OpenFace Error: {e.stderr}")
            return False
