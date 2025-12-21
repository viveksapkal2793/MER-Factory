from pathlib import Path
from typing import List, Dict
import json
import pandas as pd
from rich.console import Console

from .config import VIDEO_EXTENSIONS, IMAGE_EXTENSIONS, AUDIO_EXTENSIONS

console = Console(stderr=True)


def find_files_to_process(input_path: Path, verbose: bool = True) -> List[Path]:
    """
    Finds all valid video, image, or audio files from a given input path.

    Args:
        input_path: The path to a single file or a directory.
        verbose: If True, prints detailed output.

    Returns:
        A list of paths to the files that need to be processed.

    Raises:
        SystemExit: If the input file is invalid or no files are found.
    """
    files_to_process = []
    all_extensions = VIDEO_EXTENSIONS.union(IMAGE_EXTENSIONS).union(AUDIO_EXTENSIONS)

    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext in all_extensions:
            files_to_process.append(input_path)
        else:
            console.print(
                f"[bold red]Error: '{input_path}' is not a recognized video, image, or audio file.[/bold red]"
            )
            raise SystemExit(1)
    elif input_path.is_dir():
        if verbose:
            console.print(f"Searching for files in [cyan]{input_path}[/cyan]...")
        for ext in all_extensions:
            files_to_process.extend(input_path.rglob(f"*{ext}"))
        if not files_to_process:
            console.print(
                f"[bold red]Error: No video, image, or audio files found in '{input_path}'.[/bold red]"
            )
            raise SystemExit(1)
        if verbose:
            console.print(f"Found {len(files_to_process)} file(s).")
    return files_to_process


def load_labels_from_file(label_file: Path, verbose: bool = True) -> Dict[str, str]:
    """
    Loads ground truth labels from a specified CSV file.

    The CSV file must contain 'name' and 'label' columns.

    Args:
        label_file: Path to the CSV file.
        verbose: If True, prints detailed output.

    Returns:
        A dictionary mapping file stem to its label.

    Raises:
        SystemExit: If the file is not found, is invalid, or cannot be parsed.
    """
    if not label_file.exists():
        console.print(
            f"[bold red]Error: Label file not found at '{label_file}'[/bold red]"
        )
        raise SystemExit(1)
    try:
        df = pd.read_csv(label_file)
        if "name" not in df.columns or "label" not in df.columns:
            console.print(
                "[bold red]Error: Label file must contain 'name' and 'label' columns.[/bold red]"
            )
            raise SystemExit(1)
        labels = pd.Series(df.label.values, index=df.name).to_dict()
        if verbose:
            console.log(
                f"Successfully loaded {len(labels)} labels from [cyan]{label_file.name}[/cyan]"
            )
        return labels
    except Exception as e:
        console.print(f"[bold red]Error reading or parsing label file: {e}[/bold red]")
        raise SystemExit(1)

def check_json_completeness(json_path: Path) -> tuple[bool, list[str]]:
    """
    Check if a JSON file exists and has all required fields.
    
    Args:
        json_path: Path to the JSON file to check.
    
    Returns:
        (is_complete, missing_fields): Tuple with completion status and list of missing fields
    """
    if not json_path.exists():
        return False, ["JSON file does not exist"]
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        missing_fields = []
        
        # Check top-level required fields
        if not data.get("chronological_emotion_peaks"):
            missing_fields.append("chronological_emotion_peaks")
        
        if not data.get("overall_peak_frame_info"):
            missing_fields.append("overall_peak_frame_info")
        
        # Check nested fields in coarse_descriptions_at_peak
        coarse_desc = data.get("coarse_descriptions_at_peak", {})
        if not coarse_desc.get("visual_expression"):
            missing_fields.append("visual_expression")
        if not coarse_desc.get("visual_objective"):
            missing_fields.append("visual_objective")
        if not coarse_desc.get("audio_analysis"):
            missing_fields.append("audio_analysis")
        
        # Check final_summary
        if not data.get("final_summary"):
            missing_fields.append("final_summary")
        
        return len(missing_fields) == 0, missing_fields
        
    except json.JSONDecodeError:
        return False, ["Invalid JSON format"]
    except Exception as e:
        return False, [f"Error reading file: {str(e)}"]