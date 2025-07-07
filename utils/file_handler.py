from pathlib import Path
from typing import List, Dict

import pandas as pd
from rich.console import Console

# Use a relative import because this module is now part of the 'utils' package
from .config import VIDEO_EXTENSIONS, IMAGE_EXTENSIONS

console = Console(stderr=True)


def find_files_to_process(input_path: Path, verbose: bool = True) -> List[Path]:
    """
    Finds all valid video or image files from a given input path (file or directory).

    Args:
        input_path: The path to a single file or a directory.
        verbose: If True, prints detailed output.

    Returns:
        A list of paths to the files that need to be processed.

    Raises:
        SystemExit: If the input file is invalid or no files are found.
    """
    files_to_process = []
    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS:
            files_to_process.append(input_path)
        else:
            console.print(
                f"[bold red]Error: '{input_path}' is not a recognized video/image file.[/bold red]"
            )
            raise SystemExit(1)
    elif input_path.is_dir():
        if verbose:
            console.print(f"Searching for files in [cyan]{input_path}[/cyan]...")
        all_extensions = VIDEO_EXTENSIONS.union(IMAGE_EXTENSIONS)
        for ext in all_extensions:
            files_to_process.extend(input_path.rglob(f"*{ext}"))
        if not files_to_process:
            console.print(
                f"[bold red]Error: No video or image files found in '{input_path}'.[/bold red]"
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
