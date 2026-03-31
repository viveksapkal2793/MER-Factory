#!/usr/bin/env python3
"""
Script to check how many video subdirectories have complete JSON files.
"""

import json
from pathlib import Path
from typing import Tuple, List, Dict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


def check_json_completeness(json_path: Path) -> Tuple[bool, List[str]]:
    """
    Check if a JSON file exists and is complete.
    
    Returns:
        (is_complete, missing_fields): Tuple with completeness flag and list of missing required fields
    """
    if not json_path.exists():
        return False, ["file_missing"]
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Define required fields based on file type
        if "_audio_analysis.json" in json_path.name:
            required_fields = ["source_path", "audio_analysis"]
        elif "_merr_data.json" in json_path.name:
            required_fields = [
                "source_path",
                "chronological_emotion_peaks",
                "overall_peak_au_description",
                "audio_analysis",
                "video_summary",
                "visual_perception",
                "synthesis",
            ]
        else:
            required_fields = []
        
        missing = [field for field in required_fields if field not in data or data[field] is None]
        
        return len(missing) == 0, missing
        
    except (json.JSONDecodeError, IOError) as e:
        return False, [f"parse_error: {str(e)}"]


def scan_output_directory(output_dir: Path, json_pattern: str = "*_audio_analysis.json") -> Dict:
    """
    Scan output directory and check JSON completeness for all subdirectories.
    
    Args:
        output_dir: Path to output directory
        json_pattern: JSON file pattern to look for (e.g., "*_audio_analysis.json", "*_merr_data.json")
    
    Returns:
        Dictionary with statistics
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        console.print(f"[bold red]Error: Output directory does not exist: {output_dir}[/bold red]")
        return {}
    
    subdirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not subdirs:
        console.print(f"[bold yellow]No subdirectories found in {output_dir}[/bold yellow]")
        return {}
    
    stats = {
        "total": len(subdirs),
        "complete": 0,
        "incomplete": 0,
        "missing_json": 0,
        "parse_error": 0,
        "details": []
    }
    
    console.rule(f"[bold cyan]Checking JSON Completeness: {json_pattern}[/bold cyan]")
    console.log(f"Output directory: [green]{output_dir}[/green]")
    console.log(f"Total subdirectories: {len(subdirs)}\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Scanning...", total=len(subdirs))
        
        for subdir in sorted(subdirs):
            video_name = subdir.name
            json_path = subdir / f"{video_name}.json" if "*" in json_pattern else subdir / json_pattern
            
            # If pattern includes wildcard, find matching file
            if "*" in json_pattern:
                matching_files = list(subdir.glob(json_pattern))
                if matching_files:
                    json_path = matching_files[0]
                else:
                    json_path = None
            
            is_complete, missing_fields = check_json_completeness(json_path) if json_path else (False, ["file_missing"])
            
            if json_path and not json_path.exists():
                stats["missing_json"] += 1
                status = "❌ MISSING"
            elif missing_fields and "parse_error" in missing_fields[0]:
                stats["parse_error"] += 1
                status = "⚠️  PARSE_ERROR"
            elif is_complete:
                stats["complete"] += 1
                status = "✅ COMPLETE"
            else:
                stats["incomplete"] += 1
                status = f"⚠️  INCOMPLETE ({len(missing_fields)} missing)"
            
            stats["details"].append({
                "name": video_name,
                "status": status,
                "missing_fields": missing_fields
            })
            
            progress.update(task, advance=1)
    
    return stats


def print_summary(stats: Dict, show_details: bool = False):
    """Print a formatted summary of the scan results."""
    if not stats:
        return
    
    console.rule("[bold green]Summary[/bold green]")
    
    # Summary table
    table = Table(title="Completeness Summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="yellow")
    
    total = stats["total"]
    table.add_row("✅ Complete", str(stats["complete"]), f"{stats['complete']/total*100:.1f}%")
    table.add_row("⚠️  Incomplete", str(stats["incomplete"]), f"{stats['incomplete']/total*100:.1f}%")
    table.add_row("❌ Missing JSON", str(stats["missing_json"]), f"{stats['missing_json']/total*100:.1f}%")
    table.add_row("⚠️  Parse Error", str(stats["parse_error"]), f"{stats['parse_error']/total*100:.1f}%")
    table.add_row("[bold]TOTAL[/bold]", str(total), "100.0%")
    
    console.print(table)
    
    # Show incomplete details if requested
    if show_details and (stats["incomplete"] > 0 or stats["missing_json"] > 0):
        console.rule("[bold yellow]Details[/bold yellow]")
        for detail in stats["details"]:
            if "INCOMPLETE" in detail["status"] or "MISSING" in detail["status"]:
                console.log(f"{detail['status']}: {detail['name']}")
                if detail["missing_fields"]:
                    console.log(f"  Missing: {', '.join(detail['missing_fields'][:3])}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check how many video subdirectories have complete JSON files"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory to scan (e.g., /path/to/train_annotation)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_audio_analysis.json",
        help="JSON file pattern to look for (default: *_audio_analysis.json)"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show details of incomplete/missing files"
    )
    
    args = parser.parse_args()
    
    stats = scan_output_directory(args.output_dir, args.pattern)
    print_summary(stats, show_details=args.details)


if __name__ == "__main__":
    main()
