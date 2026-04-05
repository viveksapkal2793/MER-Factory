"""
Script to extract peak frame AUs from all videos and organize them at dialogue level.
FIXED: Handle leading spaces in AU column names.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from typing import Dict, List, Optional
import re
import numpy as np

console = Console()


class PeakAUExtractor:
    """Extracts peak frame AUs from OpenFace CSV files."""
    
    # AU name mapping (same as emotion_analyzer.py)
    AU_NAMES = {
        "AU01": "Inner brow raiser",
        "AU02": "Outer brow raiser",
        "AU04": "Brow lowerer",
        "AU05": "Upper lid raiser",
        "AU06": "Cheek raiser",
        "AU07": "Lid tightener",
        "AU09": "Nose wrinkler",
        "AU10": "Upper lip raiser",
        "AU12": "Lip corner puller",
        "AU14": "Dimpler",
        "AU15": "Lip corner depressor",
        "AU17": "Chin raiser",
        "AU20": "Lip stretcher",
        "AU23": "Lip tightener",
        "AU25": "Lips part",
        "AU26": "Jaw drop",
        "AU28": "Lip suck",
        "AU45": "Blink",
    }
    
    def __init__(self, annotation_dir: Path, dialogue_offset: int = 0):
        """
        Initialize the extractor.
        
        Args:
            annotation_dir: Path to the annotation directory containing utterance subdirectories
            dialogue_offset: Offset to add to dialogue numbers (e.g., 1039 for dev set)
        """
        self.annotation_dir = Path(annotation_dir)
        self.dialogue_offset = dialogue_offset
        if not self.annotation_dir.exists():
            raise ValueError(f"Annotation directory does not exist: {annotation_dir}")
    
    def find_all_au_csvs(self) -> List[Path]:
        """Find all AU CSV files in the annotation directory (flat structure)."""
        csv_files = []
        
        # Scan flat directory for all .csv files
        for csv_file in self.annotation_dir.glob("*.csv"):
            if csv_file.is_file():
                csv_files.append(csv_file)
        
        return sorted(csv_files)
    
    def extract_dialogue_utterance(self, video_name: str) -> tuple:
        """Extract dialogue number and utterance number from video name."""
        match = re.match(r'dia(\d+)_utt(\d+)', video_name)
        if match:
            dialogue_num = int(match.group(1))
            utterance_num = int(match.group(2))
            return dialogue_num, utterance_num
        
        numbers = re.findall(r'\d+', video_name)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        
        raise ValueError(f"Cannot parse dialogue/utterance from: {video_name}")
    
    def normalize_intensity(self, intensity: float) -> float:
        """
        Normalize abnormally large intensity values ONLY.
        Does NOT modify normal values (0-5 range).
        """
        # Handle NaN - return 0
        if pd.isna(intensity) or np.isnan(intensity):
            return 0.0
        
        # Convert to float to ensure we're working with a number
        intensity = float(intensity)
        
        # Only normalize if > 5.0 (abnormal)
        if intensity > 5.0:
            intensity_str = f"{intensity:.2f}"
            
            if '.' in intensity_str:
                parts = intensity_str.split('.')
                if len(parts[0]) > 1:
                    # Get last digit before decimal
                    normalized = float(f"{parts[0][-1]}.{parts[1]}")
                else:
                    normalized = float(f"{parts[0]}.{parts[1]}")
            else:
                normalized = float(intensity_str[-1])
            
            return min(normalized, 5.0)
        
        # Return as-is if in normal range
        return intensity
    
    def get_peak_frame_aus(self, csv_path: Path, threshold: float = 0.8) -> Optional[str]:
        """
        Extract AUs at peak frame from the CSV file.
        Uses EXACT same logic as facial_analyzer.py
        """
        try:
            # Load CSV
            try:
                au_df = pd.read_csv(csv_path)
            except pd.errors.ParserError:
                try:
                    au_df = pd.read_csv(csv_path, on_bad_lines='skip')
                except TypeError:
                    au_df = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=False)
            
            if au_df.empty:
                return "Error: Empty CSV"
            
            # Strip whitespace from column names (THE FIX!)
            au_df.columns = au_df.columns.str.strip()
            
            # Get AU intensity columns
            au_intensity_cols = [c for c in au_df.columns if c.endswith("_r")]
            if not au_intensity_cols:
                return "Error: No AU columns"
            
            # Calculate overall intensity (same as facial_analyzer.py)
            au_df["overall_intensity"] = au_df[au_intensity_cols].sum(axis=1)
            
            # Find peak frame
            peak_frame_idx = au_df["overall_intensity"].idxmax()
            peak_frame_data = au_df.loc[peak_frame_idx]
            
            # Get active AUs at peak with threshold=0.8
            active_aus = []
            for au_code, au_name in self.AU_NAMES.items():
                col_name = f"{au_code}_r"
                if col_name in peak_frame_data.index:
                    # Get the raw value and convert to float explicitly
                    raw_intensity = peak_frame_data[col_name]
                    
                    # Convert to Python float (not numpy/pandas type)
                    if pd.isna(raw_intensity):
                        continue
                    
                    intensity = float(raw_intensity)
                    
                    # Normalize ONLY if abnormal
                    normalized_intensity = self.normalize_intensity(intensity)
                    
                    # Now check threshold
                    if normalized_intensity >= threshold:
                        active_aus.append({
                            "au_code": au_code,
                            "au_name": au_name,
                            "intensity": normalized_intensity
                        })
            
            # Sort by intensity
            active_aus.sort(key=lambda x: x["intensity"], reverse=True)
            
            # Format output
            if not active_aus:
                return "No significant facial expressions detected"
            
            au_parts = [
                f"{au['au_name']} (intensity: {au['intensity']:.2f})"
                for au in active_aus
            ]
            
            return ", ".join(au_parts)
            
        except Exception as e:
            console.log(f"[red]Error processing {csv_path.name}: {e}[/red]")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def process_all_videos(self) -> Dict[str, Dict]:
        """Process all videos and organize by dialogue."""
        csv_files = self.find_all_au_csvs()
        
        if not csv_files:
            console.log("[red]No CSV files found![/red]")
            return {}
        
        console.log(f"[green]Found {len(csv_files)} CSV files[/green]")
        
        dialogue_data = defaultdict(lambda: {"visual_expressions": []})
        utterance_tracker = defaultdict(list)
        
        # Statistics
        no_expression_count = 0
        with_expression_count = 0
        error_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing videos...", total=len(csv_files))
            
            for csv_file in csv_files:
                video_name = csv_file.stem
                
                try:
                    dialogue_num, utterance_num = self.extract_dialogue_utterance(video_name)
                    visual_expression = self.get_peak_frame_aus(csv_file, threshold=0.8)
                    
                    # Track statistics
                    if visual_expression.startswith("Error"):
                        error_count += 1
                    elif "No significant facial expressions detected" in visual_expression:
                        no_expression_count += 1
                    else:
                        with_expression_count += 1
                    
                    utterance_tracker[dialogue_num].append((utterance_num, visual_expression))
                    
                except Exception as e:
                    console.log(f"[red]Error with {video_name}: {e}[/red]")
                    error_count += 1
                
                progress.update(task, advance=1)
        
        # Log statistics
        total = no_expression_count + with_expression_count + error_count
        console.log(f"\n[cyan]Statistics:[/cyan]")
        console.log(f"  With expressions: {with_expression_count} ({with_expression_count/total*100:.1f}%)")
        console.log(f"  No expressions: {no_expression_count} ({no_expression_count/total*100:.1f}%)")
        console.log(f"  Errors: {error_count} ({error_count/total*100:.1f}%)")
        
        # Sort and apply offset
        for dialogue_num, utterances in utterance_tracker.items():
            utterances.sort(key=lambda x: x[0])
            offset_dialogue_num = str(dialogue_num + self.dialogue_offset)
            dialogue_data[offset_dialogue_num]["visual_expressions"] = [
                expr for _, expr in utterances
            ]
        
        return dict(dialogue_data)
    
    def save_to_json(self, output_file: Path, data: Dict):
        """Save dialogue data to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        console.log(f"[green]✓ Saved to {output_file}[/green]")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract peak frame AUs from videos and organize by dialogue"
    )
    parser.add_argument(
        "annotation_dir",
        type=Path,
        help="Path to annotation directory (e.g., train_annotation)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("dialogue_visual_expressions.json"),
        help="Output JSON file path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (train/dev/test)"
    )
    parser.add_argument(
        "--dialogue-offset",
        type=int,
        default=0,
        help="Offset to add to dialogue numbers"
    )
    
    args = parser.parse_args()
    
    console.rule("[bold cyan]Peak AU Extraction[/bold cyan]")
    console.log(f"Annotation directory: {args.annotation_dir}")
    console.log(f"Dialogue offset: {args.dialogue_offset}")
    
    extractor = PeakAUExtractor(args.annotation_dir, dialogue_offset=args.dialogue_offset)
    
    console.log("\n[yellow]Processing videos...[/yellow]")
    dialogue_data = extractor.process_all_videos()
    
    if args.output == Path("dialogue_visual_expressions.json"):
        output_file = args.annotation_dir.parent / f"{args.split}_dialogue_visual_expressions.json"
    else:
        output_file = args.output
    
    console.log("\n[yellow]Saving results...[/yellow]")
    extractor.save_to_json(output_file, dialogue_data)
    
    console.rule("[bold green]Summary[/bold green]")
    console.log(f"Total dialogues: {len(dialogue_data)}")
    total_utterances = sum(len(d["visual_expressions"]) for d in dialogue_data.values())
    console.log(f"Total utterances: {total_utterances}")
    if dialogue_data:
        console.log(f"Dialogue range: {min(map(int, dialogue_data.keys()))} - {max(map(int, dialogue_data.keys()))}")
    console.log(f"Output file: {output_file}")


if __name__ == "__main__":
    main()