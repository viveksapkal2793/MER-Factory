"""
Organize pre-extracted AU CSV files for MER-Factory pipeline
"""
import shutil
from pathlib import Path
from rich.console import Console
from rich.progress import track
import typer

app = typer.Typer()
console = Console()


@app.command()
def organize(
    au_csv_dir: Path = typer.Argument(..., help="Directory containing AU CSV files"),
    output_dir: Path = typer.Argument(..., help="Output directory for organized structure"),
    video_dir: Path = typer.Option(None, help="Optional: Directory containing original videos"),
    dry_run: bool = typer.Option(False, help="Show what would be done without doing it"),
):
    """
    Organize AU CSV files into the directory structure expected by MER-Factory
    
    Example:
        python organize_au_files.py /path/to/csvs /path/to/output --video-dir /path/to/videos
    """
    console.rule("[bold blue]Organizing AU CSV Files[/bold blue]")
    
    # Validate input
    if not au_csv_dir.exists():
        console.print(f"[red]Error: AU CSV directory not found: {au_csv_dir}[/red]")
        raise typer.Exit(1)
    
    # Find all CSV files
    csv_files = list(au_csv_dir.glob("*.csv"))
    
    if not csv_files:
        console.print(f"[red]Error: No CSV files found in {au_csv_dir}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Found {len(csv_files)} CSV files[/blue]")
    console.print(f"[blue]Output directory: {output_dir}[/blue]")
    if video_dir:
        console.print(f"[blue]Video directory: {video_dir}[/blue]")
    
    if dry_run:
        console.print("[yellow]DRY RUN - No files will be copied[/yellow]")
    
    # Create output directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each CSV file
    success = 0
    failed = 0
    video_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mkv', '.MKV', '.mov', '.MOV']
    
    for csv_file in track(csv_files, description="Organizing files..."):
        try:
            # Get filename without extension
            filename = csv_file.stem
            
            # Create subdirectory
            video_subdir = output_dir / filename
            
            if not dry_run:
                video_subdir.mkdir(parents=True, exist_ok=True)
                
                # Copy CSV file
                dest_csv = video_subdir / f"{filename}.csv"
                shutil.copy2(csv_file, dest_csv)
            else:
                console.print(f"Would create: {video_subdir}")
                console.print(f"  Would copy: {csv_file} -> {video_subdir}/{filename}.csv")
            
            # Try to copy corresponding video if video_dir provided
            if video_dir and video_dir.exists():
                video_copied = False
                for ext in video_extensions:
                    video_file = video_dir / f"{filename}{ext}"
                    if video_file.exists():
                        if not dry_run:
                            dest_video = video_subdir / f"{filename}{ext}"
                            shutil.copy2(video_file, dest_video)
                            console.print(f"  [green]✓ {filename} (with video)[/green]")
                        else:
                            console.print(f"  Would copy video: {video_file}")
                        video_copied = True
                        break
                
                if not video_copied and not dry_run:
                    console.print(f"  [yellow]✓ {filename} (no video found)[/yellow]")
            elif not dry_run:
                console.print(f"  [green]✓ {filename}[/green]")
            
            success += 1
            
        except Exception as e:
            console.print(f"  [red]✗ {filename}: {e}[/red]")
            failed += 1
    
    # Summary
    console.rule("[bold green]Summary[/bold green]")
    console.print(f"Total files: {len(csv_files)}")
    console.print(f"[green]✓ Success: {success}[/green]")
    if failed > 0:
        console.print(f"[red]✗ Failed: {failed}[/red]")
    
    if not dry_run:
        console.print(f"\n[blue]Output structure:[/blue]")
        console.print(f"{output_dir}/")
        console.print("├── video1/")
        console.print("│   ├── video1.csv")
        console.print("│   └── video1.mp4 (if found)")
        console.print("├── video2/")
        console.print("│   ├── video2.csv")
        console.print("│   └── video2.mp4")
        console.print("└── ...")
        
        console.print("\n[green]✓ Organization complete![/green]")
        console.print("\nNow run:")
        console.print(f"python main.py process {output_dir} {output_dir} --type mer --huggingface-model model_weights --cache")


if __name__ == "__main__":
    app()