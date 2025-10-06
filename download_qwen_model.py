import os
from pathlib import Path
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

console = Console()

def download_qwen_model(target_dir: str = "/scratch/data/bikash_rs/vivek/MER-Factory/model_weights"):
    """Download Qwen2.5-Omni model to specified directory"""
    
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Downloading Qwen2.5-Omni-7B to: {target_path}[/bold blue]")
    console.print("[yellow]This will download ~14GB of data. Please ensure you have sufficient disk space.[/yellow]")
    
    # Check existing files
    existing_files = list(target_path.glob("*"))
    if existing_files:
        console.print(f"[yellow]Found {len(existing_files)} existing files. Will resume download...[/yellow]")
    
    try:
        console.print("[cyan]Starting download (this may take a while)...[/cyan]")
        
        # Download the model with better error handling
        result = snapshot_download(
            repo_id="Qwen/Qwen2.5-Omni-7B",
            local_dir=str(target_path),
            force_download=False,  # Don't re-download existing files
            token=None,            # No auth token needed
            max_workers=4,         # Reduce parallel downloads
            tqdm_class=None        # Disable internal progress bar
        )
        
        console.print(f"[bold green]✅ Model successfully downloaded to: {target_path}[/bold green]")
        
        # Verify download
        model_files = list(target_path.glob("*.bin")) + list(target_path.glob("*.safetensors"))
        config_files = list(target_path.glob("config.json"))
        
        console.print(f"\n[cyan]Downloaded files:[/cyan]")
        console.print(f"  Model files (.bin/.safetensors): {len(model_files)}")
        console.print(f"  Config files: {len(config_files)}")
        console.print(f"  Total files: {len(list(target_path.glob('*')))}")
        
        if model_files and config_files:
            console.print(f"[green]✅ Verification passed![/green]")
            return str(target_path)
        else:
            console.print("[red]❌ Verification failed: Missing model files or config[/red]")
            console.print("[yellow]Try running the script again to resume download.[/yellow]")
            return None
            
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Download interrupted by user. Run again to resume.[/yellow]")
        return None
    except Exception as e:
        console.print(f"[bold red]❌ Download failed: {e}[/bold red]")
        console.print("[yellow]This is likely a network timeout. Run the script again to resume.[/yellow]")
        return None

def verify_model_files(target_dir: str):
    """Verify all required model files are present"""
    target_path = Path(target_dir)
    
    console.print(f"\n[cyan]Verifying files in: {target_path}[/cyan]")
    
    required_files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    ]
    
    missing_files = []
    for file in required_files:
        file_path = target_path / file
        if file_path.exists():
            console.print(f"  ✅ {file}")
        else:
            console.print(f"  ❌ {file} [red](MISSING)[/red]")
            missing_files.append(file)
    
    # Check for model weight files
    safetensor_files = list(target_path.glob("*.safetensors"))
    console.print(f"\n  Safetensors files: {len(safetensor_files)}")
    for sf in safetensor_files:
        console.print(f"    - {sf.name} ({sf.stat().st_size / 1e9:.2f} GB)")
    
    if missing_files:
        console.print(f"\n[red]Missing {len(missing_files)} required files. Re-run download.[/red]")
        return False
    else:
        console.print("\n[green]All required files present! ✅[/green]")
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Qwen2.5-Omni model")
    parser.add_argument(
        "--target-dir", 
        type=str, 
        default="/scratch/data/bikash_rs/vivek/MER-Factory/model_weights",
        help="Target directory to download the model"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing files without downloading"
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_model_files(args.target_dir)
    else:
        result = download_qwen_model(args.target_dir)
        if result:
            console.print(f"\n[bold green]✅ Model downloaded successfully![/bold green]")
            verify_model_files(result)
            print(f"\nYou can now run the main script with:")
            print(f"python main.py [args] --huggingface-model \"{result}\"")
        else:
            console.print("\n[yellow]Download incomplete. Run the script again to resume.[/yellow]")