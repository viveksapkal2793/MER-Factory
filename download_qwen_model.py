# download_qwen_model.py
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def download_qwen_model(target_dir: str = "D:/Acads/BTP/MER-Factory/model_weights"):
    """Download Qwen2.5-Omni model to specified directory"""
    
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Downloading Qwen2.5-Omni-7B to: {target_path}[/bold blue]")
    console.print("[yellow]This will download ~14GB of data. Please ensure you have sufficient disk space.[/yellow]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Downloading model files...", total=None)
            
            # Download the model
            snapshot_download(
                repo_id="Qwen/Qwen2.5-Omni-7B",
                local_dir=str(target_path),
                local_dir_use_symlinks=False,  # Create actual files, not symlinks
                resume_download=True,          # Resume if interrupted
                token=None                     # No auth token needed for public models
            )
            
            progress.update(task, description="Download complete!")
        
        console.print(f"[bold green]✅ Model successfully downloaded to: {target_path}[/bold green]")
        
        # Verify download
        model_files = list(target_path.glob("*.bin")) + list(target_path.glob("*.safetensors"))
        config_files = list(target_path.glob("config.json"))
        
        if model_files and config_files:
            console.print(f"[green]✅ Verification passed: Found {len(model_files)} model files and config[/green]")
            return str(target_path)
        else:
            console.print("[red]❌ Verification failed: Missing model files or config[/red]")
            return None
            
    except Exception as e:
        console.print(f"[bold red]❌ Download failed: {e}[/bold red]")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Qwen2.5-Omni model")
    parser.add_argument(
        "--target-dir", 
        type=str, 
        default="D:/Acads/BTP/MER-Factory/model_weights",
        help="Target directory to download the model"
    )
    
    args = parser.parse_args()
    
    result = download_qwen_model(args.target_dir)
    if result:
        print(f"\nModel downloaded successfully!")
        print(f"You can now run the main script with:")
        print(f"python main.py [args] --huggingface-model \"{result}\"")
    else:
        print("\nDownload failed. Please check the error messages above.")