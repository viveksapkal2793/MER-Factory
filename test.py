import torch
from pathlib import Path
from rich.console import Console
from mer_factory.models.hf_models.qwen2_5_omni import Qwen2_5OmniModel

console = Console()

def test_local_model():
    # Use the correct Linux path for HPC
    model_path = "/scratch/data/bikash_rs/vivek/MER-Factory/model_weights"
    
    console.print(f"[blue]CUDA available: {torch.cuda.is_available()}[/blue]")
    
    if torch.cuda.is_available():
        console.print(f"[blue]CUDA version: {torch.version.cuda}[/blue]")
        console.print(f"[blue]GPU: {torch.cuda.get_device_name(0)}[/blue]")
    
    console.print(f"[blue]Testing local model loading from: {model_path}[/blue]")
    
    # Verify the path exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        console.print(f"[red]❌ Model path does not exist: {model_path}[/red]")
        return False
    
    # Check for required files
    required_files = ["preprocessor_config.json", "config.json", "tokenizer_config.json"]
    missing_files = []
    for file in required_files:
        if not (model_path_obj / file).exists():
            missing_files.append(file)
    
    if missing_files:
        console.print(f"[red]❌ Missing required files in model directory:[/red]")
        for file in missing_files:
            console.print(f"  - {file}")
        console.print(f"\n[yellow]Files in directory:[/yellow]")
        for file in model_path_obj.iterdir():
            console.print(f"  - {file.name}")
        return False
    
    console.print("[green]✅ All required config files found[/green]")
    
    try:        
        
        console.print("[yellow]Creating model instance...[/yellow]")
        model = Qwen2_5OmniModel(model_path, verbose=True)
        
        console.print("[green]✅ Model loaded successfully![/green]")
        console.print(f"Device: {model.device}")
        console.print(f"Model type: {type(model.model)}")
        
        # Test a simple text generation
        console.print("\n[yellow]Testing text generation...[/yellow]")
        test_prompt = "Hello, how are you?"
        response = model.describe_facial_expression(test_prompt)
        console.print(f"[green]Response: {response}[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Model loading failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_local_model()
    if success:
        console.print("\nReady to process your MELD dataset!")
    else:
        console.print("\n❌ Please check the error messages above.")