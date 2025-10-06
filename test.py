# test_local_model.py
import torch
from pathlib import Path
from rich.console import Console

console = Console()

def test_local_model():
    model_path = "D:/Acads/BTP/MER-Factory/model_weights"
    
    console.print(f"[blue]Testing local model loading from: {model_path}[/blue]")
    
    try:
        # Test import
        from mer_factory.models.hf_models.qwen2_5_omni import Qwen2_5OmniModel
        
        console.print("[yellow]Creating model instance...[/yellow]")
        model = Qwen2_5OmniModel(model_path, verbose=True)
        
        console.print("[green]✅ Model loaded successfully![/green]")
        console.print(f"Device: {model.device}")
        console.print(f"Model type: {type(model.model)}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Model loading failed: {e}[/red]")
        return False

if __name__ == "__main__":
    success = test_local_model()
    if success:
        print("\n🎉 Ready to process your MELD dataset!")
    else:
        print("\n❌ Please check the error messages above.")