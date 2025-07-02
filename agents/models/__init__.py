from rich.console import Console

from .api_models.gemini import GeminiModel
from .api_models.ollama import OllamaModel
from .hf_models.gemma_multimodal import GemmaMultimodalModel


console = Console(stderr=True)


class LLMModels:
    def __init__(
        self,
        api_key: str = None,
        ollama_text_model_name: str = None,
        ollama_vision_model_name: str = None,
        huggingface_model_id: str = None,
        verbose: bool = True,
    ):
        """
        Initializes the appropriate model based on the provided arguments using a
        dictionary dispatch pattern for cleaner, more scalable code.

        Args:
            api_key (str, optional): Google API key for Gemini.
            ollama_text_model_name (str, optional): Name of the Ollama text model.
            ollama_vision_model_name (str, optional): Name of the Ollama vision model.
            huggingface_model_id (str, optional): ID of the Hugging Face model.
            verbose (bool): Whether to print verbose logs.
        """
        self.verbose = verbose
        self.model_instance = None
        self.model_type = None

        model_factory = {
            "huggingface": {
                "condition": huggingface_model_id,
                "class": GemmaMultimodalModel,
                "args": {"model_id": huggingface_model_id, "verbose": verbose},
            },
            "ollama": {
                "condition": ollama_vision_model_name or ollama_text_model_name,
                "class": OllamaModel,
                "args": {
                    "text_model_name": ollama_text_model_name,
                    "vision_model_name": ollama_vision_model_name,
                    "verbose": verbose,
                },
            },
            "gemini": {
                "condition": api_key,
                "class": GeminiModel,
                "args": {"api_key": api_key, "verbose": verbose},
            },
        }

        for model_type, config in model_factory.items():
            if config["condition"]:
                self.model_type = model_type
                try:
                    self.model_instance = config["class"](**config["args"])
                except (ValueError, ImportError) as e:
                    console.print(
                        f"[bold red]Failed to initialize {config['class'].__name__}: {e}[/bold red]"
                    )
                    raise
                break
        else:
            raise ValueError(
                "No model specified. Provide --huggingface-model, --ollama-..., or a GOOGLE_API_KEY."
            )
