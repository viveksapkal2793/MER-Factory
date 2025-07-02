from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from rich.console import Console
import base64
import mimetypes
import asyncio

from agents.prompts import PromptTemplates

console = Console(stderr=True)


class OllamaModel:
    """
    A class to handle all interactions with Ollama models.
    Supports separate text and vision models, or a single vision model
    acting as both.
    """

    def __init__(
        self,
        text_model_name: str = None,
        vision_model_name: str = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.text_model = None
        self.vision_model = None

        if vision_model_name:
            self.vision_model = ChatOllama(
                model=vision_model_name, temperature=0, num_predict=512
            )
            if self.verbose:
                console.log(f"Ollama vision model '{vision_model_name}' initialized.")

        if text_model_name:
            self.text_model = ChatOllama(
                model=text_model_name, temperature=0, num_predict=1024
            )
            if self.verbose:
                console.log(f"Ollama text model '{text_model_name}' initialized.")

        if not self.text_model and self.vision_model:
            self.text_model = self.vision_model
            if self.verbose:
                console.log("Ollama vision model will also be used for text tasks.")

        if not self.text_model and not self.vision_model:
            raise ValueError(
                "At least one Ollama model (text or vision) must be specified."
            )

    async def describe_facial_expression(self, au_text: str) -> str:
        """Generates a description from AU text using Ollama."""
        if not self.text_model:
            return "Error: Ollama text model not initialized."
        try:
            prompt = PromptTemplates.describe_facial_expression().format(
                au_text=au_text
            )
            chain = self.text_model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(
                f"[bold red]❌ Error describing facial expression: {e}[/bold red]"
            )
            return f"Error generating facial description: {e}"

    async def describe_image(self, image_path: Path) -> str:
        """Generates a description for an image file using Ollama."""
        if not self.vision_model:
            return "Error: Ollama vision model not initialized."
        try:
            image_data = await asyncio.to_thread(
                lambda: base64.b64encode(image_path.read_bytes()).decode("utf-8")
            )
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
            message = HumanMessage(
                content=[
                    {"type": "text", "text": PromptTemplates.describe_image()},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                    },
                ]
            )
            chain = self.vision_model | StrOutputParser()
            return await chain.ainvoke([message])
        except Exception as e:
            console.log(f"[bold red]❌ Error describing image: {e}[/bold red]")
            return f"Error: {e}"

    async def analyze_audio(self, audio_path: Path) -> dict:
        """Audio analysis is not supported for Ollama models."""
        if self.verbose:
            console.log(
                "[yellow]Warning: Audio analysis is not supported for Ollama models.[/yellow]"
            )
        return {"transcript": "", "tone_description": ""}

    async def describe_video(self, video_path: Path) -> str:
        """Video analysis is not supported for Ollama models."""
        if self.verbose:
            console.log(
                "[yellow]Warning: Video analysis is not supported for Ollama models.[/yellow]"
            )
        return ""

    async def synthesize_summary(self, prompt: str) -> str:
        """Synthesizes a final summary from context using Ollama."""
        if not self.text_model:
            return "Error: Ollama text model not initialized."
        try:
            chain = self.text_model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(f"[bold red]❌ Error synthesizing summary: {e}[/bold red]")
            return f"Error: {e}"
