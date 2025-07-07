from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from rich.console import Console
import base64
import mimetypes
import torch
from transformers import pipeline
import asyncio


console = Console(stderr=True)


class OllamaModel:
    """
    A class to handle all interactions with Ollama models and Hugging Face models.
    Supports separate text and vision models from Ollama, and audio/video
    models from Hugging Face. Ollama calls are asynchronous, Hugging Face calls are synchronous.
    """

    def __init__(
        self,
        text_model_name: str = None,
        vision_model_name: str = None,
        audio_model_name: str = "openai/whisper-base",
        verbose: bool = True,
    ):
        """
        Initializes the models.

        Args:
            text_model_name (str, optional): The name of the Ollama text model.
            vision_model_name (str, optional): The name of the Ollama vision model.
            audio_model_name (str, optional): The name of the Hugging Face audio model.
            verbose (bool, optional): Whether to print status messages.
        """
        self.verbose = verbose
        self.text_model = None
        self.vision_model = None
        self.audio_pipeline = None

        # --- Ollama Model Initialization ---
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
            # If no Ollama models are specified, we can still proceed if HF models are available.
            console.log(
                "[yellow]Warning: No Ollama text or vision model specified.[/yellow]"
            )

        # --- Hugging Face Model Initialization ---
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if self.verbose:
            console.log(f"Using device: '{device}' for Hugging Face models.")

        if audio_model_name:
            try:
                self.audio_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=audio_model_name,
                    device=device,
                    trust_remote_code=True,
                )
                if self.verbose:
                    console.log(
                        f"Hugging Face audio model '{audio_model_name}' initialized."
                    )
            except Exception as e:
                console.log(
                    f"[bold red]❌ Error initializing audio model: {e}[/bold red]"
                )

    async def describe_facial_expression(self, prompt: str) -> str:
        """Generates a description from AU text using Ollama."""
        if not self.text_model:
            return "Error: Ollama text model not initialized."
        try:
            chain = self.text_model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(
                f"[bold red]❌ Error describing facial expression: {e}[/bold red]"
            )
            return f"Error generating facial description: {e}"

    async def describe_image(self, image_path: Path, prompt: str) -> str:
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
                    {"type": "text", "text": prompt},
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
            return ""

    async def analyze_audio(self, audio_path: Path, prompt: str) -> dict:
        return self._analyze_audio(audio_path, prompt)

    def _analyze_audio(self, audio_path: Path, prompt: str) -> dict:
        """
        Analyzes an audio file to transcribe the speech using a Hugging Face model.

        Args:
            audio_path (Path): The path to the audio file.

        Returns:
            A dictionary containing the transcript.
        """
        if not self.audio_pipeline:
            return ""

        try:
            if self.verbose:
                console.log(f"Analyzing audio file: {audio_path}")
            result = self.audio_pipeline(str(audio_path))
            transcript = result.get("text", "")
            if self.verbose:
                console.log(f"Audio transcript: '{transcript}'")
            if transcript:
                return f"The audio transcript is: '{transcript}'"
            return ""
        except Exception as e:
            console.log(f"[bold red]❌ Error analyzing audio: {e}[/bold red]")
            return ""

    async def describe_video(self, video_path: Path, prompt: str) -> str:
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
            return ""
