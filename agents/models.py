from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from pathlib import Path
from rich.console import Console
import json
import base64
import mimetypes
import asyncio

from .prompts import PromptTemplates

console = Console(stderr=True)


class LLMModels:  # Renamed for broader applicability
    """
    A class to handle all interactions with different LLM APIs (Google Gemini, Ollama).
    Supports separate text and vision models for Ollama, or a single Ollama vision model
    acting as both.
    """

    def __init__(
        self,
        api_key: str = None,
        ollama_text_model_name: str = None,
        ollama_vision_model_name: str = None,
        verbose: bool = True,
    ):
        """
        Initializes the LLM models with the provided API key or Ollama model names.
        If Ollama model names are provided, they take precedence over the API key.
        If only ollama_vision_model_name is provided, it will be used for text tasks as well.
        """
        self.verbose = verbose
        self.text_model = None
        self.vision_model = None
        self.model_type = None

        if ollama_vision_model_name:
            self.vision_model = ChatOllama(
                model=ollama_vision_model_name, temperature=0, num_predict=512
            )
            self.model_type = "ollama"
            console.log(
                f"Ollama vision model '{ollama_vision_model_name}' initialized."
            )

            if ollama_text_model_name:
                self.text_model = ChatOllama(
                    model=ollama_text_model_name, temperature=0, num_predict=512
                )
                console.log(
                    f"Ollama text model '{ollama_text_model_name}' initialized."
                )
            else:
                # If only vision model is provided, use it for text as well
                self.text_model = self.vision_model
                console.log("Ollama vision model will also be used for text tasks.")
        elif ollama_text_model_name:
            # If only text model is provided, use it for text
            self.text_model = ChatOllama(model=ollama_text_model_name, temperature=0)
            self.model_type = "ollama"
            console.log(f"Ollama text model '{ollama_text_model_name}' initialized.")
        elif api_key:
            # Initialize ChatGoogleGenerativeAI if an API key is provided
            self.text_model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite", google_api_key=api_key, temperature=0
            )
            self.vision_model = self.text_model
            self.model_type = "gemini"
            console.log("Gemini models initialized.")
        else:
            raise ValueError(
                "Either 'api_key' or at least one 'ollama_model_name' (text or vision) must be provided."
            )

    async def describe_facial_expression(self, au_text: str) -> str:
        """Generates a natural language description from AU text."""
        if self.verbose:
            console.log("Generating facial expression description from AUs...")
        if not self.text_model:
            return (
                "Error: Text model not initialized for facial expression description."
            )
        try:
            prompt = PromptTemplates.describe_facial_expression().format(
                au_text=au_text
            )
            chain = self.text_model | StrOutputParser()  # Use text_model here
            response = await chain.ainvoke(prompt)
            return response
        except Exception as e:
            console.log(
                f"[bold red]❌ Error describing facial expression: {e}[/bold red]"
            )
            return f"Error generating facial description: {e}"

    async def describe_image(self, image_path: Path) -> str:
        """Generates a description for an image file using async I/O."""
        if self.verbose:
            console.log(
                f"Generating visual description for [cyan]{image_path.name}[/cyan]..."
            )
        if not self.vision_model:
            return "Error: Vision model not initialized for image description."
        try:

            def _read_and_encode():
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")

            image_data = await asyncio.to_thread(_read_and_encode)
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"

            if self.model_type == "ollama":
                message_content = [
                    {"type": "text", "text": PromptTemplates.describe_image()},
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{image_data}",
                    },
                ]
            elif self.model_type == "gemini":
                message_content = [
                    {"type": "text", "text": PromptTemplates.describe_image()},
                    {"type": "media", "data": image_data, "mime_type": mime_type},
                ]

            message = HumanMessage(content=message_content)
            chain = self.vision_model | StrOutputParser()  # Use vision_model here
            response = await chain.ainvoke([message])
            return response
        except Exception as e:
            console.log(
                f"[bold red]❌ Error describing image {image_path}: {e}[/bold red]"
            )
            return f"Error describing image: {e}"

    async def analyze_audio(self, audio_path: Path) -> dict:
        """Transcribes audio and describes its tone using async I/O."""
        if self.model_type == "ollama":
            return {
                "transcript": "",
                "tone_description": "",
            }

        if self.verbose:
            console.log(
                f"Generating audio tone description for [cyan]{audio_path.name}[/cyan]..."
            )
        if (
            not self.vision_model
        ):  # Gemini models handle audio via vision model in this setup
            return {
                "transcript": "Error: Vision model not initialized for audio analysis.",
                "tone_description": "Error: Vision model not initialized for audio analysis.",
            }
        try:

            def _read_and_encode():
                with open(audio_path, "rb") as audio_file:
                    return base64.b64encode(audio_file.read()).decode("utf-8")

            audio_data = await asyncio.to_thread(_read_and_encode)
            mime_type = mimetypes.guess_type(audio_path)[0] or "audio/wav"

            prompt = PromptTemplates.analyze_audio()

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "media",
                        "data": audio_data,
                        "mime_type": mime_type,
                    },
                ]
            )
            chain = self.vision_model | StrOutputParser()  # Use vision_model here
            str_response = await chain.ainvoke([message])
            try:
                cleaned_response = (
                    str_response.replace("```json", "").replace("```", "").strip()
                )
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                console.log(
                    f"[bold red]❌ Failed to parse JSON response from LLM.[/bold red]"
                )
                if self.verbose:
                    console.log(f"Raw response: {str_response}")
                return {
                    "transcript": "Error: Invalid JSON response.",
                    "tone_description": "Error: Invalid JSON response.",
                }
        except Exception as e:
            console.log(
                f"[bold red]❌ Error analyzing audio {audio_path}: {e}[/bold red]"
            )
            return {
                "transcript": "Error during transcription.",
                "tone_description": f"Error analyzing audio tone: {e}",
            }

    async def describe_video(self, video_path: Path) -> str:
        """Generates a description for a video file using async I/O."""
        if self.model_type == "ollama":
            # Video input is not supported by ChatOllama via direct media input
            return ""

        if self.verbose:
            console.log(
                f"Generating description for video [cyan]{video_path.name}[/cyan]..."
            )
        if not self.vision_model:
            return "Error: Vision model not initialized for video description."
        try:

            def _read_and_encode():
                with open(video_path, "rb") as video_file:
                    return base64.b64encode(video_file.read()).decode("utf-8")

            video_data = await asyncio.to_thread(_read_and_encode)
            mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": PromptTemplates.describe_video(),
                    },
                    {
                        "type": "media",
                        "data": video_data,
                        "mime_type": mime_type,
                    },
                ]
            )
            chain = self.vision_model | StrOutputParser()  # Use vision_model here
            response = await chain.ainvoke([message])
            return response
        except Exception as e:
            console.log(
                f"[bold red]❌ Error describing video {video_path}: {e}[/bold red]"
            )
            return f"Error describing video: {e}"

    async def synthesize_summary(self, context: str) -> str:
        """Generates a fine-grained emotional summary from coarse clues."""
        if self.verbose:
            console.log("Generating fine-grained summary...")
        if not self.text_model:
            return "Error: Text model not initialized for summary synthesis."
        try:
            prompt = PromptTemplates.synthesize_summary().format(context=context)
            chain = self.text_model | StrOutputParser()  # Use text_model here
            response = await chain.ainvoke(prompt)
            return response
        except Exception as e:
            console.log(f"[bold red]❌ Error synthesizing summary: {e}[/bold red]")
            return f"Error generating summary: {e}"
