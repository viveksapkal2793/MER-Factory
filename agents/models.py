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
from .hf_model import HuggingFaceModel

console = Console(stderr=True)


class LLMModels:
    """
    A class to handle all interactions with different LLM APIs (Google Gemini, Ollama, Hugging Face).
    Supports separate text and vision models for Ollama, or a single Ollama vision model
    acting as both. Hugging Face models are expected to be multimodal.
    """

    def __init__(
        self,
        api_key: str = None,
        ollama_text_model_name: str = None,
        ollama_vision_model_name: str = None,
        huggingface_model_id: str = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.text_model = None
        self.vision_model = None
        self.model_type = None
        self.hf_model_instance = None

        if huggingface_model_id:
            self.model_type = "huggingface"
            try:
                self.hf_model_instance = HuggingFaceModel(
                    model_id=huggingface_model_id, verbose=verbose
                )
                self.text_model = self.hf_model_instance
                self.vision_model = self.hf_model_instance
            except (ValueError, ImportError) as e:
                console.print(
                    f"[bold red]Failed to initialize HuggingFaceModel: {e}[/bold red]"
                )
                raise

        elif ollama_vision_model_name:
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
                "No model specified. Provide --huggingface-model, --ollama-..., or a GOOGLE_API_KEY."
            )

    async def describe_facial_expression(self, au_text: str) -> str:
        if self.verbose:
            console.log("Generating facial expression description from AUs...")

        if self.model_type == "huggingface":
            return self.hf_model_instance.describe_facial_expression(au_text)

        if not self.text_model:
            return "Error: Text model not initialized."
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
        if self.verbose:
            console.log(
                f"Generating visual description for [cyan]{image_path.name}[/cyan]..."
            )

        if self.model_type == "huggingface":
            return self.hf_model_instance.describe_image(image_path)

        if not self.vision_model:
            return "Error: Vision model not initialized."
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
            console.log(f"[bold red]❌ Error: {e}[/bold red]")
            return f"Error: {e}"

    async def analyze_audio(self, audio_path: Path) -> dict:
        if self.verbose:
            console.log(f"Analyzing audio [cyan]{audio_path.name}[/cyan]...")

        if self.model_type == "huggingface":
            return self.hf_model_instance.analyze_audio(audio_path)

        if self.model_type == "ollama":
            if self.verbose:
                console.log(
                    "[yellow]Warning: Audio analysis is not supported for Ollama models.[/yellow]"
                )
            return {"transcript": "", "tone_description": ""}

        if not self.vision_model:
            return {
                "transcript": "Error: Model not initialized.",
                "tone_description": "Error: Model not initialized.",
            }

        try:
            audio_data = await asyncio.to_thread(
                lambda: base64.b64encode(audio_path.read_bytes()).decode("utf-8")
            )
            mime_type = mimetypes.guess_type(audio_path)[0] or "audio/wav"
            message = HumanMessage(
                content=[
                    {"type": "text", "text": PromptTemplates.analyze_audio()},
                    {"type": "media", "data": audio_data, "mime_type": mime_type},
                ]
            )
            chain = self.vision_model | StrOutputParser()
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
                return {
                    "transcript": "",
                    "tone_description": str_response,
                }
        except Exception as e:
            console.log(f"[bold red]❌ Error analyzing audio: {e}[/bold red]")
            return {
                "transcript": "",
                "tone_description": "",
            }

    async def describe_video(self, video_path: Path) -> str:
        if self.verbose:
            console.log(
                f"Generating description for video [cyan]{video_path.name}[/cyan]..."
            )

        if self.model_type == "huggingface":
            return self.hf_model_instance.describe_video(video_path)

        if self.model_type == "ollama":
            if self.verbose:
                console.log(
                    "[yellow]Warning: Video analysis is not supported for Ollama models.[/yellow]"
                )
            return ""

        if not self.vision_model:
            return "Error: Vision model not initialized."
        try:
            video_data = await asyncio.to_thread(
                lambda: base64.b64encode(video_path.read_bytes()).decode("utf-8")
            )
            mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
            message = HumanMessage(
                content=[
                    {"type": "text", "text": PromptTemplates.describe_video()},
                    {"type": "media", "data": video_data, "mime_type": mime_type},
                ]
            )
            chain = self.vision_model | StrOutputParser()
            return await chain.ainvoke([message])
        except Exception as e:
            console.log(f"[bold red]❌ Error describing video: {e}[/bold red]")
            return f"Error: {e}"

    async def synthesize_summary(self, context: str) -> str:
        if self.verbose:
            console.log("Generating fine-grained summary...")

        if self.model_type == "huggingface":
            return self.hf_model_instance.synthesize_summary(context)

        if not self.text_model:
            return "Error: Text model not initialized."
        try:
            prompt = PromptTemplates.synthesize_summary().format(context=context)
            chain = self.text_model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(f"[bold red]❌ Error synthesizing summary: {e}[/bold red]")
            return f"Error: {e}"
