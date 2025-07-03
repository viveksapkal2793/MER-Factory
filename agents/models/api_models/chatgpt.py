from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from rich.console import Console
import base64
import mimetypes
import asyncio
import json

from agents.prompts import PromptTemplates

console = Console(stderr=True)


class ChatGptModel:
    """
    A class to handle all interactions with the OpenAI ChatGPT API using LangChain.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        vision_model_name: str = "gpt-4o",
        audio_model_name: str = "gpt-4o-audio-preview",
        verbose: bool = True,
    ):
        """
        Initializes the ChatGPT models.

        Args:
            api_key (str): The OpenAI API key.
            model_name (str, optional): The name of the text model. Defaults to "gpt-4o".
            vision_model_name (str, optional): The name of the vision model. Defaults to "gpt-4o".
            audio_model_name (str, optional): The name of the audio model. Defaults to "gpt-4o-audio-preview".
            verbose (bool, optional): Whether to print status messages.
        """
        self.verbose = verbose
        self.model = None
        self.vision_model = None
        self.audio_model = None

        if not api_key:
            raise ValueError("OpenAI API key is required for ChatGptModel.")

        # Initialize the main text model
        self.model = ChatOpenAI(
            model=model_name, api_key=api_key, temperature=0, max_tokens=1024
        )
        if self.verbose:
            console.log(f"ChatGPT text model '{model_name}' initialized.")

        # Initialize the vision model
        self.vision_model = ChatOpenAI(
            model=vision_model_name, api_key=api_key, temperature=0, max_tokens=512
        )
        if self.verbose:
            console.log(f"ChatGPT vision model '{vision_model_name}' initialized.")

        # Initialize the audio model
        self.audio_model = ChatOpenAI(
            model=audio_model_name, api_key=api_key, temperature=0, max_tokens=1024
        )
        if self.verbose:
            console.log(f"ChatGPT audio model '{audio_model_name}' initialized.")

    async def describe_facial_expression(self, au_text: str) -> str:
        """Generates a description from AU text using ChatGPT."""
        if not self.model:
            return ""
        try:
            prompt = PromptTemplates.describe_facial_expression().format(
                au_text=au_text
            )
            chain = self.model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(
                f"[bold red]❌ Error describing facial expression: {e}[/bold red]"
            )
            return ""

    async def describe_image(self, image_path: Path) -> str:
        """Generates a description for an image file using ChatGPT."""
        if not self.vision_model:
            return ""
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
            return ""

    async def analyze_audio(self, audio_path: Path) -> dict:
        """Analyzes an audio file and returns a structured dictionary using ChatGPT."""
        if not self.audio_model:
            return {
                "transcript": "",
                "tone_description": "",
            }
        try:
            audio_data = await asyncio.to_thread(
                lambda: base64.b64encode(audio_path.read_bytes()).decode("utf-8")
            )

            message = HumanMessage(
                content=[
                    {"type": "text", "text": PromptTemplates.analyze_audio()},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"{audio_data}",
                            "format": "wav",
                        },
                    },
                ]
            )
            chain = self.audio_model | StrOutputParser()
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
                return {"transcript": "", "tone_description": str_response}
        except Exception as e:
            console.log(f"[bold red]❌ Error analyzing audio: {e}[/bold red]")
            return {"transcript": "", "tone_description": ""}

    async def describe_video(self, video_path: Path) -> str:
        """Video analysis is not supported in this implementation."""
        if self.verbose:
            console.log(
                "[yellow]Warning: Video analysis is not implemented as requested.[/yellow]"
            )
        return ""

    async def synthesize_summary(self, prompt: str) -> str:
        """Synthesizes a final summary from context using ChatGPT."""
        if not self.model:
            return "Error: ChatGPT text model not initialized."
        try:
            chain = self.model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(f"[bold red]❌ Error synthesizing summary: {e}[/bold red]")
            return ""
