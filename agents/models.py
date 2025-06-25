from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from pathlib import Path
from rich.console import Console
import json
import base64
import mimetypes
import asyncio

console = Console(stderr=True)


class GeminiModels:
    """
    A class to handle all interactions with the Google Gemini API using the
    LangChain integration.
    """

    def __init__(self, api_key: str, verbose: bool = True):
        """Initializes the Gemini models with the provided API key."""
        self.verbose = verbose
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=api_key, temperature=0
        )

        console.log("Gemini models initialized.")

    async def describe_facial_expression(self, au_text: str) -> str:
        """Generates a natural language description from AU text."""
        if self.verbose:
            console.log("Generating facial expression description from AUs...")
        try:
            prompt = f"""
            Based on the following detected facial Action Units (AUs), provide a concise, natural language description of the person's facial expression.

            Detected AUs:
            ---
            {au_text}
            ---

            Example: If the AUs are 'inner brow raised, lip corners pulled up', a good description would be 'The person appears to be smiling gently, with a hint of pleasant surprise or happiness.'
            """
            response = self.model.invoke(prompt)
            return response.content
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
        try:

            def _read_and_encode():
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")

            image_data = await asyncio.to_thread(_read_and_encode)
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Analyze this image. Describe the main subject, their apparent age and gender, clothing, background scene, and any discernible objects or gestures. Focus only on objective visual elements.",
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{image_data}",
                    },
                ]
            )
            response = await self.model.ainvoke([message])
            return response.content
        except Exception as e:
            console.log(
                f"[bold red]❌ Error describing image {image_path}: {e}[/bold red]"
            )
            return f"Error describing image: {e}"

    async def analyze_audio(self, audio_path: Path) -> dict:
        """Transcribes audio and describes its tone using async I/O."""
        if self.verbose:
            console.log(
                f"Generating audio tone description for [cyan]{audio_path.name}[/cyan]..."
            )
        try:

            def _read_and_encode():
                with open(audio_path, "rb") as audio_file:
                    return base64.b64encode(audio_file.read()).decode("utf-8")

            audio_data = await asyncio.to_thread(_read_and_encode)
            mime_type = mimetypes.guess_type(audio_path)[0] or "audio/wav"

            prompt = """
            Analyze this audio file. Perform two tasks:
            1. Transcribe the speech into text. If there is no speech, state "No speech detected".
            2. Describe the audio characteristics. Include descriptions of the speaker's tone (e.g., cheerful, angry, calm), pitch, speed, and any background noises.
            
            Provide the output as a single, raw JSON object string with two keys: "transcript" and "tone_description". Do not wrap it in markdown backticks or other formatting.
            """

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
            chain = self.model | StrOutputParser()
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
        if self.verbose:
            console.log(
                f"Generating description for video [cyan]{video_path.name}[/cyan]..."
            )
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
                        "text": "Describe the content of this video. What is happening? Describe the scene, any people, any emotion, and their actions.",
                    },
                    {
                        "type": "media",
                        "data": video_data,
                        "mime_type": mime_type,
                    },
                ]
            )
            response = await self.model.ainvoke([message])
            return response.content
        except Exception as e:
            console.log(
                f"[bold red]❌ Error describing video {video_path}: {e}[/bold red]"
            )
            return f"Error describing video: {e}"

    async def synthesize_summary(self, context: str) -> str:
        """Generates a fine-grained emotional summary from coarse clues."""
        if self.verbose:
            console.log("Generating fine-grained summary...")
        try:
            prompt = f"""
            You are an expert in multimodal emotion recognition. Your task is to synthesize a set of clues from different modalities (visual, audio, text) into a coherent and insightful emotional analysis.
            Reason about the connections between the clues to infer the subject's emotional state, its intensity, and the likely cause.
            
            Here are the clues:
            ---
            {context}
            ---
            
            Based on these clues, provide a single-paragraph summary of the person's emotional experience.
            """
            response = await self.model.ainvoke(prompt)
            return response.content
        except Exception as e:
            console.log(f"[bold red]❌ Error synthesizing summary: {e}[/bold red]")
            return f"Error generating summary: {e}"
