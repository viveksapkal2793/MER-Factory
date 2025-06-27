from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

import os
import torch
from pathlib import Path
from rich.console import Console
import json
import base64
import mimetypes
import asyncio
import tempfile
import subprocess

from .prompts import PromptTemplates

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
        """
        Initializes the LLM models.
        The order of precedence is: Hugging Face > Ollama > Google Gemini.
        """
        self.verbose = verbose
        self.text_model = None
        self.vision_model = None
        self.model_type = None
        self.huggingface_pipeline = None

        if huggingface_model_id:
            self.model_type = "huggingface"
            console.log(
                f"Initializing Hugging Face pipeline for '{huggingface_model_id}'..."
            )
            try:
                from transformers import AutoProcessor, AutoModelForImageTextToText

                # set TOKENIZERS_PARALLELISM=false
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                processor = AutoProcessor.from_pretrained(huggingface_model_id)

                model = AutoModelForImageTextToText.from_pretrained(
                    huggingface_model_id,
                    load_in_4bit=True if torch.cuda.is_available() else False,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                )

                def huggingface_pipeline(messages: list, max_new_tokens: int) -> str:
                    """
                    Runs the Hugging Face pipeline with the provided messages.
                    Messages should be a list of dictionaries with 'role' and 'content'.
                    """

                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(model.device, dtype=model.dtype)
                    input_len = inputs["input_ids"].shape[-1]
                    with torch.inference_mode():
                        generation = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,  # do_sample=False
                        )
                        generation = generation[0][input_len:]

                    text = processor.decode(generation, skip_special_tokens=True)

                    return text

                self.huggingface_pipeline = huggingface_pipeline

                self.text_model = self.huggingface_pipeline
                self.vision_model = self.huggingface_pipeline
                console.log("Hugging Face pipeline initialized successfully.")
            except ImportError:
                console.log(
                    "[bold red]ERROR: 'transformers' and 'torch' are required to use Hugging Face models. Please install them (`pip install transformers torch`).[/bold red]"
                )
                raise
            except Exception as e:
                console.log(
                    f"[bold red]ERROR: Could not initialize Hugging Face pipeline: {e}[/bold red]"
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

    def _run_huggingface_pipeline(self, messages: list, max_tokens: int) -> str:
        """Helper to run the synchronous Hugging Face pipeline directly."""
        try:
            # This directly calls the synchronous pipeline function
            response = self.huggingface_pipeline(
                messages,
                max_new_tokens=max_tokens,
            )
            # The custom huggingface_pipeline function returns a string directly.
            # The original code's indexing `[0]["generated_text"]` was incorrect.
            return response
        except Exception as e:
            console.log(
                f"[bold red]❌ Error during Hugging Face pipeline execution: {e}[/bold red]"
            )
            return f"Error during pipeline execution: {e}"

    async def describe_facial_expression(self, au_text: str) -> str:
        """Generates a natural language description from AU text."""
        if self.verbose:
            console.log("Generating facial expression description from AUs...")

        if self.model_type == "huggingface":
            prompt = PromptTemplates.describe_facial_expression().format(
                au_text=au_text
            )
            content = [
                {"type": "text", "text": prompt},
            ]
            messages = [{"role": "user", "content": content}]
            return self._run_huggingface_pipeline(messages, max_tokens=256)

        if not self.text_model:
            return "Error: Text model not initialized."
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

        if self.model_type == "huggingface":
            prompt = PromptTemplates.describe_image()
            content = [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": prompt},
            ]
            messages = [
                {"role": "user", "content": content},
            ]
            return self._run_huggingface_pipeline(messages, max_tokens=512)

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
        """Transcribes audio and describes its tone."""
        if self.verbose:
            console.log(f"Analyzing audio [cyan]{audio_path.name}[/cyan]...")

        if self.model_type == "huggingface":
            prompt = PromptTemplates.analyze_audio()
            content = [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt},
            ]
            messages = [{"role": "user", "content": content}]
            str_response = self._run_huggingface_pipeline(messages, max_tokens=512)
            try:
                cleaned_response = (
                    str_response.replace("```json", "").replace("```", "").strip()
                )
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                console.log(
                    "[bold red]❌ Failed to parse JSON from Hugging Face model.[/bold red]"
                )
                return {
                    "transcript": "Error: Invalid JSON.",
                    "tone_description": str_response,
                }

        if self.model_type == "ollama":
            return {"transcript": "", "tone_description": ""}  # Not supported

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
                    "transcript": "Error: Invalid JSON.",
                    "tone_description": "Error: Invalid JSON.",
                }
        except Exception as e:
            console.log(f"[bold red]❌ Error analyzing audio: {e}[/bold red]")
            return {"transcript": "Error.", "tone_description": f"Error: {e}"}

    async def describe_video(self, video_path: Path) -> str:
        """Generates a description for a video file."""
        if self.verbose:
            console.log(
                f"Generating description for video [cyan]{video_path.name}[/cyan]..."
            )

        if self.model_type == "huggingface":
            # HuggingFace pipeline requires pre-processing videos into frames and audio.
            def _run_sync_video_processing():
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    frames_dir = temp_path / "frames"
                    frames_dir.mkdir()
                    audio_clip_path = temp_path / "audio.wav"

                    try:
                        # Using subprocess as it's a dev dependency and avoids modifying other files.
                        # Extract frames at 1 FPS.
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-i",
                                str(video_path),
                                "-vf",
                                "fps=1",
                                str(frames_dir / "%04d.jpg"),
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        # Extract audio.
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-i",
                                str(video_path),
                                "-vn",
                                "-ac",
                                "1",
                                "-ar",
                                "16000",
                                str(audio_clip_path),
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        err_msg = f"FFmpeg failed. Ensure it's installed and in PATH. Details: {e}"
                        if isinstance(e, subprocess.CalledProcessError):
                            err_msg += f"\nSTDERR: {e.stderr}"
                        console.log(f"[bold red]❌ {err_msg}[/bold red]")
                        return f"Error: {err_msg}"

                    # Prepare messages with multiple frames and one audio clip
                    prompt = PromptTemplates.describe_video()
                    content = []
                    frame_files = sorted(list(frames_dir.glob("*.jpg")))[:15]
                    for f in frame_files:
                        content.append({"type": "image", "url": str(f)})
                    if audio_clip_path.exists() and audio_clip_path.stat().st_size > 0:
                        content.append({"type": "audio", "audio": str(audio_clip_path)})
                    content.append({"type": "text", "text": prompt})
                    messages = [{"role": "user", "content": content}]
                    # Run pipeline using the helper
                    return self.huggingface_pipeline(
                        messages,
                        max_new_tokens=512,
                    )

            return await asyncio.to_thread(_run_sync_video_processing)

        if self.model_type == "ollama":
            return ""  # Not supported

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
        """Generates a fine-grained emotional summary from coarse clues."""
        if self.verbose:
            console.log("Generating fine-grained summary...")

        if self.model_type == "huggingface":
            prompt = PromptTemplates.synthesize_summary().format(context=context)
            content = [
                {"type": "text", "text": prompt},
            ]
            messages = [{"role": "user", "content": content}]
            return self._run_huggingface_pipeline(messages, max_tokens=512)

        if not self.text_model:
            return "Error: Text model not initialized."
        try:
            prompt = PromptTemplates.synthesize_summary().format(context=context)
            chain = self.text_model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(f"[bold red]❌ Error synthesizing summary: {e}[/bold red]")
            return f"Error: {e}"
