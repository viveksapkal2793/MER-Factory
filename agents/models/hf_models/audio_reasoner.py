from typing import List, Dict, Any
from pathlib import Path
from rich.console import Console
from agents.prompts import PromptTemplates
import json

try:
    from swift.llm import PtEngine, InferRequest, RequestConfig
    from swift.plugin import InferStats
except ImportError as e:
    raise ImportError(
        "Failed to import Swift LLM dependencies. "
        "Please ensure you have the 'swift-llm' package installed. "
        "You can install it using: pip install ms-swift==3.0.0"
    ) from e

console = Console(stderr=True)


class AudioReasonerModel:
    """
    A wrapper for an audio reasoning model using the Swift library.
    It is designed to handle audio-based queries and provide detailed,
    structured responses.
    """

    def __init__(self, model_id: str, verbose: bool = True):
        """
        Initializes the AudioReasonerModel.

        Args:
            model_id (str): The path to the model checkpoint.
            verbose (bool): Whether to print verbose logs.
        """
        self.model_id = model_id
        self.verbose = verbose
        self.engine = None

        self.system_prompt = (
            "You are an audio deep-thinking model. Upon receiving a question, "
            "please respond in two parts: <THINK> and <RESPONSE>. The <THINK> "
            "section should be further divided into four parts: <PLANNING>, "
            "<CAPTION>, <REASONING>, and <SUMMARY>."
        )
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        if self.verbose:
            console.log(f"Initializing Swift PtEngine for '{self.model_id}'...")
        try:
            self.engine = PtEngine(
                self.model_id, max_batch_size=64, model_type="qwen2_audio"
            )
            console.log(
                f"Swift PtEngine for model '{self.model_id}' initialized successfully."
            )
        except Exception as e:
            console.log(
                f"[bold red]ERROR: Could not initialize Swift PtEngine: {e}[/bold red]"
            )
            raise

    def _run_generation(self, messages: List[Dict[str, Any]]) -> str:
        """
        Internal method to run the generation pipeline using the Swift engine.

        Args:
            messages (List[Dict[str, Any]]): The message payload for the model.

        Returns:
            str: The generated text response from the model.
        """
        try:
            infer_request = InferRequest(messages=messages)

            request_config = RequestConfig(
                max_tokens=1024, temperature=0, stream=self.verbose
            )
            metric = InferStats()
            gen = self.engine.infer([infer_request], request_config, metrics=[metric])

            output = ""
            if self.verbose:
                query_content = next(
                    (
                        part["text"]
                        for part in messages[-1]["content"]
                        if part.get("type") == "text"
                    ),
                    "",
                )
                print(f"Query: {query_content}\nResponse: ", end="")

            for resp_list in gen:
                if resp_list and resp_list[0] is not None:
                    delta = resp_list[0].choices[0].delta.content
                    if delta:
                        if self.verbose:
                            print(delta, end="", flush=True)
                        output += delta

            if self.verbose:
                console.log(f"Metric: {metric.compute()}")

            return output.strip()
        except Exception as e:
            console.log(
                f"[bold red]❌ Error during Swift engine execution: {e}[/bold red]"
            )
            return f"Error during pipeline execution: {e}"

    def _get_message(self, audiopath: Path, prompt: str) -> List[Dict[str, Any]]:
        """Constructs the message dictionary for the Swift engine."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audiopath)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    def analyze_audio(self, audio_path: Path) -> dict:
        """
        Analyzes an audio file to produce a structured response by parsing
        the model's output.

        Args:
            audio_path (Path): The path to the audio file.

        Returns:
            dict: A dictionary containing the parsed 'think' and 'response'
                    sections, along with the raw output.
        """
        if self.verbose:
            console.log(f"Analyzing audio with '{self.model_id}'...")

        prompt = PromptTemplates.analyze_audio()
        messages = self._get_message(audio_path, prompt)
        str_response = self._run_generation(messages)

        try:
            cleaned_response = (
                str_response.replace("```json", "").replace("```", "").strip()
            )
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            if self.verbose:
                console.log(
                    "[bold red]❌ Failed to parse JSON from Hugging Face model.[/bold red]"
                )
            return {
                "transcript": "",
                "tone_description": str_response,
            }

    def describe_facial_expression(self, au_text: str) -> str:
        """Not supported by this audio-focused model."""
        if self.verbose:
            console.log(
                f"[yellow]Model '{self.model_id}' does not support facial expression analysis.[/yellow]"
            )
        return ""

    def describe_image(self, image_path: Path) -> str:
        """Not supported by this audio-focused model."""
        if self.verbose:
            console.log(
                f"[yellow]Model '{self.model_id}' does not support image analysis.[/yellow]"
            )
        return ""

    def describe_video(self, video_path: Path) -> str:
        """Not supported by this audio-focused model."""
        if self.verbose:
            console.log(
                f"[yellow]Model '{self.model_id}' does not support video analysis.[/yellow]"
            )
        return ""

    def synthesize_summary(self, prompt: str) -> str:
        """Not supported, as the model is designed for audio inputs."""
        if self.verbose:
            console.log(
                f"[yellow]Model '{self.model_id}' is audio-focused and does not support text-only summary synthesis.[/yellow]"
            )
        return ""
