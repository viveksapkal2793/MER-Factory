import os
import torch
from pathlib import Path
from rich.console import Console
import tempfile
import subprocess
import json
from typing import List, Dict, Any

from .prompts import PromptTemplates

console = Console(stderr=True)


class HuggingFaceModel:
    """
    A class to encapsulate all interactions with a Hugging Face multimodal model.
    This version uses synchronous methods for all model interactions.
    """

    def __init__(self, model_id: str, verbose: bool = True):
        """
        Initializes the HuggingFaceModel by loading the model and processor.

        Args:
            model_id (str): The ID of the Hugging Face model to load.
            verbose (bool): Whether to print verbose logs.
        """
        self.model_id = model_id
        self.verbose = verbose
        self.processor = None
        self.model = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Loads the Hugging Face model and processor."""
        if self.verbose:
            console.log(f"Initializing Hugging Face pipeline for '{self.model_id}'...")
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
            )
            if self.verbose:
                console.log(
                    f"Hugging Face model '{self.model_id}' initialized successfully on device: {self.model.device}."
                )
        except ImportError:
            console.log(
                "[bold red]ERROR: 'transformers' and 'torch' are required. Please run: pip install transformers torch[/bold red]"
            )
            raise
        except Exception as e:
            console.log(
                f"[bold red]ERROR: Could not initialize Hugging Face pipeline: {e}[/bold red]"
            )
            raise

    def _validate_and_fix_inputs(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """验证并修复输入张量中的数值问题"""
        for key, tensor in inputs.items():
            if torch.isnan(tensor).any():
                console.log(f"[yellow]⚠️ 检测到 {key} 中有 NaN 值，正在修复...[/yellow]")
                tensor = torch.nan_to_num(tensor, nan=0.0)
            if torch.isinf(tensor).any():
                console.log(f"[yellow]⚠️ 检测到 {key} 中有 Inf 值，正在修复...[/yellow]")
                tensor = torch.nan_to_num(tensor, posinf=1e6, neginf=-1e6)
            inputs[key] = tensor
        return inputs

    def _run_pipeline(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        """
        Internal synchronous method to run the generation pipeline.
        """
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            input_len = inputs["input_ids"].shape[-1]
            inputs = {
                key: tensor.to(self.model.device) for key, tensor in inputs.items()
            }

            for key, tensor in inputs.items():
                if hasattr(tensor, "dtype") and tensor.dtype.is_floating_point:
                    inputs[key] = tensor.to(self.model.dtype)

            # 验证并修复输入
            inputs = self._validate_and_fix_inputs(inputs)

            # 尝试主要生成策略
            try:
                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        output_scores=False,  # 避免概率计算问题
                    )
            except RuntimeError as e:
                if "probability tensor" in str(e) or "assert" in str(e).lower():
                    console.log(
                        "[yellow]⚠️ 检测到概率张量错误，切换到保守生成策略...[/yellow]"
                    )
                    # 使用更保守的生成参数
                    with torch.inference_mode():
                        generation = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,  # 使用贪心解码
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            use_cache=False,
                            num_beams=1,
                        )
                else:
                    raise

            generation = generation[0][input_len:]
            text = self.processor.decode(generation, skip_special_tokens=True)
            return text
        except Exception as e:
            console.log(
                f"[bold red]❌ Error during Hugging Face pipeline execution: {e}[/bold red]"
            )
            return f"Error during pipeline execution: {e}"

    def describe_facial_expression(self, au_text: str) -> str:
        """Generates a description from AU text."""
        prompt = PromptTemplates.describe_facial_expression().format(au_text=au_text)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._run_pipeline(messages, 256)

    def describe_image(self, image_path: Path) -> str:
        """Generates a description for an image file."""
        prompt = PromptTemplates.describe_image()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._run_pipeline(messages, 512)

    def analyze_audio(self, audio_path: Path) -> dict:
        """Analyzes an audio file and returns a structured dictionary."""
        prompt = PromptTemplates.analyze_audio()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        str_response = self._run_pipeline(messages, 512)
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

    def describe_video(self, video_path: Path) -> str:
        """Generates a description for a video by processing its frames and audio."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            frames_dir = temp_path / "frames"
            frames_dir.mkdir()
            audio_clip_path = temp_path / "audio.wav"

            try:
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
                err_msg = f"FFmpeg failed. Ensure it's installed. Details: {e}"
                if isinstance(e, subprocess.CalledProcessError):
                    err_msg += f"\nSTDERR: {e.stderr}"
                console.log(f"[bold red]❌ {err_msg}[/bold red]")
                return f"Error: {err_msg}"

            prompt = PromptTemplates.describe_video()
            content = [{"type": "text", "text": prompt}]
            frame_files = sorted(list(frames_dir.glob("*.jpg")))[:15]
            for f in frame_files:
                content.append({"type": "image", "url": str(f)})
            if audio_clip_path.exists() and audio_clip_path.stat().st_size > 0:
                content.append({"type": "audio", "audio": str(audio_clip_path)})

            messages = [{"role": "user", "content": content}]
            return self._run_pipeline(messages, 512)

    def synthesize_summary(self, context: str) -> str:
        """Synthesizes a final summary from context."""
        prompt = PromptTemplates.synthesize_summary().format(context=context)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._run_pipeline(messages, 512)
