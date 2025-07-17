import json
from pathlib import Path
from typing import Dict


class PromptTemplates:
    """
    A class to load and manage prompt templates from a JSON file.
    """

    def __init__(self, prompts_file: Path):
        """Initializes the PromptTemplates by loading prompts from a JSON file."""
        if not prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found at: {prompts_file}")
        with open(prompts_file, "r", encoding="utf-8") as f:
            self._prompts: Dict = json.load(f)

    def get_facial_expression_prompt(self) -> str:
        """Returns the prompt for describing facial expressions from AUs."""
        return self._prompts["facial_expression_description"]

    def get_image_prompt(self, has_label: bool = False) -> str:
        """Returns the prompt for describing an image."""
        key = "with_label" if has_label else "without_label"
        return self._prompts["image_description"][key]

    def get_audio_prompt(self, has_label: bool = False) -> str:
        """Returns the prompt for analyzing audio."""
        key = "with_label" if has_label else "without_label"
        return self._prompts["audio_analysis"][key]

    def get_video_prompt(self, has_label: bool = False) -> str:
        """Returns the prompt for describing a video."""
        key = "with_label" if has_label else "without_label"
        return self._prompts["video_description"][key]

    def get_synthesis_prompt(self, task: str, has_label: bool = False) -> str:
        """Returns the synthesis prompt for a given task (e.g., Emotion Recognition)."""
        key = "with_label" if has_label else "without_label"
        return self._prompts["synthesis"][task][key]

    def get_image_synthesis_prompt(self, task: str, has_label: bool = False) -> str:
        """Returns the synthesis prompt specifically for image analysis."""
        key = "with_label" if has_label else "without_label"
        return self._prompts["image_synthesis"][task][key]
