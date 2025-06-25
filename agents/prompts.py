class PromptTemplates:
    """
    A class to store and manage all prompt templates used in the application.
    """

    @staticmethod
    def describe_facial_expression():
        """Returns the prompt for describing facial expressions from AUs."""
        return """You are an expert in the Facial Action Coding System (FACS). Based on the provided
list of detected facial Action Units (AUs) and their intensities, provide a structured
analysis of the person's facial expression.

Detected AUs:
---
{au_text}
---

Example: If the AUs are 'inner brow raised, lip corners pulled up', a good description would be 'The person appears to be smiling gently, with a hint of pleasant surprise or happiness.'
"""

    @staticmethod
    def describe_image():
        """Returns the prompt for describing an image."""
        return "Analyze this image. Describe the main subject, their apparent age and gender, clothing, background scene, and any discernible objects or gestures. Focus only on objective visual elements. DO NOT PROVIDE ANY RESPONSE OTHER THAN A RAW TEXT DESCRIPTION."

    @staticmethod
    def analyze_audio():
        """Returns the prompt for analyzing audio."""
        return """Analyze this audio file. Perform two tasks:
1. Transcribe the speech into text. If there is no speech, state "No speech detected".
2. Describe the audio characteristics. Include descriptions of the speaker's tone (e.g., cheerful, angry, calm), pitch, speed, and any background noises.

Provide the output as a single, raw JSON object string with two keys: "transcript" and "tone_description". Do not wrap it in markdown backticks or other formatting.
"""

    @staticmethod
    def describe_video():
        """Returns the prompt for describing a video."""
        return "Describe the content of this video. What is happening? Describe the scene, any people, any emotion, and their actions. DO NOT PROVIDE ANY RESPONSE OTHER THAN A RAW TEXT DESCRIPTION."

    @staticmethod
    def synthesize_summary():
        """Returns the prompt for synthesizing an emotional summary."""
        return """You are an expert psychologist specializing in multimodal emotion recognition. Your task is to synthesize a set of clues from different modalities into a coherent and insightful emotional analysis. Reason about the connections between the clues to infer the subject's emotional state, and the likely cause.

Here are the clues you've gathered:
---
{context}
---

Based on these clues, provide a single-paragraph summary of the person's emotional experience.
"""
