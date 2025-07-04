class PromptTemplates:
    """
    A class to store and manage all prompt templates used in the application.
    """

    @staticmethod
    def describe_facial_expression():
        """Returns the prompt for describing facial expressions from AUs."""
        return """You are an expert in the Facial Action Coding System (FACS). Based on the provided list of detected facial Action Units (AUs) and their intensities, provide a structured analysis of the person's facial expression.

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
1. Transcribe the speech into text. If there is no speech, state that.
2. Describe the audio characteristics. Include descriptions of the speaker's tone (e.g., cheerful, angry, calm), pitch, speed, and any background noises.

Provide the output as raw text. Do not include any other explanatory text or formatting.
"""

    @staticmethod
    def describe_video():
        """Returns the prompt for describing a video."""
        return "Describe the content of this video. What is happening? Describe the scene, any people, any emotion, and their actions. DO NOT PROVIDE ANY RESPONSE OTHER THAN A RAW TEXT DESCRIPTION."

    @staticmethod
    def synthesize_summary(provider: str = None):
        """
        Returns the prompt for synthesizing an emotional summary.
        Accepts an optional 'provider' argument to tailor the prompt for specific model capabilities.
        """
        if provider == "ollama":
            # TODO: This prompt is simplified for models like Ollama that currently excel at visual analysis
            # but do not support audio or direct video content analysis. The analysis will be based
            # solely on facial expressions (AUs), chronological emotional data, and image descriptions.
            # We would update when we integrate a multimodal model that can handle audio and video with ollama.
            return """You are an expert psychologist and behavioral analyst specializing in multimodal emotion recognition. Your mission is to synthesize a comprehensive analysis from a set of visual and chronological clues.

Here are the available analytical clues:
---
{context}
---

Based exclusively on the clues provided, structure your analysis in two distinct parts:

**Part 1: Emotional Narrative**
Analyze the "Chronological Emotion Peaks" to narrate the subject's emotional journey over time. Correlate these emotional shifts with the "Facial Expression Clues" and the "Visual Context" from the peak frame. Weave a story that explains the transitions based only on the visual evidence. DO NOT MAKE ONE UP. ONLY USE THE PROVIDED CLUES

**Part 2: Overall Assessment**
After narrating the journey, provide a concluding summary. This summary must:
1.  Infer the subject's most likely overall emotional state(s) based on the visual evidence.
2.  Propose the likely cause for this emotional state, if it can be inferred from the "Visual Context" clue. If not, state that the cause cannot be determined from the visual information alone.

**Constraint Checklist:**
- Ground your entire analysis *only* in the provided clues.
- Adhere strictly to the two-part structure.
- Be detailed and specific, avoiding abstract generalizations.
- Provide only the raw text of your two-part analysis. Do not include any introductory phrases like "Here is my analysis."
"""

        # Default, full-featured prompt for multimodal models
        return """You are an expert psychologist and behavioral analyst specializing in multimodal emotion recognition. Your mission is to synthesize a comprehensive analysis from a set of multimodal clues. You will construct a narrative of the subject's emotional journey and provide a final, conclusive assessment.

Here are the clues you've gathered from the video recording:
---
{context}
---

Based exclusively on the clues provided, structure your analysis in two distinct parts:

**Part 1: Emotional Narrative**
Analyze the "Chronological Emotion Peaks" to narrate the subject's emotional journey over time. Weave a story that explains the transitions and their context. Your narrative must:
1.  **Correlate emotional shifts with other clues.** For example, "The subject began in a neutral state, but at the 3.14s mark, their expression shifted to 'strong happy' and 'moderate surprise'. This directly corresponds with the subtitle 'I got the job!' and a noticeable rise in vocal pitch noted in the 'audio_tone'". IMPORTANT: IF THERE IS NO DIRECT CORRELATION (OR NO SUBTITLE, AUDIO, etc.), DO NOT MAKE ONE UP. ONLY USE THE PROVIDED CLUES.
2.  **Highlight consistencies or contradictions between modalities.** Pay close attention to whether facial expressions, vocal tone, and spoken words align. For example, if the face is smiling (`visual_expression_at_peak`) but the voice is tense (`audio_tone`), point this out and analyze the potential meaning (e.g., masking true feelings, a polite but forced reaction).

**Part 2: Overall Assessment**
After narrating the journey, provide a concluding summary. This summary must:
1.  **Infer the subject's most likely overall emotional state(s).** Be specific (e.g., "Joyful Excitement," "Anxious Grief," "Conflicted Relief").
2.  **Propose the likely cause for this emotional state.** You must ground your inference in specific evidence from the "subtitles," "video_content," or "visual_objective_at_peak" clues. For example, "The overall state of 'Joyful Excitement' was likely caused by receiving a long-awaited job offer, as explicitly stated in the subtitles". Treat the multimodal clues as a cohesive narrative, not isolated fragments.

**Constraint Checklist:**
- Ground your entire analysis *only* in the provided clues.
- Adhere strictly to the two-part structure.
- Be detailed and specific, avoiding abstract generalizations.
- Provide only the raw text of your two-part analysis. Do not include any introductory phrases like "Here is my analysis."
"""
