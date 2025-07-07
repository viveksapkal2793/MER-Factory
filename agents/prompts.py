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
    def describe_image(has_label: bool = False):
        """Returns the prompt for describing an image."""
        if has_label:
            return """The image's emotion state is labeled as "{label}".\n\nAnalyze this image. Describe the main subject, their apparent age Analyze this image. Describe the main subject, their apparent age and gender, clothing, background scene, and any discernible objects or gestures. Focus only on objective visual elements. DO NOT PROVIDE ANY RESPONSE OTHER THAN A RAW TEXT DESCRIPTION."""
        return "Analyze this image. Describe the main subject, their apparent age and gender, clothing, background scene, and any discernible objects or gestures. Focus only on objective visual elements. DO NOT PROVIDE ANY RESPONSE OTHER THAN A RAW TEXT DESCRIPTION."

    @staticmethod
    def analyze_audio(has_label: bool = False):
        """Returns the prompt for analyzing audio."""
        if has_label:
            return """The audio's emotion state is labeled as "{label}".\n\nAnalyze this audio file. Perform two tasks:
1. Transcribe the speech into text. If there is no speech, state that.
2. Describe the audio characteristics. Include descriptions of the speaker's tone (e.g., cheerful, angry, calm), pitch, speed, and any background noises.

Provide the output as raw text. Do not include any other explanatory text or formatting.
"""
        return """Analyze this audio file. Perform two tasks:
1. Transcribe the speech into text. If there is no speech, state that.
2. Describe the audio characteristics. Include descriptions of the speaker's tone (e.g., cheerful, angry, calm), pitch, speed, and any background noises.

Provide the output as raw text. Do not include any other explanatory text or formatting.
"""

    @staticmethod
    def describe_video(has_label: bool = False):
        """Returns the prompt for describing a video."""
        if has_label:
            return """The video is labeled as "{label}".\n\nDescribe the content of this video. What is happening? Describe the scene, any people, their emotions, and their actions. Focus on objective visual elements and behaviors. DO NOT PROVIDE ANY RESPONSE OTHER THAN A RAW TEXT DESCRIPTION."""
        return "Describe the content of this video. What is happening? Describe the scene, any people, any emotion, and their actions. DO NOT PROVIDE ANY RESPONSE OTHER THAN A RAW TEXT DESCRIPTION."

    @staticmethod
    def synthesize_summary(has_label: bool = False):
        """
        Returns the appropriate synthesis prompt.
        - If has_label is True, it returns a prompt to JUSTIFY the provided label.
        - If has_label is False, it returns a prompt to INFER an emotional state.
        """
        if has_label:
            return """You are an expert psychologist and behavioral analyst. Your primary mission is to build a detailed, evidence-based RATIONALE explaining why the provided "Ground Truth Label" is correct.

Here are the clues you've gathered from the recording:
---
{context}
---

**ANALYSIS STRUCTURE:**

**Part 1: Evidentiary Analysis**
Break down the evidence from each clue (facial, audio, content, etc.) and explain precisely how it supports the ground truth label.
- **Correlate specific data points to the label.** (e.g., "The 'happy' label is supported by the 'strong happy' peak, the subtitle 'I got the job!', and a high vocal pitch.")
- **Highlight consistencies or explain apparent contradictions.** (e.g., "While the tone was slightly tense, this is consistent with high-arousal joy and excitement.")

**Part 2: Conclusive Rationale**
Provide a concise, concluding paragraph that summarizes why the provided label is the most accurate description of the subject's state, based on the weight of the evidence.

**Constraint Checklist:**
- Ground your entire analysis *only* in the provided clues.
- Adhere strictly to the two-part structure.
- Provide only the raw text of your analysis. Do not include introductory phrases.
"""
        else:
            return """You are an expert psychologist and behavioral analyst. Your mission is to perform a deep analysis of the multimodal clues to INFER the subject's emotional state and journey.

Here are the clues you've gathered from the recording:
---
{context}
---

**ANALYSIS STRUCTURE:**

**Part 1: Emotional Narrative**
Analyze the "Chronological Emotion Peaks" to narrate the subject's emotional journey over time. Weave a story that explains the transitions and their context.
- **Correlate emotional shifts with other clues.**
- **Highlight consistencies or contradictions between modalities.**

**Part 2: Overall Assessment**
Provide a concluding summary that:
- **Infers the subject's most likely overall emotional state(s).**
- **Proposes the likely cause for this state based on the evidence.**

**Constraint Checklist:**
- Ground your entire analysis *only* in the provided clues.
- Adhere strictly to the two-part structure.
- Provide only the raw text of your analysis. Do not include introductory phrases.
"""
