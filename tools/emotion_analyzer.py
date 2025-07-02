# https://proceedings.neurips.cc/paper_files/paper/2024/hash/c7f43ada17acc234f568dc66da527418-Abstract-Conference.html
# AU and Emotion Analysis Module, logic from Emotion-LLaMA in NeurIPS 2024
AU_TO_TEXT_MAP = {
    "AU01_r": "Inner brow raiser",
    "AU02_r": "Outer brow raiser",
    "AU04_r": "Brow lowerer",
    "AU05_r": "Upper lid raiser",
    "AU06_r": "Cheek raiser",
    "AU07_r": "Lid tightener",
    "AU09_r": "Nose wrinkler",
    "AU10_r": "Upper lip raiser",
    "AU12_r": "Lip corner puller (smile)",
    "AU14_r": "Dimpler",
    "AU15_r": "Lip corner depressor",
    "AU17_r": "Chin raiser",
    "AU20_r": "Lip stretcher",
    "AU23_r": "Lip tightener",
    "AU25_r": "Lips part",
    "AU26_r": "Jaw drop",
    "AU28_r": "Lip suck",
    "AU45_r": "Blink",
}


EMOTION_TO_AU_MAP = {
    "happy": ["AU06_c", "AU12_c", "AU14_c"],
    "sad": ["AU01_c", "AU04_c", "AU15_c", "AU14_c"],
    "surprise": ["AU01_c", "AU02_c", "AU05_c", "AU26_c"],
    "fear": ["AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU07_c", "AU20_c", "AU26_c"],
    "angry": ["AU04_c", "AU05_c", "AU07_c", "AU23_c", "AU10_c", "AU17_c"],
    "contempt": ["AU12_c", "AU10_c", "AU14_c", "AU17_c"],
    "worried": ["AU28_c", "AU20_c"],
}


class EmotionAnalyzer:
    """
    A class to centralize logic for analyzing emotions and Action Units (AUs)
    from OpenFace data.
    """

    @staticmethod
    def analyze_emotions_at_peak(peak_frame, emotion_threshold=0.8):
        """
        Analyzes the emotions present in a single data frame representing an emotional peak.

        Args:
            peak_frame (pd.Series): A row from a DataFrame corresponding to a peak frame.
            emotion_threshold (float): The minimum intensity score for an emotion to be considered.

        Returns:
            list: A list of dictionaries, each containing details of a detected emotion.
        """
        emotions_at_peak = []
        for emotion, au_list_c in EMOTION_TO_AU_MAP.items():
            available_aus_c = [au for au in au_list_c if au in peak_frame]
            if not all(au in peak_frame for au in au_list_c):
                continue

            present_aus_c = [au for au in available_aus_c if peak_frame[au] == 1]
            if len(present_aus_c) < 0.5 * len(au_list_c):
                continue

            au_list_r = [au.replace("_c", "_r") for au in au_list_c]
            available_aus_r = [au for au in au_list_r if au in peak_frame]

            if not available_aus_r:
                continue

            score = peak_frame[available_aus_r].mean()
            if score >= emotion_threshold:
                strength = (
                    "strong"
                    if score >= 3.0 * emotion_threshold
                    else "moderate" if score >= 2 * emotion_threshold else "slight"
                )
                emotions_at_peak.append(
                    {"emotion": emotion, "score": score, "strength": strength}
                )
        return emotions_at_peak

    @staticmethod
    def get_active_aus(frame_data, threshold=0.8):
        """
        Extracts a dictionary of active AUs and their intensities from a dataframe row.

        Args:
            frame_data (pd.Series): A row from a DataFrame.
            threshold (float): The minimum intensity for an AU to be considered active.

        Returns:
            dict: A dictionary of active AU codes and their intensities.
        """
        au_intensity_cols = [
            c for c in frame_data.index if c.startswith("AU") and c.endswith("_r")
        ]
        active_aus = {
            au: i
            for au, i in frame_data[au_intensity_cols].items()
            if i > threshold and au in AU_TO_TEXT_MAP
        }
        return active_aus

    @staticmethod
    def extract_au_description(active_aus):
        """
        Generates a human-readable string from a dictionary of active AUs.

        Args:
            active_aus (dict): A dictionary of active AU codes and their intensities.

        Returns:
            str: A formatted string describing the active AUs.
        """
        if not active_aus:
            return "Neutral expression."

        return ", ".join(
            [
                f"{AU_TO_TEXT_MAP.get(au, au)} (intensity: {i:.2f})"
                for au, i in active_aus.items()
            ]
        )
