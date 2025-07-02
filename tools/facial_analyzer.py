import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path
from .emotion_analyzer import EmotionAnalyzer


class FacialAnalyzer:
    """
    Analyzes a full OpenFace AU data file to find emotional peaks,
    the overall peak frame, and summaries for individual frames.
    """

    def __init__(self, au_data_path: Path):
        """
        Initializes the analyzer by loading and pre-processing the OpenFace CSV.

        Args:
            au_data_path (Path): The path to the OpenFace output CSV file.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV is empty or contains no AU intensity columns.
        """
        try:
            self.df = pd.read_csv(au_data_path)
            self.df.columns = self.df.columns.str.strip()
        except FileNotFoundError:
            raise

        if self.df.empty:
            raise ValueError("OpenFace produced an empty CSV.")

        self.au_intensity_cols = [c for c in self.df.columns if c.endswith("_r")]
        if not self.au_intensity_cols:
            raise ValueError("No AU intensity columns ('_r') found in the data.")

        self.df["overall_intensity"] = self.df[self.au_intensity_cols].sum(axis=1)

    def get_chronological_emotion_summary(
        self, peak_height=0.8, peak_distance=20, emotion_threshold=0.8
    ):
        """
        Finds all significant emotional peaks and generates a human-readable summary for each.

        Returns:
            tuple: A tuple containing:
                - list: A list of strings, each describing an emotional peak.
                - bool: A flag indicating if the video was expressive.
        """
        peak_indices, _ = find_peaks(
            self.df["overall_intensity"],
            height=peak_height,
            distance=peak_distance,
        )

        if peak_indices.size == 0:
            return ["neutral"], False

        detected_emotions_summary_list = []
        for peak_idx in peak_indices:
            peak_frame = self.df.iloc[peak_idx]
            peak_timestamp = peak_frame["timestamp"]

            emotions_at_peak = EmotionAnalyzer.analyze_emotions_at_peak(
                peak_frame, emotion_threshold
            )

            if not emotions_at_peak:
                continue

            emotions_at_peak.sort(key=lambda x: x["score"], reverse=True)
            peak_desc_parts = [
                f"{e['emotion']} ({e['strength']})" for e in emotions_at_peak
            ]
            peak_summary_str = (
                f"Peak at {peak_timestamp:.2f}s: {', '.join(peak_desc_parts)}"
            )
            detected_emotions_summary_list.append(peak_summary_str)

        is_expressive = bool(detected_emotions_summary_list)
        if not is_expressive:
            return ["neutral"], False

        return detected_emotions_summary_list, is_expressive

    def get_overall_peak_frame_info(self):
        """
        Finds the single most intense frame in the entire video.

        Returns:
            dict: A dictionary containing information about the peak frame.
        """
        peak_frame_idx = self.df["overall_intensity"].idxmax()
        peak_frame_data = self.df.loc[peak_frame_idx]

        peak_frame_info = {
            "frame_number": int(peak_frame_data["frame"]),
            "timestamp": peak_frame_data["timestamp"],
            "top_aus_intensities": EmotionAnalyzer.get_active_aus(peak_frame_data),
        }
        return peak_frame_info

    def get_frame_au_summary(self, frame_index=0, threshold=0.8):
        """
        Gets the AU summary for a specific frame (defaults to the first).
        Useful for single-image analysis.

        Returns:
            str: A human-readable description of the active AUs in the frame.
        """
        if frame_index >= len(self.df):
            raise IndexError("Frame index out of bounds.")

        frame_data = self.df.iloc[frame_index]
        active_aus = EmotionAnalyzer.get_active_aus(frame_data, threshold=threshold)
        au_text_desc = EmotionAnalyzer.extract_au_description(active_aus)
        return au_text_desc
