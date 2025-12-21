import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path
from .emotion_analyzer import EmotionAnalyzer
from rich.console import Console
console = Console(stderr=True)

class FacialAnalyzer:
    """
    Analyzes a full OpenFace AU data file to find emotional peaks,
    the overall peak frame, and summaries for individual frames.
    """

    def __init__(self, au_csv_path: Path):
        """
        Initializes the analyzer by loading and pre-processing the OpenFace CSV.

        Args:
            au_csv_path (Path): The path to the OpenFace output CSV file.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV is empty or contains no AU intensity columns.
        """
        self.au_csv_path = au_csv_path
        
        # FIXED: Load CSV with error handling for malformed lines
        try:
            self.au_df = pd.read_csv(au_csv_path)
        except pd.errors.ParserError as e:
            console.log(f"[yellow]CSV parsing error in {au_csv_path.name}, attempting to fix...[/yellow]")
            
            # Try reading with on_bad_lines parameter (pandas >= 1.3.0)
            try:
                # Skip bad lines
                self.au_df = pd.read_csv(au_csv_path, on_bad_lines='skip')
                
                # Log how many lines were skipped
                total_lines = sum(1 for _ in open(au_csv_path)) - 1  # -1 for header
                valid_lines = len(self.au_df)
                skipped_lines = total_lines - valid_lines
                
                if skipped_lines > 0:
                    console.log(f"[yellow]Skipped {skipped_lines} malformed lines in {au_csv_path.name}[/yellow]")
                
            except TypeError:
                # Fallback for older pandas versions (< 1.3.0)
                self.au_df = pd.read_csv(
                    au_csv_path, 
                    error_bad_lines=False,
                    warn_bad_lines=True
                )
            
            # Check if we have any data left
            if self.au_df.empty:
                raise ValueError(f"AU CSV file {au_csv_path.name} is empty or completely malformed after removing bad lines")
        
        # Validate that we have the expected columns
        expected_columns = 40  # OpenFace typically outputs 40 columns
        if len(self.au_df.columns) < 30:  # Minimum threshold
            console.log(f"[yellow]Warning: Expected ~{expected_columns} columns, got {len(self.au_df.columns)} in {au_csv_path.name}[/yellow]")
        
        # Store original line count for reference
        self.total_frames = len(self.au_df)

        if self.au_df.empty:
            raise ValueError("OpenFace produced an empty CSV.")

        self.au_intensity_cols = [c for c in self.au_df.columns if c.endswith("_r")]
        if not self.au_intensity_cols:
            raise ValueError("No AU intensity columns ('_r') found in the data.")

        self.au_df["overall_intensity"] = self.au_df[self.au_intensity_cols].sum(axis=1)

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
            self.au_df["overall_intensity"],
            height=peak_height,
            distance=peak_distance,
        )

        if peak_indices.size == 0:
            return ["neutral"], False

        detected_emotions_summary_list = []
        for peak_idx in peak_indices:
            peak_frame = self.au_df.iloc[peak_idx]
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
        peak_frame_idx = self.au_df["overall_intensity"].idxmax()
        peak_frame_data = self.au_df.loc[peak_frame_idx]

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
        if frame_index >= len(self.au_df):
            raise IndexError("Frame index out of bounds.")

        frame_data = self.au_df.iloc[frame_index]
        active_aus = EmotionAnalyzer.get_active_aus(frame_data, threshold=threshold)
        au_text_desc = EmotionAnalyzer.extract_au_description(active_aus)
        return au_text_desc
