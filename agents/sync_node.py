import json
from rich.console import Console
from pathlib import Path
import pandas as pd
from scipy.signal import find_peaks

from tools.ffmpeg_adapter import FFMpegAdapter

from .nodes import AU_TO_TEXT_MAP, EMOTION_TO_AU_MAP

console = Console(stderr=True)

# NOTE: These are synchronous versions of the graph nodes, intended for use
# with the Hugging Face model pipeline which runs synchronously.
# Or will interrupt the torch.compile process if run asynchronously.


def setup_paths(state):
    """Synchronous version of setup_paths."""
    video_path = Path(state["video_path"])
    video_id = state.get("video_id", video_path.stem)
    video_output_dir = state.get("video_output_dir")

    if state.get("verbose", True):
        console.log(f"Processing: {video_id}")
        console.log(f"Output directory set to: [cyan]{video_output_dir}[/cyan]")

    return {}


def run_au_extraction(state):
    """Synchronous version of run_au_extraction."""
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Action Unit (AU) Extraction (Sync)[/bold]")

    au_data_path = Path(state["au_data_path"])
    if not au_data_path.exists():
        return {"error": f"AU data file not found at {au_data_path}."}
    if verbose:
        console.log(f"Confirmed OpenFace output at [green]{au_data_path}[/green]")
    return {"au_data_path": au_data_path}


def map_au_to_text(state):
    """Synchronous version of map_au_to_text."""
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Mapping Action Units to text (Sync)...")
    au_data_path = Path(state["au_data_path"])
    try:
        df = pd.read_csv(au_data_path)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        return {"error": f"AU data file not found at {au_data_path}"}

    au_presence_cols = [
        c for c in df.columns if c.startswith("AU") and c.endswith("_c")
    ]
    if not au_presence_cols:
        return {"error": "No AU presence columns found in CSV."}

    au_frequencies = (df[au_presence_cols] > 0.5).sum().sort_values(ascending=False)
    top_intensities = [c.replace("_c", "_r") for c in au_frequencies.head(5).index]
    valid_top_intensities = [
        au for au in top_intensities if au in AU_TO_TEXT_MAP and au in df.columns
    ]

    if not valid_top_intensities:
        return {"au_text_description": "Neutral expression."}

    df["peak_score"] = df[valid_top_intensities].sum(axis=1)
    peak_frame_index = df["peak_score"].idxmax()
    peak_frame_data = df.loc[peak_frame_index]

    active_aus = {
        au: i for au, i in peak_frame_data[valid_top_intensities].items() if i > 0.8
    }
    desc = (
        ", ".join(
            [
                f"{AU_TO_TEXT_MAP.get(au, au)} (intensity: {i:.2f})"
                for au, i in active_aus.items()
            ]
        )
        or "Neutral expression."
    )
    if verbose:
        console.log(f"Detected peak expression (Sync): [yellow]{desc}[/yellow]")
    return {"au_text_description": desc}


def generate_au_description(state):
    """Generates a description from AU text using the sync HF model."""
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Generating LLM description for facial expression (Sync)...")

    hf_model = state["models"].hf_model_instance
    au_text = state["au_text_description"]

    if "Neutral expression" in au_text:
        llm_description = "A neutral facial expression was detected."
    else:
        llm_description = hf_model.describe_facial_expression(au_text)

    if verbose:
        console.log(f"LLM Description (Sync): [cyan]{llm_description}[/cyan]")
    return {"llm_au_description": llm_description}


def save_au_results(state):
    """Synchronous version of save_au_results."""
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ AU Analysis Complete (Sync)[/bold green]")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_au_analysis.json"
    )
    result_data = {
        "video_id": state["video_id"],
        "peak_au_text": state["au_text_description"],
        "llm_facial_summary": state["llm_au_description"],
    }
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=4)
    if verbose:
        console.print(f"AU analysis results saved to [green]{output_path}[/green]")
    return {}


def run_audio_extraction_and_analysis(state):
    """Analyzes an audio file using the sync HF model."""
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Audio Analysis (Sync)[/bold]")

    audio_path = Path(state["audio_path"])
    if not audio_path.exists():
        return {"error": f"Audio file not found at {audio_path}."}

    hf_model = state["models"].hf_model_instance
    if verbose:
        console.log(f"Analyzing audio with HF model at [green]{audio_path}[/green]")

    audio_analysis = hf_model.analyze_audio(audio_path)
    return {"audio_analysis_results": audio_analysis}


def save_audio_results(state):
    """Saves audio analysis results synchronously."""
    verbose = state.get("verbose", True)
    results = state["audio_analysis_results"]
    if verbose:
        console.rule("[bold green]✅ Audio Analysis Complete (Sync)[/bold green]")
        console.print(f"[bold]Transcript:[/bold] {results.get('transcript', 'N/A')}")
        console.print(
            f"[bold]Tone Description:[/bold] {results.get('tone_description', 'N/A')}"
        )
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_audio_analysis.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        console.print(f"Results saved to [cyan]{output_path}[/cyan]")
    return {}


def run_video_analysis(state):
    """Generates a description for a video file using the sync HF model."""
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Video Content Analysis (Sync)[/bold]")
    video_path = Path(state["video_path"])
    hf_model = state["models"].hf_model_instance
    video_description = hf_model.describe_video(video_path)
    if verbose:
        console.log(f"Video Description (Sync): [cyan]{video_description}[/cyan]")
    return {"video_description": video_description}


def save_video_results(state):
    """Saves video analysis results synchronously."""
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ Video Analysis Complete (Sync)[/bold green]")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_video_analysis.json"
    )
    result_data = {
        "video_id": state["video_id"],
        "llm_video_summary": state["video_description"],
    }
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=4)
    if verbose:
        console.print(f"Video analysis results saved to [green]{output_path}[/green]")
    return {}


def extract_full_features(state):
    """Synchronous version of extract_full_features."""
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Full MER Feature Extraction (Sync)[/bold]")
    audio_path = Path(state["audio_path"])
    au_data_path = Path(state["au_data_path"])
    if not audio_path.exists():
        return {"error": f"Audio file not found at {audio_path}."}
    if not au_data_path.exists():
        return {"error": f"OpenFace CSV not found at {au_data_path}."}
    if verbose:
        console.log("Confirmed existence of pre-extracted audio and AU data.")
    return {}


def filter_by_emotion(state):
    """
    Finds all significant emotional peaks in the video and, for each peak,
    analyzes the primary and secondary emotions present.
    """
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Finding all emotional peaks and analyzing mixed emotions...")
    au_data_path = Path(state["au_data_path"])
    try:
        df = pd.read_csv(au_data_path)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        return {"error": f"OpenFace output not found at {au_data_path}"}

    au_intensity_cols = [c for c in df.columns if c.endswith("_r")]
    if not au_intensity_cols:
        return {"error": "No AU intensity columns ('_r') found in the data."}

    # Calculate an overall emotional intensity score for each frame
    df["overall_intensity"] = df[au_intensity_cols].sum(axis=1)

    # Use scipy to find peaks in the overall intensity
    # These parameters are tunable to control sensitivity
    peak_indices, _ = find_peaks(
        df["overall_intensity"],
        height=state.get("threshold", 0.8),  # Min intensity to be a peak
        distance=state.get("peak_distance_frames", 20),  # Min frames between peaks
    )

    if peak_indices.size == 0:
        if verbose:
            console.log(
                "😐 No significant emotional peaks found, classifying as Neutral."
            )
        return {"is_expressive": True, "detected_emotions": ["neutral"]}

    # --- Step 2: Analyze the emotions at each peak ---
    emotion_threshold = state.get("threshold", 0.8)
    detected_emotions_summary_list = []

    for peak_idx in peak_indices:
        peak_frame = df.iloc[peak_idx]
        peak_timestamp = peak_frame["timestamp"]

        emotions_at_peak = []
        for emotion, au_list_c in EMOTION_TO_AU_MAP.items():
            available_aus_c = [au for au in au_list_c if au in peak_frame]
            if not all(au in peak_frame for au in au_list_c):
                continue
            present_aus_c = [au for au in available_aus_c if peak_frame[au] == 1]
            if len(present_aus_c) >= 0.5 * len(au_list_c):

                au_list_r = [au.replace("_c", "_r") for au in au_list_c]
                available_aus_r = [au for au in au_list_r if au in peak_frame]

                if not available_aus_r:
                    continue

                # Calculate the score for this emotion at this specific peak frame
                score = peak_frame[available_aus_r].mean()

                if score >= emotion_threshold:
                    # Classify strength for better description
                    strength = (
                        "strong"
                        if score >= 3.0 * emotion_threshold
                        else "moderate" if score >= 2 * emotion_threshold else "slight"
                    )
                    emotions_at_peak.append(
                        {"emotion": emotion, "score": score, "strength": strength}
                    )

        if not emotions_at_peak:
            continue

        # Sort emotions at this peak by score to find the primary one
        emotions_at_peak.sort(key=lambda x: x["score"], reverse=True)

        # Format the description string for this peak
        peak_desc_parts = [
            f"{e['emotion']} ({e['strength']})" for e in emotions_at_peak
        ]
        peak_summary_str = (
            f"Peak at {peak_timestamp:.2f}s: {', '.join(peak_desc_parts)}"
        )
        detected_emotions_summary_list.append(peak_summary_str)

        if verbose:
            console.log(f"  - {peak_summary_str}")

    is_expressive = bool(detected_emotions_summary_list)
    if verbose:
        if is_expressive:
            console.log(
                f"😊 Expressive. Found {len(detected_emotions_summary_list)} distinct emotional peaks."
            )
        else:
            console.log("😐 Not expressive enough for multi-peak analysis.")

    return {
        "is_expressive": is_expressive,
        "detected_emotions": detected_emotions_summary_list,
    }


def find_peak_frame(state):
    """Synchronous version of find_peak_frame."""
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Finding overall peak frame for representative image...")
    au_data_path = Path(state["au_data_path"])
    df = pd.read_csv(au_data_path)
    df.columns = df.columns.str.strip()

    au_intensity_cols = [c for c in df.columns if c.endswith("_r")]
    if not au_intensity_cols:
        return {"error": "No AU intensity columns ('_r') found in the data."}

    if "overall_intensity" not in df.columns:
        df["overall_intensity"] = df[au_intensity_cols].sum(axis=1)

    # Find the single most intense frame in the entire video
    peak_frame_idx = df["overall_intensity"].idxmax()
    peak_frame_data = df.loc[peak_frame_idx]
    peak_timestamp = peak_frame_data["timestamp"]

    video_path = Path(state["video_path"])
    peak_frame_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_peak_frame.png"
    )

    if not FFMpegAdapter.extract_nearby_frame_sync(
        video_path, peak_timestamp, peak_frame_path, verbose
    ):
        return {"error": f"Failed to extract peak frame at timestamp {peak_timestamp}."}

    top_aus_intensities = {
        au: peak_frame_data.get(au, 0)
        for au in au_intensity_cols
        if peak_frame_data.get(au, 0) > 0.8
    }

    peak_frame_info = {
        "frame_number": int(peak_frame_data["frame"]),
        "timestamp": peak_timestamp,
        "top_aus_intensities": top_aus_intensities,
    }
    if verbose:
        console.log(
            f"Identified overall peak frame at [yellow]{peak_timestamp:.2f}s[/yellow] for thumbnail."
        )
    return {"peak_frame_info": peak_frame_info, "peak_frame_path": peak_frame_path}


def generate_full_descriptions(state):
    """Generates all multimodal descriptions using the sync HF model."""
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Generating full multimodal descriptions (Sync)...")
    hf_model = state["models"].hf_model_instance
    peak_aus = state["peak_frame_info"]["top_aus_intensities"]
    active_aus = {
        au: i for au, i in peak_aus.items() if i > 0.8 and au in AU_TO_TEXT_MAP
    }
    visual_expr_desc = (
        ", ".join(
            [
                f"{AU_TO_TEXT_MAP.get(au, au)} (intensity: {i:.2f})"
                for au, i in active_aus.items()
            ]
        )
        or "Neutral expression at the overall peak frame."
    )

    visual_obj_desc = hf_model.describe_image(Path(state["peak_frame_path"]))
    audio_analysis = hf_model.analyze_audio(Path(state["audio_path"]))
    video_desc = hf_model.describe_video(Path(state["video_path"]))

    descriptions = {
        "visual_expression": visual_expr_desc,
        "visual_objective": visual_obj_desc,
        "audio_tone": audio_analysis.get("tone_description", "N/A"),
        "subtitles": audio_analysis.get("transcript", "N/A"),
        "video_content": video_desc,
    }
    return {"descriptions": descriptions}


def synthesize_summary(state):
    """Synthesizes a final summary from coarse clues using the sync HF model."""
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Synthesizing final MER summary (Sync)...")
    hf_model = state["models"].hf_model_instance
    desc = state["descriptions"]
    coarse_summary = (
        f"- Chronological Emotion Peaks: {'; '.join(state.get('detected_emotions', ['N/A']))}\n"
        f"- Facial Expression Clues (at overall peak {state['peak_frame_info']['timestamp']:.2f}s): {desc['visual_expression']}\n"
        f"- Visual Context (at overall peak): {desc['visual_objective']}\n"
        f"- Audio Tone: {desc['audio_tone']}\n"
        f"- Subtitles: {desc['subtitles']}\n"
        f"- Video Content: {desc['video_content']}"
    )
    final_summary = hf_model.synthesize_summary(coarse_summary)
    return {"final_summary": final_summary}


def save_mer_results(state):
    """Saves full MER results synchronously."""
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ Full MER Pipeline Complete (Sync)[/bold green]")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_merr_data.json"
    )
    result_data = {
        "video_id": state["video_id"],
        "source_video": str(state["video_path"]),
        "chronological_emotion_peaks": state.get("detected_emotions", []),
        "overall_peak_frame_info": state["peak_frame_info"],
        "coarse_descriptions_at_peak": state["descriptions"],
        "final_summary": state["final_summary"],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)
    if verbose:
        console.print(f"Full MER analysis saved to [green]{output_path}[/green]")
    return {}


def handle_error(state):
    """Synchronous version of handle_error."""
    error_msg = state.get("error", "An unknown error occurred.")
    video_id = state.get("video_id", "unknown_video")
    error_logs_dir = state.get("error_logs_dir", Path("./error_logs"))
    error_logs_dir.mkdir(exist_ok=True)
    error_log_path = error_logs_dir / f"{video_id}_error.log"

    console.rule(f"[bold red]❌ Error processing {video_id}[/bold red]")
    console.log(error_msg)
    console.log(f"Saving error details to [cyan]{error_log_path}[/cyan]")

    with open(error_log_path, "w") as f:
        f.write(f"Error processing video: {video_id}\n" + "=" * 20 + f"\n{error_msg}")
    return {"error": error_msg}


def run_image_analysis(state):
    """Runs the full analysis for an image using the sync HF model."""
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Image Analysis (Sync)[/bold]")

    hf_model = state["models"].hf_model_instance
    image_path = Path(state["video_path"])
    au_data_path = Path(state["au_data_path"])

    if not au_data_path.exists():
        return {"error": f"AU data file not found at {au_data_path}."}
    if verbose:
        console.log(f"Confirmed OpenFace output at [green]{au_data_path}[/green]")

    try:
        df = pd.read_csv(au_data_path)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        return {"error": f"AU data file not found at {au_data_path}"}

    if df.empty:
        return {"error": "OpenFace produced an empty CSV for the image."}

    image_frame_data = df.iloc[0]
    au_intensity_cols = [
        c for c in df.columns if c.startswith("AU") and c.endswith("_r")
    ]
    active_aus = {
        au: i
        for au, i in image_frame_data[au_intensity_cols].items()
        if i > 0.8 and au in AU_TO_TEXT_MAP
    }

    au_text_desc = (
        ", ".join(
            [
                f"{AU_TO_TEXT_MAP.get(au, au)} (intensity: {i:.2f})"
                for au, i in active_aus.items()
            ]
        )
        or "Neutral expression."
    )
    if verbose:
        console.log(f"Detected AUs: [yellow]{au_text_desc}[/yellow]")

    llm_au_description = (
        "A neutral facial expression was detected."
        if "Neutral expression" in au_text_desc
        else hf_model.describe_facial_expression(au_text_desc)
    )
    image_visual_description = hf_model.describe_image(image_path)

    if verbose:
        console.log(f"LLM AU Description (Sync): [cyan]{llm_au_description}[/cyan]")
        console.log(
            f"LLM Visual Description (Sync): [cyan]{image_visual_description}[/cyan]"
        )

    return {
        "au_text_description": au_text_desc,
        "llm_au_description": llm_au_description,
        "image_visual_description": image_visual_description,
    }


def synthesize_image_summary(state):
    """Synthesizes the final summary for an image using the sync HF model."""
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Synthesizing final image summary (Sync)...")
    hf_model = state["models"].hf_model_instance
    context = (
        f"- Facial Expression Clues: {state['llm_au_description']}\n"
        f"- Visual Context: {state['image_visual_description']}"
    )
    final_summary = hf_model.synthesize_summary(context)
    if verbose:
        console.log(f"Final Summary (Sync): [magenta]{final_summary}[/magenta]")
    return {"final_summary": final_summary}


def save_image_results(state):
    """Saves image analysis results synchronously."""
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ Image Analysis Complete (Sync)[/bold green]")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_image_analysis.json"
    )
    result_data = {
        "image_id": state["video_id"],
        "source_image": str(state["video_path"]),
        "au_text_description": state["au_text_description"],
        "llm_au_description": state["llm_au_description"],
        "image_visual_description": state["image_visual_description"],
        "final_summary": state["final_summary"],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)
    if verbose:
        console.print(f"Image analysis results saved to [green]{output_path}[/green]")
    return {}
