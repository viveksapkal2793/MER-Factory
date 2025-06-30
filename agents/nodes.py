import json
from rich.console import Console
from pathlib import Path
import pandas as pd
import asyncio
from scipy.signal import find_peaks


from tools.ffmpeg_adapter import FFMpegAdapter
from .models import LLMModels

console = Console(stderr=True)


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


async def setup_paths(state):
    video_path = Path(state["video_path"])
    video_id = state.get("video_id", video_path.stem)
    video_output_dir = state.get("video_output_dir")

    if state.get("verbose", True):
        console.log(f"Processing: {video_id}")
        console.log(f"Output directory set to: [cyan]{video_output_dir}[/cyan]")

    return {}


async def run_au_extraction(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Action Unit (AU) Extraction[/bold]")

    au_data_path = Path(state["au_data_path"])
    if not au_data_path.exists():
        return {
            "error": f"AU data file not found at {au_data_path}. The background OpenFace task may have failed."
        }
    if verbose:
        console.log(f"Confirmed OpenFace output at [green]{au_data_path}[/green]")
    return {"au_data_path": au_data_path}


async def map_au_to_text(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Mapping Action Units to text...")
    au_data_path = Path(state["au_data_path"])
    try:
        df = await asyncio.to_thread(pd.read_csv, au_data_path)
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
        return {
            "au_text_description": "No significant Action Units detected in the video."
        }

    df["peak_score"] = df[valid_top_intensities].sum(axis=1)
    peak_frame_index = df["peak_score"].idxmax()
    peak_frame_data = df.loc[peak_frame_index]

    active_aus = {
        au: i for au, i in peak_frame_data[valid_top_intensities].items() if i > 0.8
    }
    if not active_aus:
        desc = "No prominent facial action units were detected at the emotional peak."
    else:
        desc = ", ".join(
            [
                f"{AU_TO_TEXT_MAP.get(au, au)} (intensity: {i:.2f})"
                for au, i in active_aus.items()
            ]
        )
    if verbose:
        console.log(f"Detected peak expression: [yellow]{desc}[/yellow]")
    return {"au_text_description": desc}


async def generate_au_description(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Generating LLM description for facial expression...")
    models: LLMModels = state["models"]
    au_text = state["au_text_description"]

    if (
        "No prominent facial action units" in au_text
        or "No Action Units detected" in au_text
        or "No significant Action Units detected" in au_text
    ):
        llm_description = "Could not generate a description as no strong facial actions were detected."
    else:
        llm_description = await models.describe_facial_expression(au_text)
    if verbose:
        console.log(f"LLM Description: [cyan]{llm_description}[/cyan]")
    return {"llm_au_description": llm_description}


async def save_au_results(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ AU Analysis Complete[/bold green]")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_au_analysis.json"
    )
    result_data = {
        "video_id": state["video_id"],
        "peak_au_text": state["au_text_description"],
        "llm_facial_summary": state["llm_au_description"],
    }

    def _save():
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=4)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"AU analysis results saved to [green]{output_path}[/green]")
    return {}


async def run_audio_extraction_and_analysis(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Audio Analysis[/bold]")

    audio_path = Path(state["audio_path"])
    if not audio_path.exists():
        return {
            "error": f"Audio file not found at {audio_path}. The background FFMpeg task may have failed."
        }

    models: LLMModels = state["models"]
    if verbose:
        console.log(f"Analyzing pre-extracted audio at [green]{audio_path}[/green]")
    audio_analysis = await models.analyze_audio(audio_path)
    return {"audio_analysis_results": audio_analysis}


async def save_audio_results(state):
    verbose = state.get("verbose", True)
    results = state["audio_analysis_results"]
    if verbose:
        console.rule("[bold green]✅ Audio Analysis Complete[/bold green]")
        console.print(f"[bold]Transcript:[/bold] {results.get('transcript', 'N/A')}")
        console.print(
            f"[bold]Tone Description:[/bold] {results.get('tone_description', 'N/A')}"
        )
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_audio_analysis.json"
    )

    def _save():
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"Results saved to [cyan]{output_path}[/cyan]")
    return {}


async def run_video_analysis(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Video Content Analysis[/bold]")
    video_path = Path(state["video_path"])
    models: LLMModels = state["models"]
    video_description = await models.describe_video(video_path)
    if verbose:
        console.log(f"Video Description: [cyan]{video_description}[/cyan]")
    return {"video_description": video_description}


async def save_video_results(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ Video Analysis Complete[/bold green]")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_video_analysis.json"
    )
    result_data = {
        "video_id": state["video_id"],
        "llm_video_summary": state["video_description"],
    }

    def _save():
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=4)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"Video analysis results saved to [green]{output_path}[/green]")
    return {}


async def extract_full_features(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Full MER Feature Extraction[/bold]")

    # Extraction is now handled by the main script. This node just verifies the paths.
    audio_path = Path(state["audio_path"])
    au_data_path = Path(state["au_data_path"])

    if not audio_path.exists():
        return {"error": f"Audio file not found at {audio_path} for MER pipeline."}
    if not au_data_path.exists():
        return {"error": f"OpenFace CSV not found at {au_data_path} for MER pipeline."}

    if verbose:
        console.log("Confirmed existence of pre-extracted audio and AU data.")

    return {}


async def filter_by_emotion(state):
    """
    Finds all significant emotional peaks in the video and, for each peak,
    analyzes the primary and secondary emotions present.
    """
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Finding all emotional peaks and analyzing mixed emotions...")
    au_data_path = Path(state["au_data_path"])
    try:
        df = await asyncio.to_thread(pd.read_csv, au_data_path)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        return {"error": f"OpenFace output not found at {au_data_path}"}

    # --- Step 1: Find all significant emotional peaks ---
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
            console.log("😐 No significant emotional peaks found based on thresholds.")
        return {"is_expressive": False, "detected_emotions": []}

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


async def find_peak_frame(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Finding overall peak frame for representative image...")
    au_data_path = Path(state["au_data_path"])
    df = await asyncio.to_thread(pd.read_csv, au_data_path)
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
    if not await FFMpegAdapter.extract_nearby_frame(
        video_path, peak_timestamp, peak_frame_path, verbose
    ):
        return {"error": f"Failed to extract peak frame at timestamp {peak_timestamp}."}

    # We still provide the top AUs at this single peak for the detailed description
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


async def generate_full_descriptions(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Generating full multimodal descriptions...")
    models: LLMModels = state["models"]

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
        or "No strong facial clues at the overall peak frame."
    )

    # Run LLM calls concurrently
    if models.model_type == "huggingface":
        visual_obj_desc = await models.describe_image(Path(state["peak_frame_path"]))
        audio_analysis = await models.analyze_audio(Path(state["audio_path"]))
        video_desc = await models.describe_video(Path(state["video_path"]))

    else:
        visual_obj_desc_task = models.describe_image(Path(state["peak_frame_path"]))
        audio_analysis_task = models.analyze_audio(Path(state["audio_path"]))
        video_desc_task = models.describe_video(Path(state["video_path"]))
        visual_obj_desc, audio_analysis, video_desc = await asyncio.gather(
            visual_obj_desc_task, audio_analysis_task, video_desc_task
        )

    descriptions = {
        "visual_expression": visual_expr_desc,
        "visual_objective": visual_obj_desc,
        "audio_tone": audio_analysis.get("tone_description", "N/A"),
        "subtitles": audio_analysis.get("transcript", "N/A"),
        "video_content": video_desc,
    }
    return {"descriptions": descriptions}


async def synthesize_summary(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Synthesizing final MER summary...")
    models: LLMModels = state["models"]
    desc = state["descriptions"]

    coarse_summary = (
        f"- Chronological Emotion Peaks: {'; '.join(state.get('detected_emotions', ['N/A']))}\n"
        f"- Facial Expression Clues (at overall peak {state['peak_frame_info']['timestamp']:.2f}s): {desc['visual_expression']}\n"
        f"- Visual Context (at overall peak): {desc['visual_objective']}\n"
        f"- Audio Tone: {desc['audio_tone']}\n"
        f"- Subtitles: {desc['subtitles']}\n"
        f"- Video Content: {desc['video_content']}"
    )
    final_summary = await models.synthesize_summary(coarse_summary)
    return {"final_summary": final_summary}


async def save_mer_results(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ Full MER Pipeline Complete[/bold green]")
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

    def _save():
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"Full MER analysis saved to [green]{output_path}[/green]")
    return {}


async def handle_error(state):
    error_msg = state.get("error", "An unknown error occurred.")
    video_id = state.get("video_id", "unknown_video")
    error_logs_dir = state.get("error_logs_dir", Path("./error_logs"))
    await asyncio.to_thread(error_logs_dir.mkdir, exist_ok=True)
    error_log_path = error_logs_dir / f"{video_id}_error.log"

    console.rule(f"[bold red]❌ Error processing {video_id}[/bold red]")
    console.log(error_msg)
    console.log(f"Saving error details to [cyan]{error_log_path}[/cyan]")

    def _save():
        with open(error_log_path, "w") as f:
            f.write(
                f"Error processing video: {video_id}\n" + "=" * 20 + f"\n{error_msg}"
            )

    await asyncio.to_thread(_save)
    return {"error": error_msg}


async def run_image_analysis(state):
    """
    Runs the full analysis for a single image: AU extraction, AU description, and visual description.
    """
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Image Analysis[/bold]")

    models: LLMModels = state["models"]
    image_path = Path(
        state["video_path"]
    )  # for simplicity, we use video_path for image_path
    au_data_path = Path(state["au_data_path"])

    if not au_data_path.exists():
        return {
            "error": f"AU data file not found at {au_data_path}. The background OpenFace task may have failed."
        }
    if verbose:
        console.log(f"Confirmed OpenFace output at [green]{au_data_path}[/green]")

    try:
        df = await asyncio.to_thread(pd.read_csv, au_data_path)
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

    if not active_aus:
        au_text_desc = "No prominent facial action units were detected in the image."
    else:
        au_text_desc = ", ".join(
            [
                f"{AU_TO_TEXT_MAP.get(au, au)} (intensity: {i:.2f})"
                for au, i in active_aus.items()
            ]
        )
    if verbose:
        console.log(f"Detected AUs: [yellow]{au_text_desc}[/yellow]")

    if "No prominent facial action units" in au_text_desc:
        llm_au_desc_task = asyncio.completed_future(
            "Could not generate a description as no strong facial actions were detected."
        )
    else:
        llm_au_desc_task = models.describe_facial_expression(au_text_desc)

    visual_desc_task = models.describe_image(image_path)

    llm_au_description, image_visual_description = await asyncio.gather(
        llm_au_desc_task, visual_desc_task
    )

    if verbose:
        console.log(f"LLM AU Description: [cyan]{llm_au_description}[/cyan]")
        console.log(f"LLM Visual Description: [cyan]{image_visual_description}[/cyan]")

    return {
        "au_text_description": au_text_desc,
        "llm_au_description": llm_au_description,
        "image_visual_description": image_visual_description,
    }


async def synthesize_image_summary(state):
    """
    Synthesizes the final summary for the image analysis.
    """
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Synthesizing final image summary...")
    models: LLMModels = state["models"]

    context = (
        f"- Facial Expression Clues: {state['llm_au_description']}\n"
        f"- Visual Context: {state['image_visual_description']}"
    )

    final_summary = await models.synthesize_summary(context)
    if verbose:
        console.log(f"Final Summary: [magenta]{final_summary}[/magenta]")

    return {"final_summary": final_summary}


async def save_image_results(state):
    """
    Saves the image analysis results to a JSON file.
    """
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ Image Analysis Complete[/bold green]")

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

    def _save():
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"Image analysis results saved to [green]{output_path}[/green]")
    return {}
