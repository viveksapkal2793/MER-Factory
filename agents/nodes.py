import json
from rich.console import Console
from pathlib import Path
import pandas as pd
import asyncio


from tools.ffmpeg_adapter import FFMpegAdapter
from tools.openface_adapter import OpenFaceAdapter
from .models import GeminiModels

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
    output_dir = Path(state["output_dir"])
    video_id = video_path.stem
    video_output_dir = output_dir / video_id
    await asyncio.to_thread(video_output_dir.mkdir, parents=True, exist_ok=True)
    if state.get("verbose", True):
        console.log(f"Processing: {video_id}")
    return {"video_id": video_id, "video_output_dir": video_output_dir}


async def run_au_extraction(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Action Unit (AU) Extraction[/bold]")
    video_path = Path(state["video_path"])
    video_output_dir = Path(state["video_output_dir"])
    if not await OpenFaceAdapter.run_feature_extraction(
        video_path, video_output_dir, verbose
    ):
        return {"error": f"Failed to run OpenFace on {video_path.name}"}

    au_data_path = video_output_dir / f"{state['video_id']}.csv"
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
        au: i for au, i in peak_frame_data[valid_top_intensities].items() if i > 0.2
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
    models: GeminiModels = state["models"]
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
    video_path = Path(state["video_path"])
    video_output_dir = Path(state["video_output_dir"])
    models: GeminiModels = state["models"]
    audio_path = video_output_dir / f"{state['video_id']}_audio_only.wav"
    if not await FFMpegAdapter.extract_audio(video_path, audio_path, verbose):
        return {"error": f"Failed to extract audio for {video_path.name}"}
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
    models: GeminiModels = state["models"]
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
    video_path = Path(state["video_path"])
    video_output_dir = Path(state["video_output_dir"])
    audio_path = video_output_dir / f"{state['video_id']}.wav"

    audio_task = FFMpegAdapter.extract_audio(video_path, audio_path, verbose)
    openface_task = OpenFaceAdapter.run_feature_extraction(
        video_path, video_output_dir, verbose
    )

    audio_ok, openface_ok = await asyncio.gather(audio_task, openface_task)

    if not audio_ok:
        return {"error": "Failed audio extraction for MER."}
    if not openface_ok:
        return {"error": "Failed OpenFace extraction for MER."}

    return {
        "audio_path": audio_path,
        "au_data_path": video_output_dir / f"{state['video_id']}.csv",
    }


async def filter_by_emotion(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Filtering by emotion...")
    au_data_path = Path(state["au_data_path"])
    try:
        df = await asyncio.to_thread(pd.read_csv, au_data_path)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        return {"error": f"OpenFace output not found at {au_data_path}"}

    detected_emotions = []
    threshold = state.get("threshold", 0.45)
    for emotion, au_list in EMOTION_TO_AU_MAP.items():
        available_aus = [au for au in au_list if au in df.columns]
        if available_aus:
            present_au_count = sum(1 for au in available_aus if (df[au] > 0.5).any())
            if (present_au_count / len(available_aus)) >= threshold:
                detected_emotions.append(emotion)

    is_expressive = bool(detected_emotions)
    if verbose:
        if is_expressive:
            console.log(f"😊 Expressive. Emotions: {', '.join(detected_emotions)}")
        else:
            console.log("😐 Not expressive enough. Halting.")
    return {"is_expressive": is_expressive, "detected_emotions": detected_emotions}


async def find_peak_frame(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Finding peak frame...")
    au_data_path = Path(state["au_data_path"])
    df = await asyncio.to_thread(pd.read_csv, au_data_path)
    df.columns = df.columns.str.strip()

    au_presence_cols = [
        c for c in df.columns if c.startswith("AU") and c.endswith("_c")
    ]
    au_frequencies = (df[au_presence_cols] > 0.5).sum().sort_values(ascending=False)
    potential_top_intensities_r = [
        c.replace("_c", "_r") for c in au_frequencies.head(5).index
    ]
    top_intensities = [au for au in potential_top_intensities_r if au in df.columns]

    if not top_intensities:
        return {"error": "No valid AU columns for peak score."}

    df["peak_score"] = df[top_intensities].sum(axis=1)
    peak_frame_data = df.loc[df["peak_score"].idxmax()]
    peak_timestamp = peak_frame_data["timestamp"]

    video_path = Path(state["video_path"])
    peak_frame_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_peak_frame.png"
    )
    if not await FFMpegAdapter.extract_frame(
        video_path, peak_timestamp, peak_frame_path, verbose
    ):
        return {"error": "Failed to extract peak frame."}

    peak_frame_info = {
        "frame_number": int(peak_frame_data["frame"]),
        "timestamp": peak_timestamp,
        "top_aus_intensities": {
            au: peak_frame_data.get(au, 0) for au in top_intensities
        },
    }
    if verbose:
        console.log(f"Identified peak frame at [yellow]{peak_timestamp:.2f}s[/yellow].")
    return {"peak_frame_info": peak_frame_info, "peak_frame_path": peak_frame_path}


async def generate_full_descriptions(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Generating full multimodal descriptions...")
    models: GeminiModels = state["models"]

    peak_aus = state["peak_frame_info"]["top_aus_intensities"]
    active_aus = {au: i for au, i in peak_aus.items() if i > 0.2}
    visual_expr_desc = (
        ", ".join([AU_TO_TEXT_MAP.get(au, au) for au in active_aus])
        or "No strong facial clues."
    )

    # Run LLM calls concurrently
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
    models: GeminiModels = state["models"]
    desc = state["descriptions"]
    coarse_summary = (
        f"- Detected Emotion Category: {', '.join(state.get('detected_emotions', ['N/A']))}\n"
        f"- Facial Expression Clues: {desc['visual_expression']}\n"
        f"- Visual Context: {desc['visual_objective']}\n"
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
        "detected_emotions": state.get("detected_emotions", []),
        "peak_frame_info": state["peak_frame_info"],
        "coarse_descriptions": state["descriptions"],
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
