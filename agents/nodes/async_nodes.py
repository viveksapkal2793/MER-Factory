import json
from rich.console import Console
from pathlib import Path
import asyncio


from tools.ffmpeg_adapter import FFMpegAdapter
from ..models import LLMModels
from tools.emotion_analyzer import EmotionAnalyzer
from tools.facial_analyzer import FacialAnalyzer

console = Console(stderr=True)


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


async def save_au_results(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ AU Analysis Complete[/bold green]")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_au_analysis.json"
    )
    result_data = {
        "video_id": state["video_id"],
        "chronological_emotion_peaks": state.get("detected_emotions", []),
    }

    def _save():
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=4)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"AU analysis results saved to [green]{output_path}[/green]")
    return {}


async def generate_audio_description(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Audio Analysis[/bold]")

    audio_path = Path(state["audio_path"])
    if not audio_path.exists():
        return {
            "error": f"Audio file not found at {audio_path}. The background FFMpeg task may have failed."
        }

    models: LLMModels = state["models"].model_instance
    if verbose:
        console.log(f"Analyzing pre-extracted audio at [green]{audio_path}[/green]")
    audio_analysis = await models.analyze_audio(audio_path)
    return {"audio_analysis_results": audio_analysis}


async def save_audio_results(state):
    verbose = state.get("verbose", True)
    results = state["audio_analysis_results"]

    if verbose and (results.get("transcript") or results.get("tone_description")):
        console.rule("[bold green]✅ Audio Analysis Complete[/bold green]")
        console.print(f"[bold]Tone Description:[/bold] {results}")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_audio_analysis.json"
    )

    def _save():
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"Results saved to [cyan]{output_path}[/cyan]")
    return {}


async def generate_video_description(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Video Content Analysis[/bold]")
    video_path = Path(state["video_path"])
    models: LLMModels = state["models"].model_instance
    video_description = await models.describe_video(video_path)
    if verbose and video_description:
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
        # Run blocking FacialAnalyzer instantiation in a separate thread
        analyzer = await asyncio.to_thread(FacialAnalyzer, au_data_path)

        # These methods are CPU-bound and don't need `to_thread` as they operate on in-memory data
        summary_list, is_expressive = analyzer.get_chronological_emotion_summary(
            peak_height=state.get("threshold", 0.8),
            peak_distance=state.get("peak_distance_frames", 20),
            emotion_threshold=state.get("threshold", 0.8),
        )
    except (FileNotFoundError, ValueError) as e:
        return {"error": f"Failed to analyze facial data: {e}"}

    if verbose:
        if is_expressive:
            console.log(
                f"😊 Expressive. Found {len(summary_list)} distinct emotional peaks."
            )
            for peak_summary in summary_list:
                console.log(f"  - {peak_summary}")
        else:
            console.log(
                "😐 Not expressive enough for multi-peak analysis, classifying as Neutral."
            )

    return {"detected_emotions": summary_list}


async def find_peak_frame(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Finding overall peak frame for representative image...")
    au_data_path = Path(state["au_data_path"])

    try:
        # Run blocking FacialAnalyzer instantiation in a separate thread
        analyzer = await asyncio.to_thread(FacialAnalyzer, au_data_path)

        peak_frame_info = analyzer.get_overall_peak_frame_info()
    except (FileNotFoundError, ValueError) as e:
        return {"error": f"Failed to find peak frame: {e}"}

    peak_timestamp = peak_frame_info["timestamp"]
    video_path = Path(state["video_path"])
    peak_frame_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_peak_frame.png"
    )

    if not await FFMpegAdapter.extract_nearby_frame(
        video_path, peak_timestamp, peak_frame_path, verbose
    ):
        return {"error": f"Failed to extract peak frame at timestamp {peak_timestamp}."}

    if verbose:
        console.log(
            f"Identified overall peak frame at [yellow]{peak_timestamp:.2f}s[/yellow] for thumbnail."
        )
    return {"peak_frame_info": peak_frame_info, "peak_frame_path": peak_frame_path}


async def generate_peak_frame_visual_description(state):
    """Generates a visual description for the peak frame image."""
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Generating visual description for peak frame...")
    models: LLMModels = state["models"].model_instance
    peak_frame_path = Path(state["peak_frame_path"])

    visual_obj_desc = await models.describe_image(peak_frame_path)

    if verbose:
        console.log(f"Peak Frame Visual Description: [cyan]{visual_obj_desc}[/cyan]")
    return {"image_visual_description": visual_obj_desc}


async def generate_peak_frame_au_description(state):
    """Generates an AU-based description for the peak frame."""
    # NOTE: Here we pass the original AU other than using LLM to convert it to text.
    # This is because the MER pipeline has other modalities can provide context,
    # we give LLM the raw AU data to avoid any potential misinterpretation.
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Generating AU description for peak frame...")
    peak_aus = state["peak_frame_info"]["top_aus_intensities"]

    visual_expr_desc = EmotionAnalyzer.extract_au_description(peak_aus)
    if "Neutral expression" in visual_expr_desc:
        visual_expr_desc = "Neutral expression at the overall peak frame."

    if verbose:
        console.log(f"Peak Frame AU Description: [yellow]{visual_expr_desc}[/yellow]")
    return {"peak_frame_au_description": visual_expr_desc}


async def synthesize_summary(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Synthesizing final MER summary...")
    models: LLMModels = state["models"].model_instance

    # Dynamically build the context based on available data
    clues = []

    # Chronological emotions
    detected_emotions = state.get("detected_emotions")
    clues.append(f"- Chronological Emotion Peaks: {'; '.join(detected_emotions)}")

    # Peak frame facial expression
    peak_frame_au_desc = state.get("peak_frame_au_description")
    timestamp = state.get("peak_frame_info", {}).get("timestamp", 0)

    clues.append(
        f"- Facial Expression Clues (at overall peak {timestamp:.2f}s): {peak_frame_au_desc}"
    )
    # Peak frame visual context
    image_visual_desc = state.get("image_visual_description")
    clues.append(f"- Visual Context (at overall peak): {image_visual_desc}")

    # Audio analysis
    audio_analysis = state.get("audio_analysis_results", "")
    if audio_analysis:
        clues.append(f"- Audio clues: {audio_analysis}")

    # Video description
    video_description = state.get("video_description", "N/A")
    if video_description and video_description.strip() and video_description != "N/A":
        clues.append(f"- Video Content Overview: {video_description}")

    coarse_summary = "\n".join(clues)

    if verbose:
        console.log("--- Sending Following Clues to LLM for Synthesis ---")
        console.log(coarse_summary)
        console.log("----------------------------------------------------")

    final_summary = await models.synthesize_summary(coarse_summary)
    return {"final_summary": final_summary}


async def save_mer_results(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ Full MER Pipeline Complete[/bold green]")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_merr_data.json"
    )

    descriptions = {
        "visual_expression": state.get("peak_frame_au_description", "N/A"),
        "visual_objective": state.get("image_visual_description", "N/A"),
        "audio_tone": state.get("audio_analysis_results", ""),
        "video_content": state.get("video_description", "N/A"),
    }

    result_data = {
        "video_id": state["video_id"],
        "source_video": str(state["video_path"]),
        "chronological_emotion_peaks": state.get("detected_emotions", []),
        "overall_peak_frame_info": state["peak_frame_info"],
        "coarse_descriptions_at_peak": descriptions,
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

    models: LLMModels = state["models"].model_instance
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
        analyzer = await asyncio.to_thread(FacialAnalyzer, au_data_path)
        au_text_desc = analyzer.get_frame_au_summary(threshold=0.8)
    except (FileNotFoundError, ValueError) as e:
        return {"error": f"Failed to analyze image facial data: {e}"}

    if verbose:
        console.log(f"Detected AUs: [yellow]{au_text_desc}[/yellow]")

    async def get_precomputed_value(value):
        return value

    if au_text_desc == "Neutral expression.":
        llm_au_desc_task = get_precomputed_value(
            "A neutral facial expression was detected."
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
    models: LLMModels = state["models"].model_instance

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
