from langgraph.graph import StateGraph, END
from rich.console import Console
from pathlib import Path
from typing import TypedDict, Dict, Any, List

from .models import LLMModels

console = Console()


class MERRState(TypedDict, total=False):
    """Represents the state of the MERR pipeline."""

    video_path: Path
    output_dir: Path
    processing_type: str
    video_id: str
    video_output_dir: Path
    models: LLMModels
    error: str
    verbose: bool
    error_logs_dir: Path
    au_data_path: Path
    au_text_description: str
    llm_au_description: str
    final_summary: str
    peak_frame_info: Dict[str, Any]
    peak_frame_path: Path
    audio_path: Path
    audio_analysis_results: str
    video_description: str
    descriptions: Dict[str, str]
    threshold: float
    detected_emotions: List
    peak_distance_frames: int
    ground_truth_label: str

    # Image-Specific State
    image_visual_description: str
    peak_frame_au_description: str


def route_by_processing_type(state: MERRState) -> str:
    """Routes flow based on the user's chosen processing type."""
    proc_type = state["processing_type"]
    if state.get("verbose", True):
        console.log(f"Routing based on processing type: [yellow]{proc_type}[/yellow]")
    if state.get("error"):
        return "handle_error"
    if proc_type == "AU":
        return "au_pipeline"
    if proc_type == "audio":
        return "audio_pipeline"
    if proc_type == "video":
        return "video_pipeline"
    if proc_type == "MER":
        return "full_pipeline"
    if proc_type == "image":
        return "image_pipeline"
    return "handle_error"


def route_after_emotion_filter(state: MERRState) -> str:
    """Routes flow after emotion filtering based on the pipeline."""
    if state.get("error"):
        return "handle_error"
    proc_type = state["processing_type"]
    if proc_type == "AU":
        return "save_au_results"
    elif proc_type == "MER":
        return "find_peak_frame"
    return "handle_error"


def route_after_audio_generation(state: MERRState) -> str:
    """Routes flow after audio description generation based on the pipeline."""
    if state.get("error"):
        return "handle_error"
    proc_type = state["processing_type"]
    if proc_type == "audio":
        return "save_audio_results"
    elif proc_type == "MER":
        return "generate_video_description"
    return "handle_error"


def route_after_video_generation(state: MERRState) -> str:
    """Routes flow after video description generation based on the pipeline."""
    if state.get("error"):
        return "handle_error"
    proc_type = state["processing_type"]
    if proc_type == "video":
        return "save_video_results"
    elif proc_type == "MER":
        return "generate_peak_frame_visual_description"
    return "handle_error"


def create_graph(use_sync_nodes: bool = False):
    """
    Creates and compiles the modular MERR construction graph.
    It can create either an asynchronous or a synchronous graph based on the flag.
    """
    if use_sync_nodes:
        console.log(
            "Creating a [bold yellow]synchronous[/bold yellow] graph for Hugging Face model."
        )
        from .nodes import sync_nodes as nodes
    else:
        console.log("Creating an [bold green]asynchronous[/bold green] graph.")
        from .nodes import async_nodes as nodes

    workflow = StateGraph(MERRState)

    # Add all nodes from the selected module
    workflow.add_node("setup_paths", nodes.setup_paths)
    workflow.add_node("handle_error", nodes.handle_error)
    # AU Pipeline
    workflow.add_node("run_au_extraction", nodes.run_au_extraction)
    workflow.add_node("save_au_results", nodes.save_au_results)
    # Audio Pipeline
    workflow.add_node("generate_audio_description", nodes.generate_audio_description)
    workflow.add_node("save_audio_results", nodes.save_audio_results)
    # Video Pipeline
    workflow.add_node("generate_video_description", nodes.generate_video_description)
    workflow.add_node("save_video_results", nodes.save_video_results)
    # MER Pipeline
    workflow.add_node("extract_full_features", nodes.extract_full_features)
    workflow.add_node("filter_by_emotion", nodes.filter_by_emotion)
    workflow.add_node("find_peak_frame", nodes.find_peak_frame)
    workflow.add_node(
        "generate_peak_frame_visual_description",
        nodes.generate_peak_frame_visual_description,
    )
    workflow.add_node(
        "generate_peak_frame_au_description",
        nodes.generate_peak_frame_au_description,
    )
    workflow.add_node("synthesize_summary", nodes.synthesize_summary)
    workflow.add_node("save_mer_results", nodes.save_mer_results)
    # Image Pipeline
    workflow.add_node("run_image_analysis", nodes.run_image_analysis)
    workflow.add_node("synthesize_image_summary", nodes.synthesize_image_summary)
    workflow.add_node("save_image_results", nodes.save_image_results)

    # --- Define Graph Structure ---
    workflow.set_entry_point("setup_paths")

    # 1. Main router from setup to pipeline entry points
    workflow.add_conditional_edges(
        "setup_paths",
        route_by_processing_type,
        {
            "au_pipeline": "run_au_extraction",
            "audio_pipeline": "generate_audio_description",
            "video_pipeline": "generate_video_description",
            "full_pipeline": "extract_full_features",
            "image_pipeline": "run_image_analysis",
            "handle_error": "handle_error",
        },
    )

    # 2. Define shared paths and routers
    # Both AU and MER pipelines run emotion filtering.
    workflow.add_edge("run_au_extraction", "filter_by_emotion")
    workflow.add_edge("extract_full_features", "filter_by_emotion")

    # After emotion filtering, route based on the original pipeline choice.
    workflow.add_conditional_edges(
        "filter_by_emotion",
        route_after_emotion_filter,
        {
            "save_au_results": "save_au_results",
            "find_peak_frame": "find_peak_frame",
            "handle_error": "handle_error",
        },
    )

    # After audio generation, route based on the original pipeline choice.
    workflow.add_conditional_edges(
        "generate_audio_description",
        route_after_audio_generation,
        {
            "save_audio_results": "save_audio_results",
            "generate_video_description": "generate_video_description",
            "handle_error": "handle_error",
        },
    )

    # After video generation, route based on the original pipeline choice.
    workflow.add_conditional_edges(
        "generate_video_description",
        route_after_video_generation,
        {
            "save_video_results": "save_video_results",
            "generate_peak_frame_visual_description": "generate_peak_frame_visual_description",
            "handle_error": "handle_error",
        },
    )

    # 3. Define MER pipeline sequence
    workflow.add_edge("find_peak_frame", "generate_audio_description")
    workflow.add_edge(
        "generate_peak_frame_visual_description", "generate_peak_frame_au_description"
    )
    workflow.add_edge("generate_peak_frame_au_description", "synthesize_summary")
    workflow.add_edge("synthesize_summary", "save_mer_results")

    # 4. Define Image pipeline sequence
    workflow.add_edge("run_image_analysis", "synthesize_image_summary")
    workflow.add_edge("synthesize_image_summary", "save_image_results")

    # 5. Define terminal nodes for all pipelines
    workflow.add_edge("save_au_results", END)
    workflow.add_edge("save_audio_results", END)
    workflow.add_edge("save_video_results", END)
    workflow.add_edge("save_mer_results", END)
    workflow.add_edge("save_image_results", END)
    workflow.add_edge("handle_error", END)

    app = workflow.compile()

    console.log("Modular graph compiled successfully.")
    # print(app.get_graph().draw_mermaid())
    return app
