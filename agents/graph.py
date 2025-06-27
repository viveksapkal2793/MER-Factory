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
    is_expressive: bool
    peak_frame_info: Dict[str, Any]
    peak_frame_path: Path
    audio_path: Path
    audio_analysis_results: dict
    video_description: str
    descriptions: Dict[str, str]
    threshold: float
    detected_emotions: List

    # Image-Specific State
    image_visual_description: str


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


def should_continue_full_pipeline(state: MERRState) -> str:
    """Router for the full pipeline after the emotion filter."""
    if state.get("error"):
        return "handle_error"
    if state.get("is_expressive"):
        return "continue_processing"
    return "end_processing"


def create_graph(use_sync_nodes: bool = False):
    """
    Creates and compiles the modular MERR construction graph.
    It can create either an asynchronous or a synchronous graph based on the flag.
    """
    if use_sync_nodes:
        console.log(
            "Creating a [bold yellow]synchronous[/bold yellow] graph for Hugging Face model."
        )
        from . import sync_node as nodes
    else:
        console.log("Creating an [bold green]asynchronous[/bold green] graph.")
        from . import nodes

    workflow = StateGraph(MERRState)

    # Add all nodes from the selected module
    workflow.add_node("setup_paths", nodes.setup_paths)
    workflow.add_node("handle_error", nodes.handle_error)
    # AU Pipeline
    workflow.add_node("run_au_extraction", nodes.run_au_extraction)
    workflow.add_node("map_au_to_text", nodes.map_au_to_text)
    workflow.add_node("generate_au_description", nodes.generate_au_description)
    workflow.add_node("save_au_results", nodes.save_au_results)
    # Audio Pipeline
    workflow.add_node("run_audio_analysis", nodes.run_audio_extraction_and_analysis)
    workflow.add_node("save_audio_results", nodes.save_audio_results)
    # Video Pipeline
    workflow.add_node("run_video_analysis", nodes.run_video_analysis)
    workflow.add_node("save_video_results", nodes.save_video_results)
    # MER Pipeline
    workflow.add_node("extract_full_features", nodes.extract_full_features)
    workflow.add_node("filter_by_emotion", nodes.filter_by_emotion)
    workflow.add_node("find_peak_frame", nodes.find_peak_frame)
    workflow.add_node("generate_full_descriptions", nodes.generate_full_descriptions)
    workflow.add_node("synthesize_summary", nodes.synthesize_summary)
    workflow.add_node("save_mer_results", nodes.save_mer_results)
    # Image Pipeline
    workflow.add_node("run_image_analysis", nodes.run_image_analysis)
    workflow.add_node("synthesize_image_summary", nodes.synthesize_image_summary)
    workflow.add_node("save_image_results", nodes.save_image_results)

    # --- Define Graph Structure ---
    workflow.set_entry_point("setup_paths")

    # 1. Main router
    workflow.add_conditional_edges(
        "setup_paths",
        route_by_processing_type,
        {
            "au_pipeline": "run_au_extraction",
            "audio_pipeline": "run_audio_analysis",
            "video_pipeline": "run_video_analysis",
            "full_pipeline": "extract_full_features",
            "image_pipeline": "run_image_analysis",
            "handle_error": "handle_error",
        },
    )

    # 2. AU pipeline
    workflow.add_edge("run_au_extraction", "map_au_to_text")
    workflow.add_edge("map_au_to_text", "generate_au_description")
    workflow.add_edge("generate_au_description", "save_au_results")
    workflow.add_edge("save_au_results", END)

    # 3. Audio pipeline
    workflow.add_edge("run_audio_analysis", "save_audio_results")
    workflow.add_edge("save_audio_results", END)

    # 4. Video pipeline
    workflow.add_edge("run_video_analysis", "save_video_results")
    workflow.add_edge("save_video_results", END)

    # 5. Full MER pipeline
    workflow.add_edge("extract_full_features", "filter_by_emotion")
    workflow.add_conditional_edges(
        "filter_by_emotion",
        should_continue_full_pipeline,
        {
            "continue_processing": "find_peak_frame",
            "end_processing": END,
            "handle_error": "handle_error",
        },
    )
    workflow.add_edge("find_peak_frame", "generate_full_descriptions")
    workflow.add_edge("generate_full_descriptions", "synthesize_summary")
    workflow.add_edge("synthesize_summary", "save_mer_results")
    workflow.add_edge("save_mer_results", END)

    # 6. Image pipeline
    workflow.add_edge("run_image_analysis", "synthesize_image_summary")
    workflow.add_edge("synthesize_image_summary", "save_image_results")
    workflow.add_edge("save_image_results", END)

    # 7. Shared error handler
    workflow.add_edge("handle_error", END)

    app = workflow.compile()

    console.log("Modular graph compiled successfully.")
    # print(app.get_graph().draw_mermaid())
    return app
