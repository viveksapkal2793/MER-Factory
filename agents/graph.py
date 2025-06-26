from ast import List
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from rich.console import Console
from pathlib import Path

from sympy import Li


from .nodes import (
    setup_paths,
    handle_error,
    run_au_extraction,
    map_au_to_text,
    generate_au_description,
    save_au_results,
    run_audio_extraction_and_analysis,
    save_audio_results,
    run_video_analysis,
    save_video_results,
    extract_full_features,
    filter_by_emotion,
    find_peak_frame,
    generate_full_descriptions,
    synthesize_summary,
    save_mer_results,
)
from .models import LLMModels

console = Console()


class MERRState(TypedDict, total=False):
    """Represents the state of the MERR pipeline."""

    video_path: Path
    output_dir: Path
    processing_type: str
    video_id: str
    video_output_dir: Path
    is_expressive: bool
    au_data_path: Path
    peak_frame_info: Dict[str, Any]
    peak_frame_path: Path
    audio_path: Path
    au_text_description: str
    llm_au_description: str
    audio_analysis_results: dict
    video_description: str
    descriptions: Dict[str, str]
    final_summary: str
    models: LLMModels
    error: str
    threshold: float
    verbose: bool
    error_logs_dir: Path
    detected_emotions: List


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
    return "handle_error"


def should_continue_full_pipeline(state: MERRState) -> str:
    """Router for the full pipeline after the emotion filter."""
    if state.get("error"):
        return "handle_error"
    if state.get("is_expressive"):
        return "continue_processing"
    return "end_processing"


def create_graph() -> StateGraph:
    """Creates and compiles the modular MERR construction graph."""
    workflow = StateGraph(MERRState)

    # Add all nodes
    workflow.add_node("setup_paths", setup_paths)
    workflow.add_node("handle_error", handle_error)
    # AU Pipeline
    workflow.add_node("run_au_extraction", run_au_extraction)
    workflow.add_node("map_au_to_text", map_au_to_text)
    workflow.add_node("generate_au_description", generate_au_description)
    workflow.add_node("save_au_results", save_au_results)
    # Audio Pipeline
    workflow.add_node("run_audio_analysis", run_audio_extraction_and_analysis)
    workflow.add_node("save_audio_results", save_audio_results)
    # Video Pipeline
    workflow.add_node("run_video_analysis", run_video_analysis)
    workflow.add_node("save_video_results", save_video_results)
    # MER Pipeline
    workflow.add_node("extract_full_features", extract_full_features)
    workflow.add_node("filter_by_emotion", filter_by_emotion)
    workflow.add_node("find_peak_frame", find_peak_frame)
    workflow.add_node("generate_full_descriptions", generate_full_descriptions)
    workflow.add_node("synthesize_summary", synthesize_summary)
    workflow.add_node("save_mer_results", save_mer_results)

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

    # 6. Shared error handler
    workflow.add_edge("handle_error", END)

    app = workflow.compile()

    console.log("Modular graph compiled successfully.")
    # print(app.get_graph().draw_mermaid())
    return app
