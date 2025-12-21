#!/usr/bin/env python3
# filepath: /scratch/data/bikash_rs/vivek/MER-Factory/cache_check.py
"""Inspect cache contents to see what's cached and for how many videos."""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table
import hashlib

console = Console()

base_dir = Path("/scratch/data/bikash_rs/vivek/dataset/Meld_feat_ext/train_annotation")
llm_cache_dir = base_dir / ".llm_cache"

def check_json_completeness(json_path: Path) -> tuple[bool, list[str]]:
    """Check if JSON is complete."""
    if not json_path.exists():
        return False, ["missing"]
    
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        missing = []
        if not data.get("chronological_emotion_peaks"):
            missing.append("emotion_peaks")
        if not data.get("overall_peak_frame_info"):
            missing.append("peak_info")
        coarse = data.get("coarse_descriptions_at_peak", {})
        if not coarse.get("visual_expression"):
            missing.append("visual_expression")
        if not coarse.get("visual_objective"):
            missing.append("visual_objective")
        if not coarse.get("audio_analysis"):
            missing.append("audio_analysis")
        if not data.get("final_summary"):
            missing.append("final_summary")
        
        return len(missing) == 0, missing
    except:
        return False, ["error_parsing"]

def analyze_json_completeness():
    """Analyze which videos have complete vs incomplete JSON."""
    console.rule("[bold cyan]JSON Completeness Analysis[/bold cyan]")
    
    complete_videos = []
    incomplete_videos = []
    missing_json = []
    
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith('.') or subdir.name == 'error_logs':
            continue
        
        video_name = subdir.name
        json_path = subdir / f"{video_name}_merr_data.json"
        
        if not json_path.exists():
            missing_json.append(video_name)
            continue
        
        is_complete, missing_fields = check_json_completeness(json_path)
        
        if is_complete:
            complete_videos.append(video_name)
        else:
            incomplete_videos.append({
                'name': video_name,
                'missing': missing_fields
            })
    
    # Print summary
    table = Table(title="JSON Completeness Summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Percentage", style="yellow", justify="right")
    
    total = len(complete_videos) + len(incomplete_videos) + len(missing_json)
    
    table.add_row(
        "✅ Complete JSON",
        str(len(complete_videos)),
        f"{len(complete_videos)/total*100:.1f}%",
        style="green"
    )
    table.add_row(
        "⚠️  Incomplete JSON",
        str(len(incomplete_videos)),
        f"{len(incomplete_videos)/total*100:.1f}%",
        style="yellow"
    )
    table.add_row(
        "❌ Missing JSON",
        str(len(missing_json)),
        f"{len(missing_json)/total*100:.1f}%",
        style="red"
    )
    table.add_row(
        "Total Videos",
        str(total),
        "100.0%",
        style="bold"
    )
    
    console.print(table)
    
    # Show missing fields breakdown
    if incomplete_videos:
        console.print(f"\n[bold yellow]Missing Fields Breakdown:[/bold yellow]")
        
        field_counts = defaultdict(int)
        for video in incomplete_videos:
            for field in video['missing']:
                field_counts[field] += 1
        
        field_table = Table()
        field_table.add_column("Missing Field", style="cyan")
        field_table.add_column("Count", style="red", justify="right")
        field_table.add_column("% of Incomplete", style="yellow", justify="right")
        
        for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
            field_table.add_row(
                field,
                str(count),
                f"{count/len(incomplete_videos)*100:.1f}%"
            )
        
        console.print(field_table)
        
        # Show sample incomplete videos
        console.print(f"\n[bold]Sample Incomplete Videos (first 10):[/bold]")
        for video in incomplete_videos[:10]:
            console.print(f"  • {video['name']}: Missing {', '.join(video['missing'])}")
    
    return complete_videos, incomplete_videos, missing_json

def inspect_llm_cache_detailed():
    """Detailed inspection of LLM cache using diskcache."""
    console.rule("[bold cyan]LLM Cache Detailed Analysis[/bold cyan]")
    
    if not llm_cache_dir.exists():
        console.print("[yellow]LLM cache directory does not exist[/yellow]")
        return
    
    try:
        import diskcache
        cache = diskcache.Cache(str(llm_cache_dir))
        
        # Statistics
        cache_size = len(cache)
        total_size_bytes = sum(f.stat().st_size for f in llm_cache_dir.rglob('*') if f.is_file())
        
        console.print(f"[green]Total cached entries: {cache_size}[/green]")
        console.print(f"[cyan]Cache directory size: {total_size_bytes / 1024 / 1024:.2f} MB[/cyan]")
        console.print(f"[cyan]Average entry size: {total_size_bytes / cache_size / 1024:.2f} KB[/cyan]")
        
        # Analyze cache keys to understand what's cached
        console.print(f"\n[bold]Analyzing cache entries (sampling 100)...[/bold]")
        
        method_counts = defaultdict(int)
        video_mentions = defaultdict(int)
        
        sample_keys = []
        for i, key in enumerate(cache.iterkeys()):
            if i >= 100:  # Sample first 100
                break
            sample_keys.append(key)
            
            # The key format from your caching.py is: (model_name, method_name, args_tuple)
            if isinstance(key, tuple) and len(key) >= 2:
                model_name = key[0]
                method_name = key[1] if len(key) > 1 else 'unknown'
                
                method_counts[method_name] += 1
                
                # Try to extract video name from arguments
                if len(key) > 2:
                    args = key[2]
                    arg_str = str(args)
                    # Look for video names in the arguments
                    for subdir in base_dir.iterdir():
                        if subdir.name in arg_str:
                            video_mentions[subdir.name] += 1
        
        # Show method breakdown
        if method_counts:
            console.print(f"\n[bold]Cached Methods (from sample):[/bold]")
            method_table = Table()
            method_table.add_column("Method", style="cyan")
            method_table.add_column("Count", style="green", justify="right")
            
            for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
                method_table.add_row(method, str(count))
            
            console.print(method_table)
        
        # Show a sample cache entry
        if sample_keys:
            console.print(f"\n[bold]Sample Cache Key Structure:[/bold]")
            sample_key = sample_keys[0]
            console.print(f"  Type: {type(sample_key)}")
            console.print(f"  Length: {len(sample_key) if isinstance(sample_key, tuple) else 'N/A'}")
            console.print(f"  Content preview: {str(sample_key)[:200]}...")
            
            # Try to get the value
            try:
                sample_value = cache[sample_key]
                console.print(f"\n[bold]Sample Cache Value:[/bold]")
                console.print(f"  Type: {type(sample_value)}")
                console.print(f"  Size: {len(str(sample_value))} chars")
                if isinstance(sample_value, str):
                    console.print(f"  Preview: {sample_value[:200]}...")
            except:
                console.print(f"  [yellow]Could not retrieve sample value[/yellow]")
        
        cache.close()
        
        return cache_size, total_size_bytes
        
    except ImportError:
        console.print("[red]diskcache not available - install with: pip install diskcache[/red]")
        return 0, 0
    except Exception as e:
        console.print(f"[red]Error inspecting LLM cache: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 0, 0

def estimate_cache_impact(incomplete_videos):
    """Estimate how much work is needed to complete incomplete videos."""
    console.rule("[bold cyan]Reprocessing Estimation[/bold cyan]")
    
    # Count what needs to be generated
    needs_audio = sum(1 for v in incomplete_videos if 'audio_analysis' in v['missing'])
    needs_visual_obj = sum(1 for v in incomplete_videos if 'visual_objective' in v['missing'])
    needs_visual_exp = sum(1 for v in incomplete_videos if 'visual_expression' in v['missing'])
    needs_summary = sum(1 for v in incomplete_videos if 'final_summary' in v['missing'])
    
    console.print(f"[bold]Videos Needing Regeneration:[/bold]")
    console.print(f"  • Audio analysis: {needs_audio} videos")
    console.print(f"  • Visual objective: {needs_visual_obj} videos")
    console.print(f"  • Visual expression: {needs_visual_exp} videos")
    console.print(f"  • Final summary: {needs_summary} videos")
    console.print(f"\n  [bold yellow]Total incomplete: {len(incomplete_videos)} videos[/bold yellow]")

def main():
    console.print("\n[bold green]🔍 Comprehensive Cache & Data Analysis[/bold green]\n")
    
    # Analyze JSON completeness
    complete_videos, incomplete_videos, missing_json = analyze_json_completeness()
    
    # Inspect LLM cache
    console.print("\n")
    cache_entries, cache_size_bytes = inspect_llm_cache_detailed()
    
    # Estimate work needed
    if incomplete_videos:
        console.print("\n")
        estimate_cache_impact(incomplete_videos)
    
    # Final recommendations
    console.rule("[bold green]Recommendations[/bold green]")
    
    if incomplete_videos or missing_json:
        console.print(f"\n⚠️  [bold yellow]Action Required:[/bold yellow]")
        
        if incomplete_videos:
            console.print(f"\n1. [cyan]{len(incomplete_videos)} videos have incomplete data[/cyan]")
            console.print(f"   The LLM cache has {cache_entries:,} entries")
            console.print(f"   These cached responses may be for incomplete videos")
            console.print(f"\n   [bold]Options:[/bold]")
            console.print(f"   a) Run with [cyan]--skip-complete[/cyan] to only process incomplete videos")
            console.print(f"      (Will use cached data where available)")
            console.print(f"   b) Clear LLM cache and reprocess:")
            console.print(f"      [cyan]rm -rf {llm_cache_dir}[/cyan]")
            console.print(f"      Then run processing again")
        
        if missing_json:
            console.print(f"\n2. [red]{len(missing_json)} videos have no JSON at all[/red]")
            console.print(f"   Run processing to generate data for these videos")
    else:
        console.print(f"\n✅ [bold green]All videos ({len(complete_videos)}) have complete data![/bold green]")
        console.print(f"   LLM cache can be safely kept for future runs")

if __name__ == "__main__":
    main()