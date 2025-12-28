"""Identify incomplete JSON files and split them into two batches"""
import json
from pathlib import Path
from collections import defaultdict

base_dir = "/scratch/data/bikash_rs/vivek/dataset/Meld_feat_ext/train_annotation"
output_dir = "/scratch/data/bikash_rs/vivek/MER-Factory"

incomplete_files = []
base_path = Path(base_dir)

print(f"🔍 Scanning directory: {base_dir}")
print(f"Looking for incomplete JSON files...\n")

# Process JSON files
for json_file in base_path.glob("*/*_merr_data.json"):
    try:
        with open(json_file) as f:
            data = json.load(f)
            
            is_incomplete = False
            missing_fields = []
            
            if not data.get("chronological_emotion_peaks"):
                missing_fields.append("chronological_emotion_peaks")
                is_incomplete = True
            if not data.get("overall_peak_frame_info"):
                missing_fields.append("overall_peak_frame_info")
                is_incomplete = True
            if not data.get("coarse_descriptions_at_peak", {}).get("visual_expression"):
                missing_fields.append("visual_expression")
                is_incomplete = True
            if not data.get("coarse_descriptions_at_peak", {}).get("visual_objective"):
                missing_fields.append("visual_objective")
                is_incomplete = True
            if not data.get("coarse_descriptions_at_peak", {}).get("audio_analysis"):
                missing_fields.append("audio_analysis")
                is_incomplete = True
            if not data.get("final_summary"):
                missing_fields.append("final_summary")
                is_incomplete = True
            
            if is_incomplete:
                # Extract video name from path (parent directory name)
                video_name = json_file.parent.name
                incomplete_files.append((video_name, missing_fields))
                
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing {json_file.name}: {e}")
        video_name = json_file.parent.name
        incomplete_files.append((video_name, ["JSON parse error"]))
    except Exception as e:
        print(f"❌ Error reading {json_file.name}: {e}")
        video_name = json_file.parent.name
        incomplete_files.append((video_name, ["Read error"]))

# Sort by video name for consistency
incomplete_files.sort(key=lambda x: x[0])

print(f"Found {len(incomplete_files)} incomplete JSON files")

# Split into two batches
mid_point = len(incomplete_files) // 2
batch1 = incomplete_files[:mid_point]
batch2 = incomplete_files[mid_point:]

print(f"Batch 1: {len(batch1)} files")
print(f"Batch 2: {len(batch2)} files")

# Write batch 1
batch1_file = Path(output_dir) / "incomplete_batch1.txt"
with open(batch1_file, 'w') as f:
    for video_name, _ in batch1:
        f.write(f"{video_name}\n")
print(f"\n✅ Batch 1 written to: {batch1_file}")

# Write batch 2
batch2_file = Path(output_dir) / "incomplete_batch2.txt"
with open(batch2_file, 'w') as f:
    for video_name, _ in batch2:
        f.write(f"{video_name}\n")
print(f"✅ Batch 2 written to: {batch2_file}")

# Write detailed report
report_file = Path(output_dir) / "incomplete_files_report.txt"
with open(report_file, 'w') as f:
    f.write(f"INCOMPLETE FILES REPORT\n")
    f.write(f"=" * 80 + "\n\n")
    f.write(f"Total incomplete files: {len(incomplete_files)}\n")
    f.write(f"Batch 1: {len(batch1)} files\n")
    f.write(f"Batch 2: {len(batch2)} files\n\n")
    f.write(f"=" * 80 + "\n\n")
    
    f.write("BATCH 1 FILES:\n")
    f.write("-" * 80 + "\n")
    for video_name, missing_fields in batch1:
        f.write(f"{video_name}: {', '.join(missing_fields)}\n")
    
    f.write("\n" + "=" * 80 + "\n\n")
    f.write("BATCH 2 FILES:\n")
    f.write("-" * 80 + "\n")
    for video_name, missing_fields in batch2:
        f.write(f"{video_name}: {', '.join(missing_fields)}\n")

print(f"✅ Detailed report written to: {report_file}")

print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
print(f"Total incomplete files: {len(incomplete_files)}")
print(f"Batch 1 size: {len(batch1)}")
print(f"Batch 2 size: {len(batch2)}")
print("\nNext steps:")
print(f"1. Run job 1: python main.py ... --filter-file incomplete_batch1.txt")
print(f"2. Run job 2: python main.py ... --filter-file incomplete_batch2.txt")