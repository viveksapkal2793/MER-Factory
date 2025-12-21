"""Quick check for missing fields"""
import json
from pathlib import Path
from collections import defaultdict

base_dir = "/scratch/data/bikash_rs/vivek/dataset/Meld_feat_ext/train_annotation"
missing = defaultdict(int)
total = 0
total_subdirs = 0
subdirs_with_json = 0
subdirs_without_json = []

print(f"🔍 Scanning directory: {base_dir}")
print(f"Looking for pattern: */*_merr_data.json\n")

# Check subdirectories
base_path = Path(base_dir)
if base_path.exists():
    all_subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    total_subdirs = len(all_subdirs)
    
    for subdir in all_subdirs:
        json_files_in_subdir = list(subdir.glob("*_merr_data.json"))
        if json_files_in_subdir:
            subdirs_with_json += 1
        else:
            subdirs_without_json.append(subdir.name)

# Process JSON files
for json_file in base_path.glob("*/*_merr_data.json"):
    total += 1
    
    try:
        with open(json_file) as f:
            data = json.load(f)
            if not data.get("chronological_emotion_peaks"):
                missing["chronological_emotion_peaks"] += 1
            if not data.get("overall_peak_frame_info"):
                missing["overall_peak_frame_info"] += 1
            # if not data.get("coarse_descriptions_at_peak", {}).get("video_content"):
            #     missing["video_content"] += 1
            if not data.get("coarse_descriptions_at_peak", {}).get("visual_expression"):
                missing["visual_expression"] += 1
            if not data.get("coarse_descriptions_at_peak", {}).get("visual_objective"):
                missing["visual_objective"] += 1
            if not data.get("coarse_descriptions_at_peak", {}).get("audio_analysis"):
                missing["audio_analysis"] += 1
            if not data.get("final_summary"):
                missing["final_summary"] += 1
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing {json_file.name}: {e}")
    except Exception as e:
        print(f"❌ Error reading {json_file.name}: {e}")

# Print results
print("="*80)
print("📊 DIRECTORY STRUCTURE ANALYSIS")
print("="*80)
print(f"Total subdirectories: {total_subdirs}")
print(f"Subdirectories with JSON files: {subdirs_with_json}")
print(f"Subdirectories without JSON files: {total_subdirs - subdirs_with_json}")

if subdirs_without_json and len(subdirs_without_json) <= 10:
    print(f"\n⚠️  Subdirectories missing JSON files:")
    for subdir in sorted(subdirs_without_json):
        print(f"  • {subdir}")

print("\n" + "="*80)
print("📊 JSON FILES ANALYSIS")
print("="*80)
print(f"Total JSON files found: {total}")
print(f"Expected JSON files: {total_subdirs}")
print(f"Coverage: {(total/total_subdirs*100):.2f}%" if total_subdirs > 0 else "N/A")

if total > 0:
    files_with_all_fields = total
    for count in missing.values():
        if count > 0:
            files_with_all_fields = min(files_with_all_fields, total - count)
    
    print(f"\n✅ Files with all fields complete: {files_with_all_fields} ({files_with_all_fields/total*100:.2f}%)")
    print(f"❌ Files with missing fields: {total - files_with_all_fields} ({(total - files_with_all_fields)/total*100:.2f}%)")
    
    if missing:
        print(f"\n{'Field Name':<40} {'Missing':<10} {'Percentage':<12}")
        print("-"*80)
        for field, count in sorted(missing.items(), key=lambda x: x[1], reverse=True):
            percentage = (count/total*100)
            status = "❌" if percentage > 50 else "⚠️" if percentage > 10 else "✓"
            print(f"{status} {field:<37} {count:<10} {percentage:>6.2f}%")
    else:
        print("\n✅ All fields are complete in all JSON files!")
else:
    print("\n❌ No JSON files found!")
    print(f"Directory exists: {base_path.exists()}")
    if base_path.exists():
        print(f"Directory is readable: {base_path.is_dir()}")