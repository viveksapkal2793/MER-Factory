# Follow https://github.com/zeroQiaoba/MERTools/ we export the label data to csv file.
# Open an issue if you have any questions or suggestions.
import os
import json
import glob
import csv
import argparse
import re
from tqdm import tqdm


def clean_text_for_csv(text):
    """
    Clean text data for CSV export by handling problematic characters.

    Args:
        text (str): Input text to clean

    Returns:
        str: Cleaned text safe for CSV export
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().replace("\n", "\\n")

    return text


def process_export_folder(output_folder, file_type, export_path):
    """
    Process folders in output directory based on file type.

    Args:
        output_folder (str): Path to the output folder
        file_type (str): Type of processing ('au', 'image', 'mer', 'audio', 'video')
        export_path (str): Path to export the CSV file
    """
    if not os.path.exists(output_folder):
        print(f"Output folder does not exist: {output_folder}")
        return

    all_data = []  # Collect all data for CSV export

    # Get all subdirectories in output folder
    folders = [
        f
        for f in os.listdir(output_folder)
        if os.path.isdir(os.path.join(output_folder, f))
    ]

    print(f"Found {len(folders)} folders to process")

    # Process folders with progress bar
    for folder_name in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(output_folder, folder_name)

        # Define file patterns based on file_type
        patterns = {
            "au": f"{folder_name}_au_analysis.json",
            "mer": f"{folder_name}_merr_data.json",
            "audio": f"{folder_name}_audio_analysis.json",
            "video": f"{folder_name}_video_analysis.json",
            "image": f"{folder_name}_image_analysis.json",
        }

        # Default to a generic pattern if file_type is unknown, though choices limit this.
        pattern = patterns.get(file_type, f"{folder_name}*.json")
        json_pattern = os.path.join(folder_path, pattern)
        json_files = glob.glob(json_pattern)

        if not json_files:
            print(f"No matching JSON files found in {folder_name} for type {file_type}")
            continue

        # Process JSON files with progress bar
        for json_file in tqdm(
            json_files, desc=f"Processing {folder_name}", leave=False
        ):
            data = process_json_file(json_file, file_type)
            if data:
                all_data.append(data)

    # Export all data to single CSV
    if all_data:
        export_to_csv(all_data, export_path, file_type)


def process_json_file(json_file, file_type):
    """
    Process JSON file based on file type.

    Args:
        json_file (str): Path to the JSON file
        file_type (str): Type of file being processed

    Returns:
        dict: Processed data or None if error
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        if file_type != "image":
            video_id = data.get("video_id", "unknown")
        else:
            video_id = data.get("image_id", "unknown")

        if file_type == "au":
            chronological_emotion_peaks = data.get("chronological_emotion_peaks", [])
            # Clean each emotion peak text
            cleaned_peaks = [
                clean_text_for_csv(peak) for peak in chronological_emotion_peaks
            ]
            emotion_peaks_text = "; ".join(cleaned_peaks)

            return {
                "video_id": video_id,
                "chronological_emotion_peaks": emotion_peaks_text,
                "file_type": file_type,
            }
        elif file_type == "mer":
            # Special handling for MER files
            chronological_emotion_peaks = data.get("chronological_emotion_peaks", [])
            # Clean each emotion peak text
            cleaned_peaks = [
                clean_text_for_csv(peak) for peak in chronological_emotion_peaks
            ]
            emotion_peaks_text = "; ".join(cleaned_peaks)
            coarse_descriptions = data.get("coarse_descriptions_at_peak", {})

            return {
                "video_id": video_id,
                "chronological_emotion_peaks": emotion_peaks_text,
                "visual_expression": clean_text_for_csv(
                    coarse_descriptions.get("visual_expression", "")
                ),
                "visual_objective": clean_text_for_csv(
                    coarse_descriptions.get("visual_objective", "")
                ),
                "audio_analysis": clean_text_for_csv(
                    coarse_descriptions.get("audio_analysis", "")
                ),
                "video_content": clean_text_for_csv(
                    coarse_descriptions.get("video_content", "")
                ),
                "final_summary": clean_text_for_csv(data.get("final_summary", "")),
                "file_type": file_type,
            }
        elif file_type == "audio":
            return {
                "video_id": video_id,
                "audio_analysis": clean_text_for_csv(data.get("audio_analysis", "")),
                "file_type": file_type,
            }
        elif file_type == "video":
            return {
                "video_id": video_id,
                "llm_video_summary": clean_text_for_csv(
                    data.get("llm_video_summary", "")
                ),
                "file_type": file_type,
            }
        elif file_type == "image":
            return {
                "image_id": video_id,
                "source_image": clean_text_for_csv(data.get("source_image", "")),
                "au_text_description": clean_text_for_csv(
                    data.get("au_text_description", "")
                ),
                "llm_au_description": clean_text_for_csv(
                    data.get("llm_au_description", "")
                ),
                "image_visual_description": clean_text_for_csv(
                    data.get("image_visual_description", "")
                ),
                "final_summary": clean_text_for_csv(data.get("final_summary", "")),
                "file_type": file_type,
            }
        else:
            # Generic handling for other file types
            return {
                "video_id": video_id,
                "data": clean_text_for_csv(str(data)),
                "file_type": file_type,
            }

    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None


def export_to_csv(all_data, export_path, file_type):
    """
    Export all processed data to a single CSV file.

    Args:
        all_data (list): List of processed data dictionaries
        export_path (str): Path to export folder
        file_type (str): Type of files processed
    """

    csv_filename = f"{file_type}_export_data.csv"
    csv_path = os.path.join(export_path, csv_filename)

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            if file_type == "au":
                fieldnames = ["video_id", "chronological_emotion_peaks", "file_type"]
            elif file_type == "mer":
                fieldnames = [
                    "video_id",
                    "chronological_emotion_peaks",
                    "visual_expression",
                    "visual_objective",
                    "audio_analysis",
                    "video_content",
                    "final_summary",
                    "file_type",
                ]
            elif file_type == "audio":
                fieldnames = ["video_id", "audio_analysis", "file_type"]
            elif file_type == "video":
                fieldnames = ["video_id", "llm_video_summary", "file_type"]
            elif file_type == "image":
                fieldnames = [
                    "image_id",
                    "source_image",
                    "au_text_description",
                    "llm_au_description",
                    "image_visual_description",
                    "final_summary",
                    "file_type",
                ]
            else:
                fieldnames = ["video_id", "data", "file_type"]

            # Use QUOTE_ALL to ensure all fields are quoted
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL
            )
            writer.writeheader()

            # Write rows with progress bar
            for row in tqdm(all_data, desc="Writing CSV"):
                writer.writerow(row)

        print(f"Exported all {file_type} data to {csv_filename}")

    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process JSON files and export to CSV")
    parser.add_argument("--output_folder", help="Path to the output folder")
    parser.add_argument(
        "--file_type",
        choices=["au", "image", "mer", "audio", "video"],
        type=str.lower,
        help="Type of files to process",
    )
    parser.add_argument(
        "--export_path",
        help="Path to export the CSV file. Defaults to output_folder if not provided.",
    )

    args = parser.parse_args()

    export_path = args.export_path if args.export_path else args.output_folder
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    process_export_folder(args.output_folder, args.file_type, export_path)


if __name__ == "__main__":
    main()
