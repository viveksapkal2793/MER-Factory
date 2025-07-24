# We provide two export options:
# 1. Follow the MERTools (https://github.com/zeroQiaoba/MERTools/) format to export label data to a CSV file.
# 2. Follow the ShareGPT format in LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory/) to export data to JSON/JSONL files.
# Open an issue if you have any questions or suggestions.

import os
import json
import glob
import csv
import argparse
import random
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


def process_export_folder(output_folder, file_type):
    """
    Process folders in output directory based on file type and return the collected data.

    Args:
        output_folder (str): Path to the output folder
        file_type (str): Type of processing ('au', 'image', 'mer', 'audio', 'video')

    Returns:
        list: A list of dictionaries containing the processed data.
    """
    if not os.path.exists(output_folder):
        print(f"Output folder does not exist: {output_folder}")
        return []

    all_data = []  # Collect all data for export

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
            # This is expected if a folder doesn't contain the specific file_type
            continue

        # Process JSON files with progress bar
        for json_file in tqdm(
            json_files, desc=f"Processing {folder_name}", leave=False
        ):
            data = process_json_file(json_file, file_type)
            if data:
                all_data.append(data)

    return all_data


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
        source_path = data.get("source_path", "unknown")

        if file_type == "au":
            chronological_emotion_peaks = data.get("chronological_emotion_peaks", [])
            # Clean each emotion peak text
            cleaned_peaks = [
                clean_text_for_csv(peak) for peak in chronological_emotion_peaks
            ]
            emotion_peaks_text = "; ".join(cleaned_peaks)

            return {
                "source_path": source_path,
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
                "source_path": source_path,
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
                "source_path": source_path,
                "audio_analysis": clean_text_for_csv(data.get("audio_analysis", "")),
                "file_type": file_type,
            }
        elif file_type == "video":
            return {
                "source_path": source_path,
                "llm_video_summary": clean_text_for_csv(
                    data.get("llm_video_summary", "")
                ),
                "file_type": file_type,
            }
        elif file_type == "image":
            return {
                "source_path": source_path,
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
                "source_path": source_path,
                "data": clean_text_for_csv(str(data)),
                "file_type": file_type,
            }

    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None


def read_from_csv(csv_filepath):
    """
    Reads data from a CSV file into a list of dictionaries.

    Args:
        csv_filepath (str): The path to the input CSV file.

    Returns:
        list: A list of dictionaries representing the rows of the CSV.
    """
    if not os.path.exists(csv_filepath):
        print(f"CSV file not found: {csv_filepath}")
        return []

    all_data = []
    try:
        with open(csv_filepath, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                all_data.append(row)
        print(f"Successfully read {len(all_data)} rows from {csv_filepath}")
        return all_data
    except Exception as e:
        print(f"Error reading from CSV file {csv_filepath}: {e}")
        return []


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
            if not all_data:
                print("No data to export to CSV.")
                return

            # Dynamically get fieldnames from the first item, assuming all items have the same structure
            fieldnames = all_data[0].keys()

            # Use QUOTE_ALL to ensure all fields are quoted
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL
            )
            writer.writeheader()

            # Write rows with progress bar
            for row in tqdm(all_data, desc="Writing CSV"):
                writer.writerow(row)

        print(f"Exported all {file_type} data to {csv_path}")

    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def load_prompts(prompt_file):
    """
    Loads prompts from a JSON file.

    Args:
        prompt_file (str): Path to the JSON file containing prompts.

    Returns:
        dict: A dictionary of prompts, or an empty dictionary if loading fails.
    """
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        return prompts.get("user_questions", {})
    except FileNotFoundError:
        print(
            f"Warning: Prompt file not found at '{prompt_file}'. Using default prompts."
        )
        return {}
    except json.JSONDecodeError:
        print(
            f"Warning: Could not decode JSON from '{prompt_file}'. Using default prompts."
        )
        return {}
    except Exception as e:
        print(f"An error occurred while loading prompts: {e}")
        return {}


def export_to_json(
    all_data, export_path, export_format, json_format, file_type, prompts
):
    """
    Exports data to a JSON or JSONL file in a specified format.

    Args:
        all_data (list): List of data dictionaries to export.
        export_path (str): The directory to save the export file.
        export_format (str): The format for the JSON structure ('sharegpt' for now).
        json_format (str): The file format ('json' or 'jsonl').
        file_type (str): The type of data being processed.
        prompts (dict): A dictionary containing lists of prompts for each file type.
    """
    # Define special tags and default instruction/output mappings
    tag_map = {
        "au": "<image>",
        "image": "<image>",
        "video": "<video>",
        "audio": "<audio>",
        "mer": "<video><audio>",
    }
    default_instruction_map = {
        "au": "Describe the chronological emotion peaks from the analysis.",
        "image": "Provide a detailed analysis of the image.",
        "video": "Summarize the content of the video.",
        "audio": "Provide an analysis of the audio.",
        "mer": "Provide a comprehensive multi-modal emotion recognition summary.",
    }
    output_key_map = {
        "au": "chronological_emotion_peaks",
        "image": "final_summary",
        "video": "llm_video_summary",
        "audio": "audio_analysis",
        "mer": "final_summary",
    }

    tag = tag_map.get(file_type, "")
    output_key = output_key_map.get(file_type)

    if not output_key:
        print(
            f"No output key mapping for file_type '{file_type}'. Cannot export to JSON."
        )
        return

    prompt_list = prompts.get(file_type)

    json_output_data = []
    for row in tqdm(all_data, desc=f"Formatting for {export_format}"):
        output_text = row.get(output_key, "")
        source_path = row.get("source_path", "")

        # Select a random prompt or use the default
        if prompt_list:
            instruction = random.choice(prompt_list)
        else:
            instruction = default_instruction_map.get(file_type, "Describe the data.")

        if export_format == "sharegpt":
            formatted_entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{tag}\n{instruction}" if tag else instruction,
                    },
                    {"role": "gpt", "content": output_text},
                ],
                (
                    "videos"
                    if file_type in ["video", "mer"]
                    else "audios" if file_type == "audio" else "images"
                ): [source_path],
            }
        else:
            raise  # Should not happen due to arg choices

        json_output_data.append(formatted_entry)

    # Write to file
    filename = f"{file_type}_{export_format}_export.{json_format}"
    filepath = os.path.join(export_path, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            if json_format == "jsonl":
                for entry in tqdm(json_output_data, desc=f"Writing {json_format}"):
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            else:  # json
                json.dump(json_output_data, f, indent=4, ensure_ascii=False)

        print(f"Successfully exported data to {filepath}")

    except Exception as e:
        print(f"Error exporting to {json_format}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process analysis files and export to CSV or JSON formats."
    )

    # Group for processing from folder
    folder_group = parser.add_argument_group("Process from JSON files in a folder")
    folder_group.add_argument(
        "--output_folder", help="Path to the output folder containing analysis results."
    )
    folder_group.add_argument(
        "--file_type",
        choices=["au", "image", "mer", "audio", "video"],
        type=str.lower,
        help="Type of analysis files to process.",
    )

    # Group for converting from CSV
    csv_group = parser.add_argument_group("Convert from existing CSV file")
    csv_group.add_argument(
        "--input_csv", help="Path to an existing CSV file to convert to JSON."
    )

    # General export options that apply to both modes
    parser.add_argument(
        "--export_path",
        help="Path to export the files. Defaults to the source folder if not provided.",
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Flag to export the output to a CSV file (only valid when processing from a folder).",
    )
    parser.add_argument(
        "--export_format",
        choices=["sharegpt"],
        default="sharegpt",
        type=str.lower,
        help="If specified, exports the data to a JSON file with this format.",
    )
    parser.add_argument(
        "--json_format",
        choices=["json", "jsonl"],
        default="json",
        type=str.lower,
        help="The file format for JSON export (json or jsonl). Default is json.",
    )
    parser.add_argument(
        "--prompt-file",
        default="utils/prompts/prompts.json",
        help="Path to a JSON file with prompts. Defaults to 'utils/prompts/prompts.json'.",
    )

    args = parser.parse_args()

    prompts = {}
    if args.export_format:
        prompts = load_prompts(args.prompt_file)

    if args.input_csv:
        # Mode 2: Convert from CSV to JSON
        if not args.export_format:
            print("Error: --export_format is required when using --input_csv.")
            parser.print_help()
            return

        all_data = read_from_csv(args.input_csv)
        if not all_data:
            print("No data read from CSV, exiting.")
            return

        # Determine file_type from the data itself, assuming it's consistent
        file_type = all_data[0].get("file_type")
        if not file_type:
            print(
                "Error: 'file_type' column not found in the CSV. Cannot determine export mapping."
            )
            return

        export_path = (
            args.export_path if args.export_path else os.path.dirname(args.input_csv)
        )
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        export_to_json(
            all_data,
            export_path,
            args.export_format,
            args.json_format,
            file_type,
            prompts,
        )

    elif args.output_folder and args.file_type:
        # Mode 1: Process from folder to CSV and/or JSON
        export_path = args.export_path if args.export_path else args.output_folder
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        all_data = process_export_folder(args.output_folder, args.file_type)

        if not all_data:
            print("No data was processed. Exiting.")
            return

        if args.export_csv:
            export_to_csv(all_data, export_path, args.file_type)

        if args.export_format:
            export_to_json(
                all_data,
                export_path,
                args.export_format,
                args.json_format,
                args.file_type,
                prompts,
            )

        if not args.export_csv and not args.export_format:
            print(
                "No export option selected. Use --export_csv or --export_format [format]."
            )
    else:
        print("Error: You must specify a mode of operation.")
        print("Mode 1: --output_folder and --file_type")
        print("Mode 2: --input_csv")
        parser.print_help()


if __name__ == "__main__":
    main()
