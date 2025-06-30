# Multimodal Emotion Recognition Reasoning Dataset Builder

A modular CLI tool for constructing multimodal emotion recognition reasoning (MERR) datasets from video/image files. This tool provides four different processing modes: Action Unit (AU) extraction, audio analysis, video analysis, image anlysis, and full multimodal emotion recognition pipeline.

This is the implementation of **[Emotion-LLaMA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c7f43ada17acc234f568dc66da527418-Abstract-Conference.html) @ NeurIPS 2024** MERR dataset construction strategy.

## Pipeline Structure

<details>
<summary>Click here to expand/collapse</summary>


```mermaid
graph TD;
        __start__([<p>__start__</p>]):::first
        setup_paths(setup_paths)
        handle_error(handle_error)
        run_au_extraction(run_au_extraction)
        save_au_results(save_au_results)
        generate_audio_description(generate_audio_description)
        save_audio_results(save_audio_results)
        generate_video_description(generate_video_description)
        save_video_results(save_video_results)
        extract_full_features(extract_full_features)
        filter_by_emotion(filter_by_emotion)
        find_peak_frame(find_peak_frame)
        generate_peak_frame_visual_description(generate_peak_frame_visual_description)
        generate_peak_frame_au_description(generate_peak_frame_au_description)
        synthesize_summary(synthesize_summary)
        save_mer_results(save_mer_results)
        run_image_analysis(run_image_analysis)
        synthesize_image_summary(synthesize_image_summary)
        save_image_results(save_image_results)
        __end__([<p>__end__</p>]):::last
        __start__ --> setup_paths;
        extract_full_features --> filter_by_emotion;
        filter_by_emotion -.-> find_peak_frame;
        filter_by_emotion -.-> handle_error;
        filter_by_emotion -.-> save_au_results;
        find_peak_frame --> generate_audio_description;
        generate_audio_description -.-> generate_video_description;
        generate_audio_description -.-> handle_error;
        generate_audio_description -.-> save_audio_results;
        generate_peak_frame_au_description --> synthesize_summary;
        generate_peak_frame_visual_description --> generate_peak_frame_au_description;
        generate_video_description -.-> generate_peak_frame_visual_description;
        generate_video_description -.-> handle_error;
        generate_video_description -.-> save_video_results;
        run_au_extraction --> filter_by_emotion;
        run_image_analysis --> synthesize_image_summary;
        setup_paths -. &nbsp;full_pipeline&nbsp; .-> extract_full_features;
        setup_paths -. &nbsp;audio_pipeline&nbsp; .-> generate_audio_description;
        setup_paths -. &nbsp;video_pipeline&nbsp; .-> generate_video_description;
        setup_paths -.-> handle_error;
        setup_paths -. &nbsp;au_pipeline&nbsp; .-> run_au_extraction;
        setup_paths -. &nbsp;image_pipeline&nbsp; .-> run_image_analysis;
        synthesize_image_summary --> save_image_results;
        synthesize_summary --> save_mer_results;
        handle_error --> __end__;
        save_au_results --> __end__;
        save_audio_results --> __end__;
        save_image_results --> __end__;
        save_mer_results --> __end__;
        save_video_results --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```

</details>

## Features

- **AU Pipeline**: Extract facial Action Units and generate natural language descriptions
- **Audio Pipeline**: Extract audio, transcribe speech, and analyze tone
- **Video Pipeline**: Generate comprehensive video content descriptions  
- **Image Pipeline**: End-to-end emotion recognition with image description and emotional synthesis
- **MER Pipeline**: Full end-to-end multimodal emotion recognition with peak frame detection and emotional synthesis

Examples of MERR can be found at [llava-llama3:latest_llama3.2_merr_data.json](examples/llava-llama3:latest_llama3.2_merr_data.json) and [gemini_merr.json](examples/gemini_merr.json)

## Prerequisites



### 1. FFmpeg
FFmpeg is required for video and audio processing.

<details>
<summary>Click here to expand/collapse</summary>

**Installation:**
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

**Verify installation:**
```bash
ffmpeg -version
ffprobe -version
```

</details>

### 2. OpenFace
OpenFace is needed for facial Action Unit extraction.

<details>
<summary>Click here to expand/collapse</summary>

**Installation:**
1. Clone OpenFace repository:
   ```bash
   git clone https://github.com/TadasBaltrusaitis/OpenFace.git
   cd OpenFace
   ```

2. Follow the installation instructions for your platform from the [OpenFace Wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki)

3. Build the project and note the path to the `FeatureExtraction` executable (typically in `build/bin/FeatureExtraction`)

</details>

## Installation

```bash
git clone git@github.com:Lum1104/MER-Dataset-Builder.git
cd MER-Dataset-Builder

conda create -n mer_dataset_builder python=3.12
conda activate mer_dataset_builder

pip install -r requirements.txt
```

**Configuration:**
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and configure your settings:


## Usage

### Basic Command Structure
```bash
python main.py [[VIDEO_FILE] | [VIDEO_DIR]] [OUTPUT_DIR] [OPTIONS]
python main.py path_to_video/ output/ --type MER --silent --threshold 0.45 # using gemini by default
python main.py path_to_video/ output/ --type MER --ollama-vision-model llava-llama3:latest --ollama-text-model llama3.2 --silent # support local ollama running
python main.py path_to_video/ output/ --type MER --huggingface-model google/gemma-3n-E4B-it --silent # huggingface model
```

Note: run `ollama pull llama3.2` etc, if Ollama model is needed. Ollama only support peak frame & AU analysis for now.

### Processing Types

#### 1. Action Unit (AU) Extraction
Extracts facial Action Units and generates natural language descriptions:
```bash
python main.py video.mp4 output/ --type AU
```

#### 2. Audio Analysis
Extracts audio, transcribes speech, and analyzes tone:
```bash
python main.py video.mp4 output/ --type audio
```

#### 3. Video Analysis
Generates comprehensive video content descriptions:
```bash
python main.py video.mp4 output/ --type video
```

#### 4. Image Analysis
Runs the pipeline with images input:
```bash
python main.py ./images ./output --type MER
```

#### 4. Full MER Pipeline (Default)
Runs the complete multimodal emotion recognition pipeline:
```bash
python main.py video.mp4 output/ --type MER
# or simply:
python main.py video.mp4 output/
```

## Testing Tools

Verify your installation of FFmpeg and OpenFace:

<details>
<summary>Click here to expand/collapse</summary>

### Test FFmpeg Integration
```bash
python test_ffmpeg.py video_file.mp4 test_output/
```

### Test OpenFace Integration
```bash
python test_openface.py video_file.mp4 test_output/
```

</details>

## Troubleshooting

### Common Issues

1. **FFmpeg not found:**
   - Ensure FFmpeg is installed and in your system PATH
   - Test with: `ffmpeg -version`

2. **OpenFace executable not found:**
   - Ensure OPENFACE_EXECUTABLE is set in your .env file
   - Verify the path points to the correct FeatureExtraction executable
   - Ensure the executable has proper permissions
   - Test with the provided `test_openface.py` script

3. **Google API errors:**
   - Verify your API key is correct in the `.env` file
   - Check your API quotas and billing in Google Cloud Console
