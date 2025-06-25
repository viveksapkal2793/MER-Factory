# MERR-AutoAnn: Multimodal Emotion Recognition and Auto-Annotation

A modular CLI tool for constructing multimodal emotion recognition (MERR) datasets from video files. This tool provides four different processing modes: Action Unit (AU) extraction, audio analysis, video analysis, and full multimodal emotion recognition pipeline.

## Features

- **AU Pipeline**: Extract facial Action Units and generate natural language descriptions
- **Audio Pipeline**: Extract audio, transcribe speech, and analyze tone
- **Video Pipeline**: Generate comprehensive video content descriptions  
- **MER Pipeline**: Full end-to-end multimodal emotion recognition with peak frame detection and emotional synthesis

## Prerequisites

### 1. FFmpeg
FFmpeg is required for video and audio processing.

**Installation:**
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

**Verify installation:**
```bash
ffmpeg -version
ffprobe -version
```

### 2. OpenFace (Required for AU extraction)
OpenFace is needed for facial Action Unit extraction.

**Installation:**
1. Clone OpenFace repository:
   ```bash
   git clone https://github.com/TadasBaltrusaitis/OpenFace.git
   cd OpenFace
   ```

2. Follow the installation instructions for your platform from the [OpenFace Wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki)

3. Build the project and note the path to the `FeatureExtraction` executable (typically in `build/bin/FeatureExtraction`)

### 3. Google Gemini API Key
The tool uses Google's Gemini AI for natural language processing.

1. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```

## Installation

```bash
git clone git@github.com:Lum1104/MER-Dataset-Builder.git
cd MER-Dataset-Builder

conda create -n mer_dataset_builder python=3.12
conda activate mer_dataset_builder

pip install -r requirements.txt
```

Create a `.env` file with:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

Edit `tools/openface_adapter.py` and update the path to your OpenFace executable:
```python
command = [
    "/path/to/your/OpenFace/build/bin/FeatureExtraction",  # Update this path
    "-f",
    str(video_path),
    # ... rest of command
]
```

## Usage

### Basic Command Structure
```bash
python main.py [[VIDEO_FILE] | [VIDEO_DIR]] [OUTPUT_DIR] [OPTIONS]
python main.py path_to_video/ output/ --type MER --silent
```

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

#### 4. Full MER Pipeline (Default)
Runs the complete multimodal emotion recognition pipeline:
```bash
python main.py video.mp4 output/ --type MER
# or simply:
python main.py video.mp4 output/
```

## Testing Tools

The project includes testing utilities to verify your setup:

### Test FFmpeg Integration
```bash
python test_ffmpeg.py video_file.mp4 test_output/
```

### Test OpenFace Integration
```bash
python test_openface.py /path/to/FeatureExtraction video_file.mp4 test_output/
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found:**
   - Ensure FFmpeg is installed and in your system PATH
   - Test with: `ffmpeg -version`

2. **OpenFace executable not found:**
   - Verify the path in `tools/openface_adapter.py`
   - Ensure the executable has proper permissions
   - Test with the provided `test_openface.py` script

3. **Google API errors:**
   - Verify your API key is correct in the `.env` file
   - Check your API quotas and billing in Google Cloud Console
