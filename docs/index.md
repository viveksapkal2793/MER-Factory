---
layout: default
title: MER-Factory - Multimodal Emotion Recognition Factory
description: Your automated factory for constructing Multimodal Emotion Recognition and Reasoning (MERR) datasets
lang: en
---

<div class="hero-section">
  <h1 class="hero-title">👉🏻 MER-Factory 👈🏻</h1>
  <p class="hero-subtitle">Your automated factory for constructing Multimodal Emotion Recognition and Reasoning (MERR) datasets</p>
  
  <div class="badges">
    <img src="https://img.shields.io/badge/Task-Multimodal_Emotion_Reasoning-red" alt="MERR">
    <img src="https://img.shields.io/badge/Task-Multimodal_Emotion_Recognition-red" alt="MER">
    <img src="https://img.shields.io/badge/Python-3.12+-blue" alt="Python">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </div>

  <img src="assets/logo.png" alt="MER-Factory Logo" class="hero-image">
</div>

## Quick Overview

MER-Factory is a **Python-based, open-source framework** designed for the Affective Computing community. It automates the creation of unified datasets for training Multimodal Large Language Models (MLLMs) by extracting multimodal features and leveraging LLMs to generate detailed analyses and emotional reasoning summaries.

### 🚀 Key Features

- **Multi-Pipeline Architecture**: Support for AU, Audio, Video, Image, and full MER processing
- **Flexible Model Integration**: Works with OpenAI, Google Gemini, Ollama, and Hugging Face models
- **Scalable Processing**: Async/concurrent processing for large datasets
- **Scientific Foundation**: Based on Facial Action Coding System (FACS) and latest research
- **Easy CLI Interface**: Simple command-line usage with comprehensive options

### 📋 Processing Types

| Pipeline | Description | Use Case |
|----------|-------------|----------|
| **AU** | Facial Action Unit extraction and description | Facial expression analysis |
| **Audio** | Speech transcription and tonal analysis | Audio emotion analysis |
| **Video** | Comprehensive video content description | Video emotion analysis |
| **Image** | Static image emotion recognition | Image-based emotion analysis |
| **MER** | Complete multimodal pipeline | Full emotion reasoning datasets |

## Quick Start

```bash
# Install MER-Factory
git clone https://github.com/Lum1104/MER-Factory.git
cd MER-Factory
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and OpenFace path

# Run full MER pipeline
python main.py video.mp4 output/ --type MER --silent
```

### 📖 Example Outputs

Check out real examples of what MER-Factory produces:
- [Gemini Model Output](https://github.com/Lum1104/MER-Factory/blob/main/examples/gemini_merr.json)
- [LLaVA Model Output](https://github.com/Lum1104/MER-Factory/blob/main/examples/llava-llama3:latest_llama3.2_merr_data.json)

## Architecture Overview

<div class="architecture-section">
  <h3>System Components</h3>
  <ul>
    <li><strong>CLI Framework:</strong> Typer for robust command-line interface</li>
    <li><strong>Workflow Management:</strong> LangGraph for stateful processing pipelines</li>
    <li><strong>Facial Analysis:</strong> OpenFace for Facial Action Units extraction</li>
    <li><strong>Media Processing:</strong> FFmpeg for audio/video manipulation</li>
    <li><strong>AI Integration:</strong> Pluggable architecture for multiple LLM providers</li>
    <li><strong>Concurrency:</strong> Asyncio for efficient parallel processing</li>
  </ul>
</div>

## Getting Started

Ready to dive in? Here's what you need to know:

1.  **[Prerequisites](/MER-Factory/getting-started#prerequisites)** - Install FFmpeg and OpenFace
2.  **[Installation Guide](/MER-Factory/getting-started#installation)** - Set up MER-Factory
3.  **[Basic Usage](/MER-Factory/getting-started#your-first-pipeline)** - Your first emotion recognition pipeline
4.  **[Model Configuration](/MER-Factory/getting-started#model-options)** - Choose and configure your AI models
5.  **[Advanced Features](/MER-Factory/getting-started#next-steps)** - Explore all capabilities

## Community & Support

- 📚 **[Technical Documentation](/MER-Factory/technical-docs)** - Deep dive into system architecture
- 🔧 **[API Reference](/MER-Factory/api-reference)** - Complete function and class documentation
- 💡 **[Examples](/MER-Factory/examples)** - Real-world usage examples and tutorials
- 🐛 **Issues & Bug Reports** - [GitHub Issues](https://github.com/Lum1104/MER-Factory/issues)
- 💬 **Discussions** - [GitHub Discussions](https://github.com/Lum1104/MER-Factory/discussions)

*Advancing together with the Affective Computing community.*

<style>
.hero-section {
  text-align: center;
  margin: 2rem 0;
  padding: 2rem;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 10px;
}

.hero-title {
  font-size: 2.5rem;
  margin-bottom: 0.2rem;
  color: #2c3e50;
}

.hero-subtitle {
  font-size: 1.2rem;
  color: #7f8c8d;
  margin-bottom: 0.2rem;
}

.badges {
  margin: 0.3rem 0;
  display: inline-block;
  text-align: center;
}

.badges img {
  height: 28px;
  margin: 0.4rem;
  vertical-align: middle;
  border: none;
  background: none;
  box-shadow: none;
}

.hero-image {
  max-width: 30%;
  width: auto;
  height: auto;
  margin: 0.5rem auto;
  display: block;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .hero-image {
    width: 90%;
  }
  
  .badges {
    display: block;
    text-align: center;
  }
}

.architecture-section {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  margin: 1.5rem 0;
}


table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
}

table th, table td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid #dee2e6;
}

table th {
  background-color: #f8f9fa;
  font-weight: 600;
}
</style>
