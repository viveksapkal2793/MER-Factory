---
layout: default
title: API Reference
description: Complete API reference for MER-Factory classes and functions
---

# API Reference

Complete reference documentation for MER-Factory's core classes, functions, and modules.

## Core Modules

### `main.py` - CLI Entry Point

The main entry point for MER-Factory command-line interface.

#### Functions

##### `main_orchestrator(config: AppConfig) -> None`

Main orchestration function that coordinates the entire processing pipeline.

**Parameters:**
- `config` (AppConfig): Application configuration object containing all processing parameters

**Process:**
1. Runs feature extraction phase (FFmpeg, OpenFace)
2. Executes main processing pipeline via LangGraph
3. Handles concurrent processing for multiple files

TODO: Coming soon...