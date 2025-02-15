# PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation

## Overview
This repository contains the implementation of **Persuasion-Augmented Chain of Thought (PCoT)**, a method that enhances zero-shot disinformation detection using Large Language Models (LLMs) by incorporating persuasion knowledge.

PCoT operates in two main stages:
1. **Persuasion Detection Step**: The model detects persuasion strategies within the text.
2. **Disinformation Detection Step**: Using the persuasion analysis from the first stage, the model classifies whether a text contains disinformation.


## Repository Structure

```
📂 project_root/
├── LICENSE.txt                           # License file
├── persuasion_groups.yaml                # Names of persuasion strategies
├── persuasion_one_detailed_multistep.sh  # Persuasion detection step
├── run_pcot_one_detailed_multistep.sh    # Final disinformation detection step using PCoT
├── simple_detection.sh                   # Baseline disinformation detection (without persuasion)
├── simple_detection_with_persuasion.sh   # Disinformation detection with PCoT in a single step
├── pyproject.toml                        # Project configuration
├── uv.lock                               # Dependencies lock file
```

## Installation
### Step 1: Install `uv`
Follow the official guide to install `uv`: [Getting Started with uv](https://docs.astral.sh/uv/getting-started/)

### Step 2: Create Project Environment
Once `uv` is installed, use the following command to create the project environment:
```bash
uv sync
```

## Usage

### 1. Run Baseline Disinformation Detection
To perform disinformation detection without persuasion knowledge infusion, run:
```bash
bash simple_detection.sh
```
This script applies competitive zero-shot disinformation detection methods to our PCoT: VaN, Z-CoT, DeF-SpeC.

### 2. Run Full PCoT Process
To execute PCoT in two detailed steps:
#### **Step 1: Persuasion Detection**
```bash
bash persuasion_one_detailed_multistep.sh
```
This step analyzes persuasion strategies in the text and generates structured output for further use.

#### **Step 2: Disinformation Detection with PCoT**
```bash
bash run_pcot_one_detailed_multistep.sh
```
This step utilizes the persuasion analysis from Step 1 to enhance disinformation detection.


### 3. Run PCoT in One Step
If you want to compare to PCoT in one step then follow below instructions.

Note: This method performs significantly worse than two-stage PCoT. To know more about it check our paper.

To apply persuasion-infused disinformation detection in a single step:
```bash
bash simple_detection_with_persuasion.sh
```
This script combines persuasion analysis and disinformation detection into a single inference pass.


## Methodology

PCoT consists of two stages:
- **Stage 1: Persuasion Detection**: 
  - The LLM detects persuasion strategies using predefined categories.
  - Generates structured JSON-like output with binary presence labels and explanations.
  
- **Stage 2: Disinformation Detection**:
  - Uses the persuasion analysis from Stage 1 as additional input.
  - Performs zero-shot binary classification to determine if the text contains disinformation.

For more details, refer to the paper.


## License
This project is licensed under the terms of the MIT License. See `LICENSE.txt` for details.

