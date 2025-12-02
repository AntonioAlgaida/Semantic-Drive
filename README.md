# Semantic-Drive
**Democratizing Long-Tail Data Curation via Open-Vocabulary Grounding and Neuro-Symbolic VLM Consensus**

![Status](https://img.shields.io/badge/Status-Research_Prototype-blue) ![Python](https://img.shields.io/badge/Python-3.10-green) ![Hardware](https://img.shields.io/badge/Hardware-RTX_3090-purple)

## Abstract
**Semantic-Drive** is a local-first, privacy-preserving framework for mining safety-critical edge cases from raw autonomous vehicle logs. Unlike cloud-based auto-labelers (e.g., Waymo) or end-to-end driving agents (e.g., DriveGPT4), this project runs entirely on consumer hardware (NVIDIA RTX 3090).

It utilizes a **Neuro-Symbolic Architecture**:
1.  **Grounding:** Real-time Open-Vocabulary Segmentation (**YOLOE-11L**) identifies hazards based on the **WOD-E2E Taxonomy**.
2.  **Reasoning:** A Vision-Language Model (**Qwen3-VL / Kimi-Thinking**) performs forensic scene analysis using Chain-of-Thought (CoT).
3.  **Consensus:** A Multi-Model "Judge" synthesizes reports to eliminate hallucinations.

## Architecture
*WIP*

The pipeline transforms "Dark Data" (unsearchable video) into a structured Semantic Database:
*   **Input:** Raw NuScenes Camera Feeds (Front-Hemisphere).
*   **Process:** YOLO Inventory Injection -> VLM Skepticism Policy -> Schema Enforcement.
*   **Output:** Rich JSON metadata describing topology, causality, and risk.

## Getting Started

### 1. Prerequisites
*   **Hardware:** NVIDIA GPU (24GB VRAM recommended for Qwen3-30B).
*   **Software:** [LM Studio](https://lmstudio.ai/) running locally on port `1234`.
*   **Data:** [NuScenes v1.0-mini](https://www.nuscenes.org/download) dataset.

### 2. Installation
```bash
# Clone repository
git clone https://github.com/AntonioAlgaida/Semantic-Drive.git
cd Semantic-Drive

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Edit `src/config.py` to point to your dataset:
```python
NUSCENES_DATAROOT = "/path/to/your/nuscenes"
```

## Usage

### Phase 1: Interactive Grounding (Unit Test)
Verify that YOLOE-11 is correctly detecting "Long-Tail" objects (e.g., debris, construction drums) using the interactive notebook:
```bash
jupyter notebook notebooks/03_yoloe_interactive.ipynb
```

### Phase 2: The Data Factory (Batch Mining)
Launch the mining agent. This connects to your local LM Studio instance.
```bash
# Example: Mining with Kimi-Thinking
python -m src.main --model "kimi-vl-thinking" --output_name "kimi_run1" --verbose
```

### Phase 3: Analysis Dashboard
Visualize the mined scenarios, reasoning traces, and visual grounding side-by-side.
```bash
jupyter notebook notebooks/04_results_viewer.ipynb
```

## Methodology (The "Scenario DNA")
We enforce a strict JSON schema derived from the **Waymo Open Dataset for End-to-End Driving (WOD-E2E)** taxonomy. The system detects:
*   **ODD:** Weather, Lighting (Glare), Sensor Fidelity (Droplets).
*   **Topology:** Map Divergence, Construction Diversions.
*   **Dynamics:** VRU Intent (Hesitation), Vehicle Aggression (Cut-ins).

## Citation
If you use this work, please cite the project repository.
*This project builds upon concepts from DriveGPT4, WOD-E2E, and recent advances in Neuro-Symbolic AI.*

---
**Author:** Antonio Guillen-Perez