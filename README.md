# CITRUS: Cluster-based Interactive Truncation for Retrieval Using Semantics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bigwisu/citrus/blob/main/citrus_dashboard.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**CITRUS** is a Python framework for **Scientometric Retrieval** that replaces "Black Box" keyword searches with auditable, vector-based semantic screening.

It introduces the **HiLAT (Human-in-the-Loop Adaptive Truncation)** protocol, allowing researchers to calibrate the "Semantic Cliff" of a literature search visually, rather than relying on arbitrary similarity thresholds.

> **Paper:** *From Static Keywords to Adaptive Vectors: The CITRUS Framework for LLM-Empowered Scientometric Retrieval* (Under Review, *Scientometrics*, 2026).

---

## Quick Start (No Coding Required)

You do not need to install Python locally. The entire framework runs in the cloud via Google Colab.

1.  **Get an API Key:**
    *   **Easiest:** [OpenAI API Key](https://platform.openai.com/) (Standard accessibility).
    *   **Fastest/Scientific:** [IBM Cloud API Key](https://cloud.ibm.com/) (For high-volume processing using the `slate-125m` model used in the paper).
2.  **Launch the Dashboard:**
    Click the "Open in Colab" badge above.
3.  **Run:**
    Click "Play" on the setup cell, enter your API key, and input your search query.

---

## Repository Structure

This repository is organized into the **Interface** (for users) and the **Engine** (for developers).

| File | Description | Audience |
| :--- | :--- | :--- |
| `citrus_dashboard.ipynb` | **The User Interface.** A clean, low-code notebook with form fields and visualizers. Loads the engine in the background. | **Researchers / Reviewers** |
| `citrus_engine.py` | **The Core Logic.** Contains the embedding pipelines, Jaccard math, SQLite-Vec operations, and Knee Detection algorithms. | **Developers** |
| `citrus_dev_scratchpad.ipynb` | **The Lab Notebook.** The raw, chronological record of the experiments conducted for the manuscript, including the full 52k corpus ingestion. | **Auditors** |

---

## Key Features

### 1. Dual-Mode Vectorization
CITRUS supports two operational modes to balance accessibility with scientific rigor:
*   **IBM Granite (`slate-125m`):** Optimized for retrieval tasks. High throughput (~50k docs/12 mins). Used for the manuscript results.
*   **OpenAI (`text-embedding-3`):** Accessible to general users. CITRUS automatically enforces **Matryoshka Representation Learning (MRL)** to truncate vectors to $d=768$, ensuring topological consistency with the Granite model.

### 2. HiLAT Calibration (The "Elbow" Plot)
Instead of returning "Top 100" papers, CITRUS calculates the **Similarity Decay Curve** and identifies the mathematical "Knee" (Elbow).
*   **Automated Suggestion:** The system suggests a cut-off rank (e.g., $n=25$).
*   **Human Audit:** The researcher reviews the "Drop Zone" (papers immediately below the cut-off) to validate or adjust the boundary.

### 3. Cluster Orthogonality Check
Includes tools to generate **Jaccard Similarity Heatmaps**, verifying that your search clusters (e.g., "Theoretical" vs. "Applied") are semantically distinct.

---

## Cloud-Native Architecture

The CITRUS framework was architected, developed, and validated entirely within **Google Colab**. 

*   **Zero Local Footprint:** No local GPU or high-performance workstation is required.
*   **Vector Database:** Utilizes `sqlite-vec` for efficient, in-process vector storage that fits within standard Colab RAM/Disk limits.
*   **Reproducibility:** The environment dependencies are standardized. If it runs in our Colab, it will run in yours.

---

## Citation

If you use CITRUS in your research, please cite the framework as:

```bibtex
@article{Suntoyo2026CITRUS,
  title={From Static Keywords to Adaptive Vectors: The CITRUS Framework for LLM-Empowered Scientometric Retrieval},
  author={Suntoyo, Wisudanto C. and Sunitiyoso, Yos and Siallagan, Manahan P. and Hermawan, Pri},
  journal={Scientometrics (Submitted)},
  year={2026}
}
```

---

## Performance Note
*   **Large Corpora (>10k docs):** We strongly recommend using the **IBM Granite** mode. Standard Generative API endpoints (OpenAI/Gemini) may experience timeouts or rate-limiting when processing massive datasets in a single session.
*   **Gemini Users:** Due to current API latency, Gemini is supported via Batch API only in the `scratchpad`, not the live dashboard.

## Legal & Copyright Compliance (The "Content-Decoupled" Artifact)

To comply with the [Scopus API Terms of Use](https://dev.elsevier.com/) regarding the mass redistribution of proprietary content, the `citrus_epistemic_artifact.sqlite` file in this repository is **Content-Decoupled**.

*   ✅ **Included:** Vector Embeddings ($\mathbb{R}^{768}$), Cluster Assignments, CITRUS Scores, Bibliometric Metadata (DOI, Title, Year, Author).
*   ❌ **Redacted:** The raw Abstract text.

### How to Restore the Text ("Rehydration")
Researchers with valid institutional access to Scopus can restore the full textual corpus using the provided script. This shifts the data retrieval from "Redistribution" to "Authorized API Consumption."

1.  Ensure you have a valid Scopus API Key.
2.  Run the script:
    ```bash
    python citrus_rehydrate.py
    ```
3.  The script will iterate through the DOIs in the database, fetch the abstracts using your credentials, and repopulate the SQLite file locally.