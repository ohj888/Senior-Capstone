# 3D Scene Graph Capstone

This project compares pretrained and LoRA-adapted Qwen and SmolVLM models within a ConceptGraphs-based scene graph pipeline, using OSMa data. It includes object naming, manually reviewed relationships, graph construction, and quantitative and qualitative evaluation.

## Setup files

Located in [`code/setup/`](code/setup/).

| File | What it contains |
| --- | --- |
| [download_dependencies.ipynb](code/setup/download_dependencies.ipynb) | Downloads project dependencies and model files. |
| [explore_osma_bench.ipynb](code/setup/explore_osma_bench.ipynb) | Explores the dataset and previews its scenes and labels. |
| [osma_explorer.py](code/setup/osma_explorer.py) | Helper functions for inspecting OSMa data. |
| [setup_conceptgraphs_runtime.py](code/setup/setup_conceptgraphs_runtime.py) | Sets up the separate ConceptGraphs mapping environment. |
| [conceptgraphs-runtime-requirements.txt](code/setup/conceptgraphs-runtime-requirements.txt) | Package versions for that mapping environment. |

## Main notebooks and training code

Located in [`code/midterm_report/`](code/midterm_report/).

| File | What it contains |
| --- | --- |
| [finetune_vlm.ipynb](code/midterm_report/finetune_vlm.ipynb) | Prepares object examples and trains object-naming adapters. |
| [finetune_vlm2.ipynb](code/midterm_report/finetune_vlm2.ipynb) | Prepares and reviews relationship examples, then trains relationship adapters. |
| [lora_baseline.py](code/midterm_report/lora_baseline.py) | Shared dataset preparation, model loading, training, and evaluation helpers. |
| [baseline_concpeptgrah.ipynb](code/midterm_report/baseline_concpeptgrah.ipynb) | Runs detection, 3D mapping, and scene graph construction. |
| [display_results.ipynb](code/midterm_report/display_results.ipynb) | Displays metrics, reconstructed maps, labeled images, and relationship examples. |

## Pipeline helpers

Located in [`code/midterm_report/conceptgraphs_support/`](code/midterm_report/conceptgraphs_support/).

| File | What it contains |
| --- | --- |
| `conceptgraphs_bridge.py` | Converts dataset inputs and launches ConceptGraphs stages. |
| `conceptgraphs_local_graph.py` | Connects the original graph builder to local VLM predictions. |
| `conceptgraphs_vlm_worker.py` | Loads a VLM and answers object or relationship requests. |
| `conceptgraphs_clip_cache.py` | Loads cached CLIP weights with lower memory overhead. |
| `scene_graph_pipeline.py` | Shared inference, voting, scene selection, and component checks. |
| `object_choices.py` | Object prompts and allowed category names. |
| `relation_choices.py` | Relationship prompts, allowed answers, and arrow-direction rules. |
| `observation_selection.py` | Selects one observation per object/frame and handles competing boxes. |
| `edge_validation.py` | Flags questionable relationships using consistency and geometry checks. |
| `results_report.py` | Calculates metrics and saves evaluation reports. |
| `image_evidence.py` | Creates object crops, labeled scene images, and graph galleries. |
| `__init__.py` | Makes the helper folder a Python package. |

## Typical workflow

Setup → inspect the dataset → train adapters → construct scene graphs → display results.

Downloaded data and dependencies live in `dataset/` and `dependencies/`. Saved adapters, graphs, and reports live in `artifacts/`. These local files are excluded from Git by `.gitignore`.

Relationship arrows and labels are model predictions. Flagged examples require review; valid formatting does not establish correctness.
