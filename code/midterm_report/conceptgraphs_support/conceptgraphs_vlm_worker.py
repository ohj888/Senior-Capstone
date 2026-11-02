"""Persistent local VLM worker for the original ConceptGraphs graph builder.

The mapping environment talks JSON over stdin/stdout to a separate local Python
environment. This lets the original Linux stack use the existing Windows CUDA
Qwen/SmolVLM environment without installing conflicting Transformers versions.
There is no HTTP server, GPT client, or remote inference endpoint.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

# Workers are launched by absolute script path, independently of their cwd.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "conceptgraphs_support"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    with contextlib.redirect_stdout(sys.stderr):
        import lora_baseline as lb
        from . import scene_graph_pipeline as sg
        from . import relation_choices as rc
        config = rc.inference_settings(config)
        sg.validate_variant(config["project_root"], config["variant"])
        audit = sg.split_audit({"scene": config["scene"]}, config["variant"], config.get("split", "validation"))
        if config.get("split", "validation") in ("validation", "test") and any(
                row.get("adapter_used") and row.get("training_overlap") for row in audit.values()):
            raise ValueError("Selected scene/group overlaps this adapter's training split")
        predictor = sg.LocalPredictor(config["project_root"], config["variant"], config["task"],
                                      config.get("loading"), config.get("generation"))
        predictor.__enter__()
    print(json.dumps({"ready": True, "task": config["task"]}), flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("op") == "close":
                break
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    if request["op"] == "object":
                        image = lb.load_crop(request["view"], config["data_root"], config.get("object_image_max_side", 384))
                        prompt = config.get("object_prompt", lb.PROMPT)
                    elif request["op"] == "relationship":
                        image = lb.relation_image(request["view"], config["data_root"], config.get("relation_image_max_side", 512))
                        prompt = config.get("relation_prompt", lb.RELATION_PROMPT)
                    else:
                        raise ValueError("Worker operation must be object or relationship")
                    choices = rc.LABELS if request["op"] == "relationship" and config.get("relationship_response_format") == rc.FORMAT else None
                    answer = predictor.predict(image, prompt, choices=choices)
                    if choices is not None:
                        raw = answer["prediction"]
                        choice = rc.parse_choice(raw)
                        if choice is None:
                            raise ValueError(f"Constrained relationship output is not a valid choice: {raw!r}")
                        answer.update(prediction=choice, raw_prediction=raw,
                                      choice_index=rc.LABELS.index(choice)+1, response_format=rc.FORMAT,
                                      constrained_decoding=True)
                print(json.dumps({"ok": True, **answer}), flush=True)
            except Exception as exc:
                print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            predictor.__exit__(None, None, None)


if __name__ == "__main__":
    main()
