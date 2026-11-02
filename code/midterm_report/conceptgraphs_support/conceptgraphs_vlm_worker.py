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
        from . import object_choices as oc
        config = rc.inference_settings(config)
        config = oc.inference_settings(config, config["data_root"], config["scene"])
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
                    relationship = request["op"] == "relationship"
                    choices = (rc.LABELS if config.get("relationship_response_format") == rc.FORMAT else None) if relationship else config.get("object_choices")
                    method = config.get("relationship_choice_decoding", "constrained_greedy") if relationship else "constrained_greedy"
                    answer = predictor.predict(image, prompt, choices=choices, choice_decoding=method)
                    if relationship and choices is not None and config.get("verify_swapped_boxes"):
                        view = request["view"]
                        swapped_view = {**view, "bbox_a": view["bbox_b"], "bbox_b": view["bbox_a"]}
                        swapped_image = lb.relation_image(swapped_view, config["data_root"], config.get("relation_image_max_side", 512))
                        swapped = predictor.predict(swapped_image, prompt, choices=choices, choice_decoding=method)
                        answer.update(swapped_prediction=swapped,
                                      swap_consistent=swapped["prediction"] == rc.SWAPPED_LABELS[answer["prediction"]],
                                      generation_seconds=answer["generation_seconds"]+swapped["generation_seconds"])
                    if choices is not None:
                        raw = answer["prediction"]
                        if raw not in choices:
                            raise ValueError(f"Decoder violated the exact output vocabulary: {raw!r}")
                        answer.update(raw_prediction=raw, choice_index=list(choices).index(raw)+1,
                                      response_format=rc.FORMAT if relationship else oc.FORMAT,
                                      constrained_decoding=True)
                        if not relationship:
                            answer.update(class_id=-1 if raw == oc.ABSTAIN else config["object_class_ids"][raw],
                                          abstained=raw == oc.ABSTAIN,
                                          vocabulary_sha256=config["object_vocabulary_sha256"])
                print(json.dumps({"ok": True, **answer}), flush=True)
            except Exception as exc:
                print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            predictor.__exit__(None, None, None)


if __name__ == "__main__":
    main()
