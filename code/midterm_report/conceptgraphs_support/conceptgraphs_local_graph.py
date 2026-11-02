"""Run ConceptGraphs' actual build_scenegraph with local Qwen/SmolVLM answers.

The original builder performs pruning, overlap computation, spanning forests,
pair traversal and graph serialization. Its ChatCompletion dependency is replaced
in-process by a local worker; an OpenAI client is never instantiated.
"""
from contextlib import contextmanager
import importlib
from importlib.machinery import ModuleSpec
import csv
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import types

from . import conceptgraphs_bridge as cg
from . import scene_graph_pipeline as sg


def _windows_path(path):
    if sys.platform == "win32":
        return str(Path(path).resolve())
    return subprocess.check_output(["wslpath", "-w", str(Path(path).resolve())], text=True).strip()


class LocalWorker:
    def __init__(self, settings, variant, task, output):
        self.settings, self.variant, self.task, self.output = settings, variant, task, Path(output)

    def __enter__(self):
        cfg = self.settings["local_graph"]
        interpreter = cfg["worker_python"]
        windows_worker = interpreter.lower().endswith(".exe")
        convert = _windows_path if windows_worker else lambda value: str(Path(value).resolve())
        root = Path(__file__).resolve().parents[3]
        variant = dict(self.variant)
        for key in ("object_run", "relationship_run"):
            if variant.get(key):
                if re.match(r"^[A-Za-z]:[\\/]", variant[key]):
                    if not windows_worker:
                        variant[key] = subprocess.check_output(["wslpath", "-u", variant[key]], text=True).strip()
                else:
                    variant[key] = convert(root / variant[key])
        worker_config = {"project_root": convert(root), "data_root": convert(self.settings["data_root"]),
                         "variant": variant, "task": self.task,
                         "scene": self.settings["scene"], "split": cfg.get("split", "validation"),
                         "loading": cfg.get("loading", {}), "generation": cfg.get("generation", {}),
                         "object_image_max_side": cfg.get("object_image_max_side", 384),
                         "relation_image_max_side": cfg.get("relation_image_max_side", 512)}
        for key in ("relationship_response_format", "relationship_choice_decoding", "object_response_format", "verify_swapped_boxes"):
            if key in cfg:
                worker_config[key] = cfg[key]
        if cfg.get("relation_choice_prompt"):
            worker_config["relation_choice_prompt"] = cfg["relation_choice_prompt"]
        for key in ("object_prompt", "relation_prompt"):
            if key in cfg:
                worker_config[key] = cfg[key]
        config_path = self.output / f"{self.task}_worker.json"
        cg._json(config_path, worker_config)
        self.log = (self.output / f"{self.task}_worker.log").open("w", encoding="utf-8")
        source = Path(__file__).with_name("conceptgraphs_vlm_worker.py")
        self.process = subprocess.Popen([interpreter, "-u", convert(source), "--config", convert(config_path)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, encoding="utf-8", bufsize=1)
        try:
            reply = self._read()
            if not reply.get("ready"):
                raise RuntimeError(f"Local {self.task} worker did not become ready")
        except BaseException:
            self.process.terminate()
            self.process.wait()
            self.log.close()
            raise
        return self

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Local {self.task} worker exited; see {self.output / (self.task + '_worker.log')}")
        return json.loads(line)

    def request(self, op, view):
        self.process.stdin.write(json.dumps({"op": op, "view": view}) + "\n")
        self.process.stdin.flush()
        result = self._read()
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Local worker failed"))
        result.pop("ok", None)
        return result

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.process.poll() is None:
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
            try:
                code = self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                code = self.process.wait()
            if code and exc_type is None:
                raise RuntimeError(f"Local {self.task} worker exited with status {code}; see its log")
        finally:
            self.log.close()
            self.process.stdin.close()
            self.process.stdout.close()


@contextmanager
def _local_completion_module(callback):
    """Supply only the interface used by upstream; prevent remote fallback."""
    previous = sys.modules.get("openai")
    stub = types.ModuleType("openai")
    # Transformers checks optional packages with find_spec() during import.
    # A manually created module otherwise has __spec__=None and breaks that
    # check before the upstream graph builder can use our local callback.
    stub.__spec__ = ModuleSpec("openai", loader=None)
    stub.ChatCompletion = types.SimpleNamespace(create=callback)
    sys.modules["openai"] = stub
    try:
        yield stub
    finally:
        if previous is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous


def run_local_graph(settings, manifest):
    from .relation_choices import inference_settings
    from .object_choices import inference_settings as object_settings
    from . import edge_validation as ev
    cfg = inference_settings(settings["local_graph"])
    cfg = object_settings(cfg, settings["data_root"], manifest["scene"])
    cfg["edge_validation"] = ev.settings(cfg, manifest)
    settings = {**settings, "local_graph": cfg}
    if not cfg.get("variants"):
        raise ValueError("Select at least one local VLM variant")
    min_views = cfg.get("min_views_per_object", 2)
    if not isinstance(min_views, int) or min_views < 0:
        raise ValueError("min_views_per_object must be a nonnegative integer")
    mapfile = cg.map_path(settings, manifest)
    if not mapfile.is_file():
        raise FileNotFoundError(f"Scene {manifest['scene']} has no completed 3D map at {mapfile}. Run ACTION='all' or ACTION='map' for this scene before ACTION='graph'.")
    canonical = cg.load_map_objects(mapfile, manifest)
    cfg["observation_selection_policy"] = canonical["metadata"]["observation_selection"]
    objects = canonical["objects"]
    if not objects:
        raise ValueError("ConceptGraphs mapping produced no objects. Inspect detection/fusion before graph construction.")
    output_root = Path(settings["run_dir"]).parent / "scene_graphs"
    summary = []
    for variant in cfg["variants"]:
        name = variant["name"]
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Variant name must be a single directory name")
        output = output_root / name
        suffix = 1
        while output.exists():
            output = output_root / f"{name}_{suffix:03d}"
            suffix += 1
        output.mkdir(parents=True, exist_ok=False)
        record = {"complete": False, "variant": variant, "scene": manifest["scene"],
                  "mapfile": str(mapfile), "backend": "local_vlm",
                  "construction": "conceptgraph.scenegraph.build_scenegraph_cfslam.build_scenegraph",
                  "settings": cfg}
        cg._json(output / "run.json", record)
        try:
            nodes = []
            responses = output / "cfslam_gpt-4_responses"  # Upstream filename only; contents are local.
            responses.mkdir()
            with LocalWorker(settings, variant, "objects", output) as worker:
                for obj in objects:
                    views = sorted(obj["observations"], key=lambda view: (-view.get("confidence", 0), -sg._area(view)))
                    views = sg._limit(views, cfg.get("object_views", 3), "object_views")
                    predictions = [{**view, **worker.request("object", view)} for view in views]
                    result = sg.vote([p["prediction"] for p in predictions], cfg.get("object_choices"))
                    tag = result["label"] or "invalid"
                    node = {"id": obj["id"], "tag": tag, "votes": result, "predictions": predictions,
                            "status": "abstained" if tag == "unknown" else "unresolved" if tag == "invalid" else "predicted"}
                    if cfg.get("object_choices") is not None:
                        node["class_id"] = -1 if tag in ("unknown", "invalid") else cfg["object_class_ids"][tag]
                    nodes.append(node)
                    # Upstream only prunes fail/invalid; preserve unknown in the
                    # audit file while giving its builder the matching sentinel.
                    upstream_tag = "invalid" if tag == "unknown" or len(obj["observations"]) < min_views else tag
                    payload = {"id": int(obj["id"]), "captions": [p["prediction"] for p in predictions],
                               "response": json.dumps({"summary": upstream_tag, "possible_tags": [upstream_tag], "object_tag": upstream_tag}),
                               "backend": "local_vlm", "variant": name}
                    cg._json(responses / f"{obj['id']}.json", payload)
            cg._json(output / "local_node_predictions.json", nodes)
            # Every original ID has a response file, preserving the upstream
            # response-index correspondence even for an unresolved object tag.
            retained = [obj for obj, node in zip(objects, nodes)
                        if node["tag"].lower() not in ("fail", "invalid", "unknown") and len(obj["observations"]) >= min_views]
            raw_pairs = []
            pair_limit = cfg.get("pair_views", 2)
            from .relation_choices import UPSTREAM_RELATIONS
            conversion = UPSTREAM_RELATIONS
            with LocalWorker(settings, variant, "relationships", output) as worker:
                def completion(**kwargs):
                    text = kwargs["messages"][-1]["content"]
                    query = json.loads(text[text.index('{"object1"'):])
                    a, b = query["object1"]["id"], query["object2"]["id"]
                    first, second = retained[a], retained[b]
                    av = {v["rgb"]: v for v in first["observations"]}
                    bv = {v["rgb"]: v for v in second["observations"]}
                    views = [{"id": f"{a}:{b}", "scene": manifest["scene"], "rgb": rgb,
                              "frame_id": av[rgb]["frame_id"], "bbox_a": [int(round(x)) for x in av[rgb]["bbox"]],
                              "bbox_b": [int(round(x)) for x in bv[rgb]["bbox"]]}
                             for rgb in sorted(av.keys() & bv.keys()) if av[rgb]["bbox"] != bv[rgb]["bbox"]]
                    views.sort(key=lambda v: -min(sg._area({"bbox": v["bbox_a"]}), sg._area({"bbox": v["bbox_b"]})))
                    predictions = [{**view, **worker.request("relationship", view)}
                                   for view in sg._limit(views, pair_limit, "pair_views")]
                    vote = sg.vote([p["prediction"] for p in predictions], list(conversion))
                    validation = ev.validate(vote["label"], first, second, vote, cfg["edge_validation"])
                    if cfg.get("verify_swapped_boxes") and any(not p.get("swap_consistent") for p in predictions):
                        validation.update(accepted=False, status="rejected", reason="inconsistent_A_B_swap")
                        validation["issues"].append("inconsistent_A_B_swap")
                    score_tied = any((p.get("choice_score_margin") is not None and p["choice_score_margin"] <= 1e-8)
                                     or (p.get("swapped_prediction", {}).get("choice_score_margin") is not None
                                         and p["swapped_prediction"]["choice_score_margin"] <= 1e-8)
                                     for p in predictions)
                    if score_tied:
                        validation.update(accepted=False, status="rejected", reason="tied_complete_label_scores")
                        validation["issues"].append("tied_complete_label_scores")
                    validation = ev.finalize(validation, cfg["edge_validation"])
                    proposed = conversion.get(vote["label"], "FAIL")
                    keep = not score_tied and (validation["accepted"] or validation["action"] == "flag")
                    relation = proposed if keep else "FAIL"
                    reason = ("no_shared_rgb_evidence" if not views else
                              "unresolved_local_vlm_output" if vote["label"] is None else
                              "edge_validation_rejected" if not keep else
                              "local_vlm_prediction_flagged" if not validation["accepted"] else "local_vlm_pair_prediction")
                    row = {"source": a, "target": b, "source_original_id": first["id"],
                           "target_original_id": second["id"], "query": query,
                           "relation": relation, "proposed_relation": proposed, "edge_validation": validation,
                           "reason": reason, "votes": vote, "predictions": predictions}
                    raw_pairs.append(row)
                    cg._json(output / "local_relation_predictions.json", raw_pairs)
                    return {"choices": [{"message": {"content": json.dumps({"object_relation": relation, "reason": reason})}}]}

                with _local_completion_module(completion) as stub:
                    upstream = importlib.import_module("conceptgraph.scenegraph.build_scenegraph_cfslam")
                    upstream.openai = stub
                    args = upstream.ProgramArgs(mode="build-scenegraph", cachedir=str(output), mapfile=str(mapfile),
                                                device=settings["device"], min_views_per_object=min_views,
                                                downsample_voxel_size=cfg.get("pair_point_distance_m") or settings["mapping"].get("downsample_voxel_size", 0.025))
                    upstream.build_scenegraph(args)
                    upstream.generate_scenegraph_json(args)
            # Preserve untouched upstream outputs, but never count upstream FAIL
            # records as valid predicted edges in the reviewable JSON export.
            with (output / "cfslam_scenegraph_edges.pkl").open("rb") as handle:
                raw_edges = pickle.load(handle)
            graph_nodes = json.loads((output / "scene_graph.json").read_text())
            graph_nodes = [{**node, "original_id": node["id"], "id": i} for i, node in enumerate(graph_nodes)]
            if len(graph_nodes) != len(retained):
                raise ValueError("Upstream and local node pruning disagree; graph IDs would be unsafe")
            tags_by_original = {node["id"]: node["tag"] for node in nodes}
            for node, obj in zip(graph_nodes, retained):
                if (node["object_tag"] != tags_by_original[obj["id"]]
                        or any(abs(float(a)-float(b)) > .051 for a, b in zip(node["bbox_center"], obj["center"]))):
                    raise ValueError("Upstream object order differs from RGB evidence; refusing to misattach edges")
                node["original_id"] = int(obj["id"])
            # Upstream can reject a slow local answer under its original 25 s
            # timeout. Record that final decision as unresolved too.
            decision_file = output / "cfslam_object_relations.json"
            final_decisions = json.loads(decision_file.read_text()) if decision_file.exists() else []
            for row, decision in zip(raw_pairs, final_decisions):
                row["upstream_relation"] = decision["object_relation"]
                if decision["object_relation"] == "FAIL" and row["relation"] != "FAIL":
                    row["reason"] = "upstream_timeout_or_parse_failure"
            cg._json(output / "local_relation_predictions.json", raw_pairs)
            validation_by_pair = {(row["source"], row["target"]): row["edge_validation"] for row in raw_pairs}
            from .relation_choices import edge_semantics
            edges = []
            for a, b, relation in raw_edges:
                if relation == "FAIL":
                    continue
                a, b = int(a), int(b)
                if not (0 <= a < len(retained) and 0 <= b < len(retained)) or a == b:
                    raise ValueError("Upstream returned invalid edge endpoints")
                validation = validation_by_pair[(a, b)]
                if validation["action"] == "reject":
                    raise ValueError("Upstream saved an edge rejected by validation")
                subject, reference, label, directed = edge_semantics(a, b, relation)
                edges.append({"source": a, "target": b, "relation": relation,
                              "subject": subject, "reference": reference, "directed": directed,
                              "edge_validation": validation})
            graph = {"scene": manifest["scene"], "variant": name,
                     "nodes": graph_nodes,
                     "edges": edges,
                     "strict_edges": [edge for edge in edges if edge["edge_validation"]["accepted"]
                                      and edge["edge_validation"]["status"] != "unchecked"],
                     "unresolved_pairs": [p for p in raw_pairs if p.get("upstream_relation", p["relation"]) == "FAIL"],
                     "node_index_to_original_id": {str(i): obj["id"] for i, obj in enumerate(retained)},
                     "construction": record["construction"], "remote_api_calls": 0,
                     "edge_validation_policy": cfg["edge_validation"],
                     "arrow_convention": "subject (supported/contained object) -> supporter/container",
                     "notes": ["Local naming adapters and vote aggregation replace LLaVA/GPT node captions.",
                               "Local image-pair adapters replace GPT relation answers; original pruning and MST construction run upstream.",
                               "next to and nearby extend the original relationship vocabulary.",
                               "Diagnostic mode flags inconsistent predictions; geometry_guard_v1 optionally rejects them.",
                               "No saved prediction or heuristic pass establishes ground-truth correctness.",
                               "Raw upstream filenames contain gpt-4 for compatibility, but no GPT API was used."]}
            cg._json(output / "graph.json", graph)
            with (output / "review.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source", "target", "source_original_id", "target_original_id",
                                                           "prediction", "reason", "reviewed_relation", "failure_stage", "notes"])
                writer.writeheader()
                for row in raw_pairs:
                    writer.writerow({**{key: row[key] for key in ("source", "target", "source_original_id", "target_original_id", "reason")},
                                     "prediction": row.get("upstream_relation", row["relation"]),
                                     "reviewed_relation": "",
                                     "failure_stage": "edge_validation" if not row["edge_validation"]["accepted"] else "",
                                     "notes": ("Proposed: " + row["proposed_relation"] + "; check: " + row["edge_validation"]["reason"])})
            record.update(complete=True, nodes=len(graph["nodes"]), edges=len(graph["edges"]),
                          candidate_pairs=len(raw_pairs), unresolved_pairs=len(graph["unresolved_pairs"]))
            cg._json(output / "run.json", record)
            summary.append({"variant": name, "output": str(output), "nodes": record["nodes"],
                            "edges": record["edges"], "candidate_pairs": len(raw_pairs),
                            "unresolved_pairs": record["unresolved_pairs"]})
        except Exception as exc:
            record.update(error=f"{type(exc).__name__}: {exc}")
            cg._json(output / "run.json", record)
            raise
    cg._json(output_root / "comparison.json", summary)
    return summary
