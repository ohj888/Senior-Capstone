"""Local VLM scene-graph pilot on a fixed, genuine ConceptGraphs object map.

Mapping is delegated to the sibling conceptgraphs_bridge module. This module never uses semantic
masks, category annotations, or VQA answers as inference inputs. Diagnostics
measure consistency and coverage, not ground-truth graph accuracy.
"""
from collections import Counter
from contextlib import nullcontext
import csv
import copy
import gc
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

import lora_baseline as lb


DEFAULT_EVIDENCE = {
    "candidate_policy": "conceptgraphs_overlap_mst",  # Or 'radius' for a pair-selection ablation.
    "object_views": 3,             # None uses every observation.
    "pair_views": 2,               # None uses every co-visible frame.
    "max_pair_distance_m": 1.5,    # AABB surface distance; None removes this filter.
    "max_pairs": 60,               # None removes the pilot budget.
    "object_image_max_side": 384,
    "relation_image_max_side": 512,
    "duplicate_distance_m": 0.15,  # Diagnostic flag, never an automatic merge.
    "object_prompt": lb.PROMPT,
    "relation_prompt": lb.RELATION_PROMPT,
    "relations": list(lb.RELATIONS),
    "relation_edges": {
        "a_on_b": {"relation": "on", "direction": "a_to_b"},
        "b_on_a": {"relation": "on", "direction": "b_to_a"},
        "a_in_b": {"relation": "in", "direction": "a_to_b"},
        "b_in_a": {"relation": "in", "direction": "b_to_a"},
        "next_to": {"relation": "next_to", "direction": "symmetric"},
        "nearby": {"relation": "nearby", "direction": "symmetric"},
        "none": None,
    },
}
DEFAULT_GENERATION = {"max_new_tokens": 24, "do_sample": False, "seed": 42}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def local_runs(root):
    """List completed saved runs without loading weights or selecting a 'latest'."""
    rows = []
    for task in ("objects", "relationships"):
        for path in sorted((Path(root) / "artifacts" / task).glob("*/*/run.json")):
            record = read_json(path)
            rows.append({"task": task, "model": record.get("model_key"),
                         "run": path.parent.name, "path": str(path.parent),
                         "complete": bool(record.get("complete")),
                         "dataset_fingerprint": record.get("dataset_fingerprint")})
    return rows


def default_variants(root):
    """Explicit model/task conditions. A missing requested adapter is an error.

    SmolVLM has an object adapter in the current repository, but no relationship
    adapter. Its condition name records that relationship inference is pretrained.
    Set relationship_run to a completed SmolVLM run to add that experiment later.
    """
    root = Path(root)
    qobj = str(root / "artifacts/objects/qwen/objects_3epochs")
    sobj = str(root / "artifacts/objects/smolvlm/objects_3epochs")
    qrel = str(root / "artifacts/relationships/qwen/relations_3epochs")
    return [
        {"name": "qwen_pretrained", "model_key": "qwen", "object_run": qobj,
         "relationship_run": qrel, "object_adapter": False, "relationship_adapter": False},
        {"name": "qwen_adapted", "model_key": "qwen", "object_run": qobj,
         "relationship_run": qrel, "object_adapter": True, "relationship_adapter": True},
        {"name": "smolvlm_pretrained", "model_key": "smolvlm", "object_run": sobj,
         "relationship_run": None, "object_adapter": False, "relationship_adapter": False},
        {"name": "smolvlm_objects_adapted_relations_pretrained", "model_key": "smolvlm",
         "object_run": sobj, "relationship_run": None, "object_adapter": True,
         "relationship_adapter": False},
    ]


def select_scenes(data_root, source_manifest, split="validation", scenes=None, max_scenes=1):
    """Use the existing layout split; never re-split adjacent trajectory frames."""
    data_root = Path(data_root)
    manifest = read_json(source_manifest) if not isinstance(source_manifest, dict) else source_manifest
    assignments = manifest["scenes"]
    available = sorted(name for name, row in assignments.items() if split is None or row["split"] == split)
    if scenes is not None:
        invalid = set(scenes) - set(available)
        if invalid:
            raise ValueError(f"Scenes absent from requested split {split}: {sorted(invalid)}")
        available = list(scenes)
    if max_scenes is not None:
        if max_scenes < 1:
            raise ValueError("max_scenes must be positive or None.")
        available = available[:max_scenes]
    if not available:
        raise ValueError("No selected scenes. Check dataset root, manifest and split.")
    for name in available:
        if not (safe_source(data_root, name) / "traj.txt").is_file():
            raise FileNotFoundError(f"Missing local trajectory for {name}")
    return [{"scene": name, **assignments[name]} for name in available]


def safe_source(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evidence source must be inside the dataset root.")
    return path


def _limit(rows, limit, name):
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError(f"{name} must be a positive integer or None.")
    return rows if limit is None else rows[:limit]


def _area(observation):
    x0, y0, x1, y1 = observation["bbox"]
    return (x1 - x0) * (y1 - y0)


def _bounds(obj):
    if "bbox_min" in obj and "bbox_max" in obj:
        low, high = np.asarray(obj["bbox_min"], float), np.asarray(obj["bbox_max"], float)
    else:
        center, extent = np.asarray(obj["center"], float), np.asarray(obj["extent"], float)
        low, high = center - extent / 2, center + extent / 2
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high < low):
        raise ValueError(f"Invalid 3D bounds for object {obj.get('id')}")
    return low, high


def box_distance(a, b):
    alo, ahi = _bounds(a)
    blo, bhi = _bounds(b)
    return float(np.linalg.norm(np.maximum(np.maximum(alo - bhi, blo - ahi), 0)))


def prepare_evidence(canonical_map, data_root, settings=None):
    """Freeze nodes, crop observations, 3D candidates and co-visible pair images.

    Each unordered node pair is evaluated in a shared RGB frame. A 3D-nearby
    pair without such evidence is retained in diagnostics, never fabricated into
    an image or passed off as a negative relation.
    """
    cfg = copy.deepcopy({**DEFAULT_EVIDENCE, **(settings or {})})
    unknown = set(cfg) - set(DEFAULT_EVIDENCE)
    if unknown:
        raise ValueError(f"Unknown evidence settings: {sorted(unknown)}")
    for name in ("object_views", "pair_views", "max_pairs"):
        _limit([], cfg[name], name)
    if cfg["max_pair_distance_m"] is not None and (not np.isfinite(cfg["max_pair_distance_m"]) or cfg["max_pair_distance_m"] < 0):
        raise ValueError("max_pair_distance_m must be nonnegative or None.")
    if not np.isfinite(cfg["duplicate_distance_m"]) or cfg["duplicate_distance_m"] < 0:
        raise ValueError("duplicate_distance_m must be finite and nonnegative.")
    for key in ("object_image_max_side", "relation_image_max_side"):
        if cfg[key] is not None and (not isinstance(cfg[key], int) or cfg[key] < 1):
            raise ValueError(f"{key} must be a positive integer or None.")
    if cfg["candidate_policy"] not in ("conceptgraphs_overlap_mst", "radius"):
        raise ValueError("candidate_policy must be conceptgraphs_overlap_mst or radius.")
    if len({lb.normalize_label(label) for label in cfg["relations"]}) != len(cfg["relations"]):
        raise ValueError("Relation labels must be unique after normalization.")
    for label in cfg["relations"]:
        if label not in cfg["relation_edges"]:
            raise ValueError(f"Configure relation_edges for relation {label!r}.")
        rule = cfg["relation_edges"][label]
        if rule is not None and (not rule.get("relation") or rule.get("direction") not in ("a_to_b", "b_to_a", "symmetric")):
            raise ValueError(f"Invalid relation_edges rule for {label!r}.")
    scene = canonical_map["scene"]
    nodes, seen_ids = [], set()
    image_sizes = {}
    for original in canonical_map["objects"]:
        obj = copy.deepcopy(original)
        obj["id"] = str(obj["id"])
        if obj["id"] in seen_ids:
            raise ValueError(f"Duplicate object ID: {obj['id']}")
        seen_ids.add(obj["id"])
        low, high = _bounds(obj)
        obj.update(bbox_min=low.tolist(), bbox_max=high.tolist(), center=((low+high)/2).tolist(),
                   extent=(high-low).tolist())
        unique = {}
        for entry in obj.get("observations", []):
            observation = dict(entry)
            relative = observation["rgb"]
            source = safe_source(data_root, relative)
            if relative not in image_sizes:
                with Image.open(source) as im:
                    image_sizes[relative] = im.size
            width, height = image_sizes[relative]
            box = observation["bbox"]
            if len(box) != 4 or not np.isfinite(box).all():
                raise ValueError("Evidence box must contain four finite coordinates.")
            x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
            x1, y1 = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"Empty crop for {obj['id']} in {relative}")
            observation.update(bbox=[x0, y0, x1, y1], frame_id=int(observation["frame_id"]))
            key = (relative, observation["frame_id"])
            if key not in unique or _area(observation) > _area(unique[key]):
                unique[key] = observation
        obj["observations"] = sorted(unique.values(), key=lambda r: (-_area(r), r["rgb"]))
        obj["naming_views"] = _limit(obj["observations"], cfg["object_views"], "object_views")
        nodes.append(obj)
    nodes.sort(key=lambda r: r["id"])
    original_pairs = {}
    if cfg["candidate_policy"] == "conceptgraphs_overlap_mst":
        if "candidate_pairs" not in canonical_map:
            raise ValueError("Canonical map has no ConceptGraphs candidate_pairs. Run the bridge export stage, or explicitly select candidate_policy='radius' for an ablation.")
        for pair in canonical_map["candidate_pairs"]:
            a, b = str(pair["source"]), str(pair["target"])
            if a == b or a not in seen_ids or b not in seen_ids:
                raise ValueError("Invalid ConceptGraphs candidate pair endpoint.")
            original_pairs[tuple(sorted([a, b]))] = pair
    pairs, no_common_view, out_of_range, candidate_log = [], [], 0, []
    for a, b in itertools.combinations(nodes, 2):
        distance = box_distance(a, b)
        if cfg["candidate_policy"] == "conceptgraphs_overlap_mst" and (a["id"], b["id"]) not in original_pairs:
            candidate_log.append({"a": a["id"], "b": b["id"], "distance_m": distance, "status": "not_selected_by_conceptgraphs_mst"})
            continue
        if cfg["candidate_policy"] == "radius" and cfg["max_pair_distance_m"] is not None and distance > cfg["max_pair_distance_m"]:
            out_of_range += 1
            candidate_log.append({"a": a["id"], "b": b["id"], "distance_m": distance, "status": "outside_distance_filter"})
            continue
        a_views = {(r["rgb"], r["frame_id"]): r for r in a["observations"]}
        b_views = {(r["rgb"], r["frame_id"]): r for r in b["observations"]}
        pair_id = hashlib.sha256(json.dumps([scene, a["id"], b["id"]]).encode()).hexdigest()[:20]
        views = []
        for key in sorted(a_views.keys() & b_views.keys()):
            av, bv = a_views[key], b_views[key]
            if av["bbox"] == bv["bbox"]:
                continue
            views.append({"id": pair_id, "scene": scene, "rgb": key[0], "frame_id": key[1],
                          "bbox_a": av["bbox"], "bbox_b": bv["bbox"],
                          "visible_area": min(_area(av), _area(bv))})
        views.sort(key=lambda r: (-r["visible_area"], r["rgb"]))
        if not views:
            no_common_view.append({"a": a["id"], "b": b["id"], "distance_m": distance})
            candidate_log.append({"a": a["id"], "b": b["id"], "distance_m": distance, "status": "no_distinct_common_rgb_evidence"})
            continue
        pairs.append({"id": pair_id, "a": a["id"], "b": b["id"], "distance_m": distance,
                      "conceptgraphs_overlap": original_pairs.get((a["id"], b["id"]), {}).get("overlap"),
                      "views": _limit(views, cfg["pair_views"], "pair_views")})
    pairs.sort(key=lambda r: (r["distance_m"], r["a"], r["b"]))
    eligible_count = len(pairs)
    unbudgeted = pairs
    pairs = _limit(pairs, cfg["max_pairs"], "max_pairs")
    selected_ids = {row["id"] for row in pairs}
    candidate_log.extend({"a": row["a"], "b": row["b"], "distance_m": row["distance_m"],
                          "status": "selected" if row["id"] in selected_ids else "excluded_by_pair_budget"}
                         for row in unbudgeted)
    core = {"schema_version": 1, "scene": scene, "settings": cfg, "nodes": nodes, "pairs": pairs,
            "candidate_log": candidate_log,
            "mapping_metadata": canonical_map.get("metadata", {}),
            "coverage": {"all_possible_pairs": len(nodes)*(len(nodes)-1)//2,
                         "outside_distance_filter": out_of_range,
                         "nearby_without_common_view": no_common_view,
                         "eligible_pairs_before_budget": eligible_count,
                         "pairs_excluded_by_budget": eligible_count-len(pairs)},
            "sources": [{"rgb": name, "sha256": hashlib.sha256(safe_source(data_root, name).read_bytes()).hexdigest()}
                        for name in sorted(image_sizes)]}
    return {**core, "fingerprint": lb.fingerprint(core)}


def vote(answers, vocabulary=None):
    """No invented probability: report observed vote support and invalid outputs."""
    labels = [lb.normalize_label(a) for a in answers]
    # Relationship tokens contain underscores, so normalize the allowed labels too.
    allowed = {lb.normalize_label(v): v for v in vocabulary} if vocabulary is not None else None
    valid = [label for label in labels if label and (allowed is None or label in allowed)]
    counts = Counter(valid)
    invalid = len(labels) - len(valid)
    if not counts:
        return {"label": None, "support": 0, "views": len(labels), "agreement": None, "invalid": invalid, "tied": False}
    ordered = sorted(counts.items(), key=lambda r: (-r[1], r[0]))
    label, support = ordered[0]
    tied = len(ordered) > 1 and support == ordered[1][1]
    return {"label": None if tied else (allowed[label] if allowed is not None else label),
            "support": support, "views": len(labels), "agreement": support / len(labels),
            "invalid": invalid, "tied": tied}


class LocalPredictor:
    """One local base/adapter in memory at a time. No API or model downloads."""
    def __init__(self, root, variant, task, loading=None, generation=None):
        self.root, self.variant, self.task = Path(root), variant, task
        self.loading = dict(loading or {})
        self.generation = {**DEFAULT_GENERATION, **(generation or {})}
        self.seed = self.generation.pop("seed")
        self.model = self.processor = None

    def __enter__(self):
        from transformers import set_seed
        set_seed(self.seed)
        key = "object" if self.task == "objects" else "relationship"
        self.disabled = not self.variant[f"{key}_adapter"]
        run = self.variant.get(f"{key}_run")
        model_key = self.variant["model_key"]
        if run is None and not self.disabled:
            raise ValueError(f"{self.variant['name']} requests a {key} adapter without a saved run.")
        use_4bit = self.loading.pop("use_4bit", None)
        if run is not None:
            expected_task = "gt_assisted_region_category" if self.task == "objects" else "visual_pair_relationship"
            self.model, self.processor, self.record = lb.reload_adapter(
                self.root, run, model_key=model_key, task=expected_task,
                loading=self.loading, use_4bit=use_4bit)
        else:
            self.model, self.processor = lb.load_local_base(
                self.root, model_key, use_4bit=True if use_4bit is None else use_4bit,
                loading=self.loading)
            self.record = {"model_key": model_key, "model": lb.MODELS[model_key]}
        self.model.eval()
        return self

    def predict(self, image, prompt):
        import torch
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                    {"type": "text", "text": prompt}]}]
        batch = lb.encode_messages(self.processor, messages).to(self.model.device)
        disabled = self.disabled and hasattr(self.model, "disable_adapter")
        context = self.model.disable_adapter() if disabled else nullcontext()
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        started = time.perf_counter()
        with context, torch.inference_mode():
            generated = self.model.generate(**batch, **self.generation)
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        answer = self.processor.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {"prediction": answer, "generation_seconds": time.perf_counter()-started,
                "input_tokens": int(batch["input_ids"].shape[1]),
                "output_tokens": int(generated.shape[1]-batch["input_ids"].shape[1])}

    def __exit__(self, *exc):
        self.model = self.processor = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _edge(pair, relation, rules=None):
    a, b = pair["a"], pair["b"]
    rule = (DEFAULT_EVIDENCE["relation_edges"] if rules is None else rules).get(relation)
    if rule is None:
        return None
    source, target = (b, a) if rule["direction"] == "b_to_a" else (a, b)
    return {"source": source, "target": target, "relation": rule["relation"],
            "directed": rule["direction"] != "symmetric", "pair_id": pair["id"]}


def validate_variant(root, variant):
    if variant["model_key"] not in lb.MODELS:
        raise ValueError(f"Unsupported model: {variant['model_key']}")
    if not variant["name"] or Path(variant["name"]).name != variant["name"] or variant["name"] in (".", ".."):
        raise ValueError("Variant name must be a single nonempty directory name.")
    records = {}
    for key, task in (("object", "gt_assisted_region_category"), ("relationship", "visual_pair_relationship")):
        run = variant.get(f"{key}_run")
        if variant[f"{key}_adapter"] and run is None:
            raise ValueError(f"Missing {key}_run for enabled adapter")
        if run is not None:
            records[key] = lb.read_run_record(run, model_key=variant["model_key"], task=task)
            if not (Path(run) / "adapter/adapter_model.safetensors").is_file():
                raise FileNotFoundError(f"Missing saved adapter weights at {run}")
    return records


def split_audit(evidence, variant, split="validation"):
    """Make training overlap visible before a pilot is interpreted as held out."""
    result = {}
    selected_group = evidence.get("mapping_metadata", {}).get("scene_assignment", {}).get("group")
    if selected_group is None and variant.get("object_run"):
        source = Path(variant["object_run"]) / "data_manifest.json"
        if source.is_file():
            selected_group = read_json(source).get("scenes", {}).get(evidence["scene"], {}).get("group")
    for key in ("object", "relationship"):
        run = variant.get(f"{key}_run")
        if run is None:
            result[key] = {"adapter_used": False, "saved_scene_split": None}
            continue
        manifest_path = Path(run) / "data_manifest.json"
        if not manifest_path.is_file():
            result[key] = {"adapter_used": variant[f"{key}_adapter"], "saved_scene_split": "unknown"}
            continue
        manifest = read_json(manifest_path)
        assignment = manifest.get("scenes", {}).get(evidence["scene"], {})
        saved_splits = {r["split"] for r in manifest.get("examples", []) if r.get("scene") == evidence["scene"]}
        if assignment.get("split"):
            saved_splits.add(assignment["split"])
        group_training = bool(selected_group) and any(
            r.get("group") == selected_group and r.get("split") == "train" for r in manifest.get("examples", []))
        result[key] = {"adapter_used": variant[f"{key}_adapter"], "saved_scene_split": sorted(saved_splits),
                       "requested_split": split, "scene_group": selected_group,
                       "training_overlap": "train" in saved_splits or group_training}
    return result


def run_variant(root, evidence, data_root, output_dir, variant, *, loading=None, generation=None,
                predictor_factory=LocalPredictor, split="validation"):
    """Run object naming, release GPU, then relation prediction; save raw evidence.

    output_dir must be fresh. Interrupted/failed runs remain marked incomplete.
    No training occurs here: saved adapters are loaded frozen for inference.
    """
    output_dir = Path(output_dir)
    if lb.fingerprint({key: value for key, value in evidence.items() if key != "fingerprint"}) != evidence["fingerprint"]:
        raise ValueError("Frozen evidence was modified. Re-run prepare_evidence before inference.")
    records = validate_variant(root, variant)
    audits = split_audit(evidence, variant, split)
    if split in ("validation", "test") and any(r.get("adapter_used") and r.get("training_overlap") for r in audits.values()):
        raise ValueError("Selected scene appears in adapter training data; choose a held-out scene or explicitly set split='train' for a training diagnostic.")
    # Fail before GPU allocation if the frozen evidence changed on disk.
    for item in evidence["sources"]:
        actual = hashlib.sha256(safe_source(data_root, item["rgb"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Evidence RGB changed after preparation: {item['rgb']}")
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = evidence["settings"]
    record = {"complete": False, "scene": evidence["scene"], "variant": variant,
              "evidence_fingerprint": evidence["fingerprint"], "loading": loading or {},
              "generation": {**DEFAULT_GENERATION, **(generation or {})}, "adapter_runs": records,
              "split_audit": audits, "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    lb.save_json(output_dir / "run.json", record)
    graph = {"schema_version": 1, "scene": evidence["scene"], "variant": variant["name"],
             "evidence_fingerprint": evidence["fingerprint"], "nodes": [], "pairs": [], "edges": []}
    try:
        with predictor_factory(root, variant, "objects", loading, generation) as predictor:
            for node in evidence["nodes"]:
                predictions = []
                for view in node["naming_views"]:
                    crop = lb.load_crop(view, data_root, cfg["object_image_max_side"])
                    predictions.append({"rgb": view["rgb"], "frame_id": view["frame_id"], "bbox": view["bbox"],
                                        **predictor.predict(crop, cfg["object_prompt"])})
                summary = vote([row["prediction"] for row in predictions])
                graph["nodes"].append({"id": node["id"], "label": summary["label"], "center": node["center"],
                                       "extent": node["extent"], "observations": len(node["observations"]),
                                       "votes": summary, "predictions": predictions})
        lb.save_json(output_dir / "object_predictions.json", graph["nodes"])
        if evidence["pairs"]:
            with predictor_factory(root, variant, "relationships", loading, generation) as predictor:
                for pair in evidence["pairs"]:
                    predictions = []
                    for view in pair["views"]:
                        image = lb.relation_image(view, data_root, cfg["relation_image_max_side"])
                        predictions.append({"rgb": view["rgb"], "frame_id": view["frame_id"],
                                            "bbox_a": view["bbox_a"], "bbox_b": view["bbox_b"],
                                            **predictor.predict(image, cfg["relation_prompt"])})
                    summary = vote([row["prediction"] for row in predictions], cfg["relations"])
                    graph["pairs"].append({"id": pair["id"], "a": pair["a"], "b": pair["b"],
                                           "relation": summary["label"], "votes": summary, "predictions": predictions})
                    edge = _edge(pair, summary["label"], cfg["relation_edges"])
                    if edge is not None:
                        graph["edges"].append(edge)
        graph["diagnostics"] = diagnostics(graph, evidence)
        lb.save_json(output_dir / "graph.json", graph)
        export_review(output_dir / "review.csv", graph)
        record.update(complete=True, diagnostics=graph["diagnostics"])
        lb.save_json(output_dir / "run.json", record)
        return graph
    except Exception as exc:
        record.update(error=f"{type(exc).__name__}: {exc}")
        lb.save_json(output_dir / "run.json", record)
        raise


def diagnostics(graph, evidence):
    nodes, pairs = graph["nodes"], graph["pairs"]
    label_for = {node["id"]: node["label"] for node in nodes}
    possible_duplicates = []
    for a, b in itertools.combinations(evidence["nodes"], 2):
        if label_for.get(a["id"]) and label_for.get(a["id"]) == label_for.get(b["id"]):
            distance = box_distance(a, b)
            if distance <= evidence["settings"]["duplicate_distance_m"]:
                possible_duplicates.append({"a": a["id"], "b": b["id"], "distance_m": distance})
    object_answers = sum(n["votes"]["views"] for n in nodes)
    relation_answers = sum(p["votes"]["views"] for p in pairs)
    connected = {end for edge in graph["edges"] for end in (edge["source"], edge["target"])}
    return {"interpretation": "Consistency/coverage diagnostics only; no ground-truth instance graph is available.",
            "nodes": len(nodes), "candidate_pairs_evaluated": len(pairs), "edges": len(graph["edges"]),
            "nodes_without_rgb_evidence": sum(not n["predictions"] for n in nodes),
            "nodes_with_naming_disagreement": sum(n["votes"]["views"] > 1 and n["votes"]["agreement"] != 1 for n in nodes),
            "pairs_with_relation_disagreement": sum(p["votes"]["views"] > 1 and p["votes"]["agreement"] != 1 for p in pairs),
            "unresolved_object_votes": sum(n["label"] is None for n in nodes),
            "unresolved_relation_votes": sum(p["relation"] is None for p in pairs),
            "object_invalid_output_rate": sum(n["votes"]["invalid"] for n in nodes)/object_answers if object_answers else None,
            "relation_invalid_output_rate": sum(p["votes"]["invalid"] for p in pairs)/relation_answers if relation_answers else None,
            "isolated_nodes": sum(n["id"] not in connected for n in nodes),
            "possible_duplicate_nodes": possible_duplicates,
            "coverage": evidence["coverage"],
            "generation_seconds": sum(p.get("generation_seconds", 0) for item in nodes+pairs for p in item["predictions"])}


def export_review(path, graph):
    """Blank human review targets; generated predictions never become truth."""
    columns = ["kind", "id", "a", "b", "prediction", "reviewed_target", "error_stage", "notes"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for node in graph["nodes"]:
            writer.writerow({"kind": "object", "id": node["id"], "prediction": node["label"]})
        for pair in graph["pairs"]:
            writer.writerow({"kind": "relationship", "id": pair["id"], "a": pair["a"], "b": pair["b"],
                             "prediction": pair["relation"]})


def compare_graphs(graphs):
    if len({g["evidence_fingerprint"] for g in graphs}) > 1:
        raise ValueError("VLM comparisons require exactly the same frozen evidence.")
    return [{"variant": graph["variant"], **{key: value for key, value in graph["diagnostics"].items()
             if not isinstance(value, (dict, list))}} for graph in graphs]


def adapter_smoke(root, data_root, variants, *, samples=1, loading=None, generation=None):
    """Actual local generations on saved validation examples; no graph claim.

    This isolates model/processor/adapter loading from unavailable detector or
    mapping dependencies. Targets are retained only in returned audit records.
    The same object and relation examples are used across all variants.
    """
    if not variants:
        raise ValueError("Select at least one variant.")
    for variant in variants:
        validate_variant(root, variant)
    object_source = next((v["object_run"] for v in variants if v.get("object_run")), None)
    relation_source = next((v["relationship_run"] for v in variants if v.get("relationship_run")), None)
    rows_by_task = {}
    for task, source in (("objects", object_source), ("relationships", relation_source)):
        if source:
            manifest = read_json(Path(source) / "data_manifest.json")
            rows_by_task[task] = _limit(lb.evaluation_rows(manifest, "validation", None), samples, "samples")
    results = []
    for variant in variants:
        for task, rows in rows_by_task.items():
            print(f"Local smoke: {variant['name']} / {task} ({len(rows)} example(s))", flush=True)
            with LocalPredictor(root, variant, task, loading, generation) as predictor:
                for row in rows:
                    if task == "objects":
                        image, prompt = lb.load_crop(row, data_root), lb.PROMPT
                    else:
                        image, prompt = lb.relation_image(row, data_root), lb.RELATION_PROMPT
                    results.append({"variant": variant["name"], "task": task, "id": row["id"],
                                    "scene": row["scene"], "target": row["target"],
                                    **predictor.predict(image, prompt)})
    return {"kind": "adapter_component_smoke", "not_a_scene_graph_benchmark": True,
            "generation": {**DEFAULT_GENERATION, **(generation or {})}, "results": results}


def plot_graph(graph):
    """Small local 3D inspection plot, with actual map coordinates."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    positions = {node["id"]: node["center"] for node in graph["nodes"]}
    for node in graph["nodes"]:
        ax.scatter(*node["center"], s=35)
        ax.text(*node["center"], f"{node['id']}: {node['label'] or '?'}", fontsize=8)
    for edge in graph["edges"]:
        start, end = np.array(positions[edge["source"]]), np.array(positions[edge["target"]])
        ax.plot(*np.stack([start, end]).T, alpha=0.5)
        ax.text(*((start+end)/2), edge["relation"], fontsize=7)
    ax.set(xlabel="world x (m)", ylabel="world y (m)", zlabel="world z (m)",
           title=f"{graph['scene']} — {graph['variant']}")
    return fig
