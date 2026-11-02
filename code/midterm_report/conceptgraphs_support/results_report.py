"""Evaluate saved ConceptGraphs maps against observed OSMa semantic RGB-D.

No model inference or pipeline files are changed. The reference is reconstructed
from the exact staged frames and their poses, not a complete-room mesh. Run the
CPU evaluator in the mapping environment (SciPy/OmegaConf are already installed).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "conceptgraphs_support"


DEFAULT_EXCLUDED = ("other", "unknown", "unlabeled", "background", "wall", "floor", "ceiling")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize(name):
    return " ".join(str(name).replace("_", " ").strip(" \t\n.\"'").casefold().split())


def backproject(depth, camera, pose, pixel_stride=4, max_depth_m=10.0):
    """OpenCV pinhole projection into the saved mapper's source-world frame."""
    if pixel_stride < 1 or max_depth_m <= 0:
        raise ValueError("Use a positive pixel stride and depth limit")
    v, u = np.mgrid[0:depth.shape[0]:pixel_stride, 0:depth.shape[1]:pixel_stride]
    z = depth[v, u].astype(np.float64) / camera["png_depth_scale"]
    valid = np.isfinite(z) & (z > 0) & (z <= max_depth_m)
    u, v, z = u[valid], v[valid], z[valid]
    xyz = np.column_stack(((u - camera["cx"]) * z / camera["fx"],
                           (v - camera["cy"]) * z / camera["fy"], z))
    return (xyz @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32), v, u


def voxel_indices(xyz, size):
    if size <= 0:
        raise ValueError("Voxel size must be positive")
    if not len(xyz):
        return np.array([], dtype=np.int64)
    # Keep an actual sample and its label instead of averaging class IDs.
    _, indices = np.unique(np.floor(xyz / size).astype(np.int64), axis=0, return_index=True)
    return np.sort(indices)


def observed_reference(data_root, scene, manifest, mapping_dir, *, pixel_stride, voxel_size, max_depth_m):
    camera = manifest["camera"]
    poses = np.loadtxt(Path(mapping_dir) / "input/scene/traj.txt", ndmin=2).reshape(-1, 4, 4)
    if len(poses) != len(manifest["frames"]):
        raise ValueError("Staged pose/frame count mismatch")
    points, colors, labels = [], [], []
    for row, pose in zip(manifest["frames"], poses):
        rgb_path = Path(data_root) / row["rgb"]
        depth_path = Path(data_root) / row["depth"]
        semantic_path = Path(data_root) / scene / "results" / f"semantic{row['frame_id']:06d}.png"
        with Image.open(depth_path) as image:
            depth = np.asarray(image)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(semantic_path) as image:
            semantic = np.asarray(image)
        if depth.shape != semantic.shape or rgb.shape[:2] != depth.shape:
            raise ValueError(f"RGB/depth/semantic dimensions differ at frame {row['frame_id']}")
        xyz, v, u = backproject(depth, camera, pose, pixel_stride, max_depth_m)
        points.append(xyz); colors.append(rgb[v, u]); labels.append(semantic[v, u].astype(np.int32))
    if not points or not sum(map(len, points)):
        raise ValueError("The selected frames have no valid depth")
    xyz, rgb, labels = np.concatenate(points), np.concatenate(colors), np.concatenate(labels)
    indices = voxel_indices(xyz, voxel_size)
    return {"xyz": xyz[indices], "rgb": rgb[indices], "labels": labels[indices]}


def load_prediction(mapfile, voxel_size):
    # Input is the trusted pickle written by this project's own mapper.
    with gzip.open(mapfile, "rb") as stream:
        saved = pickle.load(stream)
    objects = saved["objects"] if isinstance(saved, dict) else saved
    points, colors, ids, names = [], [], [], {}
    for i, obj in enumerate(objects):
        xyz = np.asarray(obj["pcd_np"], dtype=np.float32)
        rgb = np.asarray(obj["pcd_color_np"], dtype=np.float32)
        valid = np.isfinite(xyz).all(axis=1)
        xyz, rgb = xyz[valid], rgb[valid]
        points.append(xyz); colors.append(np.clip(rgb * 255, 0, 255).astype(np.uint8))
        ids.append(np.full(len(xyz), i, dtype=np.int32))
        names[i] = Counter(map(str, obj.get("class_name", []))).most_common(1)
        names[i] = names[i][0][0] if names[i] else "unknown"
    if not points or not sum(map(len, points)):
        raise ValueError("Saved map has no finite points")
    xyz, rgb, ids = np.concatenate(points), np.concatenate(colors), np.concatenate(ids)
    indices = voxel_indices(xyz, voxel_size)
    return {"xyz": xyz[indices], "rgb": rgb[indices], "object_ids": ids[indices]}, names, len(objects)


def nearest(source, target):
    from scipy.spatial import cKDTree
    if not len(target):
        return np.full(len(source), np.inf), np.zeros(len(source), dtype=np.int64)
    return cKDTree(target).query(source, k=1, workers=1)


def geometry_metrics(pred_to_gt, gt_to_pred, thresholds):
    def finite_mean(values):
        return float(np.mean(values)) if len(values) and np.isfinite(values).all() else None
    accuracy, completeness = finite_mean(pred_to_gt), finite_mean(gt_to_pred)
    result = {"pred_points": len(pred_to_gt), "reference_points": len(gt_to_pred),
              "mean_pred_to_gt_m": accuracy, "mean_gt_to_pred_m": completeness,
              "chamfer_L1_sum_m": accuracy + completeness if accuracy is not None and completeness is not None else None}
    for threshold in thresholds:
        precision = float(np.mean(pred_to_gt <= threshold)) if len(pred_to_gt) else 0.0
        recall = float(np.mean(gt_to_pred <= threshold)) if len(gt_to_pred) else 0.0
        suffix = f"{threshold * 100:g}cm"
        result.update({"precision_" + suffix: precision, "recall_" + suffix: recall,
                       "fscore_" + suffix: 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    return result


def semantic_metrics(gt, predicted, class_ids):
    """GT-reference confusion matrix; final column includes unknown/uncovered."""
    lookup = {int(value): i for i, value in enumerate(class_ids)}
    n = len(lookup)
    if not n:
        raise ValueError("No evaluation classes")
    rows = np.array([lookup[int(v)] for v in gt], dtype=np.int64)
    cols = np.array([lookup.get(int(v), n) for v in predicted], dtype=np.int64)
    matrix = np.bincount(rows * (n + 1) + cols, minlength=n * (n + 1)).reshape(n, n + 1)
    support, predicted_count = matrix.sum(axis=1), matrix[:, :n].sum(axis=0)
    tp = matrix[np.arange(n), np.arange(n)]
    recall = np.divide(tp, support, out=np.zeros(n, dtype=float), where=support > 0)
    union = support + predicted_count - tp
    iou = np.divide(tp, union, out=np.zeros(n, dtype=float), where=union > 0)
    active = support > 0
    total = int(support.sum())
    metrics = {"mAcc": float(recall[active].mean()) if active.any() else None,
               "mIoU": float(iou[active].mean()) if active.any() else None,
               "f_mIoU": float((iou * support).sum() / total) if total else None,
               "point_accuracy": float(tp.sum() / total) if total else None,
               "unknown_or_uncovered_rate": float(matrix[:, -1].sum() / total) if total else None,
               "evaluated_classes": int(active.sum()), "reference_points": total}
    per_class = [{"class_id": int(class_ids[i]), "support": int(support[i]),
                  "predicted": int(predicted_count[i]), "recall": float(recall[i]), "IoU": float(iou[i])}
                 for i in range(n) if active[i] or predicted_count[i]]
    return metrics, per_class, matrix


def map_names(names, classes, aliases):
    from .object_choices import effective_aliases
    vocabulary = {normalize(name): int(key) for key, name in classes.items()}
    alias_lookup = effective_aliases(classes, aliases)
    mapped, unresolved = {}, []
    for object_id, name in names.items():
        label = alias_lookup.get(normalize(name), normalize(name))
        mapped[int(object_id)] = vocabulary.get(label, -1)
        if mapped[int(object_id)] == -1:
            unresolved.append({"object_id": int(object_id), "prediction": name})
    return mapped, unresolved


def prediction_contract(folder, classes):
    """Reject malformed predictions before scoring a run claiming strict labels."""
    from . import object_choices as oc, relation_choices as rc
    cfg = read_json(folder / "run.json").get("settings", {})
    object_format = cfg.get("object_response_format", "legacy")
    relation_format = cfg.get("relationship_response_format", "legacy")
    object_views = relation_views = 0
    if object_format == oc.FORMAT:
        expected = hashlib.sha256(json.dumps(classes, sort_keys=True).encode("utf-8")).hexdigest()
        if cfg.get("object_vocabulary_sha256") != expected:
            raise ValueError("Object vocabulary differs from scoring vocabulary; rebuild graph")
        allowed = set(cfg["object_choices"])
        class_by_name = {name: key for key, name in classes.items()}
        for node in read_json(folder / "local_node_predictions.json"):
            if node["tag"] not in allowed | {"invalid"}:
                raise ValueError("Noncanonical voted object label in strict run")
            for view in node["predictions"]:
                label = view["prediction"]
                expected_id = -1 if label == oc.ABSTAIN else class_by_name.get(label)
                if (label not in allowed or not view.get("contract_valid")
                        or view.get("raw_prediction") != label or view.get("class_id") != expected_id
                        or view.get("vocabulary_sha256") != expected):
                    raise ValueError("Object prediction violates its saved label/class-ID contract")
                object_views += 1
    if relation_format in rc.AUDITED_FORMATS:
        for pair in read_json(folder / "local_relation_predictions.json"):
            for view in pair["predictions"]:
                if (view["prediction"] not in rc.LABELS or not view.get("contract_valid")
                        or view.get("raw_prediction") != view["prediction"]
                        or view.get("choice_decoding") != cfg["relationship_choice_decoding"]):
                    raise ValueError("Relationship prediction violates its saved label contract")
                checked = [view]
                if cfg.get("verify_swapped_boxes"):
                    swapped = view.get("swapped_prediction", {})
                    if (swapped.get("prediction") not in rc.LABELS or not swapped.get("contract_valid")
                            or swapped.get("choice_decoding") != cfg["relationship_choice_decoding"]
                            or view.get("swap_consistent") != (swapped["prediction"] == rc.SWAPPED_LABELS[view["prediction"]])):
                        raise ValueError("Missing or inconsistent A/B swap audit")
                    checked.append(swapped)
                for prediction in checked:
                    if cfg["relationship_choice_decoding"] == "sequence_score":
                        scores = prediction.get("choice_scores", [])
                        if (len(scores) != len(rc.LABELS) or {r["label"] for r in scores} != set(rc.LABELS)
                                or any(not np.isfinite(r["mean_log_probability"]) for r in scores)
                                or max(scores, key=lambda r: r["mean_log_probability"])["label"] != prediction["prediction"]):
                            raise ValueError("Complete-label scores disagree with the selected relationship")
                    relation_views += 1
    condition = {key: cfg.get(key) for key in (
        "object_response_format", "object_vocabulary_sha256", "object_prompt",
        "relationship_response_format", "relationship_choice_decoding", "relationship_protocol", "relation_prompt",
        "observation_selection_policy", "verify_swapped_boxes", "edge_validation")}
    return {"object_format": object_format, "relationship_format": relation_format,
            "relationship_decoder": cfg.get("relationship_choice_decoding", "legacy"),
            "swap_check_enabled": cfg.get("verify_swapped_boxes", False),
            "edge_policy": cfg.get("edge_validation", {}).get("policy", "unchecked"),
            "verified_object_views": object_views, "verified_relationship_views": relation_views,
            "comparison_signature": hashlib.sha256(json.dumps(condition, sort_keys=True).encode("utf-8")).hexdigest()}


def completed_runs(scene_dir, mapfile):
    selected, status = {}, []
    for path in sorted((Path(scene_dir) / "scene_graphs").glob("*/run.json")):
        record = read_json(path)
        variant = record["variant"]["name"]
        usable = (record.get("complete") is True and (path.parent / "graph.json").is_file()
                  and Path(record["mapfile"]).resolve() == Path(mapfile).resolve())
        status.append({"folder": path.parent.name, "variant": variant, "complete": bool(record.get("complete")),
                       "eligible": usable, "error": record.get("error", "")})
        if usable and (variant not in selected or path.stat().st_mtime_ns > selected[variant].stat().st_mtime_ns):
            selected[variant] = path
    return {key: path.parent for key, path in sorted(selected.items())}, status


def palette(labels):
    import matplotlib
    cmap = matplotlib.colormaps["turbo"]
    labels = np.asarray(labels)
    rgb = (cmap(((labels.astype(float) * 0.61803398875) % 1))[:, :3] * 255).astype(np.uint8)
    rgb[labels < 0] = [140, 140, 140]
    return rgb


def save_ply(path, xyz, rgb, labels):
    with Path(path).open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\nelement vertex " + str(len(xyz)) + "\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                     "property int semantic_id\nend_header\n")
        np.savetxt(stream, np.column_stack((xyz, rgb, labels)), fmt="%.5f %.5f %.5f %d %d %d %d")


def plot_maps(output, reference, prediction, variants, max_points=12000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    extent = np.concatenate((reference["xyz"], prediction["xyz"]))
    lower, upper = extent.min(0), extent.max(0)
    middle, radius = (lower + upper) / 2, max(float(np.max(upper - lower)) / 2, .1)
    ri = np.linspace(0, len(reference["xyz"]) - 1, min(max_points, len(reference["xyz"])), dtype=int)
    pi = np.linspace(0, len(prediction["xyz"]) - 1, min(max_points, len(prediction["xyz"])), dtype=int)
    for name, labels in variants.items():
        fig = plt.figure(figsize=(16, 5))
        clouds = [(reference["xyz"][ri], reference["rgb"][ri], "Observed reference: RGB"),
                  (reference["xyz"][ri], palette(reference["labels"][ri]), "Observed reference: GT semantic labels"),
                  (prediction["xyz"][pi], palette(labels[pi]), f"Proposed shared map: {name}")]
        for number, (xyz, rgb, title) in enumerate(clouds, 1):
            ax = fig.add_subplot(1, 3, number, projection="3d")
            ax.scatter(*xyz.T, c=rgb / 255, s=.5, depthshade=False)
            ax.set(xlim=(middle[0]-radius, middle[0]+radius), ylim=(middle[1]-radius, middle[1]+radius),
                   zlim=(middle[2]-radius, middle[2]+radius), xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title=title)
            ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=20, azim=-60, vertical_axis="y")
        fig.tight_layout(); fig.savefig(Path(output) / f"maps_{name}.png", dpi=160); plt.close(fig)


def interactive_maps(path, reference, prediction, variants, max_points=12000):
    ri = np.linspace(0, len(reference["xyz"])-1, min(max_points, len(reference["xyz"])), dtype=int)
    pi = np.linspace(0, len(prediction["xyz"])-1, min(max_points, len(prediction["xyz"])), dtype=int)
    data = {"reference": np.round(reference["xyz"][ri], 4).tolist(), "predicted": np.round(prediction["xyz"][pi], 4).tolist(),
            "rgb": reference["rgb"][ri].tolist(), "gt": palette(reference["labels"][ri]).tolist(),
            "variants": {k: palette(v[pi]).tolist() for k, v in variants.items()}}
    template = '''<!doctype html><html><meta charset="utf-8"><title>Observed GT and proposed maps</title>
<style>body{font:15px system-ui;background:#f5f7fa;color:#182333;margin:24px} .panes{display:flex;gap:12px}section{flex:1;min-width:0;background:white;padding:12px;border-radius:10px}canvas{width:100%;height:450px;background:#101725;touch-action:none}select,button{padding:8px;margin:8px}small{display:block;margin:12px 0}</style>
<h2>Observed ground-truth reference and proposed map</h2><p>Same saved RGB-D frames and source-world poses. This reference covers observed surfaces; it is not a full-room mesh.</p>
<label>Condition <select id="variant"></select></label><button id="reset">Reset view</button>
<small>Drag any panel to rotate all panels; scroll to zoom. Identical semantic IDs use identical colors; gray indicates an unmapped prediction label. Display is sampled; metrics use the voxelized evaluation clouds.</small>
<div class="panes"><section>Reference RGB<canvas></canvas></section><section>Reference semantic labels<canvas></canvas></section><section>Proposed semantic labels<canvas></canvas></section></div>
<script>const d=__DATA__, canvases=[...document.querySelectorAll('canvas')], select=document.getElementById('variant');
for(const name of Object.keys(d.variants)){const option=document.createElement('option');option.value=name;option.textContent=name;select.append(option)}
let yaw=.5,pitch=-.3,zoom=1,drag=null; const all=d.reference.concat(d.predicted),low=[0,1,2].map(k=>Math.min(...all.map(p=>p[k]))),high=[0,1,2].map(k=>Math.max(...all.map(p=>p[k]))),center=low.map((x,k)=>(x+high[k])/2),span=Math.max(...high.map((x,k)=>x-low[k]),.1);
function draw(){canvases.forEach((canvas,index)=>{canvas.width=Math.max(300,canvas.clientWidth);canvas.height=450;const ctx=canvas.getContext('2d');ctx.fillStyle='#101725';ctx.fillRect(0,0,canvas.width,canvas.height);const points=index===2?d.predicted:d.reference,colors=index===0?d.rgb:index===1?d.gt:d.variants[select.value],scale=Math.min(canvas.width,canvas.height)/span*.8*zoom,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
const projected=points.map((p,i)=>{const x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],a=cy*x+sy*z,b=-sy*x+cy*z;return [canvas.width/2+a*scale,canvas.height/2-(cp*y-sp*b)*scale,sp*y+cp*b,i]}).sort((a,b)=>a[2]-b[2]);for(const [x,y,z,i] of projected){ctx.fillStyle='rgb('+colors[i].join(',')+')';ctx.fillRect(x,y,2,2)}})}
for(const canvas of canvases){canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw()};canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.15,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));draw()}}
select.onchange=draw;document.getElementById('reset').onclick=()=>{yaw=.5;pitch=-.3;zoom=1;draw()};window.onresize=draw;draw();</script></html>'''
    Path(path).write_text(template.replace("__DATA__", json.dumps(data).replace("<", "\\u003c")), encoding="utf-8")


def graph_diagnostics(folder, object_count):
    graph = read_json(folder / "graph.json")
    nodes = read_json(folder / "local_node_predictions.json")
    pairs = read_json(folder / "local_relation_predictions.json")
    views = sum(p["votes"]["views"] for p in pairs)
    times = [v["generation_seconds"] for p in pairs for v in p["predictions"]]
    object_views = [view for node in nodes for view in node["predictions"]]
    checked_objects = [view for view in object_views if "contract_valid" in view]
    return {"variant": graph["variant"], "folder": folder.name, "mapped_objects": object_count,
            "retained_nodes": len(graph["nodes"]), "node_retention": len(graph["nodes"]) / object_count,
            "predicted_edges": len(graph["edges"]), "candidate_pairs": len(pairs),
            "flagged_saved_edges": sum(edge.get("edge_validation", {}).get("accepted") is False for edge in graph["edges"]),
            "strict_check_passing_edges": sum(edge.get("edge_validation", {}).get("accepted") is True
                                             and edge["edge_validation"].get("status") != "unchecked" for edge in graph["edges"]),
            "unresolved_pairs": len(graph["unresolved_pairs"]),
            "resolved_pair_rate": (len(pairs)-len(graph["unresolved_pairs"])) / len(pairs) if pairs else None,
            "invalid_relation_output_rate": sum(p["votes"]["invalid"] for p in pairs) / views if views else None,
            "object_contract_checked_views": len(checked_objects),
            "object_contract_violation_rate": sum(not view["contract_valid"] for view in checked_objects) / len(checked_objects) if checked_objects else None,
            "object_abstained_views": sum(view.get("abstained", False) for view in object_views),
            "edge_check_rejected_pairs": sum(p.get("reason") == "edge_validation_rejected" for p in pairs),
            "swap_inconsistent_views": sum(view.get("swap_consistent") is False for pair in pairs for view in pair["predictions"]),
            "unchecked_saved_edges": sum("edge_validation" not in edge or edge["edge_validation"].get("status") == "unchecked" for edge in graph["edges"]),
            "tied_relation_pairs": sum(p["votes"]["tied"] for p in pairs),
            "median_relation_generation_s": float(np.median(times)) if times else None,
            "mean_object_vote_agreement": float(np.mean([n["votes"]["agreement"] for n in nodes if n["votes"]["agreement"] is not None]))}


def pair_case_gallery(output, runs, data_root, max_cases=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cards = []
    cases_dir = Path(output) / "cases"; cases_dir.mkdir(exist_ok=True)
    for variant, folder in runs.items():
        pairs = read_json(folder / "local_relation_predictions.json")
        with (folder / "review.csv").open(newline="", encoding="utf-8") as stream:
            reviews = {(r["source_original_id"], r["target_original_id"]): r for r in csv.DictReader(stream)}
        ordered = sorted(pairs, key=lambda p: p.get("upstream_relation", p["relation"]) != "FAIL")
        for number, pair in enumerate(ordered[:max_cases]):
            decision = pair.get("upstream_relation", pair["relation"])
            review = reviews.get((pair["source_original_id"], pair["target_original_id"]), {})
            reference = review.get("reviewed_relation", "").strip()
            normalize_relation = lambda r: normalize(r).replace("none of these", "none")
            status = ("reviewed success" if normalize_relation(reference) == normalize_relation(decision) else "reviewed failure") if reference else (
                "rejected by edge checks" if pair.get("reason") == "edge_validation_rejected" else
                "unresolved pipeline case" if decision == "FAIL" else
                "flagged prediction: review required" if pair.get("edge_validation", {}).get("accepted") is False else
                "unverified prediction: review required")
            views = pair["predictions"][:2]
            fig, axes = plt.subplots(1, max(1, len(views)), figsize=(12, 4), squeeze=False)
            if not views:
                axes[0, 0].text(.1, .5, "No shared RGB evidence", transform=axes[0, 0].transAxes)
            for ax, view in zip(axes[0], views):
                with Image.open(Path(data_root) / view["rgb"]) as image:
                    ax.imshow(image)
                for key, color, tag in (("bbox_a", "red", "A"), ("bbox_b", "cyan", "B")):
                    x0, y0, x1, y1 = view[key]
                    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor=color, linewidth=2))
                    ax.text(x0, y0, tag, color=color, backgroundcolor="black")
                ax.set_title(f"Frame {view['frame_id']}: {view['prediction']}")
            for ax in axes[0]: ax.axis("off")
            fig.suptitle(f"{variant} | {pair['source_original_id']} / {pair['target_original_id']} | {decision}\n{status} | {pair['reason']}")
            filename = f"{variant}_{number:02d}.png"
            fig.tight_layout(); fig.savefig(cases_dir / filename, dpi=140); plt.close(fig)
            cards.append({"variant": variant, "source_original_id": pair["source_original_id"],
                          "target_original_id": pair["target_original_id"], "prediction": decision,
                          "proposed_relation": pair.get("proposed_relation", pair["relation"]),
                          "edge_check": pair.get("edge_validation", {}).get("reason", "not_checked"),
                          "reviewed_relation": reference, "status": status, "reason": pair["reason"],
                          "image": "cases/" + filename})
    write_csv(Path(output) / "qualitative_cases.csv", cards)
    body = "".join(f"<article><h3>{html.escape(c['variant'])}: {html.escape(c['status'])}</h3>"
                   f"<p>{html.escape(c['prediction'])} — {html.escape(c['reason'])}</p>"
                   f"<img style='max-width:100%' src='{c['image']}'></article>" for c in cards)
    (Path(output) / "qualitative_cases.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Graph case review</title>"
        "<body style='font:16px system-ui;max-width:1200px;margin:auto'><h1>Graph case review</h1>"
        "<p>Valid predictions require human review before they are classified as successful. "
        "An unresolved case is a pipeline abstention, not proof of an incorrect physical relation.</p>" + body, encoding="utf-8")


def build_report(project_root, experiment, scene, output, *, pixel_stride=4, voxel_size=.025,
                 max_depth_m=10.0, thresholds=(.05, .1), semantic_threshold=.05,
                 excluded=DEFAULT_EXCLUDED, aliases=None):
    project_root, output = Path(project_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_dir = project_root / "artifacts/scene_graphs" / experiment / scene.replace("/", "__")
    mapping_dir = scene_dir / "mapping"
    manifest = read_json(mapping_dir / "manifest.json")
    mapfile = Path(read_json(mapping_dir / "map_complete.json")["output"])
    if not mapfile.is_file():
        raise FileNotFoundError("Run this evaluator inside the mapping runtime so its saved paths resolve")
    data_root = project_root / "dataset/osma-bench/data"
    classes = {int(c["id"]): c["name"] for c in read_json(data_root / scene / "embed_semseg_classes.json")["classes"]}
    from .object_choices import effective_aliases
    aliases = effective_aliases(classes, aliases)
    excluded_ids = {k for k, v in classes.items() if normalize(v) in set(map(normalize, excluded))}
    class_ids = sorted(set(classes) - excluded_ids)
    print("Reconstructing observed reference from", len(manifest["frames"]), "saved frames...", flush=True)
    reference = observed_reference(data_root, scene, manifest, mapping_dir, pixel_stride=pixel_stride,
                                   voxel_size=voxel_size, max_depth_m=max_depth_m)
    keep = np.isin(reference["labels"], class_ids)
    foreground = {k: v[keep] for k, v in reference.items()}
    if not len(foreground["xyz"]): raise ValueError("No foreground reference points after class exclusions")
    prediction, detector_names, object_count = load_prediction(mapfile, voxel_size)
    runs, status = completed_runs(scene_dir, mapfile)
    write_csv(output / "run_status.csv", status)
    if not runs: raise ValueError("No completed graph variants yet; rerun the report after a graph finishes")
    contracts = [{"variant": name, **prediction_contract(folder, classes)} for name, folder in runs.items()]
    if len({row["comparison_signature"] for row in contracts}) != 1:
        raise ValueError("Variants use different prompts/decoders/edge checks. Complete all variants under the same settings before comparing them.")
    print("Reference voxels:", len(reference["xyz"]), "foreground:", len(foreground["xyz"]),
          "predicted:", len(prediction["xyz"]), flush=True)
    np.savez_compressed(output / "observed_gt.npz", **reference)
    np.savez_compressed(output / "proposed_shared_map.npz", **prediction)
    save_ply(output / "observed_gt_semantic.ply", reference["xyz"], palette(reference["labels"]), reference["labels"])
    save_ply(output / "proposed_shared_rgb.ply", prediction["xyz"], prediction["rgb"], np.full(len(prediction["xyz"]), -1))
    pred_to_gt, _ = nearest(prediction["xyz"], foreground["xyz"])
    gt_to_pred, gt_index = nearest(foreground["xyz"], prediction["xyz"])
    geometry_rows = [{"scope": "shared object map", **geometry_metrics(pred_to_gt, gt_to_pred, thresholds)}]
    semantic_rows, per_class_rows, unknown_rows, variants, diagnostics = [], [], [], {}, []
    sources, evidence_runs = [], {}
    for name, folder in runs.items():
        print("Evaluating saved condition:", name, flush=True)
        nodes = read_json(folder / "local_node_predictions.json")
        labels_by_object, unresolved = map_names({int(n["id"]): n["tag"] for n in nodes}, classes, aliases)
        labels = np.array([labels_by_object.get(int(i), -1) for i in prediction["object_ids"]], dtype=np.int32)
        variants[name] = labels
        for row in unresolved: unknown_rows.append({"variant": name, **row})
        graph = read_json(folder / "graph.json")
        retained = set(map(int, graph["node_index_to_original_id"].values()))
        retained_mask = np.isin(prediction["object_ids"], list(retained))
        for scope in ("shared_map", "retained_graph"):
            if scope == "shared_map":
                distance, index, scope_labels = gt_to_pred, gt_index, labels
            else:
                distance, index = nearest(foreground["xyz"], prediction["xyz"][retained_mask])
                scope_labels = labels[retained_mask]
                reverse, _ = nearest(prediction["xyz"][retained_mask], foreground["xyz"])
                geometry_rows.append({"scope": name + " retained graph", **geometry_metrics(reverse, distance, thresholds)})
            assigned = np.full(len(distance), -1, dtype=np.int32)
            covered = distance <= semantic_threshold
            if len(scope_labels): assigned[covered] = scope_labels[index[covered]]
            metrics, rows, matrix = semantic_metrics(foreground["labels"], assigned, class_ids)
            semantic_rows.append({"variant": name, "scope": scope, **metrics,
                                  "geometric_coverage": float(covered.mean())})
            for row in rows: per_class_rows.append({"variant": name, "scope": scope, "class_name": classes[row["class_id"]], **row})
            np.save(output / f"confusion_{name}_{scope}.npy", matrix)
        diagnostics.append(graph_diagnostics(folder, object_count))
        from .image_evidence import snapshot_run
        evidence_folder = "evidence/" + name
        evidence_runs[name] = snapshot_run(folder, output / evidence_folder)
        sources.append({"variant": name, "folder": str(folder), "evidence_folder": evidence_folder,
                        "run_record": read_json(folder / "run.json")})
    write_csv(output / "semantic_metrics.csv", semantic_rows)
    write_csv(output / "geometry_metrics.csv", geometry_rows)
    write_csv(output / "per_class_metrics.csv", per_class_rows)
    write_csv(output / "unmapped_labels.csv", unknown_rows)
    write_csv(output / "graph_diagnostics.csv", diagnostics)
    write_csv(output / "class_vocabulary.csv", [{"class_id": k, "class_name": classes[k],
                                                "excluded": k in excluded_ids} for k in sorted(classes)])
    plot_maps(output, reference, prediction, variants)
    interactive_maps(output / "map_viewer.html", reference, prediction, variants)
    pair_case_gallery(output, runs, data_root)
    from .image_evidence import export_node_images
    export_node_images(output, evidence_runs, data_root)
    configuration = {"experiment": experiment, "scene": scene, "frames": [r["frame_id"] for r in manifest["frames"]],
                     "pixel_stride": pixel_stride, "voxel_size_m": voxel_size, "max_depth_m": max_depth_m,
                     "geometry_thresholds_m": list(thresholds), "semantic_association_threshold_m": semantic_threshold,
                     "excluded_classes": list(excluded), "class_ids": class_ids, "label_aliases": aliases,
                     "category_matching_protocol": "exact dataset names plus explicit spelling aliases; no broad/fine category merging",
                     "mapfile": str(mapfile), "map_bytes": mapfile.stat().st_size,
                     "map_mtime_ns": mapfile.stat().st_mtime_ns, "mapping_fingerprint": manifest["settings_fingerprint"],
                     "coordinate_frame": "source world; staged camera-to-world OpenCV poses; no fitted registration",
                     "reference_kind": "observed semantic RGB-D reference, not full-room mesh or instance graph",
                     "protocol": "GT-reference foreground voxels; missing/unknown predictions count as errors; mAcc/mIoU average GT-supported classes; Chamfer is sum of unsquared directional means",
                     "sources": sources, "inference_contracts": contracts,
                     "report_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(output / "report.json", configuration)
    print("Report saved:", output, flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=.025)
    parser.add_argument("--max-depth", type=float, default=10.)
    parser.add_argument("--semantic-threshold", type=float, default=.05)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.05, .1])
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--excluded", nargs="*", default=list(DEFAULT_EXCLUDED))
    args = parser.parse_args()
    build_report(args.project_root, args.experiment, args.scene, args.output,
                 pixel_stride=args.pixel_stride, voxel_size=args.voxel_size, max_depth_m=args.max_depth,
                 thresholds=args.thresholds, semantic_threshold=args.semantic_threshold,
                 excluded=args.excluded, aliases=read_json(args.aliases) if args.aliases else {})
