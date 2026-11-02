"""Connect saved graph nodes to their RGB observations and image crops."""
import base64
import colorsys
from collections import defaultdict
import csv
import html
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def node_records(graph, predictions):
    """Join by ORIGINAL map ID: graph IDs are renumbered after pruning."""
    by_original = {str(row["id"]): row for row in predictions}
    mapping = graph["node_index_to_original_id"]
    records = []
    for node in graph["nodes"]:
        original = str(mapping[str(node["id"])])
        prediction = by_original.get(original, {})
        views = sorted(prediction.get("predictions", []), key=lambda v: (
            -float(v.get("confidence") or 0), -((v["bbox"][2]-v["bbox"][0]) * (v["bbox"][3]-v["bbox"][1])),
            int(v["frame_id"])))
        records.append({"graph_id": int(node["id"]), "original_id": original,
                        "label": str(node["object_tag"]), "views": views,
                        "agreement": prediction.get("votes", {}).get("agreement")})
    return records


def add_relation_views(records, pairs):
    """Include saved relationship evidence, beyond the naming-view budget."""
    enriched = [{**record, "views": list(record["views"])} for record in records]
    by_id = {r["graph_id"]: r for r in enriched}
    for pair in pairs:
        for key, node_id in (("bbox_a", pair["source"]), ("bbox_b", pair["target"])):
            if node_id not in by_id:
                continue
            record = by_id[node_id]
            existing = {v["rgb"] for v in record["views"]}
            for view in pair.get("predictions", []):
                if view["rgb"] not in existing:
                    record["views"].append({"rgb": view["rgb"], "frame_id": view["frame_id"],
                                            "bbox": view[key], "evidence_task": "relationship"})
                    existing.add(view["rgb"])
    return enriched


def projected_edges(observations, edges):
    """Project only actual graph edges whose two endpoints have image boxes."""
    by_id = {r["graph_id"]: r for r in observations}
    result = []
    for edge in edges:
        a, b = edge["source"], edge["target"]
        relation = " ".join(edge["relation"].replace("_", " ").casefold().split())
        if a not in by_id or b not in by_id or relation in ("fail", "none", "none of these"):
            continue
        source, target = (b, a) if relation in ("b on a", "b in a") else (a, b)
        label = {"a on b": "on", "b on a": "on", "a in b": "in", "b in a": "in"}.get(relation, relation)
        centers = []
        for node_id in (source, target):
            x0, y0, x1, y1 = by_id[node_id]["view"]["bbox"]
            centers.append(((x0+x1)/2, (y0+y1)/2))
        result.append({"source": source, "target": target, "label": label,
                       "directed": label not in ("nearby", "next to"), "start": centers[0], "end": centers[1],
                       "key": (a, b, edge["relation"])})
    return result


def choose_frames(records, limit=6, edges=()):
    """Greedily select frames that reveal different retained graph nodes."""
    grouped = defaultdict(dict)
    for record in records:
        for view in record["views"]:
            grouped[view["rgb"]].setdefault(record["graph_id"], {**record, "view": view})
    selected, covered = [], set()
    while grouped and len(selected) < limit:
        def priority(key):
            visible_edges = len(projected_edges(list(grouped[key].values()), edges))
            new_nodes = len(set(grouped[key])-covered)
            # Start with visible relationships; then expand object coverage.
            return ((-new_nodes, -visible_edges) if selected else (-visible_edges, -new_nodes)) + (-len(grouped[key]), key)
        key = min(grouped, key=priority)
        observations = sorted(grouped.pop(key).values(), key=lambda r: r["graph_id"])
        covered.update(r["graph_id"] for r in observations)
        selected.append((key, observations))
    return selected


def _color(original_id):
    return tuple(round(c * 255) for c in colorsys.hsv_to_rgb((int(original_id) * .61803398875) % 1, .85, 1))


def _source(data_root, relative):
    root = Path(data_root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("RGB evidence must stay within the dataset root")
    return path


def annotated_frame(path, observations, edges=()):
    with Image.open(path) as image:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, round(image.width / 70)))
    except OSError:
        # Matplotlib ships this font even when Windows has no DejaVu font.
        from matplotlib import get_data_path
        font = ImageFont.truetype(str(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"),
                                  max(14, round(image.width / 70)))
    caption = f"Saved relationships in this frame: {len(projected_edges(observations, edges))} / {len(edges)} in the full graph"
    caption_bounds = draw.textbbox((0, 0), caption, font=font)
    banner_height = caption_bounds[3]-caption_bounds[1]+10
    visible_edges = projected_edges(observations, edges)
    for edge in visible_edges:
        a, b = edge["start"], edge["end"]
        draw.line((a, b), fill=(0, 0, 0, 255), width=7)
        draw.line((a, b), fill=(255, 232, 70, 255), width=3)
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if edge["directed"] and length > 1:
            ux, uy = dx/length, dy/length
            head = min(14, length*.4)
            draw.polygon((b, (b[0]-head*ux+head*.45*uy, b[1]-head*uy-head*.45*ux),
                           (b[0]-head*ux-head*.45*uy, b[1]-head*uy+head*.45*ux)), fill=(255, 232, 70, 255))
    for record in observations:
        box = record["view"]["bbox"]
        color = _color(record["original_id"])
        draw.rectangle(box, outline=color + (255,), width=3)
        label = f"G{record['graph_id']} / M{record['original_id']}: {record['label']}"
        bounds = draw.textbbox((0, 0), label, font=font)
        width, height = bounds[2]-bounds[0]+8, bounds[3]-bounds[1]+8
        x = min(max(0, box[0]), max(0, image.width-width))
        y = min(max(banner_height+2, box[1]-height), max(banner_height+2, image.height-height))
        draw.rectangle((x, y, x+width, y+height), fill=(0, 0, 0, 210))
        draw.text((x+4, y+2), label, font=font, fill=color+(255,))
    for number, edge in enumerate(visible_edges):
        connector = "->" if edge["directed"] else "--"
        label = f"G{edge['source']} {connector} G{edge['target']}: {edge['label']}"
        bounds = draw.textbbox((0, 0), label, font=font)
        width, height = bounds[2]-bounds[0]+8, bounds[3]-bounds[1]+8
        x = min(max(0, (edge["start"][0]+edge["end"][0])/2-width/2), max(0, image.width-width))
        y = min(max(32, (edge["start"][1]+edge["end"][1])/2+number*5), max(32, image.height-height))
        draw.rectangle((x, y, x+width, y+height), fill=(0, 0, 0, 230))
        draw.text((x+4, y+2), label, font=font, fill=(255, 232, 70, 255))
    draw.rectangle((0, 0, image.width, banner_height), fill=(0, 0, 0, 225))
    draw.text((5, 3), caption, font=font, fill=(255, 255, 255, 255))
    return image


def snapshot_run(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("run.json", "graph.json", "local_node_predictions.json", "local_relation_predictions.json", "review.csv"):
        path = source / name
        if path.is_file() and path.resolve() != (destination / name).resolve():
            shutil.copy2(path, destination / name)
    return destination


def resolve_evidence(report_dir, source, scene_dir):
    report_dir = Path(report_dir)
    snapshot = report_dir / source.get("evidence_folder", "evidence/" + source["variant"])
    if (snapshot / "graph.json").is_file() and (snapshot / "local_node_predictions.json").is_file():
        return snapshot
    original = Path(scene_dir) / "scene_graphs" / source["folder"].replace("\\", "/").rstrip("/").split("/")[-1]
    report_record = report_dir / "report.json"
    if ((original / "graph.json").is_file() and (original / "local_node_predictions.json").is_file()
            and (original / "run.json").is_file() and report_record.is_file()
            and (original / "run.json").stat().st_mtime_ns <= report_record.stat().st_mtime_ns
            and read_json(original / "run.json") == source["run_record"]):
        return original
    return None


def _thumbnail_graph(path, graph, records, crop_paths):
    """An image-backed graph layout, deliberately distinct from 3D positions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import FancyArrowPatch
    columns = max(1, math.ceil(math.sqrt(len(records))))
    rows = max(1, math.ceil(len(records) / columns))
    fig, ax = plt.subplots(figsize=(max(8, columns*2.4), max(5, rows*2.1)))
    positions = {r["graph_id"]: (i % columns * 2, -(i // columns) * 2) for i, r in enumerate(records)}
    for edge in graph["edges"]:
        a, b = positions[edge["source"]], positions[edge["target"]]
        relation = edge["relation"]
        if relation in ("b on a", "b in a"):
            a, b = b, a
        style = "-" if relation in ("nearby", "next to") else "->"
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, shrinkA=38, shrinkB=38,
                                    color="#44566c", linewidth=1.3, mutation_scale=12, alpha=.7, zorder=1))
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.15,
                f"G{edge['source']} / G{edge['target']}: {relation}",
                fontsize=7, color="#8b2525", bbox={"facecolor":"white", "alpha":.8, "edgecolor":"none"}, zorder=2)
    for record in records:
        point = positions[record["graph_id"]]
        crop_path = crop_paths.get(record["graph_id"])
        if crop_path:
            with Image.open(crop_path) as image:
                image = image.copy(); image.thumbnail((90, 70))
            ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=.9), point, frameon=True,
                                        bboxprops={"edgecolor":tuple(c/255 for c in _color(record["original_id"]))}, zorder=3))
        else:
            ax.scatter(*point, color="gray", s=150, zorder=3)
        ax.text(point[0], point[1]-.65, f"G{record['graph_id']} / M{record['original_id']}\n{record['label']}",
                ha="center", va="top", fontsize=8, zorder=4)
    ax.set(xlim=(-1.2, max(1.2, (columns-1)*2+1.2)), ylim=(-rows*2, 1.2))
    ax.axis("off")
    fig.suptitle(graph["variant"] + " — retained scene-graph nodes with RGB evidence\n"
                 "G = graph ID; M = original map ID. Layout is for readability, not 3D position.", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def export_node_images(output, runs, data_root, max_frames=6):
    output = Path(output)
    base = output / "node_images"; base.mkdir(parents=True, exist_ok=True)
    node_rows, frame_rows, summaries, sections = [], [], [], []
    for variant, folder in runs.items():
        graph = read_json(Path(folder) / "graph.json")
        records = node_records(graph, read_json(Path(folder) / "local_node_predictions.json"))
        variant_dir = base / variant; variant_dir.mkdir(exist_ok=True)
        crops, cards, shown = {}, [], set()
        for record in records:
            view = record["views"][0] if record["views"] else None
            crop_file = ""
            if view:
                with Image.open(_source(data_root, view["rgb"])) as image:
                    crop = image.convert("RGB").crop(tuple(view["bbox"]))
                crop.thumbnail((320, 240))
                crop_path = variant_dir / f"node_G{record['graph_id']}_M{record['original_id']}.jpg"
                crop.save(crop_path, quality=90)
                crops[record["graph_id"]] = crop_path
                crop_file = crop_path.relative_to(output).as_posix()
            row = {"variant": variant, "graph_id": record["graph_id"], "original_map_id": record["original_id"],
                   "predicted_label": record["label"], "vote_agreement": record["agreement"],
                   "available_naming_views": len(record["views"]), "frame_id": view["frame_id"] if view else "",
                   "rgb_source": view["rgb"] if view else "", "bbox": json.dumps(view["bbox"]) if view else "",
                   "crop": crop_file}
            node_rows.append(row)
            cards.append("<article><h4>G" + str(record["graph_id"]) + " / M" + html.escape(record["original_id"]) +
                         ": " + html.escape(record["label"]) + "</h4>" +
                         (f"<img src='{crop_file}'><p>Frame {view['frame_id']}; {len(record['views'])} naming views</p>" if view else "<p>No saved RGB crop</p>") + "</article>")
        frame_cards, shown_edges = [], set()
        pair_path = Path(folder) / "local_relation_predictions.json"
        frame_records = add_relation_views(records, read_json(pair_path) if pair_path.is_file() else [])
        for number, (relative, observations) in enumerate(choose_frames(frame_records, max_frames, graph["edges"])):
            visible_edges = projected_edges(observations, graph["edges"])
            shown_edges.update(e["key"] for e in visible_edges)
            image = annotated_frame(_source(data_root, relative), observations, graph["edges"])
            path = variant_dir / f"scene_frame_{observations[0]['view']['frame_id']:06d}.jpg"
            image.save(path, quality=94)
            frame_rows.append({"variant": variant, "frame_id": observations[0]["view"]["frame_id"],
                               "visible_graph_nodes": len(observations), "rgb_source": relative,
                               "visible_saved_edges": len(visible_edges), "total_saved_edges": len(graph["edges"]),
                               "image": path.relative_to(output).as_posix()})
            shown.update(r["graph_id"] for r in observations)
            frame_cards.append(f"<figure><img src='{path.relative_to(output).as_posix()}'><figcaption>"
                               f"Frame {observations[0]['view']['frame_id']}: {len(observations)} retained graph nodes, "
                               f"{len(visible_edges)} / {len(graph['edges'])} saved relationships visible</figcaption></figure>")
        graph_path = output / f"image_graph_{variant}.png"
        _thumbnail_graph(graph_path, graph, records, crops)
        summaries.append({"variant": variant, "retained_nodes": len(records), "nodes_with_crops": len(crops),
                          "nodes_in_selected_frames": len(shown), "annotated_frames": len(frame_cards),
                          "total_saved_edges": len(graph["edges"]), "edges_in_selected_frames": len(shown_edges),
                          "image_graph": graph_path.name})
        sections.append(f"<section><h2>{html.escape(variant)}</h2><img src='{graph_path.name}'>"
                        "<h3>Scene images with retained node labels</h3>" + "".join(frame_cards) +
                        "<h3>Every retained node and its best saved crop</h3><div class='cards'>" + "".join(cards) + "</div></section>")
    for name, rows, fields in (
            ("node_image_index.csv", node_rows, ["variant", "graph_id", "original_map_id", "predicted_label", "vote_agreement",
                                               "available_naming_views", "frame_id", "rgb_source", "bbox", "crop"]),
            ("scene_frame_index.csv", frame_rows, ["variant", "frame_id", "visible_graph_nodes", "rgb_source",
                                                 "visible_saved_edges", "total_saved_edges", "image"])):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    if not runs:
        sections.append("<p>No raw node evidence is available for this report. The source graph outputs may have been "
                        "removed. Generate a new report after the clean graph run finishes.</p>")
    page = ("<!doctype html><meta charset='utf-8'><title>Image-grounded scene graphs</title>"
            "<style>body{font:16px system-ui;max-width:1500px;margin:24px auto;padding:16px;background:#f5f7fa;color:#182333}"
            "img{max-width:100%}section{margin:32px 0}figure,article{background:white;padding:12px;border-radius:8px}"
            ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}h4{margin:0 0 10px}</style>"
            "<h1>Constructed scene graphs linked to real images</h1><p>Labels are model predictions. Boxes are the saved detector observations used for naming. "
            "G is the graph node ID; M is its original map ID. Colors link map objects across conditions. Only retained graph nodes are shown; "
            "a frame shows only nodes with a saved observation in that frame, not every physically visible object. "
            "Yellow lines are saved graph edges with both endpoints visible. Arrows show the subject-to-object direction of on/in; "
            "nearby and next-to are undirected. Missing or unresolved relationships are not invented.</p>" + "".join(sections))
    (output / "scene_nodes.html").write_text(page, encoding="utf-8")
    result = {"render_version": 2, "variants": summaries, "max_frames_per_variant": max_frames, "node_index": "node_image_index.csv",
              "frame_index": "scene_frame_index.csv", "gallery": "scene_nodes.html"}
    (output / "node_images.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def add_image_evidence(report_dir, project_root, max_frames=6):
    """Upgrade an existing report when its original node evidence still exists."""
    report_dir, project_root = Path(report_dir), Path(project_root)
    record = read_json(report_dir / "report.json")
    scene_dir = project_root / "artifacts/scene_graphs" / record["experiment"] / record["scene"].replace("/", "__")
    available, missing = {}, []
    for source in record["sources"]:
        folder = resolve_evidence(report_dir, source, scene_dir)
        if folder is None:
            missing.append(source["variant"])
        else:
            available[source["variant"]] = snapshot_run(folder, report_dir / "evidence" / source["variant"])
    result = export_node_images(report_dir, available, project_root / "dataset/osma-bench/data", max_frames)
    result["missing_source_evidence"] = missing
    (report_dir / "node_images.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
