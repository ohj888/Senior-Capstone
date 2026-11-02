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


def select_display_edges(graph, pairs=(), mode="proposals"):
    """A display view only: never modify the saved graph or invent a relation."""
    from .relation_choices import UPSTREAM_RELATIONS, edge_semantics
    if mode not in ("proposals", "saved", "strict"):
        raise ValueError("Edge view must be proposals, saved, or strict")
    saved = list(graph["edges"])
    if mode == "saved":
        return saved
    if mode == "strict":
        return [edge for edge in saved if edge.get("edge_validation", {}).get("accepted") is True
                and edge["edge_validation"].get("status") != "unchecked"]
    by_pair = {(int(edge["source"]), int(edge["target"])): edge for edge in saved}
    node_ids = {int(node["id"]) for node in graph["nodes"]}
    for pair in pairs:
        a, b = int(pair["source"]), int(pair["target"])
        if (a, b) in by_pair or a not in node_ids or b not in node_ids or a == b:
            continue
        # A tied vote has no single prediction: never arbitrarily draw a winner.
        label = pair.get("votes", {}).get("label")
        relation = UPSTREAM_RELATIONS.get(label)
        if relation is None or label == "none":
            continue
        subject, reference, _, directed = edge_semantics(a, b, relation)
        validation = dict(pair.get("edge_validation", {}))
        validation.setdefault("accepted", False)
        validation.setdefault("reason", pair.get("reason", "not_checked"))
        validation.setdefault("status", "flagged")
        by_pair[(a, b)] = {"source": a, "target": b, "subject": subject,
                           "reference": reference, "directed": directed, "relation": relation,
                           "display_only_proposal": True, "edge_validation": validation}
    return list(by_pair.values())


def edge_appearance(edge):
    validation = edge.get("edge_validation", {})
    flagged = validation.get("accepted") is False or edge.get("display_only_proposal", False)
    color = (255, 170, 50) if flagged else (85, 225, 245)
    return flagged, color


def node_records(graph, predictions):
    """Join by ORIGINAL map ID: graph IDs are renumbered after pruning."""
    from .observation_selection import select_observations
    by_original = {str(row["id"]): row for row in predictions}
    mapping = graph["node_index_to_original_id"]
    records = []
    for node in graph["nodes"]:
        original = str(mapping[str(node["id"])])
        prediction = by_original.get(original, {})
        views = sorted(select_observations(prediction.get("predictions", [])), key=lambda v: (
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
    from .relation_choices import edge_semantics
    by_id = {r["graph_id"]: r for r in observations}
    result = []
    for edge in edges:
        a, b = edge["source"], edge["target"]
        relation = " ".join(edge["relation"].replace("_", " ").casefold().split())
        if a not in by_id or b not in by_id or relation in ("fail", "none", "none of these"):
            continue
        flagged, color = edge_appearance(edge)
        source, target, label, directed = edge_semantics(a, b, relation)
        if "subject" in edge and (edge["subject"], edge["reference"], edge["directed"]) != (source, target, directed):
            raise ValueError("Saved edge direction disagrees with its relationship label")
        centers = []
        for node_id in (source, target):
            x0, y0, x1, y1 = by_id[node_id]["view"]["bbox"]
            centers.append(((x0+x1)/2, (y0+y1)/2))
        result.append({"source": source, "target": target, "label": label,
                       "directed": directed, "start": centers[0], "end": centers[1],
                       "flagged": flagged, "color": color,
                       "warning": edge.get("edge_validation", {}).get("reason", ""),
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
    caption = f"Displayed edges: {len(projected_edges(observations, edges))} / {len(edges)} | amber dashed = flagged; cyan = unverified prediction"
    caption_bounds = draw.textbbox((0, 0), caption, font=font)
    banner_height = caption_bounds[3]-caption_bounds[1]+10
    visible_edges = projected_edges(observations, edges)
    for edge in visible_edges:
        a, b = edge["start"], edge["end"]
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        color = edge["color"] + (255,)
        if edge["flagged"] and length > 0:
            for offset in range(0, math.ceil(length), 18):
                t0, t1 = offset / length, min(offset+11, length) / length
                segment = ((a[0]+dx*t0, a[1]+dy*t0), (a[0]+dx*t1, a[1]+dy*t1))
                draw.line(segment, fill=(0, 0, 0, 255), width=7)
                draw.line(segment, fill=color, width=3)
        else:
            draw.line((a, b), fill=(0, 0, 0, 255), width=7)
            draw.line((a, b), fill=color, width=3)
        if edge["directed"] and length > 1:
            ux, uy = dx/length, dy/length
            head = min(14, length*.4)
            draw.polygon((b, (b[0]-head*ux+head*.45*uy, b[1]-head*uy-head*.45*ux),
                           (b[0]-head*ux-head*.45*uy, b[1]-head*uy+head*.45*ux)), fill=color)
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
        label = f"G{edge['source']} {connector} G{edge['target']}: {edge['label']}" + (" [FLAGGED]" if edge["flagged"] else "")
        bounds = draw.textbbox((0, 0), label, font=font)
        width, height = bounds[2]-bounds[0]+8, bounds[3]-bounds[1]+8
        x = min(max(0, (edge["start"][0]+edge["end"][0])/2-width/2), max(0, image.width-width))
        y = min(max(32, (edge["start"][1]+edge["end"][1])/2+number*5), max(32, image.height-height))
        draw.rectangle((x, y, x+width, y+height), fill=(0, 0, 0, 230))
        draw.text((x+4, y+2), label, font=font, fill=edge["color"]+(255,))
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
    from .relation_choices import edge_semantics
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
        subject, reference, relation, directed = edge_semantics(edge["source"], edge["target"], edge["relation"])
        a, b = positions[subject], positions[reference]
        style = "->" if directed else "-"
        flagged, rgb = edge_appearance(edge)
        color = tuple(channel/255 for channel in rgb)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, shrinkA=38, shrinkB=38,
                                    color=color, linestyle="--" if flagged else "-",
                                    linewidth=1.3, mutation_scale=12, alpha=.8, zorder=1))
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.15,
                f"G{subject} {'->' if directed else '--'} G{reference}: {relation}" + (" [FLAGGED]" if flagged else ""),
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


def export_node_images(output, runs, data_root, max_frames=6, edge_view="proposals"):
    output = Path(output)
    base = output / "node_images"; base.mkdir(parents=True, exist_ok=True)
    node_rows, frame_rows, summaries, sections = [], [], [], []
    for variant, folder in runs.items():
        graph = read_json(Path(folder) / "graph.json")
        pair_path = Path(folder) / "local_relation_predictions.json"
        pairs = read_json(pair_path) if pair_path.is_file() else []
        display_edges = select_display_edges(graph, pairs, edge_view)
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
                   "bbox_selection": view.get("bbox_selection", "") if view else "",
                   "same_frame_alternatives": view.get("same_frame_alternatives", 1) if view else 0,
                   "crop": crop_file}
            node_rows.append(row)
            cards.append("<article><h4>G" + str(record["graph_id"]) + " / M" + html.escape(record["original_id"]) +
                         ": " + html.escape(record["label"]) + "</h4>" +
                         (f"<img src='{crop_file}'><p>Frame {view['frame_id']}; {len(record['views'])} naming views</p>" if view else "<p>No saved RGB crop</p>") + "</article>")
        frame_cards, shown_edges, shown_saved_edges = [], set(), set()
        frame_records = add_relation_views(records, pairs)
        for number, (relative, observations) in enumerate(choose_frames(frame_records, max_frames, display_edges)):
            visible_edges = projected_edges(observations, display_edges)
            saved_visible = projected_edges(observations, graph["edges"])
            shown_edges.update(e["key"] for e in visible_edges)
            shown_saved_edges.update(e["key"] for e in saved_visible)
            image = annotated_frame(_source(data_root, relative), observations, display_edges)
            path = variant_dir / f"scene_frame_{observations[0]['view']['frame_id']:06d}.jpg"
            image.save(path, quality=94)
            frame_rows.append({"variant": variant, "frame_id": observations[0]["view"]["frame_id"],
                               "visible_graph_nodes": len(observations), "rgb_source": relative,
                               "visible_saved_edges": len(saved_visible), "total_saved_edges": len(graph["edges"]),
                               "visible_display_edges": len(visible_edges), "total_display_edges": len(display_edges),
                               "visible_flagged_edges": sum(edge["flagged"] for edge in visible_edges), "edge_view": edge_view,
                               "image": path.relative_to(output).as_posix()})
            shown.update(r["graph_id"] for r in observations)
            frame_cards.append(f"<figure><img src='{path.relative_to(output).as_posix()}'><figcaption>"
                               f"Frame {observations[0]['view']['frame_id']}: {len(observations)} retained graph nodes, "
                               f"{len(visible_edges)} / {len(display_edges)} displayed proposals, "
                               f"{sum(edge['flagged'] for edge in visible_edges)} flagged; {len(saved_visible)} saved edges visible</figcaption></figure>")
        graph_path = output / f"image_graph_{variant}.png"
        _thumbnail_graph(graph_path, {**graph, "edges": display_edges}, records, crops)
        summaries.append({"variant": variant, "retained_nodes": len(records), "nodes_with_crops": len(crops),
                          "nodes_in_selected_frames": len(shown), "annotated_frames": len(frame_cards),
                          "total_saved_edges": len(graph["edges"]), "saved_edges_in_selected_frames": len(shown_saved_edges),
                          "displayed_edges_in_selected_frames": len(shown_edges),
                          "total_display_edges": len(display_edges), "edge_view": edge_view,
                          "flagged_display_edges": sum(edge_appearance(edge)[0] for edge in display_edges),
                          "image_graph": graph_path.name})
        sections.append(f"<section><h2>{html.escape(variant)}</h2><img src='{graph_path.name}'>"
                        "<h3>Scene images with retained node labels</h3>" + "".join(frame_cards) +
                        "<h3>Every retained node and its best saved crop</h3><div class='cards'>" + "".join(cards) + "</div></section>")
    for name, rows, fields in (
            ("node_image_index.csv", node_rows, ["variant", "graph_id", "original_map_id", "predicted_label", "vote_agreement",
                                               "available_naming_views", "frame_id", "rgb_source", "bbox",
                                               "bbox_selection", "same_frame_alternatives", "crop"]),
            ("scene_frame_index.csv", frame_rows, ["variant", "frame_id", "visible_graph_nodes", "rgb_source",
                                                 "visible_saved_edges", "total_saved_edges", "visible_display_edges", "total_display_edges",
                                                 "visible_flagged_edges", "edge_view", "image"])):
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
            "Cyan lines are unverified predictions; amber dashed lines are proposals flagged by consistency or geometry checks. "
            "The proposals view can show rejected predictions from the saved per-pair audit without changing graph files. "
            "Arrows run from the supported/contained object to its supporter/container; nearby and next-to are undirected. "
            "Tied votes and missing relationships are not invented.</p>" + "".join(sections))
    (output / "scene_nodes.html").write_text(page, encoding="utf-8")
    result = {"render_version": 5, "edge_view": edge_view, "variants": summaries, "max_frames_per_variant": max_frames, "node_index": "node_image_index.csv",
              "frame_index": "scene_frame_index.csv", "gallery": "scene_nodes.html"}
    (output / "node_images.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def add_image_evidence(report_dir, project_root, max_frames=6, edge_view="proposals"):
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
    result = export_node_images(report_dir, available, project_root / "dataset/osma-bench/data", max_frames, edge_view)
    result["missing_source_evidence"] = missing
    (report_dir / "node_images.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
