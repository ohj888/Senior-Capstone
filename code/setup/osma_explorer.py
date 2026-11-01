"""Read-only helpers for explore_osma_bench.ipynb. No downloads or data edits."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image
import yaml


QA_COLUMNS = ["family", "scene", "frame", "frame_id", "question", "answer", "category", "source"]


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def frame_id(name):
    match = re.fullmatch(r"frame(\d+)\.jpg", name)
    if not match:
        raise ValueError(f"Unexpected RGB frame name: {name!r}")
    return int(match.group(1))


def qa_table(document, family, source):
    """Flatten scene -> parameters[frame] -> qa into one row per question."""
    rows = []
    if not isinstance(document, dict) or not isinstance(document.get("parameters"), list):
        raise ValueError(f"Expected a scene object with a parameters list in {source}")
    for entry in document["parameters"]:
        for qa in entry["qa"]:
            rows.append({"family": family, "scene": document["scene_name"],
                         "frame": entry["frame"], "frame_id": frame_id(entry["frame"]),
                         "question": qa["question"], "answer": qa["answer"],
                         "category": qa.get("category", "Uncategorized"), "source": str(source)})
    return pd.DataFrame(rows, columns=QA_COLUMNS)


def load_poses(path):
    # Keep every row's original index; never silently drop invalid poses.
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] != 16:
        raise ValueError(f"Expected 16 numbers per trajectory row: {path}; got {rows.shape}")
    return rows.reshape(-1, 4, 4)


def modality_files(scene, prefix, extension):
    return {int(p.stem[len(prefix):]): p for p in (scene / "results").glob(f"{prefix}*{extension}")
            if p.stem[len(prefix):].isdigit()}


def load_scene(scene):
    scene = Path(scene)
    name = scene.name
    validated_path = scene / "vqa" / f"{name}_validated_questions.json"
    generated_path = scene / "vqa" / f"{name}_questions.json"
    descriptions_path = scene / "vqa" / f"{name}_descriptions.json"
    with (scene / "camera_params.yaml").open(encoding="utf-8") as handle:
        camera = yaml.safe_load(handle)["camera_params"]
    return {
        "path": scene, "name": name, "family": scene.parent.name,
        "rgb": modality_files(scene, "frame", ".jpg"),
        "depth": modality_files(scene, "depth", ".png"),
        "semantic": modality_files(scene, "semantic", ".png"),
        "poses": load_poses(scene / "traj.txt"), "camera": camera,
        "qa": qa_table(read_json(validated_path), scene.parent.name, validated_path),
        "generated_qa": qa_table(read_json(generated_path), scene.parent.name, generated_path),
        "raw_qa": read_json(validated_path),
        "descriptions": read_json(descriptions_path)["parameters"],
        "classes": {int(row["id"]): row["name"] for row in read_json(scene / "embed_semseg_classes.json")["classes"]},
    }


def audit_scene(scene):
    poses = scene["poses"]
    finite = np.isfinite(poses).all(axis=(1, 2))
    safe = poses[finite]
    rotations = safe[:, :3, :3]
    bad_rotation = (np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3)), axis=(1, 2)) > 1e-4)
    bad_rotation |= ~np.isclose(np.linalg.det(rotations), 1, atol=1e-4)
    rgb, depth, semantic = (set(scene[k]) for k in ("rgb", "depth", "semantic"))
    qa_ids = set(scene["qa"]["frame_id"])
    description_disagreements = 0
    for item in scene["descriptions"]:
        idx = frame_id(item["frame"])
        position = np.asarray(item.get("camera", {}).get("position", []), dtype=float)
        if idx >= len(poses) or position.shape != (3,) or not np.allclose(position, poses[idx, :3, 3], atol=1e-5):
            description_disagreements += 1
    return {
        "family": scene["family"], "scene": scene["name"],
        "rgb": len(rgb), "depth": len(depth), "semantic": len(semantic), "poses": len(poses),
        "generated_qa": len(scene["generated_qa"]), "validated_qa": len(scene["qa"]),
        "qa_frames": len(qa_ids), "qa_frames_missing_rgb": len(qa_ids - rgb),
        "rgb_depth_id_differences": len(rgb ^ depth), "rgb_semantic_id_differences": len(rgb ^ semantic),
        "rgb_pose_id_differences": len(rgb ^ set(range(len(poses)))),
        "nonfinite_poses": int((~finite).sum()),
        "invalid_last_rows": int((~np.isclose(safe[:, 3, :], [0, 0, 0, 1], atol=1e-5).all(axis=1)).sum()),
        "invalid_rotations": int(bad_rotation.sum()),
        "description_position_disagreements": description_disagreements,
    }


def inventory(data_root):
    paths = sorted(p.parent for p in Path(data_root).glob("*/*/traj.txt"))
    if not paths:
        raise FileNotFoundError(f"No scenes found in {data_root}. Run the OSMa-Bench download cell first.")
    summaries, questions = [], []
    for path in paths:
        scene = load_scene(path)
        summaries.append(audit_scene(scene))
        questions.append(scene["qa"])
    return paths, pd.DataFrame(summaries), pd.concat(questions, ignore_index=True)


def camera_to_world(poses, convention="camera_to_world"):
    if convention not in {"camera_to_world", "world_to_camera"}:
        raise ValueError("Use camera_to_world or world_to_camera.")
    if not np.isfinite(poses).all():
        raise ValueError("Non-finite poses found; inspect the audit before plotting.")
    return poses if convention == "camera_to_world" else np.linalg.inv(poses)


def read_image(path):
    with Image.open(path) as image:
        return np.array(image)


def frame_data(scene, idx):
    if idx not in scene["rgb"] or idx not in scene["depth"] or idx not in scene["semantic"]:
        raise ValueError(f"Frame {idx} lacks an RGB, depth, or semantic file. Choose an ID from the inventory.")
    if not 0 <= idx < len(scene["poses"]):
        raise ValueError(f"Frame {idx} has no matching trajectory row.")
    rgb, raw_depth, labels = (read_image(scene[k][idx]) for k in ("rgb", "depth", "semantic"))
    if rgb.shape[:2] != raw_depth.shape or raw_depth.shape != labels.shape:
        raise ValueError("RGB, depth, and semantic image dimensions disagree.")
    params = scene["camera"]
    expected_shape = (params["image_height"], params["image_width"])
    if raw_depth.shape != expected_shape:
        raise ValueError(f"Image dimensions disagree with camera_params.yaml: {raw_depth.shape} vs {expected_shape}")
    scale = float(params["png_depth_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("png_depth_scale must be positive and finite.")
    depth_m = raw_depth.astype(float) / scale
    depth_m[raw_depth == 0] = np.nan
    return rgb, raw_depth, depth_m, labels


def show_frame(scene, idx):
    import matplotlib.pyplot as plt
    rgb, raw_depth, depth_m, labels = frame_data(scene, idx)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB: frame{idx:06d}.jpg")
    heat = axes[1].imshow(depth_m, cmap="viridis", vmin=0)
    axes[1].set_title("Depth (metres); blank pixels have raw value 0")
    fig.colorbar(heat, ax=axes[1], shrink=.65, label="metres")
    # Deterministic categorical colours. Never treat class IDs as distances.
    palette = plt.get_cmap("turbo")(np.linspace(0, 1, max(scene["classes"], default=0) + 2))[:, :3]
    axes[2].imshow(palette[labels % len(palette)])
    axes[2].set_title("Semantic class IDs (display colours)")
    for ax in axes:
        ax.axis("off")
    ids, counts = np.unique(labels, return_counts=True)
    table = pd.DataFrame({"class_id": ids, "class_name": [scene["classes"].get(int(i), "UNMAPPED") for i in ids],
                          "pixels": counts, "percent_of_image": np.round(counts / labels.size * 100, 2)})
    return fig, table.sort_values("pixels", ascending=False)


def plot_trajectory(scene, idx, convention="camera_to_world", forward_sign=1):
    import matplotlib.pyplot as plt
    poses = camera_to_world(scene["poses"], convention)
    centers = poses[:, :3, 3]
    fig = plt.figure(figsize=(13, 5), layout="constrained")
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(*centers.T, linewidth=.8, color="gray")
    scatter = ax.scatter(*centers.T, c=np.arange(len(centers)), cmap="viridis", s=5)
    ax.scatter(*centers[idx], s=70, color="red", label=f"selected frame {idx}")
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="3D camera centers")
    midpoint = (centers.max(axis=0) + centers.min(axis=0)) / 2
    radius = max(np.ptp(centers, axis=0).max() / 2, .1)
    ax.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
    ax.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
    ax.set_zlim(midpoint[2] - radius, midpoint[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    fig.colorbar(scatter, ax=ax, shrink=.5, label="frame row (not seconds)")
    top = fig.add_subplot(122)
    top.plot(centers[:, 0], centers[:, 2], color="gray", linewidth=.8)
    top.scatter(centers[:, 0], centers[:, 2], c=np.arange(len(centers)), cmap="viridis", s=6)
    direction = forward_sign * poses[idx, :3, 2]
    scale = max(np.ptp(centers[:, [0, 2]], axis=0).max() * .08, .1)
    top.quiver(centers[idx, 0], centers[idx, 2], direction[0], direction[2],
               angles="xy", scale_units="xy", scale=1 / scale, color="red")
    top.scatter(centers[idx, 0], centers[idx, 2], color="red", s=40)
    top.set(xlabel="X", ylabel="Z", title="X-Z projection; arrow uses chosen forward sign")
    top.set_aspect("equal", adjustable="datalim")
    return fig


def animate_scene(scene, max_frames=24, convention="camera_to_world"):
    """Bounded inline player with its own play/pause buttons and frame slider."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    ids = sorted(scene["rgb"])
    if not ids:
        raise ValueError("Scene contains no RGB frames.")
    chosen = [ids[i] for i in np.linspace(0, len(ids) - 1, min(max_frames, len(ids)), dtype=int)]
    centers = camera_to_world(scene["poses"], convention)[:, :3, 3]
    thumbnails = []
    for idx in chosen:
        with Image.open(scene["rgb"][idx]) as im:
            im = im.convert("RGB")
            im.thumbnail((480, 320))
            thumbnails.append(np.array(im))
    fig, (view, path) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    artist = view.imshow(thumbnails[0])
    view.axis("off")
    path.plot(centers[:, 0], centers[:, 2], color="gray", linewidth=.7)
    marker, = path.plot([], [], "ro")
    path.set(xlabel="X", ylabel="Z", title="Selected camera center")
    path.set_aspect("equal", adjustable="datalim")
    def update(i):
        idx = chosen[i]
        artist.set_data(thumbnails[i])
        view.set_title(f"Frame {idx} (sample {i + 1}/{len(chosen)})")
        marker.set_data([centers[idx, 0]], [centers[idx, 2]])
        return artist, marker
    try:
        animation = FuncAnimation(fig, update, frames=len(chosen), interval=200, blit=False)
        return animation.to_jshtml(default_mode="once")
    finally:
        plt.close(fig)
