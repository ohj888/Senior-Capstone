"""Local OSMa -> original ConceptGraphs detection/mapping -> portable map.

Preparation and JSON conversion only need NumPy, Pillow and PyYAML. Heavy stages
run in ``python_executable`` with the user's installed ConceptGraphs environment.
No semantic masks, category dictionaries, QA answers, or remote APIs are read.
The dependency checkout is never edited. See BridgeConfig for all pilot knobs.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import subprocess
import sys

# Stage subprocesses execute this file directly, including from WSL.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "conceptgraphs_support"

import numpy as np
from PIL import Image
import yaml


DETECTOR_DEFAULTS = {
    "class_set": "ram", "detector": "dino", "sam_variant": "sam",
    "box_threshold": 0.25, "text_threshold": 0.25, "nms_threshold": 0.5,
    "add_bg_classes": False, "accumu_classes": True,
    "save_video": False, "use_slow_vis": False, "exp_suffix": "pilot",
}
RUNTIME_DEFAULTS = {
    "gsa_path": None, "dino_config": None, "dino_checkpoint": None,
    "dino_text_encoder_path": None, "sam_checkpoint": None,
    "sam_encoder_version": "vit_h", "ram_checkpoint": None,
    "tag2text_checkpoint": None, "clip_checkpoint": None,
    "clip_model": "ViT-H-14", "clip_tokenizer_path": None, "clip_device": None,
    "clip_crop_padding": 20, "generic_classes": None, "yolo_checkpoint": None,
    "mobile_sam_checkpoint": None, "light_hq_sam_checkpoint": None,
}
# Explicitly override the stale upstream base.yaml match_method=sep_thresh:
# this checkout's aggregate_similarities only implements sim_sum.
MAPPING_DEFAULTS = {
    "spatial_sim_type": "overlap", "match_method": "sim_sum",
    "sim_threshold": 1.2, "mask_conf_threshold": 0.25,
    "dbscan_eps": 0.1, "skip_bg": True, "max_bbox_area_ratio": 0.5,
    "obj_min_detections": 3, "save_suffix": "pilot", "save_pcd": True,
    "vis_render": False, "debug_render": False,
}


@dataclass
class BridgeConfig:
    data_root: str
    scene: str
    run_dir: str
    conceptgraphs_root: str
    python_executable: str = sys.executable
    device: str = "cuda"
    seed: int | None = 42
    frame_start: int = 0
    frame_end: int | None = None  # exclusive index in sorted source RGB files
    frame_stride: int = 1
    max_frames: int | None = None  # None is unlimited, including all scenes
    frame_ids: list[int] | None = None  # explicit original numeric frame IDs
    image_height: int | None = None  # mapping resolution; None uses scene YAML
    image_width: int | None = None
    camera_overrides: dict = field(default_factory=dict)
    pose_convention: str = "camera_to_world"
    camera_axes: str = "opencv"  # use opengl to flip camera Y and Z explicitly
    pose_atol: float = 1e-4
    file_mode: str = "copy"  # copy or hardlink; no symlink/admin requirement
    hash_sources: bool = True
    detector: dict = field(default_factory=dict)
    detector_runtime: dict = field(default_factory=dict)
    mapping: dict = field(default_factory=dict)  # all upstream Hydra keys accepted
    pair_overlap_threshold: float = 0.01
    pair_distance_threshold: float | None = None  # saved voxel size by default
    local_graph: dict = field(default_factory=dict)


def _cfg(value):
    return value if isinstance(value, BridgeConfig) else BridgeConfig(**value)


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _run_logged(command, log_path, *, cwd=None, env=None):
    """Forward child output to Jupyter and keep the traceback in a UTF-8 log."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=60)
    child_env = {**(os.environ if env is None else env),
                 "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    print("Pipeline log:", log_path, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=cwd, env=child_env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    tail.append(line)
                return_code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    if return_code:
        raise RuntimeError(
            f"ConceptGraphs process failed (exit {return_code}). Full log: {log_path}\n"
            + "".join(tail))


def _digest(value):
    # VLM changes reuse the same detection/map evidence; each graph run records
    # its own local_graph settings separately.
    value = {key: item for key, item in value.items() if key != "local_graph"}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _path(value):
    return str(Path(value).expanduser().resolve())


def _settings(config):
    cfg = _cfg(config)
    value = asdict(cfg)
    for key in ("data_root", "run_dir", "conceptgraphs_root", "python_executable"):
        value[key] = _path(value[key])
    value["detector"] = {**DETECTOR_DEFAULTS, **cfg.detector}
    value["detector_runtime"] = {**RUNTIME_DEFAULTS, **cfg.detector_runtime}
    for key, item in value["detector_runtime"].items():
        if item is not None and (key.endswith("_path") or key.endswith("_checkpoint") or key == "dino_config"):
            value["detector_runtime"][key] = _path(item)
    value["mapping"] = {**MAPPING_DEFAULTS, **cfg.mapping}
    return value


def discover_scenes(data_root):
    root = Path(data_root)
    return sorted(p.parent.relative_to(root).as_posix() for p in root.glob("*/*/traj.txt"))


def _source(path, root, hash_source):
    result = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
    if hash_source:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def prepare_scene(config):
    """Stage selected RGB/depth/poses without modifying original dataset files.

    Frames are renumbered contiguously so upstream ReplicaDataset's positional
    trajectory loader cannot confuse sparse frame IDs with pose row indices.
    The manifest retains the exact reverse mapping and per-scene calibration.
    """
    cfg, settings = _cfg(config), _settings(config)
    root, run = Path(settings["data_root"]), Path(settings["run_dir"])
    scene = (root / cfg.scene).resolve()
    if not scene.is_relative_to(root) or scene == root:
        raise ValueError("scene must identify a directory inside data_root")
    if cfg.frame_stride < 1 or cfg.frame_start < 0 or (cfg.max_frames is not None and cfg.max_frames < 1):
        raise ValueError("Use positive frame_stride/max_frames and nonnegative frame_start")
    if cfg.frame_end is not None and cfg.frame_end <= cfg.frame_start:
        raise ValueError("frame_end must exceed frame_start, or be None")
    if cfg.file_mode not in {"copy", "hardlink"}:
        raise ValueError("file_mode must be copy or hardlink")
    if cfg.pose_convention not in {"camera_to_world", "world_to_camera"}:
        raise ValueError("Unknown pose_convention")
    if cfg.camera_axes not in {"opencv", "opengl"}:
        raise ValueError("camera_axes must be opencv or opengl")
    if cfg.seed is not None and not 0 <= cfg.seed < 2 ** 32:
        raise ValueError("seed must be in [0, 2**32), or None for unseeded runs")
    if any(value is not None and value <= 0 for value in (cfg.image_height, cfg.image_width)):
        raise ValueError("image_height/image_width must be positive or None")
    manifest_path = run / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["settings_fingerprint"] != _digest(settings):
            raise ValueError("This run directory contains another configuration; choose a new run_dir")
        for row in manifest["frames"]:
            for key in ("rgb", "depth"):
                if not (Path(manifest["staged_scene"]) / "results" / row[f"staged_{key}"]).is_file():
                    raise FileNotFoundError("Staged run is incomplete; choose a new run_dir")
        for record in manifest["sources"]:
            source = root / record["path"]
            if not source.is_file() or _source(source, root, cfg.hash_sources) != record:
                raise ValueError("Source data changed after staging; use a new run_dir")
        for record in manifest.get("staged_sources", []):
            source = run / record["path"]
            if not source.is_file() or _source(source, run, cfg.hash_sources) != record:
                raise ValueError("Staged data changed; use a new run_dir")
        _json(run / "bridge_config.json", settings)
        return manifest
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Nonempty run directory has no completed manifest: {run}")
    frames = {int(p.stem[5:]): p for p in (scene / "results").glob("frame*.jpg") if p.stem[5:].isdigit()}
    if cfg.frame_ids is None:
        selected = sorted(frames)[cfg.frame_start:cfg.frame_end:cfg.frame_stride]
        if cfg.max_frames is not None:
            selected = selected[:cfg.max_frames]
    else:
        selected = list(cfg.frame_ids)
        if len(set(selected)) != len(selected) or selected != sorted(selected):
            raise ValueError("frame_ids must be unique and chronological")
    if not selected:
        raise ValueError(f"Frame selection is empty for {scene}")
    poses = np.loadtxt(scene / "traj.txt", ndmin=2)
    if poses.shape[1] != 16:
        raise ValueError("traj.txt must have sixteen numbers per pose row")
    poses = poses.reshape(-1, 4, 4)
    camera_path = scene / "camera_params.yaml"
    camera = yaml.safe_load(camera_path.read_text(encoding="utf-8"))["camera_params"]
    camera = {**camera, **cfg.camera_overrides}
    for key in ("image_height", "image_width", "fx", "fy", "png_depth_scale"):
        if not np.isfinite(camera[key]) or camera[key] <= 0:
            raise ValueError(f"Camera {key} must be finite and positive")
    for key in ("cx", "cy"):
        if not np.isfinite(camera[key]):
            raise ValueError(f"Camera {key} must be finite")
    # Upstream crop/resizing changes pixel coordinates. Avoid an invisible crop
    # by requiring zero crop_edge; users can supply a custom preprocessed dataset.
    if camera.get("crop_edge", 0) or camera.get("crop_size") is not None:
        raise ValueError("Nonzero camera crop needs an explicit bbox transform; use uncropped RGB-D calibration")
    camera["crop_edge"] = 0
    rows, chosen_poses, sources = [], [], []
    for index, frame_id in enumerate(selected):
        if frame_id not in frames or not 0 <= frame_id < len(poses):
            raise ValueError(f"Frame {frame_id} is missing RGB or its trajectory row")
        rgb, depth = frames[frame_id], scene / "results" / f"depth{frame_id:06d}.png"
        if not depth.is_file():
            raise FileNotFoundError(depth)
        pose = poses[frame_id].copy()
        if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=cfg.pose_atol):
            raise ValueError(f"Invalid homogeneous pose at original frame {frame_id}")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=cfg.pose_atol) or not np.isclose(np.linalg.det(rotation), 1, atol=cfg.pose_atol):
            raise ValueError(f"Invalid camera rotation at original frame {frame_id}")
        if cfg.pose_convention == "world_to_camera":
            pose = np.linalg.inv(pose)
        if cfg.camera_axes == "opengl":
            pose = pose @ np.diag([1, -1, -1, 1])
        with Image.open(rgb) as image, Image.open(depth) as depth_image:
            size = (int(camera["image_width"]), int(camera["image_height"]))
            if image.size != size or depth_image.size != size:
                raise ValueError(f"RGB/depth/calibration dimensions disagree for frame {frame_id}")
            if np.asarray(depth_image).ndim != 2:
                raise ValueError("Depth must be a single-channel numeric PNG")
        rows.append({"staged_index": index, "frame_id": frame_id,
                     "rgb": rgb.relative_to(root).as_posix(), "depth": depth.relative_to(root).as_posix(),
                     "staged_rgb": f"frame{index:06d}.jpg", "staged_depth": f"depth{index:06d}.png",
                     "width": size[0], "height": size[1]})
        chosen_poses.append(pose)
        sources.extend([_source(rgb, root, cfg.hash_sources), _source(depth, root, cfg.hash_sources)])
    sources.extend([_source(camera_path, root, cfg.hash_sources), _source(scene / "traj.txt", root, cfg.hash_sources)])
    staged = run / "input" / "scene"
    (staged / "results").mkdir(parents=True, exist_ok=True)
    for row in rows:
        for key in ("rgb", "depth"):
            source, destination = root / row[key], staged / "results" / row[f"staged_{key}"]
            if cfg.file_mode == "hardlink":
                os.link(source, destination)
            else:
                shutil.copy2(source, destination)
    np.savetxt(staged / "traj.txt", np.stack(chosen_poses).reshape(-1, 16), fmt="%.12g")
    dataset_config = run / "dataset.yaml"
    dataset_config.write_text(yaml.safe_dump({"dataset_name": "replica", "camera_params": camera}), encoding="utf-8")
    manifest = {"schema_version": 1, "scene": cfg.scene, "data_root": str(root),
                "settings_fingerprint": _digest(settings), "settings": settings,
                "staged_scene": str(staged), "dataset_config": str(dataset_config),
                "frames": rows, "camera": camera, "sources": sources,
                "coordinate_frame": "source world with configured camera convention/axes converted to OpenCV",
                "proposal_source": "predicted_grounded_sam", "uses_semantic_ground_truth": False}
    staged_files = [dataset_config, staged / "traj.txt"]
    staged_files.extend(staged / "results" / row[f"staged_{key}"] for row in rows for key in ("rgb", "depth"))
    manifest["staged_sources"] = [_source(path, run, cfg.hash_sources) for path in staged_files]
    _json(run / "bridge_config.json", settings)
    _json(manifest_path, manifest)
    return manifest


def _detector_name(settings):
    args = settings["detector"]
    name = args["class_set"]
    if args["sam_variant"] != "sam":
        name += "_" + args["sam_variant"]
    if args.get("exp_suffix"):
        name += "_" + args["exp_suffix"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Detector name must not contain path separators")
    return name


def map_path(config, manifest):
    settings = _settings(config)
    name = _detector_name(settings)
    suffix = settings["mapping"]["save_suffix"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(suffix)):
        raise ValueError("save_suffix must not contain path separators")
    return Path(manifest["staged_scene"]) / "pcd_saves" / f"full_pcd_{name}_{suffix}_post.pkl.gz"


def build_commands(config, manifest):
    """Return argument lists (never shell strings), safe for paths with spaces."""
    settings = _settings(config)
    python, script = settings["python_executable"], str(Path(__file__).resolve())
    config_path = str(Path(settings["run_dir"]) / "bridge_config.json")
    name = _detector_name(settings)
    base = {"dataset_root": str(Path(manifest["staged_scene"]).parent).replace("\\", "/"),
            "dataset_config": manifest["dataset_config"].replace("\\", "/"), "scene_id": "scene",
            "start": 0, "end": -1, "stride": 1, "device": settings["device"],
            "image_height": settings["image_height"] or manifest["camera"]["image_height"],
            "image_width": settings["image_width"] or manifest["camera"]["image_width"],
            "gsa_variant": name, "detection_folder_name": "gsa_detections_" + name,
            "det_vis_folder_name": "gsa_vis_" + name, "color_file_name": "gsa_classes_" + name,
            **settings["mapping"]}
    # Selection belongs in prepare_scene; changing it in mapping would break
    # observation-to-original-frame identity. All mapping algorithm keys remain free.
    protected = {"dataset_root", "dataset_config", "scene_id", "start", "end", "stride", "gsa_variant",
                 "detection_folder_name", "det_vis_folder_name", "color_file_name", "image_height", "image_width"}
    if protected.intersection(settings["mapping"]):
        raise ValueError("Change dataset/frame/resolution settings on BridgeConfig, not inside mapping")
    return {
        "detect": [python, "-u", script, "--stage", "detect", "--config", config_path],
        "map": [python, "-u", script, "--stage", "map", "--config", config_path],
        "map_upstream": [python, "-u", "-m", "conceptgraph.slam.cfslam_pipeline_batch"] +
               [f"++{key}={json.dumps(value, ensure_ascii=False)}" for key, value in base.items()] +
               [f"hydra.run.dir={json.dumps(str(Path(settings['run_dir']) / 'hydra').replace(chr(92), '/'))}", "hydra.job.chdir=false"],
        "export": [python, "-u", script, "--stage", "export", "--config", config_path],
        "graph": [python, "-u", script, "--stage", "graph", "--config", config_path],
    }


def _environment(settings):
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg"})
    source_paths = [settings["conceptgraphs_root"]]
    if settings["detector_runtime"].get("gsa_path"):
        env["GSA_PATH"] = settings["detector_runtime"]["gsa_path"]
        source_paths.extend([env["GSA_PATH"], str(Path(env["GSA_PATH"]) / "EfficientSAM")])
    env["PYTHONPATH"] = os.pathsep.join(source_paths + [env.get("PYTHONPATH", "")])
    if settings["seed"] is not None:
        env["PYTHONHASHSEED"] = str(settings["seed"])
    return env


def preflight(config, check_imports=True):
    """Inspect local files/import availability without loading model weights."""
    settings = _settings(config)
    runtime, detector = settings["detector_runtime"], settings["detector"]
    checks = []
    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})
    for key in ("data_root", "conceptgraphs_root", "python_executable"):
        check(key, Path(settings[key]).exists(), settings[key])
    for key in ("gsa_path", "dino_config", "dino_checkpoint", "clip_checkpoint"):
        check(key, runtime[key] is not None and Path(runtime[key]).exists(), runtime[key] or "Set detector_runtime." + key)
    sam_key = {"sam": "sam_checkpoint", "mobilesam": "mobile_sam_checkpoint", "lighthqsam": "light_hq_sam_checkpoint"}.get(detector["sam_variant"])
    if sam_key:
        check(sam_key, runtime[sam_key] is not None and Path(runtime[sam_key]).is_file(), runtime[sam_key] or "Set detector_runtime." + sam_key)
    else:
        check("sam_variant", False, "This upstream checkout implements sam, mobilesam and lighthqsam")
    if detector["class_set"] in {"ram", "tag2text"}:
        key = detector["class_set"] + "_checkpoint"
        check(key, runtime[key] is not None and Path(runtime[key]).is_file(), runtime[key] or "Set detector_runtime." + key)
    if detector["class_set"] == "scene":
        check("class_set", False, "scene uses upstream GT object metadata; use ram, tag2text, generic, minimal, or none")
    if detector["class_set"] == "none" and detector["sam_variant"] != "sam":
        check("dense_sam_variant", False, "This upstream checkout's dense mask generator requires sam_variant='sam'")
    if detector["detector"] == "yolo":
        check("yolo_checkpoint", runtime["yolo_checkpoint"] is not None and Path(runtime["yolo_checkpoint"]).is_file(), runtime["yolo_checkpoint"] or "Set a local YOLO-World checkpoint")
    for key in ("dino_text_encoder_path", "clip_tokenizer_path"):
        if runtime[key] is not None:
            check(key, Path(runtime[key]).exists(), runtime[key])
    if check_imports and Path(settings["python_executable"]).exists():
        modules = ["torch", "torchvision", "cv2", "open_clip", "open3d", "hydra", "omegaconf", "faiss", "pytorch3d", "supervision", "groundingdino", "segment_anything", "ram", "scipy", "gradslam", "natsort", "matplotlib", "imageio", "tqdm"]
        code = "import importlib.util,json; names=" + repr(modules) + "; print(json.dumps({n: importlib.util.find_spec(n) is not None for n in names}))"
        result = subprocess.run([settings["python_executable"], "-c", code], capture_output=True, text=True, env=_environment(settings))
        if result.returncode:
            check("python_import_probe", False, result.stderr.strip())
        else:
            for module, exists in json.loads(result.stdout.strip().splitlines()[-1]).items():
                check("import:" + module, exists, module)
    return {"ready": all(row["ok"] for row in checks), "checks": checks,
            "notes": ["Local checkpoints only; no downloads are performed.",
                      "GroundingDINO's BERT tokenizer/encoder must also be cached locally or supplied through dino_text_encoder_path.",
                      "The configured seed controls Python/NumPy/Torch and upstream 4mm point jitter; some GPU kernels can remain nondeterministic.",
                      "Package discovery does not verify binary ABI compatibility; the first heavy stage is the integration check."]}


def run_stage(config, manifest, stage):
    """Run one genuine local stage and record completion; errors propagate."""
    settings = _settings(config)
    if stage not in {"detect", "map", "export", "graph"}:
        raise ValueError("stage must be detect, map, export or graph")
    if manifest["settings_fingerprint"] != _digest(settings):
        raise ValueError("Configuration changed after staging; use a new run directory")
    commands = build_commands(config, manifest)
    if stage == "detect":
        report = preflight(config)
        if not report["ready"]:
            failures = [row["check"] + ": " + row["detail"] for row in report["checks"] if not row["ok"]]
            raise RuntimeError("Local ConceptGraphs dependencies are not ready:\n" + "\n".join(failures))
    log_path = Path(settings["run_dir"]) / "logs" / f"{stage}-{datetime.now():%Y%m%d-%H%M%S-%f}.log"
    _run_logged(commands[stage], log_path, cwd=settings["conceptgraphs_root"], env=_environment(settings))
    output = (Path(settings["run_dir"]).parent / "scene_graphs/comparison.json" if stage == "graph" else
              Path(settings["run_dir"]) / "canonical_map.json" if stage == "export" else
              map_path(config, manifest) if stage == "map" else
              Path(manifest["staged_scene"]) / ("gsa_detections_" + _detector_name(settings)))
    if not output.exists():
        raise FileNotFoundError(f"Stage {stage} returned without its expected artifact: {output}")
    _json(Path(settings["run_dir"]) / f"{stage}_complete.json", {"stage": stage, "settings_fingerprint": manifest["settings_fingerprint"], "output": str(output)})
    return output


def load_map_objects(path, manifest):
    """Convert a TRUSTED local upstream pickle into JSON-safe observations.

    Pickle is the upstream on-disk format and must only come from local runs.
    No Open3D is needed here; the saved arrays already contain the geometry.
    """
    with gzip.open(path, "rb") as handle:
        saved = pickle.load(handle)
    objects = saved["objects"] if isinstance(saved, dict) else saved
    if isinstance(saved, dict) and saved.get("cfg") is not None:
        map_cfg = saved["cfg"]
        if "dataset_root" in map_cfg and "scene_id" in map_cfg:
            expected_scene = (Path(map_cfg["dataset_root"]) / str(map_cfg["scene_id"])).resolve()
            if expected_scene != Path(manifest["staged_scene"]).resolve():
                raise ValueError("Saved map belongs to another staged scene")
    rows = manifest["frames"]
    settings = manifest["settings"]
    height = settings["image_height"] or manifest["camera"]["image_height"]
    width = settings["image_width"] or manifest["camera"]["image_width"]
    converted = []
    for index, obj in enumerate(objects):
        bounds = np.asarray(obj["bbox_np"], dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 3 or not len(bounds) or not np.isfinite(bounds).all():
            raise ValueError(f"Object {index} has invalid 3D bounds")
        minimum, maximum = bounds.min(axis=0), bounds.max(axis=0)
        image_indices, boxes = obj.get("image_idx", []), obj.get("xyxy", [])
        if len(image_indices) != len(boxes):
            raise ValueError(f"Object {index} has inconsistent observation arrays")
        observations = []
        for position, (image_index, box) in enumerate(zip(image_indices, boxes)):
            if int(image_index) != image_index or not 0 <= int(image_index) < len(rows):
                raise ValueError("Map observation index is outside the staged frame manifest")
            source = rows[int(image_index)]
            box = np.asarray(box, dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("Invalid detection bbox")
            box = box * np.array([source["width"] / width, source["height"] / height] * 2)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, source["width"])
            box[[1, 3]] = np.clip(box[[1, 3]], 0, source["height"])
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Degenerate detection bbox")
            obs = {"frame_id": source["frame_id"], "rgb": source["rgb"], "bbox": box.tolist()}
            confidence = obj.get("conf", [])
            if position < len(confidence) and confidence[position] is not None:
                obs["confidence"] = float(confidence[position])
            observations.append(obs)
        names = [str(value) for value in obj.get("class_name", [])]
        converted.append({"id": str(index), "center": ((minimum + maximum) / 2).tolist(),
                          "extent": (maximum - minimum).tolist(), "bbox_min": minimum.tolist(), "bbox_max": maximum.tolist(),
                          "bbox_corners": bounds.tolist(), "observations": observations,
                          "detection_class": Counter(names).most_common(1)[0][0] if names else None,
                          "point_count": len(obj.get("pcd_np", [])), "num_detections": int(obj.get("num_detections", len(observations)))})
    return {"scene": manifest["scene"], "objects": converted, "metadata": {"scene": manifest["scene"], "map_path": str(Path(path).resolve()),
            "settings_fingerprint": manifest["settings_fingerprint"], "coordinate_frame": manifest["coordinate_frame"],
            "bbox_representation": "axis-aligned bounds of original oriented bbox corners", "uses_semantic_ground_truth": False}}


def original_candidate_pairs(path, distance_threshold=None, overlap_threshold=0.01):
    """Exact geometric candidate policy from build_scenegraph_cfslam.py.

    Upstream calls this 'bbox overlap', but it is directional nearest-point
    overlap gated by bounding-box intersection. It keeps overlap[i,j] for i<j
    and computes a MINIMUM spanning forest on that overlap. Do not substitute
    a maximum spanning tree or symmetrize directional overlaps here.
    Caption-dependent GPT pruning is not part of this local geometric export.
    """
    from types import SimpleNamespace
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.slam.utils import compute_overlap_matrix
    with gzip.open(path, "rb") as handle:
        saved = pickle.load(handle)
    if not isinstance(saved, dict):
        raise ValueError("Candidate export needs the full map dictionary with its saved mapping config")
    threshold = distance_threshold
    if threshold is None:
        threshold = float(saved["cfg"]["downsample_voxel_size"])
    if threshold <= 0 or overlap_threshold < 0:
        raise ValueError("Pair distance must be positive; overlap threshold must be nonnegative")
    scene_map = MapObjectList()
    scene_map.load_serializable(saved["objects"])
    if not scene_map:
        return [], [], float(threshold)
    overlaps = compute_overlap_matrix(SimpleNamespace(downsample_voxel_size=threshold), scene_map)
    weights, rows, cols, all_edges = [], [], [], []
    for i in range(len(scene_map)):
        for j in range(i + 1, len(scene_map)):
            score = float(overlaps[i, j])
            if score > overlap_threshold:
                weights.extend([score, score]); rows.extend([i, j]); cols.extend([j, i])
                all_edges.append({"source": str(i), "target": str(j), "overlap": score})
    adjacency = csr_matrix((weights, (rows, cols)), shape=(len(scene_map), len(scene_map)))
    mst = minimum_spanning_tree(adjacency).tocoo()
    pairs = [{"source": str(int(i)), "target": str(int(j)), "overlap": float(score)} for i, j, score in zip(mst.row, mst.col, mst.data)]
    return pairs, all_edges, float(threshold)


def _clip_features(image, detections, clip_model, preprocess, tokenizer, classes, device, padding):
    """Upstream normalized crop/text features with configurable device/padding."""
    import torch
    crops, images, texts = [], [], []
    source = Image.fromarray(image)
    with torch.inference_mode():
        for box, category in zip(detections.xyxy, detections.class_id):
            x0, y0, x1, y1 = box
            crop = source.crop((max(0, x0 - padding), max(0, y0 - padding), min(source.width, x1 + padding), min(source.height, y1 + padding)))
            visual = clip_model.encode_image(preprocess(crop).unsqueeze(0).to(device))
            text = clip_model.encode_text(tokenizer([classes[int(category)]]).to(device))
            visual = visual / visual.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            text = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            crops.append(crop); images.append(visual.cpu().numpy()); texts.append(text.cpu().numpy())
    return crops, np.concatenate(images) if images else [], np.concatenate(texts) if texts else []


class _DeferredClip:
    """Load CLIP after RAM initialization has released its CPU checkpoint."""
    def __init__(self, loader, device=None):
        self.loader, self.device, self.components = loader, device, None

    def to(self, device):
        # Preserve an explicit CPU placement when upstream requests CUDA.
        if self.device is None:
            self.device = device
        return self

    def load(self):
        if self.components is None:
            print("Loading local CLIP after detector/tagger initialization...", flush=True)
            model, _, preprocess = self.loader()
            if self.device is not None:
                model = model.to(self.device)
            self.components = model, preprocess
        return self.components


def _detect(settings, manifest):
    """Run original detector main, replacing filesystem/model-loading defaults."""
    import importlib
    from contextlib import ExitStack
    from unittest.mock import patch
    from .conceptgraphs_clip_cache import ensure_clip_cache, load_clip_state_dict
    runtime, arguments = settings["detector_runtime"], settings["detector"]
    if arguments["class_set"] == "scene":
        raise ValueError("GT scene class lists are excluded from this baseline")
    if not runtime["clip_checkpoint"] or not Path(runtime["clip_checkpoint"]).is_file():
        raise FileNotFoundError("Supply detector_runtime.clip_checkpoint; downloads are disabled")
    clip_cache = ensure_clip_cache(runtime["clip_checkpoint"])
    import matplotlib
    import open_clip
    import open_clip.factory
    import torch
    os.environ.update(_environment(settings))
    for path in (settings["conceptgraphs_root"], runtime["gsa_path"]):
        if path:
            sys.path.insert(0, path)
    # Original detector selects TkAgg unconditionally. Use a noninteractive
    # backend in this child process; no dependency files are changed.
    matplotlib.use("Agg")
    with patch.object(matplotlib, "use"):
        upstream = importlib.import_module("conceptgraph.scripts.generate_gsa_results")
    upstream.GROUNDING_DINO_CONFIG_PATH = runtime["dino_config"]
    # Keep the upstream encoder identifier: GroundingDINO branches on the
    # literal 'bert-base-uncased'. Redirect its loaders to local files below.
    upstream.GROUNDING_DINO_CHECKPOINT_PATH = runtime["dino_checkpoint"]
    upstream.SAM_CHECKPOINT_PATH = runtime["sam_checkpoint"]
    upstream.SAM_ENCODER_VERSION = runtime["sam_encoder_version"]
    upstream.RAM_CHECKPOINT_PATH = runtime["ram_checkpoint"]
    upstream.TAG2TEXT_CHECKPOINT_PATH = runtime["tag2text_checkpoint"]
    if runtime["generic_classes"] is not None:
        upstream.FOREGROUND_GENERIC_CLASSES = list(runtime["generic_classes"])
    original_clip = open_clip.create_model_and_transforms
    original_tokenizer = open_clip.get_tokenizer
    original_clip_state_dict = open_clip.factory.load_state_dict
    def local_clip_state_dict(checkpoint, *args, **kwargs):
        if Path(checkpoint).resolve() == Path(runtime["clip_checkpoint"]).resolve():
            return load_clip_state_dict(clip_cache)
        return original_clip_state_dict(checkpoint, *args, **kwargs)
    def local_clip(*args, **kwargs):
        # ViT-H in CPU memory overlaps RAM's model/checkpoint allocations in
        # upstream's loading order, exhausting small WSL VMs before inference.
        model = _DeferredClip(lambda: original_clip(runtime["clip_model"],
                              pretrained=runtime["clip_checkpoint"], **kwargs), runtime["clip_device"])
        return model, None, None
    def local_tokenizer(*args, **kwargs):
        return original_tokenizer(runtime["clip_tokenizer_path"] or runtime["clip_model"], **kwargs)
    def local_clip_features(image, detections, model, prep, tok, classes, device):
        if isinstance(model, _DeferredClip):
            model, prep = model.load()
        return _clip_features(image, detections, model, prep, tok, classes,
                              runtime["clip_device"] or device, runtime["clip_crop_padding"])
    upstream.compute_clip_features = local_clip_features
    if arguments["sam_variant"] in {"mobilesam", "lighthqsam"}:
        def small_sam(variant, device):
            if variant == "mobilesam":
                from MobileSAM.setup_mobile_sam import setup_model
                checkpoint = runtime["mobile_sam_checkpoint"]
            else:
                from LightHQSAM.setup_light_hqsam import setup_model
                checkpoint = runtime["light_hq_sam_checkpoint"]
            model = setup_model()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            return upstream.SamPredictor(model.to(device))
        upstream.get_sam_predictor = small_sam
    args = upstream.get_parser().parse_args(["--dataset_root", str(Path(manifest["staged_scene"]).parent), "--dataset_config", manifest["dataset_config"]])
    # Attribute assignment also fixes upstream parser excluding its own default
    # sam_variant='sam' from explicit choices, without touching that source.
    vars(args).update(arguments)
    args.scene_id, args.start, args.end, args.stride = "scene", 0, -1, 1
    args.desired_height = settings["image_height"] or manifest["camera"]["image_height"]
    args.desired_width = settings["image_width"] or manifest["camera"]["image_width"]
    args.device = settings["device"]
    with ExitStack() as stack:
        stack.enter_context(patch.object(open_clip.factory, "load_state_dict", side_effect=local_clip_state_dict))
        stack.enter_context(patch.object(open_clip, "create_model_and_transforms", side_effect=local_clip))
        stack.enter_context(patch.object(open_clip, "get_tokenizer", side_effect=local_tokenizer))
        if runtime["dino_text_encoder_path"]:
            import transformers
            def redirect_loader(original):
                def load(identifier, *args, **kwargs):
                    if identifier == "bert-base-uncased":
                        identifier = runtime["dino_text_encoder_path"]
                        kwargs["local_files_only"] = True
                    return original(identifier, *args, **kwargs)
                return load
            for cls in (transformers.AutoTokenizer, transformers.BertTokenizer, transformers.BertModel):
                stack.enter_context(patch.object(cls, "from_pretrained", side_effect=redirect_loader(cls.from_pretrained)))
        if arguments["detector"] == "yolo":
            import ultralytics
            real_yolo = ultralytics.YOLO
            with patch.object(ultralytics, "YOLO", side_effect=lambda *a, **k: real_yolo(runtime["yolo_checkpoint"], **k)):
                upstream.main(args)
        else:
            upstream.main(args)


def run_local_pipeline(config, actions=("prepare", "detect", "map", "graph"), *, wsl_distro=None,
                       linux_python=None):
    """Notebook entry point, optionally using local WSL for the original stack.

    Configuration and data remain in the shared repository. All paths crossing
    the Windows/Linux boundary are translated before the child process starts.
    """
    settings = _settings(config)
    control = Path(settings["run_dir"]).parent / "pipeline_control.json"
    runtime = dict(settings)
    if wsl_distro is not None:
        if sys.platform != "win32" or not linux_python:
            raise ValueError("WSL dispatch needs a Windows notebook and linux_python.")
        def translate(value):
            if value is None or not re.match(r"^[A-Za-z]:[\\/]", str(value)):
                return value
            # --exec bypasses the WSL shell, which would consume backslashes
            # in unquoted Windows paths such as C:\ProgramData\...\python.exe.
            return subprocess.check_output(["wsl.exe", "-d", wsl_distro, "--exec", "wslpath", "-u", str(value)], text=True).strip()
        for key in ("data_root", "run_dir", "conceptgraphs_root"):
            runtime[key] = translate(settings[key])
        runtime["python_executable"] = linux_python
        runtime["detector_runtime"] = {key: translate(value) if isinstance(value, str) else value
                                       for key, value in settings["detector_runtime"].items()}
        runtime["local_graph"] = dict(settings["local_graph"])
        if runtime["local_graph"].get("worker_python"):
            runtime["local_graph"]["worker_python"] = translate(runtime["local_graph"]["worker_python"])
        script = translate(str(Path(__file__).resolve()))
        command = ["wsl.exe", "-d", wsl_distro, "--exec", linux_python, "-u", script,
                   "--stage", "pipeline", "--config", translate(str(control)), "--actions", *actions]
    else:
        command = [settings["python_executable"], "-u", str(Path(__file__).resolve()),
                   "--stage", "pipeline", "--config", str(control), "--actions", *actions]
    _json(control, runtime)
    log_path = control.parent / "logs" / f"pipeline-{datetime.now():%Y%m%d-%H%M%S-%f}.log"
    _run_logged(command, log_path)
    return control


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["detect", "map", "export", "graph", "pipeline"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--actions", nargs="+", default=["prepare", "detect", "map", "graph"])
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if args.stage == "pipeline":
        invalid = set(args.actions) - {"inspect", "prepare", "detect", "map", "export", "graph"}
        if invalid:
            raise ValueError(f"Unknown pipeline actions: {sorted(invalid)}")
        if "inspect" in args.actions:
            report = preflight(settings)
            _json(args.config.parent / "preflight.json", report)
            print(json.dumps(report, indent=2))
            if len(args.actions) == 1:
                return
        manifest = prepare_scene(settings)
        for stage in args.actions:
            if stage in ("inspect", "prepare"):
                continue
            marker = Path(settings["run_dir"]) / f"{stage}_complete.json"
            if stage != "graph" and marker.is_file():
                completed = json.loads(marker.read_text())
                if completed["settings_fingerprint"] != manifest["settings_fingerprint"] or not Path(completed["output"]).exists():
                    raise ValueError("Completed stage/config mismatch; use a new run directory.")
                print("Using completed stage:", stage, flush=True)
                continue
            print("Running original ConceptGraphs stage:", stage, flush=True)
            run_stage(settings, manifest, stage)
        return
    manifest = json.loads((Path(settings["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    os.environ.update(_environment(settings))
    sys.path.insert(0, settings["conceptgraphs_root"])
    if settings["seed"] is not None:
        import torch
        random.seed(settings["seed"])
        np.random.seed(settings["seed"])
        torch.manual_seed(settings["seed"])
    if args.stage == "detect":
        _detect(settings, manifest)
    elif args.stage == "map":
        import runpy
        command = build_commands(settings, manifest)["map_upstream"]
        sys.argv = ["conceptgraph.slam.cfslam_pipeline_batch", *command[4:]]
        runpy.run_module("conceptgraph.slam.cfslam_pipeline_batch", run_name="__main__")
    elif args.stage == "graph":
        from .conceptgraphs_local_graph import run_local_graph
        run_local_graph(settings, manifest)
    else:
        path = map_path(settings, manifest)
        canonical = load_map_objects(path, manifest)
        pairs, overlaps, distance = original_candidate_pairs(path, settings["pair_distance_threshold"], settings["pair_overlap_threshold"])
        canonical["candidate_pairs"], canonical["overlap_candidates"] = pairs, overlaps
        canonical["metadata"].update({"candidate_policy": "conceptgraphs_overlap_mst", "pair_distance_threshold": distance,
            "seed": settings["seed"],
            "pair_overlap_threshold": settings["pair_overlap_threshold"], "node_pruning": "upstream mapping filters only; original GPT caption-dependent pruning is excluded",
            "relation_backend": "local adapters supplied downstream; original remote GPT graph stage is not invoked"})
        revision = subprocess.run(["git", "-C", settings["conceptgraphs_root"], "rev-parse", "HEAD"], capture_output=True, text=True)
        canonical["metadata"]["conceptgraphs_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
        _json(Path(settings["run_dir"]) / "canonical_map.json", canonical)


if __name__ == "__main__":
    _main()
