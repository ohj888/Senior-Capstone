"""Relation diagnostics with optional strict rejection; never GT verification."""
from pathlib import Path

import numpy as np


POLICY = "diagnostic_v1"
STRICT_POLICY = "geometry_guard_v1"
DEFAULTS = {"policy": POLICY, "min_views": 2, "min_agreement": 1.0,
            "contact_tolerance_m": 0.10, "height_margin_m": 0.02,
            "min_footprint_overlap": 0.10, "containment_tolerance_m": 0.05,
            "min_contained_volume_fraction": 0.80,
            "next_to_max_gap_m": 0.30, "nearby_max_gap_m": 1.50}


def settings(config, manifest):
    cfg = {**DEFAULTS, **config.get("edge_validation", {})}
    if cfg["policy"] not in (POLICY, STRICT_POLICY, "off"):
        raise ValueError("Unknown edge validation policy")
    if not isinstance(cfg["min_views"], int) or cfg["min_views"] < 1:
        raise ValueError("Edge validation min_views must be a positive integer")
    for key in ("min_agreement", "min_footprint_overlap", "min_contained_volume_fraction"):
        if not 0 < cfg[key] <= 1:
            raise ValueError(f"Invalid edge validation fraction: {key}")
    for key in ("contact_tolerance_m", "height_margin_m", "containment_tolerance_m",
                "next_to_max_gap_m", "nearby_max_gap_m"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid edge validation distance: {key}")
    up = cfg.get("world_up")
    if up is not None:
        up = np.asarray(up, dtype=float)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) == 0:
            raise ValueError("world_up must be a finite nonzero 3-vector")
        cfg["world_up"] = (up / np.linalg.norm(up)).tolist()
        cfg["world_up_source"] = "explicit configuration"
    else:
        poses = np.loadtxt(Path(manifest["staged_scene"]) / "traj.txt", ndmin=2).reshape(-1, 4, 4)
        # Camera axes are already staged as OpenCV: negative camera Y is up.
        ups = -poses[:, :3, 1]
        mean = ups.mean(axis=0)
        norm = np.linalg.norm(mean)
        axis = int(np.argmax(np.abs(mean)))
        direction = np.zeros(3)
        direction[axis] = np.sign(mean[axis])
        # Do not assume arbitrary camera tilt is gravity. Require a stable,
        # nearly axis-aligned camera-up direction; otherwise abstain on support.
        reliable = norm > 0 and abs(mean[axis]) / norm >= .98 and np.min(ups @ direction) >= .94
        cfg["world_up"] = direction.tolist() if reliable else None
        cfg["world_up_source"] = "stable axis-aligned camera-up estimate" if reliable else "unavailable"
    return cfg


def _bounds(obj):
    low, high = np.asarray(obj["bbox_min"], float), np.asarray(obj["bbox_max"], float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError("Missing or degenerate object bounds for edge validation")
    return low, high


def finalize(result, cfg):
    """Separate a failed heuristic check from the decision to drop an edge."""
    result = dict(result)
    if result["accepted"]:
        result["action"] = "keep"
    else:
        result["action"] = "flag" if cfg["policy"] == POLICY else "reject"
        result["status"] = "flagged" if cfg["policy"] == POLICY else "rejected"
    return result


def validate(label, first, second, votes, cfg):
    result = {"policy": cfg["policy"], "accepted": False,
              "status": "rejected", "ground_truth_verified": False}

    def finish(accepted, reason, **measurements):
        return finalize({**result, "accepted": accepted,
                "status": "geometry_consistent_unreviewed" if accepted else "rejected",
                "reason": reason, "issues": [] if accepted else [reason], **measurements}, cfg)

    if label is None:
        return finish(False, "unresolved_view_vote")
    if cfg["policy"] == "off":
        return {**finish(True, "checks_disabled"), "status": "unchecked"}
    if votes["views"] < cfg["min_views"] or votes["invalid"] or (votes["agreement"] or 0) < cfg["min_agreement"]:
        return finish(False, "insufficient_consistent_views")
    if label == "none":
        return {**finish(True, "no_relationship_predicted"), "status": "no_edge"}
    try:
        a_low, a_high = _bounds(first)
        b_low, b_high = _bounds(second)
    except ValueError:
        return finish(False, "insufficient_3d_geometry")
    gap = float(np.linalg.norm(np.maximum(np.maximum(a_low-b_high, b_low-a_high), 0)))
    if label in ("next_to", "nearby"):
        threshold = cfg["next_to_max_gap_m"] if label == "next_to" else cfg["nearby_max_gap_m"]
        return finish(gap <= threshold, "bbox_proximity_check", bbox_gap_m=gap)
    if label not in ("a_on_b", "b_on_a", "a_in_b", "b_in_a"):
        return finish(False, "unknown_relation_label")
    # Subject is the supported/contained object; reference is supporter/container.
    subject, reference = (first, second) if label.startswith("a_") else (second, first)
    s_low, s_high = _bounds(subject)
    r_low, r_high = _bounds(reference)
    if "_in_" in label:
        tolerance = cfg["containment_tolerance_m"]
        overlap = np.maximum(np.minimum(s_high, r_high+tolerance) - np.maximum(s_low, r_low-tolerance), 0)
        fraction = float(np.prod(overlap) / np.prod(s_high-s_low))
        container_larger = np.prod(r_high-r_low) > np.prod(s_high-s_low)
        return finish(bool(container_larger and fraction >= cfg["min_contained_volume_fraction"]),
                      "bbox_containment_check", contained_volume_fraction=fraction)
    if cfg["world_up"] is None:
        return finish(False, "support_requires_reliable_world_up")
    up = np.asarray(cfg["world_up"])
    helper = np.eye(3)[int(np.argmin(np.abs(up)))]
    horizontal = np.cross(up, helper)
    horizontal /= np.linalg.norm(horizontal)
    basis = np.column_stack((horizontal, np.cross(up, horizontal), up))

    def projected(obj):
        corners = np.asarray(obj["bbox_corners"], float) @ basis
        return corners.min(axis=0), corners.max(axis=0)

    s_low, s_high = projected(subject)
    r_low, r_high = projected(reference)
    height_difference = float((s_low[2]+s_high[2]-r_low[2]-r_high[2]) / 2)
    contact_gap = float(s_low[2] - r_high[2])
    overlap = np.maximum(np.minimum(s_high[:2], r_high[:2])-np.maximum(s_low[:2], r_low[:2]), 0)
    area = min(float(np.prod(s_high[:2]-s_low[:2])), float(np.prod(r_high[:2]-r_low[:2])))
    overlap_fraction = float(np.prod(overlap) / area) if area > 0 else 0.0
    accepted = (height_difference > cfg["height_margin_m"]
                and abs(contact_gap) <= cfg["contact_tolerance_m"]
                and overlap_fraction >= cfg["min_footprint_overlap"])
    return finish(bool(accepted), "support_height_contact_footprint_check",
                  subject_above_supporter_m=height_difference, contact_gap_m=contact_gap,
                  footprint_overlap_fraction=overlap_fraction)
