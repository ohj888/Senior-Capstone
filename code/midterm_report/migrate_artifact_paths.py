"""Move legacy LoRA runs to artifacts/<task>/<model>/<run> without retraining.

Dry run: python code/midterm_report/migrate_artifact_paths.py
Apply:   python code/midterm_report/migrate_artifact_paths.py --apply

Only use on this project's trusted local training outputs: training_args.bin
contains pickled Transformers arguments. Changed metadata is backed up under
artifacts/path_migration_backup; model/optimizer/RNG files are hash-verified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import lora_baseline as lb


TASKS = {"gt_assisted_region_category": "objects", "visual_pair_relationship": "relationships"}


def contained(path, root):
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside artifact directory: {resolved}")
    return resolved


def plan_migration(project_root):
    artifacts = (Path(project_root) / "artifacts").resolve()
    plans = []
    for record_path in sorted((artifacts / "lora_baselines").rglob("run.json")):
        source = contained(record_path.parent, artifacts)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        task = TASKS[record["task"]]
        model = record["model_key"]
        name = source.name
        if task == "objects":
            name = name.removeprefix(f"{model}_")
        target = contained(lb.artifact_run_directory(project_root, task, model, name), artifacts)
        if target.exists() or any(p["target"] == target for p in plans):
            raise FileExistsError(f"Migration would overwrite a run: {target}")
        # An earlier manual rename may have left stale run names in checkpoint state.
        prefixes = {source.as_posix()}
        best = record.get("best_checkpoint")
        if best and "/checkpoints/" in best.replace("\\", "/"):
            prefixes.add(best.replace("\\", "/").rsplit("/checkpoints/", 1)[0])
        plans.append({"source": source, "target": target, "prefixes": sorted(prefixes),
                      "run_name": record.get("run_name", name)})
    return plans


def relocate(value, plan):
    if isinstance(value, dict):
        return {k: relocate(v, plan) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, plan) for v in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for prefix in plan["prefixes"]:
            if normalized.casefold() == prefix.casefold() or normalized.casefold().startswith(prefix.casefold() + "/"):
                return plan["target"].as_posix() + normalized[len(prefix):]
    return value


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def migrate(project_root, apply=False):
    artifacts = (Path(project_root) / "artifacts").resolve()
    plans = plan_migration(project_root)
    prepared = []
    # Read and validate every run before any move; keep a full inventory for verification.
    for plan in plans:
        source, target = plan["source"], plan["target"]
        changes, hashes = {}, {}
        for path in sorted(source.rglob("*")):
            contained(path, artifacts)
            if path.is_symlink():
                raise ValueError(f"Migration requires regular files/directories: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            hashes[relative.as_posix()] = checksum(path)
            if path.suffix == ".json":
                original = json.loads(path.read_text(encoding="utf-8"))
                updated = relocate(original, plan)
                if relative == Path("run.json"):
                    updated["run_name"] = plan["run_name"]
                    if updated.get("best_checkpoint"):
                        checkpoint = Path(updated["best_checkpoint"])
                        if checkpoint.is_absolute():
                            checkpoint = checkpoint.relative_to(target)
                        if not (source / checkpoint).is_dir():
                            raise FileNotFoundError(f"Final best checkpoint missing: {source / checkpoint}")
                        updated["best_checkpoint"] = checkpoint.as_posix()
                if updated != original:
                    changes[relative] = ("json", updated)
            elif path.name == "training_args.bin":
                import torch
                args = torch.load(path, map_location="cpu", weights_only=False)
                changed = False
                for key, value in vars(args).items():
                    if isinstance(value, str):
                        updated = relocate(value, plan)
                        if updated != value:
                            setattr(args, key, updated)
                            changed = True
                if changed:
                    changes[relative] = ("args", args)
        backup = contained(artifacts / "path_migration_backup" / source.relative_to(artifacts), artifacts)
        if backup.exists():
            raise FileExistsError(f"Metadata backup already exists: {backup}")
        prepared.append((plan, changes, hashes, backup))
        print(f"{source.relative_to(artifacts)} -> {target.relative_to(artifacts)} ({len(hashes)} files; {len(changes)} metadata updates)")
    if not apply:
        print("Dry run only. Use --apply to move these runs.")
        return plans
    for plan, changes, hashes, backup in prepared:
        source, target = plan["source"], plan["target"]
        for relative in changes:
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination)
        lb.save_json(backup / "migration.json", {"source": str(source), "target": str(target), "sha256_before": hashes})
        contained(source, artifacts)
        contained(target, artifacts)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        for relative, (kind, value) in changes.items():
            if kind == "json":
                lb.save_json(target / relative, value)
            else:
                torch.save(value, target / relative)
        after_files = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
        if after_files != set(hashes):
            raise RuntimeError(f"File inventory changed for {target}")
        for relative, digest in hashes.items():
            if Path(relative) not in changes and checksum(target / relative) != digest:
                raise RuntimeError(f"File content changed: {target / relative}")
        print(f"Verified unchanged weights and other non-metadata files: {target.relative_to(artifacts)}")
    # Remove only verified empty legacy directories, never recursively delete files.
    legacy = contained(artifacts / "lora_baselines", artifacts)
    if legacy.exists():
        for path in sorted(legacy.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            contained(path, artifacts)
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        if not any(legacy.iterdir()):
            legacy.rmdir()
    return plans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Move runs after validating the entire migration")
    args = parser.parse_args()
    migrate(Path(__file__).resolve().parents[2], apply=args.apply)
