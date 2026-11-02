"""Disk-backed CLIP weights for the older, memory-limited mapping runtime.

Convert the trusted local PyTorch checkpoint in a separate process before any
detector models load. Small safetensors shards preserve tensor values/dtypes
and let OpenCLIP copy file-backed weights into its model without a second
full anonymous-memory checkpoint allocation (Torch 2.0 has no mmap loader).
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _cache_path(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    stat = checkpoint.stat()
    stamp = {"path": str(checkpoint), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
             "format_version": 1}
    identity = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:16]
    return checkpoint.with_name(checkpoint.name + ".safetensors-" + identity), stamp


def ensure_clip_cache(checkpoint):
    directory, stamp = _cache_path(checkpoint)
    index = directory / "index.json"
    if index.is_file():
        record = json.loads(index.read_text(encoding="utf-8"))
        if record["source"] == stamp and all(
                (directory / name).is_file() and (directory / name).stat().st_size == size
                for name, size in record["files"].items()):
            return directory
    print("Preparing disk-backed CLIP weights (one-time local conversion)...", flush=True)
    subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                    "--checkpoint", str(Path(checkpoint).resolve())], check=True)
    return directory


def build_clip_cache(checkpoint, shard_bytes=128 * 1024 * 1024):
    import torch
    from safetensors.torch import save_file

    directory, stamp = _cache_path(checkpoint)
    directory.mkdir(parents=True, exist_ok=True)
    # This worker holds the checkpoint alone, without CLIP/SAM/RAM models.
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint_data.get("state_dict", checkpoint_data)
    files, keys, batch, batch_bytes = {}, [], {}, 0

    def save_batch():
        name = f"weights-{len(files):04d}.safetensors"
        path = directory / name
        save_file(batch, str(path))
        files[name] = path.stat().st_size
        print("Cached CLIP shard:", name, flush=True)
        batch.clear()

    strip_prefix = next(iter(state)).startswith("module.")
    for key, value in state.items():
        key = key[7:] if strip_prefix else key
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"CLIP checkpoint entry {key} is not a tensor")
        size = value.numel() * value.element_size()
        if batch and batch_bytes + size > shard_bytes:
            save_batch()
            batch_bytes = 0
        # Independent contiguous storage also handles tied checkpoint tensors.
        batch[key] = value.detach().contiguous().clone()
        batch_bytes += size
        keys.append(key)
    if batch:
        save_batch()
    index = directory / "index.json"
    temporary = index.with_suffix(".tmp")
    temporary.write_text(json.dumps({"source": stamp, "files": files, "keys": keys}, indent=2),
                         encoding="utf-8")
    temporary.replace(index)
    return directory


def load_clip_state_dict(directory):
    from safetensors.torch import load_file

    directory = Path(directory)
    record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    state = {}
    for name in record["files"]:
        tensors = load_file(str(directory / name), device="cpu")
        if state.keys() & tensors.keys():
            raise ValueError("Duplicate CLIP checkpoint keys across cache shards")
        state.update(tensors)
    if set(state) != set(record["keys"]):
        raise ValueError("Incomplete CLIP checkpoint cache")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    build_clip_cache(parser.parse_args().checkpoint)
