"""Shared local OSMa object-naming and relationship LoRA pipelines.

The semantic masks supply weak REGION proposals and category targets. They do
not supply instance identities, captions, or scene-graph relationship labels.
"""
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image


MODELS = {
    "qwen": {"folder": "qwen3-vl-2b-instruct", "repo_id": "Qwen/Qwen3-VL-2B-Instruct",
             "revision": "89644892e4d85e24eaac8bacfd4f463576704203"},
    "smolvlm": {"folder": "smolvlm2-2.2b-instruct", "repo_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                "revision": "482adb537c021c86670beed01cd58990d01e72e4"},
}
PROMPT = "Name the main object or semantic region in this RGB crop. Reply with only a short category name."
DEFAULT_DATA = {
    "seed": 42, "frames_per_scene": 6, "max_regions_per_frame": 4,
    "max_examples_per_class_per_scene": 4, "component_grid_size": 128,
    "min_grid_area": 12, "min_box_side": 24, "min_category_fraction": 0.45,
    "crop_margin": 0.06, "image_max_side": 384,
    "excluded_classes": ["other", "unknown", "unlabeled", "background", "wall", "floor", "ceiling"],
    "group_overrides": {},
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def artifact_run_directory(project_root, task, model_key, run_name):
    """Readable layout: artifacts/<task>/<model>/<run>; data identity lives in run.json."""
    if task not in ("objects", "relationships") or model_key not in MODELS:
        raise ValueError("Unknown artifact task or model.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        raise ValueError("RUN_NAME must contain only letters, numbers, underscores or hyphens.")
    return Path(project_root) / "artifacts" / task / model_key / run_name


def read_run_record(run_dir, *, model_key=None, task=None, dataset_fingerprint=None):
    """Validate saved-run identity before allocating GPU memory or reusing results."""
    record = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    if not record.get("complete"):
        raise ValueError("Saved run is incomplete.")
    for key, expected in (("model_key", model_key), ("task", task),
                          ("dataset_fingerprint", dataset_fingerprint)):
        if expected is not None and record.get(key) != expected:
            raise ValueError(f"Saved run {key} does not match the requested model/task/dataset.")
    if record.get("model") != MODELS.get(record.get("model_key")):
        raise ValueError("Saved run base-model revision does not match this code.")
    return record


def relative_best_checkpoint(output_dir, checkpoint):
    """Keep our metadata portable; Trainer maintains its own checkpoint state."""
    return Path(checkpoint).resolve().relative_to(Path(output_dir).resolve()).as_posix() if checkpoint else None


def available_run_directory(requested):
    """Pick a fresh sibling without deleting, overwriting, or resuming a run.

    This only selects a name. train_pilot still creates it with exist_ok=False
    so a concurrent process cannot cause an existing run to be overwritten.
    """
    requested = Path(requested)
    candidate = requested
    suffix = 1
    while candidate.exists() or candidate.is_symlink():
        candidate = requested.with_name(f"{requested.name}_{suffix:03d}")
        suffix += 1
    return candidate


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_label(label):
    return " ".join(str(label).replace("_", " ").strip(" \n\t.\"'").casefold().split())


def scene_group(relative_scene, overrides=None):
    if relative_scene in (overrides or {}):
        return overrides[relative_scene]
    family, name = relative_scene.split("/")
    # Conservatively group v3_sc0_staging_* together, etc. This is a filename
    # heuristic, not proof that all remaining layouts are independent.
    match = re.match(r"(v\d+_sc\d+)_staging_", name)
    return f"{family}/{match.group(1) if match else name}"


def assign_splits(scenes, seed=42, overrides=None):
    groups_by_family = defaultdict(set)
    group_for_scene = {}
    for name in sorted(scenes):
        group = scene_group(name, overrides)
        group_for_scene[name] = group
        groups_by_family[name.split("/")[0]].add(group)
    split_for_group = {}
    for family, groups in sorted(groups_by_family.items()):
        groups = sorted(groups)
        if len(groups) < 3:
            raise ValueError(f"Need at least three independent groups in {family} to split train/validation/test.")
        random.Random(f"{seed}:{family}").shuffle(groups)
        held_out = max(1, round(len(groups) * 0.2))
        for i, group in enumerate(groups):
            split = "test" if i < held_out else "validation" if i < 2 * held_out else "train"
            if group in split_for_group and split_for_group[group] != split:
                raise ValueError("A group override spans families with conflicting splits.")
            split_for_group[group] = split
    return {name: {"group": group, "split": split_for_group[group]} for name, group in group_for_scene.items()}


def semantic_components(labels, grid_size=128):
    """Coarse 4-connected components, using only NumPy/Python (no SciPy/OpenCV).

    A component can merge touching same-class objects or split an occluded one.
    It is a region proposal, NEVER a ground-truth object instance.
    """
    if labels.ndim != 2:
        raise ValueError("Expected a 2D semantic class-ID image.")
    step = max(1, math.ceil(max(labels.shape) / grid_size))
    grid = labels[::step, ::step]
    height, width = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue
            category = int(grid[y, x])
            stack = [(y, x)]
            visited[y, x] = True
            minx = maxx = x
            miny = maxy = y
            area = 0
            while stack:
                cy, cx = stack.pop()
                minx, maxx = min(minx, cx), max(maxx, cx)
                miny, maxy = min(miny, cy), max(maxy, cy)
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and grid[ny, nx] == category:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            yield category, [minx * step, miny * step, min((maxx + 1) * step, labels.shape[1]),
                             min((maxy + 1) * step, labels.shape[0])], area


def prepare_examples(data_root, settings=None):
    data_root = Path(data_root)
    cfg = {**DEFAULT_DATA, **(settings or {})}
    scenes = sorted(p.parent for p in data_root.glob("*/*/traj.txt"))
    if not scenes:
        raise FileNotFoundError(f"No OSMa-Bench scenes at {data_root}")
    names = [p.relative_to(data_root).as_posix() for p in scenes]
    splits = assign_splits(names, cfg["seed"], cfg["group_overrides"])
    rows, sources = [], []
    excluded = {normalize_label(x) for x in cfg["excluded_classes"]}
    def source_record(path):
        # Hash selected source files so both models can use exactly the same data.
        sources.append({"path": path.relative_to(data_root).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for scene in scenes:
        name = scene.relative_to(data_root).as_posix()
        # 1. OSMa provides a category dictionary INSIDE EACH SCENE:
        #    dataset/osma-bench/data/<family>/<scene>/embed_semseg_classes.json
        #    Its "classes" list contains {"id": integer, "name": category}.
        #    Read the scene's own dictionary; do not assume IDs match across scenes.
        classes_path = scene / "embed_semseg_classes.json"
        classes = {int(c["id"]): normalize_label(c["name"])
                   for c in json.loads(classes_path.read_text(encoding="utf-8"))["classes"]}
        source_record(classes_path)
        frames = sorted((scene / "results").glob("frame*.jpg"))
        selected = np.linspace(0, len(frames) - 1, min(cfg["frames_per_scene"], len(frames)), dtype=int)
        class_counts = Counter()
        for index in selected:
            rgb_path = frames[index]
            frame = int(rgb_path.stem.removeprefix("frame"))
            # 2. The matching semantic PNG is the pixel-level supervision:
            #    results/frame000000.jpg <-> results/semantic000000.png.
            #    labels[y, x] is a CATEGORY ID, not a color or an object-instance ID.
            #    Preserve the integer values (some IDs exceed 255); converting
            #    this PNG to RGB or 8-bit grayscale would lose its meaning.
            semantic_path = rgb_path.with_name(f"semantic{frame:06d}.png")
            with Image.open(semantic_path) as im:
                labels = np.array(im)
            with Image.open(rgb_path) as im:
                if im.size != (labels.shape[1], labels.shape[0]):
                    raise ValueError(f"RGB/semantic dimensions differ: {rgb_path}")
            source_record(rgb_path)
            source_record(semantic_path)
            candidates = []
            for category, box, area in semantic_components(labels, cfg["component_grid_size"]):
                # 3. A connected region gives us a crop box and one category ID.
                #    Look that ID up in the JSON to obtain the answer text.
                #    For example, IF this scene maps an ID to "chair", the
                #    target is "chair". The VLM does not predict the numeric ID.
                target = classes.get(category)
                x0, y0, x1, y1 = box
                if not target or target in excluded or area < cfg["min_grid_area"]:
                    continue
                if min(x1 - x0, y1 - y0) < cfg["min_box_side"]:
                    continue
                purity = float(np.mean(labels[y0:y1, x0:x1] == category))
                if purity < cfg["min_category_fraction"]:
                    continue
                margin = round(max(x1 - x0, y1 - y0) * cfg["crop_margin"])
                crop_box = [max(0, x0 - margin), max(0, y0 - margin),
                            min(labels.shape[1], x1 + margin), min(labels.shape[0], y1 + margin)]
                candidates.append((target, category, crop_box, purity))
            # Deterministic random selection avoids always taking only the largest objects.
            random.Random(f"{cfg['seed']}:{name}:{frame}").shuffle(candidates)
            accepted = 0
            for target, category, box, purity in candidates:
                if class_counts[target] >= cfg["max_examples_per_class_per_scene"]:
                    continue
                key = {"scene": name, "frame": frame, "bbox": box, "category_id": category}
                # Store provenance separately from the VLM input:
                # rgb + bbox describe its image; target is its supervised answer.
                # Neither the semantic PNG nor category_id is sent to the VLM.
                rows.append({"id": fingerprint(key)[:20], **key, **splits[name],
                             "rgb": rgb_path.relative_to(data_root).as_posix(),
                             "target": target, "category_fraction": purity,
                             "proposal_source": "gt_semantic_connected_region"})
                class_counts[target] += 1
                accepted += 1
                if accepted >= cfg["max_regions_per_frame"]:
                    break
        print(f"Prepared {name}: {sum(class_counts.values())} region examples ({splits[name]['split']}).")
    for split in ("train", "validation", "test"):
        if not any(row["split"] == split for row in rows):
            raise ValueError(f"No examples in {split}; inspect filters and grouping before training.")
    core = {"schema_version": 1, "task": "gt_assisted_region_category", "settings": cfg,
            "prompt": PROMPT, "scenes": splits, "sources": sources, "examples": rows}
    return {**core, "fingerprint": fingerprint(core)}


def load_crop(row, data_root, max_side=384):
    root = Path(data_root).resolve()
    source = (root / row["rgb"]).resolve()
    if not source.is_relative_to(root):
        raise ValueError("RGB path must stay inside the dataset.")
    with Image.open(source) as image:
        x0, y0, x1, y1 = row["bbox"]
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError("Invalid crop bounds")
        # 4. Crop the ordinary RGB photograph, NOT the semantic mask.
        # The mask guided localization, but the VLM sees natural RGB pixels.
        crop = image.convert("RGB").crop((x0, y0, x1, y1))
        crop.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return crop


def user_message(image):
    # Only RGB pixels and this fixed prompt enter the model. No GT mask, class
    # ID, source filename, QA answer, or scene name is included in the prompt.
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": PROMPT}]}]


def encode_example(processor, image, target=None):
    """Batch size one; loss covers only the assistant answer and its ending."""
    return encode_messages(processor, user_message(image), target)


def encode_messages(processor, messages, target=None):
    """Shared answer-only loss masking for object names and relation labels."""
    import torch
    messages = list(messages)  # Appending a training answer must not alter caller input.
    # SmolVLM adds video-processing defaults even for an image-only chat. In
    # current Transformers it puts them in legacy **kwargs unless it receives
    # a NONEMPTY processor_kwargs dict, producing a warning for every crop.
    # Route tensor conversion through that dict; chat-format controls stay above it.
    prompt = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, processor_kwargs={"return_tensors": "pt"})
    if target is None:
        # Inference: ask the question without supplying the correct category.
        return prompt
    # Training: the object category or reviewed relation is the assistant response.
    # Teacher forcing includes this response in the training sequence; the
    # causal next-token loss trains the model to predict it from earlier tokens.
    messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    full = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                         return_dict=True, processor_kwargs={"return_tensors": "pt"})
    prefix = prompt["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][:, :prefix], prompt["input_ids"]):
        raise ValueError("Chat-template prefix mismatch. Stop and fix label masking for this processor.")
    # These "labels" are TOKEN IDs for the language-model loss. They are not
    # the pixel-category IDs in the semantic PNG. Same word, different purpose.
    labels = full["input_ids"].clone()
    labels[:, :prefix] = -100  # PyTorch ignores these prompt/image positions.
    labels[full["attention_mask"] == 0] = -100
    if not (labels != -100).any():
        raise ValueError("There are no assistant target tokens to train on.")
    full["labels"] = labels
    return full


class RegionCollator:
    def __init__(self, processor, data_root, max_side=384):
        self.processor, self.data_root, self.max_side = processor, data_root, max_side

    def __call__(self, examples):
        if len(examples) != 1:
            raise ValueError("This teaching collator supports microbatch size 1; use gradient accumulation.")
        row = examples[0]
        # Complete bridge: RGB crop + fixed question -> category text from OSMa.
        # During training the loss updates LoRA parameters, not the dataset labels.
        return encode_example(self.processor, load_crop(row, self.data_root, self.max_side), row["target"])


def decoder_lora_targets(model):
    # Explicit full module names: a broad 'q_proj' match could also adapt the
    # vision encoder. The two architectures have different decoder prefixes.
    targets = [name for name, _ in model.named_modules()
               if any(part in name for part in (".language_model.", ".text_model."))
               and name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj"))]
    if not targets:
        raise ValueError("No supported language-decoder attention layers found.")
    return targets


def attach_lora(model, rank=8, alpha=16, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    targets = decoder_lora_targets(model)
    config = LoraConfig(task_type="CAUSAL_LM", r=rank, lora_alpha=alpha,
                        lora_dropout=dropout, target_modules=targets, bias="none")
    model = get_peft_model(model, config)
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Unexpected trainable base parameters; stop before training.")
    model.print_trainable_parameters()
    return model, config, trainable


def local_model_path(project_root, model_key):
    spec = MODELS[model_key]
    path = Path(project_root) / "dependencies" / "models" / spec["folder"]
    manifest_path = path / ".baseline-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run the baseline model cell in download_dependencies.ipynb first: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or any(manifest.get(k) != spec[k] for k in ("repo_id", "revision")):
        raise ValueError("Model download is incomplete or has the wrong revision. Rerun the download verification cell.")
    for item in manifest["files"]:
        file = path / item["name"]
        if not file.is_file() or file.stat().st_size != item["size"]:
            raise ValueError(f"Missing/incomplete model file: {file}. Rerun download verification.")
    return path


def check_processor_dependencies(model_key):
    """Check SmolVLM's extra dependency before loading weights (also on reload)."""
    if model_key != "smolvlm":
        return
    import sys
    try:
        from num2words import num2words
    except ImportError as exc:
        raise ImportError(
            f"SmolVLM needs num2words in this kernel's Python: {sys.executable}. "
            "Run `import sys, subprocess; subprocess.check_call([sys.executable, "
            "'-m', 'pip', 'install', 'num2words'])` in a notebook cell, then "
            "RESTART THE KERNEL before rerunning. Installing in another terminal "
            "environment or only reloading lora_baseline is not sufficient."
        ) from exc
    # Transformers sets a module-level num2words=None if it is missing when
    # SmolVLM's processor first imports. A later pip install does not reset it.
    cached = sys.modules.get("transformers.models.smolvlm.processing_smolvlm")
    if cached is not None and getattr(cached, "num2words", None) is None:
        raise ImportError("num2words is installed, but SmolVLM cached it as unavailable. Restart the notebook kernel, then rerun.")


def load_local_base(project_root, model_key, use_4bit=True):
    check_processor_dependencies(model_key)
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, SmolVLMForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("This training pilot requires CUDA. Preparation/preview cells work without a GPU.")
    path = local_model_path(project_root, model_key)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if model_key == "qwen":
        # Bound the image-token budget; the class crop still preserves aspect ratio.
        processor.image_processor.size = {"shortest_edge": 4 * 32 * 32, "longest_edge": 160 * 32 * 32}
        cls = Qwen3VLForConditionalGeneration
    else:
        processor.image_processor.do_image_splitting = False
        cls = SmolVLMForConditionalGeneration
    kwargs = {"local_files_only": True, "trust_remote_code": False, "dtype": dtype,
              "device_map": {"": 0}, "attn_implementation": "sdpa"}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = cls.from_pretrained(path, **kwargs)
    return model, processor


def load_trainable(project_root, model_key, rank=8, alpha=16, dropout=0.05, use_4bit=True, seed=42):
    from peft import prepare_model_for_kbit_training
    from transformers import set_seed
    set_seed(seed)  # Seed adapter initialization, not just the later training loop.
    model, processor = load_local_base(project_root, model_key, use_4bit)
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model, config, parameters = attach_lora(model, rank, alpha, dropout)
    return model, processor, config, parameters


def evaluation_rows(manifest, split="validation", limit=30):
    # Fixed selection shared by both models, including pretrained/adapted runs.
    return sorted((r for r in manifest["examples"] if r["split"] == split), key=lambda r: r["id"])[:limit]


def classification_metrics(predictions):
    by_class = defaultdict(list)
    for item in predictions:
        by_class[item["target"]].append(item["correct"])
    return {"examples": len(predictions),
            "exact_match": sum(x["correct"] for x in predictions) / max(1, len(predictions)),
            "macro_class_recall": float(np.mean([np.mean(v) for v in by_class.values()])) if by_class else None,
            "classes_evaluated": len(by_class)}


def predict_rows(model, processor, rows, data_root, max_side=384, disable_adapter=False):
    import torch
    was_training = model.training
    model.eval()
    predictions = []
    context = model.disable_adapter() if disable_adapter else nullcontext()
    try:
        with context, torch.inference_mode():
            for row in rows:
                batch = encode_example(processor, load_crop(row, data_root, max_side)).to(model.device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                generated = model.generate(**batch, max_new_tokens=24, do_sample=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                answer = processor.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                predictions.append({"id": row["id"], "scene": row["scene"], "target": row["target"],
                                    "prediction": answer, "correct": normalize_label(answer) == row["target"],
                                    "generation_seconds": time.perf_counter() - start})
    finally:
        model.train(was_training)
    return predictions


def training_arguments(output_dir, training, seed, train_examples, use_cpu=False):
    """Validate Trainer options before running expensive baseline predictions."""
    import torch
    from transformers import TrainingArguments
    # warmup_ratio was removed from newer Transformers releases. An explicit
    # integer warmup_steps also works with older releases supporting these VLMs.
    # This notebook uses one GPU and a microbatch of one.
    steps_per_epoch = math.ceil(train_examples / training["accumulation"])
    total_steps = (training["max_steps"] if training["max_steps"] > 0
                   else math.ceil(steps_per_epoch * training["epochs"]))
    bf16 = not use_cpu and torch.cuda.is_bf16_supported()
    return TrainingArguments(
        output_dir=str(Path(output_dir) / "checkpoints"), max_steps=training["max_steps"], num_train_epochs=training["epochs"],
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=training["accumulation"],
        learning_rate=training["learning_rate"], warmup_steps=math.ceil(total_steps * 0.05), optim="adamw_torch",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cpu=use_cpu, bf16=bf16, fp16=not use_cpu and not bf16,
        eval_strategy="steps", eval_steps=training["eval_steps"],
        save_strategy="steps", save_steps=training["eval_steps"], save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=5, report_to="none", push_to_hub=False, remove_unused_columns=False,
        label_names=["labels"], dataloader_num_workers=0, seed=seed,
    )


def train_pilot(model, processor, manifest, data_root, output_dir, model_key, training, use_4bit=True):
    import torch
    from transformers import Trainer, set_seed
    output_dir = Path(output_dir)
    train_rows = [r for r in manifest["examples"] if r["split"] == "train"]
    val_rows = [r for r in manifest["examples"] if r["split"] == "validation"]
    args = training_arguments(output_dir, training, manifest["settings"]["seed"], len(train_rows))
    # A new run directory protects previous adapters and measurements.
    output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(manifest["settings"]["seed"])
    eval_rows = evaluation_rows(manifest, "validation", training["eval_examples"])
    metadata = {"model": MODELS[model_key], "model_key": model_key, "run_name": output_dir.name,
                "dataset_fingerprint": manifest["fingerprint"],
                "task": manifest["task"], "training": training, "use_4bit": use_4bit,
                "lora": model.peft_config["default"].to_dict(), "complete": False,
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")}}
    metadata["helper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if use_4bit:
        metadata["versions"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
    # PEFT config may contain sets/enums; serialize them to ordinary JSON values.
    metadata = json.loads(json.dumps(metadata, default=lambda x: sorted(x) if isinstance(x, set) else str(x)))
    save_json(output_dir / "run.json", metadata)
    save_json(output_dir / "data_manifest.json", manifest)
    max_side = manifest["settings"]["image_max_side"]
    trainer = Trainer(model=model, args=args, train_dataset=train_rows, eval_dataset=val_rows,
                      data_collator=RegionCollator(processor, data_root, max_side))
    before = predict_rows(model, processor, eval_rows, data_root, max_side, disable_adapter=True)
    save_json(output_dir / "pretrained_validation.json", {"metrics": classification_metrics(before), "predictions": before})
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    # Save the small adapter and processor. The original weights stay untouched.
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    after = predict_rows(model, processor, eval_rows, data_root, max_side)
    save_json(output_dir / "adapted_validation.json", {"metrics": classification_metrics(after), "predictions": after})
    metadata.update(complete=True, optimizer_steps=trainer.state.global_step,
                    best_checkpoint=relative_best_checkpoint(output_dir, trainer.state.best_model_checkpoint),
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
    save_json(output_dir / "run.json", metadata)
    return {"adapter_dir": str(adapter_dir), "pretrained": classification_metrics(before), "adapted": classification_metrics(after)}


def reload_adapter(project_root, run_dir, *, model_key=None, task=None, dataset_fingerprint=None):
    run_dir = Path(run_dir)
    record = read_run_record(run_dir, model_key=model_key, task=task, dataset_fingerprint=dataset_fingerprint)
    from peft import PeftModel
    from transformers import AutoProcessor
    base, _ = load_local_base(project_root, record["model_key"], record["use_4bit"])
    model = PeftModel.from_pretrained(base, run_dir / "adapter", is_trainable=False, local_files_only=True)
    processor = AutoProcessor.from_pretrained(run_dir / "adapter", local_files_only=True, trust_remote_code=False)
    model.eval()
    return model, processor, record


# ---------------------------------------------------------------------------
# Relationship fine-tuning (finetune_vlm2.ipynb). Object-naming APIs above stay
# available to notebook 1. This task labels GIVEN pairs; it does not learn which
# pairs ConceptGraphs should select or solve its tracking/fusion errors.
# ---------------------------------------------------------------------------
RELATIONS = ("a_on_b", "b_on_a", "a_in_b", "b_in_a", "next_to", "nearby", "none")
# Secondary reporting preference, NOT confidence or a training-loss multiplier.
# Primary exact match and per-relation F1 remain unweighted.
RELATION_SCORE_WEIGHTS = {r: (0.5 if r == "nearby" else 1.0) for r in RELATIONS}
RELATION_PROMPT = (
    "Determine the physical relationship between the two marked objects. "
    "A has a red rectangle; B has a cyan rectangle. "
    "Reply with exactly one label: a_on_b, b_on_a, a_in_b, b_in_a, next_to, nearby, none. "
    "on means resting on and supported by the other object, not merely above it. "
    "in means physically contained inside the other object. "
    "next_to means physically side-by-side immediate neighbors with a small gap and no intervening object. "
    "nearby means physically close in the same local area but not immediate side-by-side neighbors. "
    "next_to and nearby are symmetric: swapping A and B does not change them. "
    "Choose the most specific applicable relation: in first, then on, then next_to, then nearby. "
    "none means none of these relationships applies. Image-plane closeness alone is not physical proximity."
)
RELATION_DATA_DEFAULTS = {"pairs_per_scene": 12, "image_max_side": 512, "seed": 42}


def _read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {number}") from exc
    return rows


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    temporary.replace(path)


def _dataset_file(data_root, relative):
    root = Path(data_root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Dataset file missing or outside data root: {relative}")
    return path


def _validate_pair(row, data_root):
    with Image.open(_dataset_file(data_root, row["rgb"])) as im:
        for key in ("bbox_a", "bbox_b"):
            box = row[key]
            if len(box) != 4 or not all(isinstance(x, int) for x in box):
                raise ValueError(f"Invalid {key}: {row['id']}")
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height):
                raise ValueError(f"Out-of-image {key}: {row['id']}")
    if row["bbox_a"] == row["bbox_b"]:
        raise ValueError(f"Pair uses the same rectangle twice: {row['id']}")


def relation_image(row, data_root, max_side=512):
    """Keep spatial context; identify A/B without drawing a predicted relation."""
    from PIL import ImageDraw, ImageFont
    with Image.open(_dataset_file(data_root, row["rgb"])) as im:
        original_size = im.size
        image = im.convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    sx, sy = image.width / original_size[0], image.height / original_size[1]
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    for tag, key, color in (("A", "bbox_a", "red"), ("B", "bbox_b", "cyan")):
        x0, y0, x1, y1 = row[key]
        box = (round(x0*sx), round(y0*sy), min(image.width-1, round(x1*sx)), min(image.height-1, round(y1*sy)))
        draw.rectangle(box, outline=color, width=2)
        tx, ty = min(box[0], max(0, image.width-20)), max(0, box[1]-21)
        draw.rectangle((tx, ty, tx+19, ty+20), fill=color)
        draw.text((tx+3, ty+1), tag, fill="black", font=font)
    return image


def relation_messages(row, data_root, max_side=512):
    # Category hints, semantic masks, reviewer notes, split and label are NOT inputs.
    return [{"role": "user", "content": [
        {"type": "image", "image": relation_image(row, data_root, max_side)},
        {"type": "text", "text": RELATION_PROMPT},
    ]}]


def prepare_relation_dataset(project_root, dataset_dir, source_manifest=None, settings=None):
    """Prepare reviewable pairs, NEVER infer GT relationships from class masks.

    This initial dataset uses semantic-region proposals from notebook 1. It is
    GT-assisted localization, not a full ConceptGraphs edge-construction score.
    The reviewer must verify that each marked region identifies ONE distinct
    physical object. Touching same-class instances can invalidate a proposal.
    """
    from itertools import combinations
    project_root, dataset_dir = Path(project_root), Path(dataset_dir)
    cfg = {**RELATION_DATA_DEFAULTS, **(settings or {})}
    if cfg["pairs_per_scene"] < 1 or cfg["image_max_side"] < 64:
        raise ValueError("Use positive pairs_per_scene and image_max_side >= 64.")
    if source_manifest is None:
        choices = sorted((project_root / "dataset/_training/osma_regions").glob("*/manifest.json"))
        if len(choices) != 1:
            raise ValueError(f"Set SOURCE_MANIFEST to one object manifest. Found: {[str(p) for p in choices]}")
        source_manifest = choices[0]
    source = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    core = {k: v for k, v in source.items() if k != "fingerprint"}
    if fingerprint(core) != source["fingerprint"]:
        raise ValueError("Object manifest fingerprint is invalid; regenerate it in notebook 1.")
    meta_path = dataset_dir / "dataset.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["source_fingerprint"] != source["fingerprint"] or meta["settings"] != cfg:
            raise ValueError("Dataset already uses other settings. Choose a new DATASET_NAME; existing labels are preserved.")
        upgrade_relation_vocabulary(dataset_dir)
        return relation_dataset_summary(dataset_dir)
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        raise ValueError("Partial dataset directory found. Choose a new DATASET_NAME; no files were overwritten.")
    data_root = project_root / "dataset/osma-bench/data"
    by_scene_frame = defaultdict(list)
    for row in source["examples"]:
        by_scene_frame[(row["scene"], row["frame"])].append(row)
    by_scene = defaultdict(list)
    for (scene, frame), objects in sorted(by_scene_frame.items()):
        for a, b in combinations(sorted(objects, key=lambda x: x["id"]), 2):
            # Vary A/B order deterministically; do not make A always the lamp, etc.
            if random.Random(f"{cfg['seed']}:{a['id']}:{b['id']}").random() < 0.5:
                a, b = b, a
            info = source["scenes"][scene]
            identity = {"scene": scene, "frame": frame, "a": a["id"], "b": b["id"]}
            by_scene[scene].append({"id": fingerprint(identity)[:20], "scene": scene, "frame": frame,
                **info, "rgb": a["rgb"], "bbox_a": a["bbox"], "bbox_b": b["bbox"],
                "hint_a": a["target"], "hint_b": b["target"],
                "proposal_source": "gt_semantic_regions_not_instances"})
    pairs = []
    for scene, candidates in sorted(by_scene.items()):
        random.Random(f"{cfg['seed']}:{scene}").shuffle(candidates)
        pairs.extend(candidates[:cfg["pairs_per_scene"]])
    if not pairs:
        raise ValueError("No frame has two prepared regions. Increase frame/region sampling in notebook 1.")
    sources = {s["path"]: s["sha256"] for s in source["sources"]}
    rgb_hashes = {}
    for pair in pairs:
        _validate_pair(pair, data_root)
        if pair["rgb"] not in rgb_hashes:
            actual = hashlib.sha256(_dataset_file(data_root, pair["rgb"]).read_bytes()).hexdigest()
            if actual != sources.get(pair["rgb"]):
                raise ValueError(f"RGB changed since object manifest: {pair['rgb']}")
            rgb_hashes[pair["rgb"]] = actual
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(dataset_dir / "pairs.jsonl", pairs)
    _write_jsonl(dataset_dir / "labels.jsonl", [
        {"pair_id": r["id"], "relation": None, "reviewed": False, "objects_valid": False,
         "annotator": "", "notes": ""} for r in pairs])
    meta = {"schema_version": 1, "task": "visual_pair_relationship", "settings": cfg,
            "source_fingerprint": source["fingerprint"], "scenes": source["scenes"],
            "pairs_fingerprint": fingerprint(pairs), "rgb_sha256": rgb_hashes,
            "prompt": RELATION_PROMPT, "relations": list(RELATIONS), "score_weights": RELATION_SCORE_WEIGHTS}
    save_json(meta_path, meta)  # Completion marker written last.
    return relation_dataset_summary(dataset_dir)


def upgrade_relation_vocabulary(dataset_dir):
    """Preserve old reviews; old none now requires checking the added relations."""
    dataset_dir = Path(dataset_dir)
    meta_path, label_path = dataset_dir / "dataset.json", dataset_dir / "labels.jsonl"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["relations"] == list(RELATIONS) and meta["prompt"] == RELATION_PROMPT:
        return
    if meta["relations"] != ["a_on_b", "b_on_a", "a_in_b", "b_in_a", "none"]:
        raise ValueError("Unsupported annotation vocabulary; do not silently reinterpret its labels.")
    # Back up the exact originals before changing metadata or review eligibility.
    backup = dataset_dir / "vocabulary_backups" / fingerprint(meta)[:12]
    backup.mkdir(parents=True, exist_ok=True)
    for original in (meta_path, label_path):
        saved = backup / original.name
        if not saved.exists():
            saved.write_bytes(original.read_bytes())
    labels = _read_jsonl(label_path)
    reset = 0
    for label in labels:
        if label["reviewed"] and label["relation"] == "none":
            label["reviewed"] = False
            label["review_note"] = "Previously none under the old vocabulary; check next_to and nearby."
            reset += 1
    _write_jsonl(label_path, labels)
    meta.update(relations=list(RELATIONS), prompt=RELATION_PROMPT, score_weights=RELATION_SCORE_WEIGHTS)
    save_json(meta_path, meta)
    print(f"Added next_to/nearby; {reset} previous none reviews need checking. Originals saved at {backup}.")


def _relation_files(dataset_dir):
    dataset_dir = Path(dataset_dir)
    if not (dataset_dir / "dataset.json").is_file():
        raise FileNotFoundError("Run ACTION='prepare' first to create the pair-review dataset.")
    meta = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    pairs = _read_jsonl(dataset_dir / "pairs.jsonl")
    labels = _read_jsonl(dataset_dir / "labels.jsonl")
    if fingerprint(pairs) != meta["pairs_fingerprint"]:
        raise ValueError("pairs.jsonl was modified. Edit only labels.jsonl or use the review form.")
    if len({r["id"] for r in pairs}) != len(pairs):
        raise ValueError("Duplicate pair IDs")
    if len({r["pair_id"] for r in labels}) != len(labels):
        raise ValueError("Duplicate annotations for a pair")
    by_id = {r["pair_id"]: r for r in labels}
    if set(by_id) != {r["id"] for r in pairs}:
        raise ValueError("labels.jsonl must contain exactly one entry per prepared pair.")
    for label in labels:
        if label["relation"] not in (*RELATIONS, "unknown", None):
            raise ValueError(f"Invalid relation for {label['pair_id']}: {label['relation']}")
        if not isinstance(label["reviewed"], bool) or not isinstance(label["objects_valid"], bool):
            raise ValueError("reviewed and objects_valid must be JSON booleans.")
        if label["reviewed"] and (not str(label["annotator"]).strip() or label["relation"] is None):
            raise ValueError("Reviewed pairs need an annotator and a relation (or unknown).")
    return meta, pairs, by_id


def relation_dataset_summary(dataset_dir):
    _, pairs, labels = _relation_files(dataset_dir)
    counts = []
    for split in ("train", "validation", "test"):
        rows = [r for r in pairs if r["split"] == split]
        usable = [labels[r["id"]] for r in rows if labels[r["id"]]["reviewed"]
                  and labels[r["id"]]["objects_valid"] and labels[r["id"]]["relation"] in RELATIONS]
        counts.append({"split": split, "proposed": len(rows), "usable": len(usable),
                       "unreviewed": sum(not labels[r["id"]]["reviewed"] for r in rows),
                       **{name: sum(x["relation"] == name for x in usable) for name in RELATIONS}})
    return {"action": "prepare", "dataset_dir": str(Path(dataset_dir).resolve()), "counts": counts}


def save_relation_review(dataset_dir, pair_id, relation, objects_valid, annotator, notes=""):
    _, _, labels = _relation_files(dataset_dir)
    if pair_id not in labels or relation not in (*RELATIONS, "unknown") or not annotator.strip():
        raise ValueError("Select a pair, choose a relation/unknown, and enter your annotator name.")
    if not objects_valid and relation != "unknown":
        raise ValueError("Invalid object proposals must be marked unknown, not none or a positive relation.")
    labels[pair_id] = {"pair_id": pair_id, "relation": relation, "reviewed": True,
                      "objects_valid": bool(objects_valid), "annotator": annotator.strip(), "notes": notes}
    _write_jsonl(Path(dataset_dir) / "labels.jsonl", list(labels.values()))


def review_relation_pairs(dataset_dir, data_root):
    """Small notebook review form; all work stays local and labels persist to disk."""
    import io
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print(f"Install ipywidgets in this kernel for the review form, or edit {Path(dataset_dir) / 'labels.jsonl'}.")
        return None
    meta, pairs, _ = _relation_files(dataset_dir)
    index = {r["id"]: r for r in pairs}
    picker = widgets.Dropdown(options=[(f"{r['split']} | {r['scene']} | frame {r['frame']} | {r['id'][:6]}", r["id"]) for r in pairs],
                              description="Pair:", layout=widgets.Layout(width="95%"))
    picture = widgets.Image(format="png")
    info = widgets.Label()
    guide = widgets.HTML(
        "<b>next_to:</b> immediate side-by-side neighbors; small gap, no intervening object.<br>"
        "<b>nearby:</b> physically close in the same local area, but not immediate neighbors.<br>"
        "Prefer <b>in → on → next_to → nearby</b> when several apply. Proximity labels are symmetric.<br>"
        "If depth/proximity is unclear, use <b>unknown</b>; image closeness alone is insufficient.<br>"
        "nearby has weight 0.5 only in the secondary weighted score; ordinary accuracy/F1 and training are unchanged."
    )
    label = widgets.Dropdown(options=[("Choose...", None)] + [(r, r) for r in (*RELATIONS, "unknown")], description="Relation:")
    valid = widgets.Checkbox(value=False, description="A and B identify two distinct, clearly visible objects", indent=False)
    annotator = widgets.Text(description="Reviewer:", placeholder="Your name or initials")
    notes = widgets.Textarea(description="Notes:")
    save = widgets.Button(description="Save + next", button_style="success")
    status = widgets.Output()
    def show(change=None):
        row = index[picker.value]
        annotation = _relation_files(dataset_dir)[2][row["id"]]
        stream = io.BytesIO()
        relation_image(row, data_root, meta["settings"]["image_max_side"]).save(stream, format="PNG")
        picture.value = stream.getvalue()
        info.value = f"Review hints: A={row['hint_a']}; B={row['hint_b']}. {annotation.get('review_note', '')} Unclear? Choose unknown."
        label.value, valid.value, notes.value = annotation["relation"], annotation["objects_valid"], annotation["notes"]
        if annotation["annotator"]:
            annotator.value = annotation["annotator"]
    def commit(button):
        with status:
            status.clear_output()
            try:
                save_relation_review(dataset_dir, picker.value, label.value, valid.value, annotator.value, notes.value)
                print("Saved. unknown/invalid pairs are excluded from training and scoring.")
                current = next(i for i, r in enumerate(pairs) if r["id"] == picker.value)
                if current + 1 < len(pairs):
                    picker.value = pairs[current + 1]["id"]
            except ValueError as exc:
                print(exc)
    picker.observe(show, names="value")
    save.on_click(commit)
    show()
    form = widgets.VBox([picker, picture, info, guide, label, valid, annotator, notes, save, status])
    display(form)
    return form


def load_relation_manifest(dataset_dir, data_root):
    """Validate labels and scene separation before importing/loading a model."""
    meta, pairs, annotations = _relation_files(dataset_dir)
    if meta["prompt"] != RELATION_PROMPT or meta["relations"] != list(RELATIONS):
        raise ValueError("This dataset uses a different relationship task definition.")
    group_splits = defaultdict(set)
    for row in pairs:
        expected = meta["scenes"][row["scene"]]
        if row["group"] != expected["group"] or row["split"] != expected["split"]:
            raise ValueError("Pair split does not match the original scene split.")
        group_splits[row["group"]].add(row["split"])
        _validate_pair(row, data_root)
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError("Shared scene/layout group crosses data splits.")
    for relative, expected_hash in meta["rgb_sha256"].items():
        if hashlib.sha256(_dataset_file(data_root, relative).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Source RGB changed: {relative}")
    examples = []
    for row in pairs:
        label = annotations[row["id"]]
        if label["reviewed"] and label["objects_valid"] and label["relation"] in RELATIONS:
            examples.append({**row, "target": label["relation"]})
    for split in ("train", "validation", "test"):
        if not any(r["split"] == split for r in examples):
            raise ValueError(f"No reviewed usable {split} pairs. Use ACTION='prepare' and label pairs in all three splits first.")
    train_labels = {r["target"] for r in examples if r["split"] == "train"}
    if "none" not in train_labels or len(train_labels) < 2:
        raise ValueError("Training needs verified 'none' pairs and at least one positive relation. Never label unknown pairs as none.")
    core = {"schema_version": 1, "task": meta["task"], "settings": meta["settings"],
            "prompt": RELATION_PROMPT, "relations": list(RELATIONS), "score_weights": RELATION_SCORE_WEIGHTS,
            "scenes": meta["scenes"],
            "source_fingerprint": meta["source_fingerprint"], "rgb_sha256": meta["rgb_sha256"],
            "examples": examples, "annotations": list(annotations.values()),
            "proposal_source": "gt_semantic_regions_not_instances"}
    return {**core, "fingerprint": fingerprint(core)}


class RelationCollator:
    def __init__(self, processor, data_root, max_side=512):
        self.processor, self.data_root, self.max_side = processor, data_root, max_side

    def __call__(self, examples):
        if len(examples) != 1:
            raise ValueError("Use microbatch 1 and gradient accumulation for this collator.")
        row = examples[0]
        return encode_messages(self.processor, relation_messages(row, self.data_root, self.max_side), row["target"])


def relation_metrics(predictions):
    """Exact relation scores on GIVEN pairs, plus a declared secondary weighted score."""
    per_class = {}
    for name in RELATIONS:
        tp = sum(p["target"] == name and p["prediction"] == name for p in predictions)
        fp = sum(p["target"] != name and p["prediction"] == name for p in predictions)
        fn = sum(p["target"] == name and p["prediction"] != name for p in predictions)
        per_class[name] = {"support": tp+fn, "precision": tp/(tp+fp) if tp+fp else 0.0,
                           "recall": tp/(tp+fn) if tp+fn else None,
                           "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
    positives = [v["f1"] for k, v in per_class.items() if k != "none" and v["support"] > 0]
    count = len(predictions)
    # A wrong positive direction is both a false positive and a false negative.
    # Count all predicted positives, including relations absent from the GT subset.
    tp = sum(p["target"] != "none" and p["target"] == p["prediction"] for p in predictions)
    predicted_positive = sum(p["prediction"] in RELATIONS[:-1] for p in predictions)
    actual_positive = sum(p["target"] != "none" for p in predictions)
    weight_total = sum(RELATION_SCORE_WEIGHTS[p["target"]] for p in predictions)
    weighted_correct = sum(RELATION_SCORE_WEIGHTS[p["target"]] for p in predictions if p["target"] == p["prediction"])
    return {"examples": count,
            "exact_match": sum(p["target"] == p["prediction"] for p in predictions)/count if count else None,
            "weighted_exact_match": weighted_correct/weight_total if weight_total else None,
            "nearby_score_weight": RELATION_SCORE_WEIGHTS["nearby"],
            "macro_relation_f1": float(np.mean(positives)) if positives else None,
            "relation_precision": tp/predicted_positive if predicted_positive else 0.0,
            "relation_recall": tp/actual_positive if actual_positive else None,
            "relation_f1": 2*tp/(predicted_positive+actual_positive) if predicted_positive+actual_positive else None,
            "invalid_output_rate": sum(p["prediction"] not in RELATIONS for p in predictions)/count if count else None,
            "positive_relations_evaluated": len(positives), "per_relation": per_class}


def predict_relations(model, processor, rows, data_root, max_side=512, disable_adapter=False):
    import torch
    was_training = model.training
    model.eval()
    output = []
    context = model.disable_adapter() if disable_adapter else nullcontext()
    try:
        with context, torch.inference_mode():
            for row in rows:
                batch = encode_messages(processor, relation_messages(row, data_root, max_side)).to(model.device)
                generated = model.generate(**batch, max_new_tokens=16, do_sample=False)
                raw = processor.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                # Keep a strict label contract. Wrong direction and verbose/invalid
                # answers remain errors; never convert a failed answer into none.
                predicted = raw.strip().casefold()
                output.append({"id": row["id"], "scene": row["scene"], "target": row["target"],
                               "prediction": predicted, "raw_prediction": raw,
                               "correct": predicted == row["target"]})
    finally:
        model.train(was_training)
    return output


def _relation_fit(model, processor, manifest, data_root, output_dir, args, record):
    import torch
    from transformers import Trainer
    max_side = manifest["settings"]["image_max_side"]
    train = [r for r in manifest["examples"] if r["split"] == "train"]
    validation = evaluation_rows(manifest, "validation", limit=None)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=validation,
                      data_collator=RelationCollator(processor, data_root, max_side))
    record["stage"] = "pretrained_validation"
    save_json(output_dir / "run.json", record)
    before = predict_relations(model, processor, validation, data_root, max_side, disable_adapter=True)
    save_json(output_dir / "pretrained_validation.json", {"metrics": relation_metrics(before), "predictions": before})
    record["stage"] = "training"
    save_json(output_dir / "run.json", record)
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    record["stage"] = "saving_adapter"
    save_json(output_dir / "run.json", record)
    model.save_pretrained(output_dir / "adapter", safe_serialization=True)
    processor.save_pretrained(output_dir / "adapter")
    after = predict_relations(model, processor, validation, data_root, max_side)
    save_json(output_dir / "adapted_validation.json", {"metrics": relation_metrics(after), "predictions": after})
    record.update(complete=True, stage="complete", optimizer_steps=trainer.state.global_step,
                  best_checkpoint=relative_best_checkpoint(output_dir, trainer.state.best_model_checkpoint),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    save_json(output_dir / "run.json", record)
    return {"pretrained": relation_metrics(before), "adapted": relation_metrics(after)}


def run_relation_pipeline(project_root, dataset_dir, model_key="qwen", action="prepare", run_name="relations_3epochs",
                          training=None, lora=None, use_4bit=True, source_manifest=None, data_settings=None,
                          test_run=None):
    """One action controls preparation, fresh training, or saved-adapter testing.

    No install/download calls. Models exist only inside this call and are freed
    afterwards, avoiding notebook state flags for loaded/trained/reloaded models.
    """
    import gc
    project_root, dataset_dir = Path(project_root), Path(dataset_dir)
    if action not in ("prepare", "train", "test") or model_key not in MODELS:
        raise ValueError("ACTION must be prepare/train/test; MODEL_KEY must be qwen/smolvlm.")
    if action == "prepare":
        return prepare_relation_dataset(project_root, dataset_dir, source_manifest, data_settings)
    requested_run = artifact_run_directory(project_root, "relationships", model_key, run_name)
    data_root = project_root / "dataset/osma-bench/data"
    manifest = load_relation_manifest(dataset_dir, data_root)  # Fails before GPU loading if labels are missing.
    if data_settings is not None and manifest["settings"] != {**RELATION_DATA_DEFAULTS, **data_settings}:
        raise ValueError("DATA_SETTINGS differ from this prepared dataset. Restore them or prepare a new DATASET_NAME.")
    root = requested_run.parent
    if action == "test":
        if test_run is None:
            options = []
            for path in root.glob("*/run.json"):
                saved = json.loads(path.read_text(encoding="utf-8"))
                if (saved.get("complete") and saved.get("run_name") == run_name
                        and saved.get("model_key") == model_key
                        and saved.get("task") == manifest["task"]
                        and saved.get("dataset_fingerprint") == manifest["fingerprint"]):
                    options.append(path)
            if not options:
                raise FileNotFoundError("No completed relationship run for this model, RUN_NAME and annotation fingerprint. Train first or set TEST_RUN to its exact folder.")
            output_dir = max(options, key=lambda p: p.stat().st_mtime_ns).parent
        else:
            output_dir = Path(test_run)
        record = read_run_record(output_dir, model_key=model_key, task=manifest["task"],
                                 dataset_fingerprint=manifest["fingerprint"])
        cached = output_dir / "held_out_test.json"
        if cached.exists():
            result = json.loads(cached.read_text(encoding="utf-8"))
            if result.get("dataset_fingerprint") != manifest["fingerprint"]:
                raise ValueError("Cached test belongs to a different dataset.")
            return {"action": action, "run_dir": str(output_dir), "metrics": result["metrics"], "cached": True}
    else:
        cfg = {"max_steps": -1, "epochs": 3, "accumulation": 4, "learning_rate": 1e-4, "eval_steps": 10,
               **(training or {})}
        lora = {"rank": 8, "alpha": 16, "dropout": 0.05, **(lora or {})}
        if cfg["accumulation"] < 1 or cfg["epochs"] <= 0 or cfg["max_steps"] == 0 or cfg["eval_steps"] < 1:
            raise ValueError("Invalid training budget; use max_steps=-1 for epochs or a positive step limit.")
        if cfg["max_steps"] > 0:
            cfg["eval_steps"] = min(cfg["eval_steps"], cfg["max_steps"])
        output_dir = available_run_directory(requested_run)
    # Heavy imports and CUDA allocation happen only after data/run validation.
    import torch
    model = processor = None
    try:
        if action == "test":
            print("Testing saved relationship adapter:", output_dir)
            model, processor, record = reload_adapter(project_root, output_dir)
            rows = evaluation_rows(manifest, "test", limit=None)
            evaluations = {}
            for mode, disabled in (("pretrained", True), ("adapted", False)):
                predictions = predict_relations(model, processor, rows, data_root,
                                                 manifest["settings"]["image_max_side"], disabled)
                evaluations[mode] = {"metrics": relation_metrics(predictions), "predictions": predictions}
            metrics = {k: v["metrics"] for k, v in evaluations.items()}
            save_json(output_dir / "held_out_test.json", {"dataset_fingerprint": manifest["fingerprint"],
                       "metrics": metrics, "evaluations": evaluations})
        else:
            args = training_arguments(output_dir, cfg, manifest["settings"]["seed"],
                                      sum(r["split"] == "train" for r in manifest["examples"]))
            if cfg["max_steps"] < 0:
                # Evaluate/save every epoch, including the final one.
                from transformers.trainer_utils import IntervalStrategy, SaveStrategy
                args.eval_strategy, args.save_strategy = IntervalStrategy.EPOCH, SaveStrategy.EPOCH
            args.prediction_loss_only = True
            output_dir.mkdir(parents=True, exist_ok=False)
            record = {"model": MODELS[model_key], "model_key": model_key, "run_name": run_name,
                      "dataset_fingerprint": manifest["fingerprint"], "task": manifest["task"],
                      "use_4bit": use_4bit, "training": cfg, "lora": lora, "complete": False, "stage": "loading_model",
                      "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "versions": {p: importlib.metadata.version(p) for p in
                                   ("torch", "transformers", "peft", "accelerate")}}
            if use_4bit:
                record["versions"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
            save_json(output_dir / "run.json", record)
            save_json(output_dir / "data_manifest.json", manifest)
            print("Training fresh relationship adapter:", output_dir)
            model, processor, _, _ = load_trainable(project_root, model_key, **lora, use_4bit=use_4bit,
                                                    seed=manifest["settings"]["seed"])
            metrics = _relation_fit(model, processor, manifest, data_root, output_dir, args, record)
            print("Saved relationship adapter:", output_dir / "adapter")
        return {"action": action, "run_dir": str(output_dir), "metrics": metrics, "cached": False}
    except Exception as exc:
        if action == "train" and (output_dir / "run.json").is_file():
            record.update(complete=False, error=f"{type(exc).__name__}: {exc}")
            save_json(output_dir / "run.json", record)
        raise
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
