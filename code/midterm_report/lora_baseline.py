"""Local OSMa-Bench region-classification pilot; no network calls or implicit training.

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
    import torch
    messages = user_message(image)
    # SmolVLM adds video-processing defaults even for an image-only chat. In
    # current Transformers it puts them in legacy **kwargs unless it receives
    # a NONEMPTY processor_kwargs dict, producing a warning for every crop.
    # Route tensor conversion through that dict; chat-format controls stay above it.
    prompt = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, processor_kwargs={"return_tensors": "pt"})
    if target is None:
        # Inference: ask the question without supplying the correct category.
        return prompt
    # Training: the JSON-derived category is the desired assistant response.
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
    metadata = {"model": MODELS[model_key], "model_key": model_key, "dataset_fingerprint": manifest["fingerprint"],
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
                    best_checkpoint=trainer.state.best_model_checkpoint,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
    save_json(output_dir / "run.json", metadata)
    return {"adapter_dir": str(adapter_dir), "pretrained": classification_metrics(before), "adapted": classification_metrics(after)}


def reload_adapter(project_root, run_dir):
    from peft import PeftModel
    from transformers import AutoProcessor
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if not record["complete"] or record["model"] != MODELS[record["model_key"]]:
        raise ValueError("Run is incomplete or its base-model revision does not match this notebook.")
    base, _ = load_local_base(project_root, record["model_key"], record["use_4bit"])
    model = PeftModel.from_pretrained(base, run_dir / "adapter", is_trainable=False, local_files_only=True)
    processor = AutoProcessor.from_pretrained(run_dir / "adapter", local_files_only=True, trust_remote_code=False)
    model.eval()
    return model, processor, record
