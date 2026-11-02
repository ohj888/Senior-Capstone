"""Declared category vocabulary for closed-set object naming and scoring.

Only the category dictionary is read at inference: no semantic masks, object
assignments, reviewed relationships, or answers are supplied to the VLM.
"""
import hashlib
import json
from pathlib import Path


FORMAT = "scene_categories_v1"
ABSTAIN = "unknown"
CATEGORY_ALIASES = {"stair": "stairs", "staircase": "stairs"}
PROMPT = """Classify the main object or semantic region in this RGB crop.
Choose exactly one category from the allowed vocabulary below.
Use the most specific category supported by the visible object. Use unknown
if no listed category fits or the crop does not provide enough evidence.
Copy the category exactly, including spaces and plurals. Do not invent a
synonym, add a sentence, return a number, or include punctuation or explanations.
Allowed categories (one per line):
{categories}
Reply with exactly one category name."""


def normalize(name):
    return " ".join(str(name).replace("_", " ").strip(" \t\r\n.\"'").casefold().split())


def effective_aliases(classes, overrides=None):
    """Only spelling-equivalent defaults; never merge broad/fine categories."""
    vocabulary = {normalize(name): int(key) for key, name in classes.items()}
    if len(vocabulary) != len(classes):
        raise ValueError("Dataset category names must be distinct after normalization")
    aliases = {source: target for source, target in CATEGORY_ALIASES.items()
               if source not in vocabulary and target in vocabulary}
    for source, target in (overrides or {}).items():
        source, target = normalize(source), normalize(target)
        if target not in vocabulary:
            raise ValueError("Every alias target must be an exact dataset class name")
        if source in vocabulary and source != target:
            raise ValueError("Aliases must not overwrite an existing dataset category")
        aliases[source] = target
    return aliases


def inference_settings(config, data_root, scene):
    config = dict(config)
    mode = config.get("object_response_format", FORMAT)
    config["object_response_format"] = mode
    if mode == "legacy":
        return config
    if mode != FORMAT:
        raise ValueError("object_response_format must be scene_categories_v1 or legacy")
    root = Path(data_root).resolve()
    source = (root / scene / "embed_semseg_classes.json").resolve()
    if not source.is_relative_to(root):
        raise ValueError("Scene vocabulary must be inside the dataset root")
    rows = json.loads(source.read_text(encoding="utf-8-sig"))["classes"]
    classes = {int(row["id"]): str(row["name"]) for row in rows}
    if len(classes) != len(rows) or not classes:
        raise ValueError("Scene category IDs must be unique and nonempty")
    effective_aliases(classes)  # Reject ambiguous normalized names.
    labels = [classes[key] for key in sorted(classes)
              if normalize(classes[key]) not in ("other", "unknown", "unlabeled", "background")]
    if any(label != label.strip() or "\n" in label or "\r" in label for label in labels):
        raise ValueError("Category names must be single lines without outer whitespace")
    labels.append(ABSTAIN)
    prompt = PROMPT.format(categories="\n".join(labels))
    digest = hashlib.sha256(json.dumps(classes, sort_keys=True).encode("utf-8")).hexdigest()
    config.update(object_choices=labels, object_prompt=prompt,
                  object_class_ids={name: key for key, name in classes.items()},
                  object_vocabulary_sha256=digest,
                  object_choice_decoding="constrained_greedy",
                  object_vocabulary_source="declared scene category dictionary; no per-object GT")
    return config
