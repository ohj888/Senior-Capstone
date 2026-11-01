"""CPU regression tests; synthetic annotations only, no downloads or real training.

Run: python code/midterm_report/test_relation_pipeline.py
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import lora_baseline as lb


class RelationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_root = self.root / "dataset/osma-bench/data"
        self.dataset = self.root / "dataset/_training/pairs"
        examples, scenes, sources = [], {}, []
        for i, split in enumerate(("train", "train", "validation", "test")):
            name = f"hm3d/scene{i}"
            scenes[name] = {"group": name, "split": split}
            path = self.data_root / name / "results/frame000000.jpg"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (96, 64), (100, 110, 120)).save(path)
            relative = path.relative_to(self.data_root).as_posix()
            sources.append({"path": relative, "sha256": lb.hashlib.sha256(path.read_bytes()).hexdigest()})
            for j, box in enumerate(([2, 2, 30, 50], [50, 10, 90, 58])):
                examples.append({"id": f"{i}-{j}", "scene": name, "frame": 0,
                                 **scenes[name], "rgb": relative, "bbox": box, "target": "chair"})
        core = {"scenes": scenes, "examples": examples, "sources": sources}
        self.source = self.root / "dataset/_training/osma_regions/test/manifest.json"
        lb.save_json(self.source, {**core, "fingerprint": lb.fingerprint(core)})
        self.settings = {**lb.RELATION_DATA_DEFAULTS, "pairs_per_scene": 1}
        lb.prepare_relation_dataset(self.root, self.dataset, self.source, self.settings)
        self.pairs = lb._read_jsonl(self.dataset / "pairs.jsonl")

    def tearDown(self):
        self.temp.cleanup()

    def label_all(self):
        train_seen = 0
        for row in self.pairs:
            label = "a_on_b"
            if row["split"] == "train":
                label = "none" if train_seen else "a_on_b"
                train_seen += 1
            lb.save_relation_review(self.dataset, row["id"], label, True, "test reviewer")

    def test_prepare_never_invents_labels_and_preserves_reviews(self):
        labels = lb._read_jsonl(self.dataset / "labels.jsonl")
        self.assertTrue(all(r["relation"] is None and not r["reviewed"] for r in labels))
        with self.assertRaisesRegex(ValueError, "reviewed usable train"):
            lb.load_relation_manifest(self.dataset, self.data_root)
        self.label_all()
        before = (self.dataset / "labels.jsonl").read_bytes()
        lb.prepare_relation_dataset(self.root, self.dataset, self.source, self.settings)
        self.assertEqual(before, (self.dataset / "labels.jsonl").read_bytes())
        with self.assertRaisesRegex(ValueError, "other settings"):
            lb.prepare_relation_dataset(self.root, self.dataset, self.source, {**self.settings, "seed": 99})

    def test_annotation_quality_gate_runs_before_loading_model(self):
        with patch.object(lb, "load_trainable", side_effect=AssertionError("Must not load a VLM")):
            with self.assertRaisesRegex(ValueError, "reviewed usable train"):
                lb.run_relation_pipeline(self.root, self.dataset, action="train")
        with self.assertRaisesRegex(ValueError, "Invalid object proposals"):
            lb.save_relation_review(self.dataset, self.pairs[0]["id"], "none", False, "reviewer")
        lb.save_relation_review(self.dataset, self.pairs[0]["id"], "unknown", False, "reviewer")
        summary = lb.relation_dataset_summary(self.dataset)
        self.assertEqual(sum(r["usable"] for r in summary["counts"]), 0)

    def test_fixed_splits_source_integrity_and_independent_labels(self):
        self.label_all()
        manifest = lb.load_relation_manifest(self.dataset, self.data_root)
        self.assertEqual(len(manifest["examples"]), 4)
        self.assertEqual({r["split"] for r in manifest["examples"]}, {"train", "validation", "test"})
        for group in {r["group"] for r in manifest["examples"]}:
            self.assertEqual(len({r["split"] for r in manifest["examples"] if r["group"] == group}), 1)
        # A wrong source image must not silently invalidate our annotations.
        image = self.data_root / self.pairs[0]["rgb"]
        Image.new("RGB", (96, 64), "blue").save(image)
        with self.assertRaisesRegex(ValueError, "Source RGB changed"):
            lb.load_relation_manifest(self.dataset, self.data_root)

    def test_pair_editing_and_path_escape_fail(self):
        pairs = copy.deepcopy(self.pairs)
        pairs[0]["split"] = "test"
        lb._write_jsonl(self.dataset / "pairs.jsonl", pairs)
        with self.assertRaisesRegex(ValueError, "pairs.jsonl was modified"):
            lb.load_relation_manifest(self.dataset, self.data_root)
        with self.assertRaisesRegex(ValueError, "outside data root"):
            lb.relation_image({**self.pairs[0], "rgb": "../../outside.jpg"}, self.data_root)

    def test_input_contains_only_marked_rgb_and_fixed_prompt(self):
        row = {**self.pairs[0], "target": "a_on_b", "hint_a": "SECRET_CATEGORY"}
        messages = lb.relation_messages(row, self.data_root)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertIsInstance(content[0]["image"], Image.Image)
        self.assertEqual(content[1]["text"], lb.RELATION_PROMPT)
        self.assertNotIn("SECRET_CATEGORY", content[1]["text"])
        swapped = {**row, "bbox_a": row["bbox_b"], "bbox_b": row["bbox_a"]}
        self.assertNotEqual(content[0]["image"].tobytes(), lb.relation_image(swapped, self.data_root).tobytes())

    def test_direction_errors_and_invalid_answers_are_not_none(self):
        metrics = lb.relation_metrics([
            {"target": "a_on_b", "prediction": "b_on_a"},
            {"target": "a_on_b", "prediction": "a_on_b"},
            {"target": "none", "prediction": "not sure"},
            {"target": "none", "prediction": "none"},
        ])
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["invalid_output_rate"], 0.25)
        self.assertEqual(metrics["per_relation"]["a_on_b"]["recall"], 0.5)
        self.assertEqual(metrics["per_relation"]["b_on_a"]["precision"], 0)
        self.assertAlmostEqual(metrics["macro_relation_f1"], 2/3)
        self.assertEqual(metrics["relation_precision"], 0.5)
        self.assertEqual(metrics["relation_recall"], 0.5)
        self.assertEqual(metrics["relation_f1"], 0.5)
        self.assertIsNone(metrics["per_relation"]["a_in_b"]["recall"])

    def test_proximity_labels_and_secondary_weighting(self):
        lb.save_relation_review(self.dataset, self.pairs[0]["id"], "next_to", True, "reviewer")
        lb.save_relation_review(self.dataset, self.pairs[1]["id"], "nearby", True, "reviewer")
        summary = lb.relation_dataset_summary(self.dataset)
        self.assertEqual(sum(r["next_to"] for r in summary["counts"]), 1)
        self.assertEqual(sum(r["nearby"] for r in summary["counts"]), 1)
        metrics = lb.relation_metrics([
            {"target": "next_to", "prediction": "next_to"},
            {"target": "nearby", "prediction": "none"},
        ])
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertAlmostEqual(metrics["weighted_exact_match"], 1/1.5)
        self.assertEqual(metrics["per_relation"]["nearby"]["recall"], 0)
        # Guessing the broad label does not earn partial credit for next_to.
        self.assertEqual(lb.relation_metrics([{"target": "next_to", "prediction": "nearby"}])["weighted_exact_match"], 0)

    def test_vocabulary_upgrade_backs_up_and_reopens_only_old_none(self):
        self.label_all()
        meta_path = self.dataset / "dataset.json"
        meta = json.loads(meta_path.read_text())
        meta.update(relations=["a_on_b", "b_on_a", "a_in_b", "b_in_a", "none"], prompt="old task")
        lb.save_json(meta_path, meta)
        original_labels = (self.dataset / "labels.jsonl").read_bytes()
        lb.upgrade_relation_vocabulary(self.dataset)
        labels = lb._read_jsonl(self.dataset / "labels.jsonl")
        for row in labels:
            self.assertEqual(row["reviewed"], row["relation"] != "none")
            self.assertEqual(row["annotator"], "test reviewer")
        backups = list((self.dataset / "vocabulary_backups").glob("*/labels.jsonl"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original_labels)
        updated_bytes = (self.dataset / "labels.jsonl").read_bytes()
        lb.upgrade_relation_vocabulary(self.dataset)
        self.assertEqual((self.dataset / "labels.jsonl").read_bytes(), updated_bytes)
        self.assertEqual(json.loads(meta_path.read_text())["relations"], list(lb.RELATIONS))

    def test_cached_final_test_uses_existing_matching_run(self):
        self.label_all()
        manifest = lb.load_relation_manifest(self.dataset, self.data_root)
        run = lb.artifact_run_directory(self.root, "relationships", "qwen", "run_001")
        lb.save_json(run / "run.json", {"complete": True, "run_name": "run", "model_key": "qwen",
                     "model": lb.MODELS["qwen"],
                     "task": "visual_pair_relationship", "dataset_fingerprint": manifest["fingerprint"]})
        lb.save_json(run / "held_out_test.json", {"dataset_fingerprint": manifest["fingerprint"], "metrics": {"saved": True}})
        with patch.object(lb, "reload_adapter", side_effect=AssertionError("Do not rerun cached evaluation")):
            result = lb.run_relation_pipeline(self.root, self.dataset, action="test", run_name="run")
        self.assertTrue(result["cached"])
        self.assertEqual(Path(result["run_dir"]), run)
        self.assertEqual(result["metrics"], {"saved": True})
        with self.assertRaises(FileNotFoundError):
            lb.run_relation_pipeline(self.root, self.dataset, model_key="smolvlm", action="test", run_name="run")

    def test_run_selection_skips_other_datasets_and_rejects_explicit_mismatch(self):
        self.label_all()
        manifest = lb.load_relation_manifest(self.dataset, self.data_root)
        good = lb.artifact_run_directory(self.root, "relationships", "qwen", "run")
        bad = lb.artifact_run_directory(self.root, "relationships", "qwen", "run_001")
        record = {"complete": True, "run_name": "run", "model_key": "qwen", "model": lb.MODELS["qwen"],
                  "task": manifest["task"], "dataset_fingerprint": manifest["fingerprint"]}
        lb.save_json(good / "run.json", record)
        lb.save_json(good / "held_out_test.json", {"dataset_fingerprint": manifest["fingerprint"], "metrics": {}})
        lb.save_json(bad / "run.json", {**record, "dataset_fingerprint": "other-dataset"})
        with patch.object(lb, "reload_adapter", side_effect=AssertionError("Must validate before loading")):
            result = lb.run_relation_pipeline(self.root, self.dataset, action="test", run_name="run")
            self.assertEqual(Path(result["run_dir"]), good)
            with self.assertRaisesRegex(ValueError, "dataset_fingerprint"):
                lb.run_relation_pipeline(self.root, self.dataset, action="test", run_name="run", test_run=bad)
            lb.save_json(good / "held_out_test.json", {"dataset_fingerprint": "other-dataset", "metrics": {}})
            with self.assertRaisesRegex(ValueError, "Cached test"):
                lb.run_relation_pipeline(self.root, self.dataset, action="test", run_name="run")

    def test_readable_paths_preserve_runs_and_validate_object_identity(self):
        run = lb.artifact_run_directory(self.root, "objects", "smolvlm", "objects_3epochs")
        self.assertEqual(run.relative_to(self.root).as_posix(), "artifacts/objects/smolvlm/objects_3epochs")
        record = {"complete": True, "model_key": "smolvlm", "model": lb.MODELS["smolvlm"],
                  "task": "gt_assisted_region_category", "dataset_fingerprint": "original"}
        lb.save_json(run / "run.json", record)
        self.assertEqual(lb.available_run_directory(run).name, "objects_3epochs_001")
        self.assertEqual(lb.read_run_record(run, model_key="smolvlm", dataset_fingerprint="original"), record)
        with patch.object(lb, "load_local_base", side_effect=AssertionError("Must validate before loading")):
            for expected in ({"model_key": "qwen"}, {"task": "visual_pair_relationship"},
                             {"dataset_fingerprint": "changed"}):
                with self.assertRaises(ValueError):
                    lb.reload_adapter(self.root, run, **expected)
        for name in ("../escape", "a/b", "a\\b", "C:escape", ""):
            with self.assertRaises(ValueError):
                lb.artifact_run_directory(self.root, "objects", "qwen", name)

    def test_training_stage_with_tiny_multimodal_model_and_saved_adapter(self):
        import torch
        from transformers import BatchFeature, Qwen3VLConfig, Qwen3VLForConditionalGeneration
        self.label_all()
        manifest = lb.load_relation_manifest(self.dataset, self.data_root)
        text = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                    pad_token_id=0, bos_token_id=1, eos_token_id=2,
                    rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]})
        config = Qwen3VLConfig(text_config=text,
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                               patch_size=8, spatial_merge_size=2, temporal_patch_size=2,
                               out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[]),
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        torch.manual_seed(42)
        model, _, _ = lb.attach_lora(Qwen3VLForConditionalGeneration(config), rank=2, alpha=4, dropout=0)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        class Processor:
            # Synthetic tokens and image patches; this tests Trainer/LoRA plumbing,
            # not model quality. Real local processors have separate smoke checks.
            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["processor_kwargs"] == {"return_tensors": "pt"}
                assert len(messages[0]["content"]) == 2
                tokens = [1, 62, 60, 60, 60, 60, 63, 3, 4]
                if len(messages) == 2:
                    target = messages[1]["content"][0]["text"]
                    tokens += [5 if target == "a_on_b" else 6, 2]
                ids = torch.tensor([tokens])
                return BatchFeature({"input_ids": ids, "attention_mask": torch.ones_like(ids),
                                     "pixel_values": torch.ones(16, 3*2*8*8),
                                     "image_grid_thw": torch.tensor([[1, 4, 4]]),
                                     "mm_token_type_ids": (ids == 60).long()})
            def save_pretrained(self, path):
                lb.save_json(Path(path) / "processor_config.json", {"test_fixture": True})
        processor = Processor()
        sample = manifest["examples"][0]
        messages = lb.relation_messages(sample, self.data_root)
        encoded = lb.encode_messages(processor, messages, sample["target"])
        self.assertEqual(len(messages), 1)  # Encoding must not leak the answer back into input messages.
        self.assertTrue((encoded["labels"][:, :9] == -100).all())
        self.assertTrue((encoded["labels"][:, 9:] != -100).all())
        output = lb.artifact_run_directory(self.root, "relationships", "qwen", "tiny_test")
        output.mkdir(parents=True)
        # Two steps: with one warmup step the first update has LR=0. This test
        # checks an actual nonzero LoRA update, not merely a successful forward.
        settings = {"max_steps": 2, "epochs": 1, "accumulation": 1, "learning_rate": 0.01, "eval_steps": 1}
        args = lb.training_arguments(output, settings, 42, train_examples=2, use_cpu=True)
        args.disable_tqdm, args.dataloader_pin_memory = True, False
        record = {"complete": False}
        def predicted(model, processor, rows, *args, **kwargs):
            return [{"id": r["id"], "target": r["target"], "prediction": "none"} for r in rows]
        with patch.object(lb, "predict_relations", side_effect=predicted), \
             patch.object(torch.cuda, "reset_peak_memory_stats"), \
             patch.object(torch.cuda, "max_memory_allocated", return_value=0):
            result = lb._relation_fit(model, processor, manifest, self.data_root, output, args, record)
        self.assertTrue((output / "adapter/adapter_model.safetensors").is_file())
        self.assertTrue((output / "pretrained_validation.json").is_file())
        self.assertTrue((output / "adapted_validation.json").is_file())
        self.assertTrue(json.loads((output / "run.json").read_text())["complete"])
        best = json.loads((output / "run.json").read_text())["best_checkpoint"]
        self.assertFalse(Path(best).is_absolute())
        self.assertTrue((output / best).is_dir())
        self.assertEqual(result["adapted"]["examples"], 1)
        self.assertTrue(all(torch.equal(before[n], p) for n, p in model.named_parameters() if not p.requires_grad))
        # Inspect the final saved checkpoint as best-checkpoint selection may
        # legitimately select the pre-update first checkpoint on synthetic data.
        from safetensors.torch import load_file
        last = load_file(output / "checkpoints/checkpoint-2/adapter_model.safetensors")
        self.assertTrue(any(v.abs().sum() > 0 for k, v in last.items() if "lora_B" in k))


if __name__ == "__main__":
    unittest.main(verbosity=2)
