"""Migration regression checks using disposable runs, never real weights."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import lora_baseline as lb
import migrate_artifact_paths as migration


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old = self.root / "artifacts/lora_baselines/abcdef123456/qwen/qwen_objects_3epochs"
        self.new = lb.artifact_run_directory(self.root, "objects", "qwen", "objects_3epochs")
        # Model a previous manual rename with a stale checkpoint path.
        stale = self.old.with_name("pilot_01_001") / "checkpoints/checkpoint-2"
        lb.save_json(self.old / "run.json", {"task": "gt_assisted_region_category", "model_key": "qwen",
                     "model": lb.MODELS["qwen"], "complete": True, "dataset_fingerprint": "original",
                     "best_checkpoint": str(stale)})
        self.checkpoint = Path("checkpoints/checkpoint-2")
        lb.save_json(self.old / self.checkpoint / "trainer_state.json", {"best_model_checkpoint": str(stale)})
        torch.save(SimpleNamespace(output_dir=str(stale.parent)), self.old / self.checkpoint / "training_args.bin")
        (self.old / "adapter").mkdir()
        (self.old / "adapter/adapter_model.safetensors").write_bytes(b"synthetic unchanged weight bytes")

    def tearDown(self):
        self.temp.cleanup()

    def test_dry_run_then_migration_repairs_metadata_and_preserves_weights(self):
        original = (self.old / "run.json").read_bytes()
        weights = (self.old / "adapter/adapter_model.safetensors").read_bytes()
        migration.migrate(self.root)
        self.assertTrue(self.old.exists())
        self.assertFalse(self.new.exists())
        migration.migrate(self.root, apply=True)
        self.assertFalse(self.old.exists())
        self.assertEqual((self.new / "adapter/adapter_model.safetensors").read_bytes(), weights)
        record = lb.read_run_record(self.new, model_key="qwen", dataset_fingerprint="original")
        self.assertEqual(record["run_name"], "objects_3epochs")
        self.assertEqual(record["best_checkpoint"], self.checkpoint.as_posix())
        state = json.loads((self.new / self.checkpoint / "trainer_state.json").read_text())
        self.assertEqual(Path(state["best_model_checkpoint"]), self.new / self.checkpoint)
        args = torch.load(self.new / self.checkpoint / "training_args.bin", weights_only=False)
        self.assertEqual(Path(args.output_dir), self.new / "checkpoints")
        backups = list((self.root / "artifacts/path_migration_backup").rglob("run.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        self.assertEqual(migration.migrate(self.root, apply=True), [])

    def test_collision_aborts_before_moving(self):
        self.new.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            migration.migrate(self.root, apply=True)
        self.assertTrue((self.old / "adapter/adapter_model.safetensors").is_file())

    def test_missing_best_checkpoint_aborts_before_moving(self):
        path = self.old / "run.json"
        record = json.loads(path.read_text())
        record["best_checkpoint"] = str(self.old / "checkpoints/checkpoint-999")
        lb.save_json(path, record)
        with self.assertRaises(FileNotFoundError):
            migration.migrate(self.root, apply=True)
        self.assertTrue(self.old.exists())
        self.assertFalse(self.new.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
