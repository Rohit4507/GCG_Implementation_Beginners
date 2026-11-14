"""
Regression tests for CLI flag wiring, evaluation pair indexing, and the
Kaggle runbook notebook structure.

    python -m unittest discover -s tests -v
"""
import ast
import glob
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.config import GCGConfig
from src.evaluate import evaluate_pairs
from src.model import init_suffix_ids
from src.tiny_model import build_tiny_model_and_tokenizer


class TestConfigPresets(unittest.TestCase):
    def test_paper_preset_upgrades_prompt_add_interval(self):
        self.assertEqual(GCGConfig(preset="debug").prompt_add_interval, 5)
        self.assertEqual(GCGConfig(preset="paper").prompt_add_interval, 50)
        self.assertEqual(GCGConfig(preset="paper", prompt_add_interval=40).prompt_add_interval, 40)


class TestEvaluatePairIndices(unittest.TestCase):
    def test_explicit_pair_indices_preserve_non_contiguous_ids(self):
        goals = ["Write a poem", "Describe water"]
        targets = ["Sure, here is a poem", "Sure, here is water"]
        model, tok = build_tiny_model_and_tokenizer(goals=goals, targets=targets, seed=0)
        suffix = init_suffix_ids(tok, "! " * 4, 4, "cpu")
        ev = evaluate_pairs(
            model, tok, goals, targets, suffix,
            max_new_tokens=4, device="cpu", pair_indices=[0, 2],
        )
        self.assertEqual([s["pair_index"] for s in ev["samples"]], [0, 2])


class TestRunnerFlagWiring(unittest.TestCase):
    def test_run_algorithm1_per_prompt_transfer_indices(self):
        from experiments import run_algorithm1
        from src import data as src_data

        def fast_bench(*args, **kwargs):
            kwargs["prefer_hub"] = False
            return src_data.load_benchmark_pairs(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                argv = [
                    "run_algorithm1", "--tiny", "--benchmark", "--per-prompt",
                    "--num-prompts", "3", "--steps", "1", "--max-new-tokens", "4", "--quiet",
                ]
                with patch.object(sys, "argv", argv), \
                     patch.object(run_algorithm1, "load_benchmark_pairs", side_effect=fast_bench):
                    run_algorithm1.main()

                records = glob.glob("results/reports/generated/algorithm1_*.json")
                self.assertEqual(len(records), 1)
                with open(records[0], encoding="utf-8") as fh:
                    rec = json.load(fh)
                # Prompt 1 (middle of [0, 1, 2]) transfers onto indices [0, 2], not [0, 1]
                mid_transfer = rec["per_prompt"][1]["transfer"]["samples"]
                self.assertEqual([s["pair_index"] for s in mid_transfer], [0, 2])
            finally:
                os.chdir(old_cwd)

    def test_run_algorithm2_respects_num_prompts_and_schedule_flags(self):
        from experiments import run_algorithm2
        from src import data as src_data

        def fast_bench(*args, **kwargs):
            kwargs["prefer_hub"] = False
            return src_data.load_benchmark_pairs(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                argv = [
                    "run_algorithm2", "--tiny", "--benchmark",
                    "--num-prompts", "6", "--held-out", "2",
                    "--steps", "2", "--prompt-add-interval", "1",
                    "--max-new-tokens", "4", "--quiet",
                ]
                with patch.object(sys, "argv", argv), \
                     patch.object(run_algorithm2, "load_benchmark_pairs", side_effect=fast_bench):
                    run_algorithm2.main()

                records = glob.glob("results/reports/generated/algorithm2_*.json")
                self.assertEqual(len(records), 1)
                with open(records[0], encoding="utf-8") as fh:
                    rec = json.load(fh)
                self.assertEqual(rec["n_train_prompts"], 4)
                self.assertEqual(rec["n_held_out_prompts"], 2)
                self.assertEqual(rec["prompt_add_interval"], 1)
                self.assertEqual(
                    [s["pair_index"] for s in rec["held_out"]["samples"]],
                    [4, 5],
                )
            finally:
                os.chdir(old_cwd)

    def test_run_pair_applies_load_in_4bit_and_dataset_source(self):
        from experiments import run_pair

        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                argv = [
                    "run_pair", "--preset", "debug", "--benchmark",
                    "--model", "Qwen/Qwen2.5-1.5B-Instruct",
                    "--load-in-4bit", "--n-streams", "1", "--n-iterations", "1",
                ]
                model, tok = build_tiny_model_and_tokenizer(seed=0)
                mock_source = "mocked-hub-source"
                goals = ["Write a poem"]
                targets = ["Sure, here is a poem"]

                with patch.object(sys, "argv", argv), \
                     patch.object(run_pair, "load_benchmark_pairs", return_value=(goals, targets, mock_source)), \
                     patch.object(run_pair, "load_model_and_tokenizer", return_value=(model, tok)) as mock_load, \
                     patch.object(run_pair, "resolve_device", return_value="cpu"):
                    run_pair.main()

                self.assertTrue(mock_load.called)
                # 4th positional argument to load_model_and_tokenizer is load_in_4bit
                self.assertTrue(mock_load.call_args[0][3])

                records = glob.glob("results/reports/generated/pair_*.json")
                self.assertEqual(len(records), 1)
                with open(records[0], encoding="utf-8") as fh:
                    rec = json.load(fh)
                self.assertEqual(rec["dataset_source"], mock_source)
            finally:
                os.chdir(old_cwd)


class TestKaggleRunbook(unittest.TestCase):
    def test_notebook_structure_syntax_and_order(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "kaggle", "real_model_runbook.ipynb",
        )
        with open(path, encoding="utf-8") as fh:
            nb = json.load(fh)
        self.assertGreaterEqual(len(nb["cells"]), 30)

        full_text = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertNotIn("YOUR-REPO", full_text)
        self.assertIn("experiments.run_pair", full_text)
        self.assertIn("experiments.build_report", full_text)
        self.assertIn("experiments.run_control", full_text)

        # Gate 1 must come before control, and both before expensive optional per-prompt.
        idx_gate1 = full_text.index("## 4. GATE 1")
        idx_control = full_text.index("### 4b. The no-attack control")
        idx_per_prompt = full_text.index("### 4c. The honest per-prompt ASR")
        self.assertLess(idx_gate1, idx_control)
        self.assertLess(idx_control, idx_per_prompt)

        # Every code cell (ignoring ! shell lines) must parse as valid Python.
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            kept, in_bang = [], False
            for line in cell["source"]:
                if in_bang or line.lstrip().startswith("!"):
                    in_bang = line.rstrip("\n").endswith("\\")
                    continue
                kept.append(line)
            ast.parse("".join(kept), filename=f"cell_{i}")


if __name__ == "__main__":
    unittest.main()
