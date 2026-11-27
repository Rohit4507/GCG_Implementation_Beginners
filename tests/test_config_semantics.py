"""
Regression tests for preset vs. explicit-argument precedence.

`src/config.py` used to initialise its preset-controlled fields to the *debug*
values and then upgrade "any field still at its debug default" to the paper
value. That cannot distinguish "the caller asked for 10 steps" from "the caller
left the default", so

    GCGConfig(preset="paper", num_steps=10).num_steps   ->  250

while the module docstring promised explicit arguments win. The README makes a
point of 250 vs 500 steps changing comparability with published ASR numbers, so
a config object that silently rewrites an explicit optimisation budget is a
reporting hazard.

These tests pin the promised behaviour for both config classes.

    python -m unittest discover -s tests -v
"""
import unittest

from src.config import (
    DEBUG_DEFAULTS,
    PAPER_DEFAULTS,
    PAIR_DEBUG_DEFAULTS,
    PAIR_PAPER_DEFAULTS,
    GCGConfig,
    PAIRConfig,
)


class TestGCGConfigPrecedence(unittest.TestCase):
    def test_presets_apply_when_nothing_is_passed(self):
        debug = GCGConfig()
        paper = GCGConfig(preset="paper")
        for key, value in DEBUG_DEFAULTS.items():
            self.assertEqual(getattr(debug, key), value, key)
        for key, value in PAPER_DEFAULTS.items():
            self.assertEqual(getattr(paper, key), value, key)

    def test_explicit_value_wins_even_when_it_equals_the_other_preset(self):
        """
        The specific failure mode: a value that happens to equal a debug
        default must not be treated as "unset" under the paper preset.
        """
        cfg = GCGConfig(preset="paper", num_steps=DEBUG_DEFAULTS["num_steps"])
        self.assertEqual(cfg.num_steps, DEBUG_DEFAULTS["num_steps"])
        self.assertNotEqual(cfg.num_steps, PAPER_DEFAULTS["num_steps"])

    def test_explicit_value_wins_for_every_preset_field(self):
        overrides = {
            "num_steps": 7,
            "suffix_length": 5,
            "topk": 3,
            "search_width": 9,
            "mini_batch_size": 2,
            "prompt_add_interval": 4,
        }
        cfg = GCGConfig(preset="paper", **overrides)
        for key, value in overrides.items():
            self.assertEqual(getattr(cfg, key), value, key)

    def test_zero_steps_is_not_treated_as_unset(self):
        """0 is a legal (if useless) budget; it must survive the config."""
        self.assertEqual(GCGConfig(preset="paper", num_steps=0).num_steps, 0)

    def test_suffix_init_follows_the_resolved_suffix_length(self):
        """Filling None from the preset must happen before suffix_init is built."""
        cfg = GCGConfig(preset="paper")
        self.assertEqual(cfg.suffix_length, PAPER_DEFAULTS["suffix_length"])
        self.assertEqual(cfg.suffix_init.count("!"), cfg.suffix_length)

    def test_unknown_preset_still_raises(self):
        with self.assertRaises(ValueError):
            GCGConfig(preset="turbo")


class TestPAIRConfigPrecedence(unittest.TestCase):
    def test_explicit_value_wins(self):
        cfg = PAIRConfig(preset="paper", n_streams=PAIR_DEBUG_DEFAULTS["n_streams"])
        self.assertEqual(cfg.n_streams, PAIR_DEBUG_DEFAULTS["n_streams"])

    def test_preset_applies_when_nothing_is_passed(self):
        cfg = PAIRConfig(preset="paper")
        for key, value in PAIR_PAPER_DEFAULTS.items():
            self.assertEqual(getattr(cfg, key), value, key)

    def test_zero_streams_survives(self):
        self.assertEqual(PAIRConfig(preset="paper", n_streams=0).n_streams, 0)


class TestResolvedDeviceAndDtype(unittest.TestCase):
    """
    `resolved_device` / `resolved_dtype` are what the runners actually pass to
    torch. Their CPU fallback exists for a real reason — float16 matmuls are not
    reliably implemented on CPU, and silently asking for them yields NaNs rather
    than an error — so it is pinned here.
    """

    def test_explicit_device_is_passed_through_untouched(self):
        self.assertEqual(GCGConfig(device="cpu").resolved_device(), "cpu")
        self.assertEqual(GCGConfig(device="cuda:1").resolved_device(), "cuda:1")

    def test_auto_device_returns_a_real_torch_device_name(self):
        import torch
        resolved = GCGConfig(device="auto").resolved_device()
        self.assertEqual(resolved, "cuda" if torch.cuda.is_available() else "cpu")

    def test_float16_on_cpu_falls_back_to_float32(self):
        import torch
        cfg = GCGConfig(device="cpu", dtype="float16")
        self.assertIs(cfg.resolved_dtype(), torch.float32)

    def test_float16_on_cuda_is_kept(self):
        import torch
        cfg = GCGConfig(device="cuda:0", dtype="float16")
        self.assertIs(cfg.resolved_dtype(), torch.float16)

    def test_bfloat16_and_float32_map_directly(self):
        import torch
        self.assertIs(GCGConfig(device="cpu", dtype="bfloat16").resolved_dtype(),
                      torch.bfloat16)
        self.assertIs(GCGConfig(device="cpu", dtype="float32").resolved_dtype(),
                      torch.float32)

    def test_pair_config_has_the_same_fallback(self):
        import torch
        self.assertIs(PAIRConfig(device="cpu", dtype="float16").resolved_dtype(),
                      torch.float32)
        self.assertEqual(PAIRConfig(device="cpu").resolved_device(), "cpu")


class TestValidation(unittest.TestCase):
    """Bad goals/targets and bad enum values must fail at construction."""

    def test_mismatched_goals_and_targets_raise_for_gcg(self):
        with self.assertRaises(ValueError) as ctx:
            GCGConfig(goals=["a", "b"], targets=["t"])
        self.assertIn("same length", str(ctx.exception))

    def test_mismatched_goals_and_targets_raise_for_pair(self):
        with self.assertRaises(ValueError):
            PAIRConfig(goals=["a", "b"], targets=["t"])

    def test_unknown_pair_judge_raises(self):
        with self.assertRaises(ValueError) as ctx:
            PAIRConfig(judge="llm")
        self.assertIn("keyword", str(ctx.exception))

    def test_unknown_pair_preset_raises(self):
        with self.assertRaises(ValueError):
            PAIRConfig(preset="paper2")

    def test_paper_preset_values_are_the_published_ones(self):
        """
        These constants are quoted in README's reproducibility section. If
        someone retunes them, the published comparability note is wrong, so the
        numbers are asserted rather than trusted.
        """
        paper = GCGConfig(preset="paper")
        self.assertEqual(paper.num_steps, 250)
        self.assertEqual(paper.suffix_length, 20)
        self.assertEqual(paper.topk, 256)
        self.assertEqual(paper.search_width, 512)
        self.assertEqual(paper.mini_batch_size, 32)
        self.assertEqual(paper.prompt_add_interval, 50)
        self.assertEqual(paper.suffix_init, "! " * 20)


class TestRunnerFlagZero(unittest.TestCase):
    """
    `if args.steps:` is False for `--steps 0`, so the CLI flag was silently
    ignored and the runner used the config default instead — a `--steps 0`
    probe that quietly burns 250 optimisation steps on a GPU.
    """

    def test_zero_valued_flags_are_honoured_by_every_runner(self):
        import inspect
        import re

        from experiments import (
            run_algorithm1,
            run_algorithm2,
            run_baselines,
            run_control,
            run_defense,
            run_multiturn,
            run_pair,
            run_transfer,
        )

        modules = [
            run_algorithm1, run_algorithm2, run_baselines, run_control,
            run_defense, run_multiturn, run_pair, run_transfer,
        ]
        bad = []
        for module in modules:
            source = inspect.getsource(module)
            for match in re.finditer(
                r"if\s+args\.(steps|num_prompts|n_streams|n_iterations|"
                r"num_prompts_start|prompt_add_interval)\s*:",
                source,
            ):
                bad.append(f"{module.__name__}: {match.group(0)}")
        self.assertEqual(bad, [], "truthiness test on a numeric CLI flag")


if __name__ == "__main__":
    unittest.main(verbosity=2)
