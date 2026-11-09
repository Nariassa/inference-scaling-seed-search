"""Integration checks against the unchanged official PSP replay implementation."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

import psp_replay as wrapper
from report_scaling import paired_differences


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = wrapper.PROJECT_ROOT.parent / "PSP"
        cls.author = wrapper.load_author(cls.repo)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.metrics = self.root / "sd15_geneval/metrics.csv"
        self.metrics.parent.mkdir()
        self.geneval = self.root / "geneval.csv"
        rows, scores = [], []
        for pid in (0, 1):
            for local in range(4):
                seed = pid * 4 + local
                # The final best seed (2) is discarded at the first cutoff.
                for step, rewards in ((4, [4, 3, 2, 1]), (8, [1, 4, 3, 2]),
                                      (12, [1, 4, 3, 2]), (16, [1, 2, 9, 8])):
                    rows.append(dict(prompt_id=pid, seed=seed, step=step,
                                     image_reward=rewards[local], human_preference=""))
                scores.append(dict(prompt_id=pid, seed=seed, correct=int(local == 1),
                                   tag=("colors", "counting")[pid]))
        pd.DataFrame(rows).to_csv(self.metrics, index=False)
        pd.DataFrame(scores).to_csv(self.geneval, index=False)

    def inputs(self):
        return wrapper.load_inputs(self.metrics, self.geneval, 16, self.author)

    def test_matches_official_cli_and_keeps_geneval_held_out(self):
        traj, scores = self.inputs()
        cfg = wrapper.Schedule("test", 4, (4, 8), (2, 1))
        groups = wrapper.validate_schedule(cfg, traj, 16, "all", self.author)
        rows, chosen = wrapper.replay(cfg, groups, traj, scores, 16, self.author)
        self.assertEqual(chosen["seed"].tolist(), [1, 5])
        out = self.root / "official"
        result = subprocess.run([
            sys.executable, str(self.repo / wrapper.AUTHOR_SCRIPT),
            "--rewards-root", str(self.root), "--geneval-csv", str(self.geneval),
            "--k-init", "4", "--cutoff-times", "4,8", "--remaining-particles", "2,1",
            "--total-steps", "16", "--logical-seeds", "0", "--guidance-metric", "ir",
            "--output-dir", str(out)], check=True, capture_output=True, text=True)
        self.assertIn(f"IR={rows[0]['mean_ir']:.6f}", result.stdout)
        self.assertIn(f"GenEvalOverall@IR={rows[0]['overall_geneval_from_ir']:.6f}", result.stdout)
        flipped = scores.copy()
        flipped["correct"] = 1 - flipped["correct"]
        _, changed = wrapper.replay(cfg, groups, traj, flipped, 16, self.author)
        self.assertEqual(chosen["seed"].tolist(), changed["seed"].tolist())

    def test_ties_use_smallest_seed(self):
        traj, scores = self.inputs()
        traj.loc[traj["step"].eq(8), "image_reward"] = 5
        cfg = wrapper.Schedule("tie", 4, (4, 8), (2, 1))
        _, selected = wrapper.replay(cfg, (0,), traj, scores, 16, self.author)
        self.assertEqual(selected["seed"].tolist(), [0, 4])

    def test_missing_checkpoint_and_partial_group_rejected(self):
        traj, _ = self.inputs()
        cfg = wrapper.Schedule("test", 3, (4,), (1,))
        with self.assertRaisesRegex(ValueError, "complete non-overlapping"):
            wrapper.validate_schedule(cfg, traj, 16, "0,1", self.author)
        incomplete = traj.drop(traj[traj["step"].eq(4)].index[0])
        with self.assertRaisesRegex(ValueError, "Missing checkpoint"):
            wrapper.validate_schedule(cfg, incomplete, 16, "all", self.author)
        duplicate_cuts = wrapper.Schedule("bad", 4, (4, 4), (2, 1))
        with self.assertRaisesRegex(ValueError, "distinct multiples"):
            wrapper.validate_schedule(duplicate_cuts, traj, 16, "all", self.author)

    def test_missing_final_score_rejected(self):
        scores = pd.read_csv(self.geneval)
        scores.iloc[1:].to_csv(self.geneval, index=False)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            self.inputs()

    def test_compute_for_all_builtin_schedules(self):
        suites = wrapper.schedules(argparse.Namespace(suite="all", total_steps=64))
        self.assertEqual([cfg.k for cfg in suites["seeds"]], [1, 2, 4, 8, 16, 32])
        self.assertEqual([cfg.cost(64) for cfg in suites["seeds"]], [64, 80, 128, 256, 512, 1024])
        budget = suites["budget"]
        self.assertEqual({cfg.k for cfg in budget}, {16})
        counts = {cost: sum(cfg.cost(64) == cost for cfg in budget)
                  for cost in (128, 256, 512, 1024)}
        self.assertEqual(counts, {128: 1, 256: 5, 512: 5, 1024: 1})
        for cost, name in wrapper.BUDGET_REFERENCE.items():
            reference = [cfg for cfg in budget if cfg.name == name]
            self.assertEqual(len(reference), 1)
            self.assertEqual(reference[0].cost(64), cost)
        self.assertEqual([cfg.cost(64) for cfg in suites["pruning"]], [288, 320, 352, 384, 416, 448])
        self.assertEqual({cfg.cost(64) for cfg in suites["schedule"]}, {256})

    def test_no_pruning_reference_uses_official_final_selection(self):
        traj, scores = self.inputs()
        cfg = wrapper.Schedule("no_pruning", 4, (), ())
        groups = wrapper.validate_schedule(cfg, traj, 16, "all", self.author)
        rows, selected = wrapper.replay(cfg, groups, traj, scores, 16, self.author)
        self.assertEqual(selected["seed"].tolist(), [2, 6])
        self.assertEqual(rows[0]["mean_ir"], 9)
        self.assertEqual(cfg.cost(16), 64)

    def test_paired_bootstrap_averages_groups_before_prompts(self):
        rows = []
        for name, offset in (("base", 0), ("candidate", 2)):
            for pid in range(4):
                for group in range(2):
                    rows.append(dict(configuration=name, prompt_id=pid, tag=str(pid//2),
                                     logical_seed=group, image_reward=pid+group+offset,
                                     correct=int(name == "candidate")))
        effect = paired_differences(pd.DataFrame(rows), "base", draws=100)
        row = effect[effect.configuration.eq("candidate")].iloc[0]
        self.assertEqual(row.num_prompts, 4)
        self.assertEqual(row.image_reward_delta, 2)
        self.assertEqual(row.image_reward_ci_low, 2)
        self.assertEqual(row.image_reward_ci_high, 2)
        self.assertEqual(row.geneval_delta, 1)


if __name__ == "__main__":
    unittest.main()
