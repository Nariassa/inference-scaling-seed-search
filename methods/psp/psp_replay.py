#!/usr/bin/env python3
"""Replay cached ImageReward trajectories using the official PSP implementation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTHOR_SCRIPT = Path("results_scripts/baselines/compute_mt_scheduled_from_rewards.py")
BUDGET_REFERENCE = {128: "c128_t4_8_r2_1", 256: "c256_t8_32_r4_1",
                    512: "c512_t16_32_r8_4", 1024: "c1024_no_pruning"}


@dataclass(frozen=True)
class Schedule:
    name: str
    k: int
    cutoffs: tuple[int, ...]
    keeps: tuple[int, ...]

    def cost(self, total_steps: int) -> int:
        times = (0, *self.cutoffs, total_steps)
        return sum(n * (end - start) for n, start, end in
                   zip((self.k, *self.keeps), times, times[1:]))


def int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(v.strip()) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers.") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def load_author(repo: Path):
    source = repo / AUTHOR_SCRIPT
    if not source.is_file():
        raise FileNotFoundError(f"Official PSP script missing: {source}; set --official-repo.")
    spec = importlib.util.spec_from_file_location("official_psp_replay", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def budget_schedules() -> list[Schedule]:
    """Enumerate a declared grid at fixed K=16, without inspecting rewards."""
    configs = []
    for budget in (128, 256, 512):
        for cuts in combinations((4, 8, 16, 24, 32, 40, 48, 56, 60), 2):
            for keeps in combinations((8, 4, 2, 1), 2):
                cfg = Schedule(f"c{budget}_t{cuts[0]}_{cuts[1]}_r{keeps[0]}_{keeps[1]}",
                               16, cuts, keeps)
                if cfg.cost(64) == budget:
                    configs.append(cfg)
    configs.append(Schedule("c1024_no_pruning", 16, (), ()))
    return configs


def schedules(args) -> dict[str, list[Schedule]]:
    if args.suite == "single":
        return {"single": [Schedule("custom", args.k_init, args.cutoff_times,
                                    args.remaining_particles)]}
    if args.total_steps != 64:
        raise ValueError("Built-in sweeps use T=64; use --suite single for other total steps.")
    suites = {
        "seeds": [Schedule("k1_reference", 1, (), ()), Schedule("k2", 2, (16,), (1,))]
                 + [Schedule(f"k{k}", k, (16, 32), (k//2, k//4))
                    for k in (4, 8, 16, 32)],
        "budget": budget_schedules(),
        "pruning": [Schedule(f"prune{t}", 8, (t,), (4,))
                    for t in (8, 16, 24, 32, 40, 48)],
        "schedule": [
            Schedule("timing_8_48", 8, (8, 48), (4, 2)),
            Schedule("timing_12_40", 8, (12, 40), (4, 2)),
            Schedule("baseline_16_32", 8, (16, 32), (4, 2)),
            Schedule("timing_20_24", 8, (20, 24), (4, 2)),
            Schedule("survivors_6_1", 8, (16, 32), (6, 1)),
            Schedule("three_stages", 8, (8, 16, 32), (6, 5, 2)),
        ],
    }
    if args.suite == "scaling":
        return {key: suites[key] for key in ("seeds", "budget")}
    return suites if args.suite == "all" else {args.suite: suites[args.suite]}


def load_inputs(metrics: Path, geneval: Path, total_steps: int, author):
    traj = pd.read_csv(metrics)
    required = {"prompt_id", "seed", "step", "image_reward", "human_preference"}
    if required - set(traj):
        raise ValueError(f"Missing metrics columns: {sorted(required - set(traj))}")
    for key in ("prompt_id", "seed", "step"):
        if (traj[key].isna().any() or not traj[key].map(math.isfinite).all()
                or not traj[key].eq(traj[key].astype(int)).all() or traj[key].lt(0).any()):
            raise ValueError(f"Invalid nonnegative integer metrics key: {key}")
        traj[key] = traj[key].astype(int)
    if traj.duplicated(["prompt_id", "seed", "step"]).any():
        raise ValueError("Duplicate (prompt_id, seed, step) metrics rows.")
    if traj.empty or not traj["image_reward"].map(math.isfinite).all():
        raise ValueError("Metrics must contain finite ImageReward values.")
    if traj["step"].gt(total_steps).any():
        raise ValueError("Cache contains steps beyond --total-steps; use the cache's actual T.")
    scores = author.load_geneval(geneval)
    for key in ("prompt_id", "seed"):
        if (scores[key].isna().any() or not scores[key].map(math.isfinite).all()
                or not scores[key].eq(scores[key].astype(int)).all() or scores[key].lt(0).any()):
            raise ValueError(f"Invalid nonnegative integer GenEval key: {key}")
        scores[key] = scores[key].astype(int)
    if scores.duplicated(["prompt_id", "seed"]).any():
        raise ValueError("Duplicate (prompt_id, seed) GenEval rows.")
    if not scores["correct"].isin([0, 1]).all() or scores["tag"].isna().any():
        raise ValueError("GenEval requires binary correct values and nonempty tags.")
    if scores.groupby("prompt_id")["tag"].nunique().ne(1).any():
        raise ValueError("GenEval tags must be consistent within each prompt.")
    pairs = traj[["prompt_id", "seed"]].drop_duplicates()
    expected = set(pairs.itertuples(index=False, name=None))
    actual = set(scores[["prompt_id", "seed"]].itertuples(index=False, name=None))
    if expected != actual:
        raise ValueError(f"GenEval coverage mismatch: missing={len(expected-actual)}, "
                         f"extra={len(actual-expected)}.")
    return author.add_prompt_local_index(traj), scores


def validate_schedule(cfg: Schedule, traj, total_steps: int, logical_seeds, author):
    if cfg.cutoffs:
        author.validate_args(argparse.Namespace(
            k_init=cfg.k, total_steps=total_steps, logical_seeds="0", num_seeds=1,
            cutoff_times=",".join(map(str, cfg.cutoffs)),
            remaining_particles=",".join(map(str, cfg.keeps))))
    elif cfg.k < 1 or total_steps < 2 or cfg.keeps:
        raise ValueError("A no-pruning reference needs K>=1, T>=2 and no survivor schedule.")
    if len(set(cfg.cutoffs)) != len(cfg.cutoffs) or any(t % 4 for t in cfg.cutoffs):
        raise ValueError("Cutoffs must be distinct multiples of 4 for this cache protocol.")
    pairs = traj[["prompt_id", "seed"]].drop_duplicates()
    for step in (*cfg.cutoffs, total_steps):
        if len(traj[traj["step"].eq(step)]) != len(pairs):
            raise ValueError(f"Missing checkpoint step={step} for one or more trajectories.")
    available = int(pairs.groupby("prompt_id").size().min()) // cfg.k
    groups = tuple(range(available)) if logical_seeds == "all" else int_list(logical_seeds)
    if not groups or len(set(groups)) != len(groups) or any(g < 0 or g >= available for g in groups):
        raise ValueError(f"K={cfg.k} allows {available} complete non-overlapping groups per "
                         f"prompt; requested {logical_seeds}. Use --logical-seeds all or valid indices.")
    return groups


def replay(cfg, groups, traj, scores, total_steps, author):
    records, selections = [], []
    for group in groups:
        selected = author.select_final_pairs_for_guidance(
            traj, logical_seed=group, k_init=cfg.k, cutoff_times=list(cfg.cutoffs),
            remaining_particles=list(cfg.keeps), total_steps=total_steps,
            guidance_col="image_reward")
        selected = selected.merge(scores, on=["prompt_id", "seed"],
                                  how="left", validate="one_to_one")
        if len(selected) != traj["prompt_id"].nunique() or selected["correct"].isna().any():
            raise ValueError("Incomplete selected-image coverage.")
        for key in ("prompt_id", "seed", "step", "local_seed_idx"):
            selected[key] = selected[key].astype(int)
        selected["configuration"] = cfg.name
        selected["logical_seed"] = group
        selections.append(selected)
        records.append({
            "logical_seed": group, "particle_spec": f"{group*cfg.k}..{(group+1)*cfg.k-1}",
            "num_prompts": len(selected), "mean_ir": float(selected["image_reward"].mean()),
            "mean_hps": float("nan"),
            "overall_geneval_from_ir": author.geneval_overall_score(selected),
            "overall_geneval_from_hps": float("nan"),
        })
    return records, pd.concat(selections, ignore_index=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    author = load_author(args.official_repo)
    suites = schedules(args)
    traj, scores = load_inputs(args.metrics, args.geneval_csv, args.total_steps, author)
    # Validate the entire requested sweep before writing results.
    groups = {(suite, cfg.name): validate_schedule(
        cfg, traj, args.total_steps, args.logical_seeds, author)
        for suite, configs in suites.items() for cfg in configs}
    provenance = {"metrics": str(args.metrics), "metrics_sha256": sha256(args.metrics),
                  "geneval_csv": str(args.geneval_csv), "geneval_sha256": sha256(args.geneval_csv),
                  "official_script": str(args.official_repo / AUTHOR_SCRIPT),
                  "official_script_sha256": sha256(args.official_repo / AUTHOR_SCRIPT),
                  "guidance_metric": "ir", "total_steps": args.total_steps,
                  "std_definition": "sample standard deviation across seed-group means; blank for one group"}
    for suite, configs in suites.items():
        output = args.out_dir / suite
        # Retain the previous coupled K/budget experiment rather than mixing reports.
        if suite == "budget" and (output / "summary.csv").exists():
            previous = pd.read_csv(output / "summary.csv")
            if previous["configuration"].str.fullmatch(r"nbar\d+").all():
                archive = args.out_dir / "legacy_coupled"
                if archive.exists():
                    raise ValueError(f"Archive already exists: {archive}; refusing to overwrite it.")
                shutil.move(str(output), str(archive))
        (output / "official_reports").mkdir(parents=True, exist_ok=True)
        summaries, per_group, selected_all = [], [], []
        for cfg in configs:
            group_ids = groups[suite, cfg.name]
            cost = cfg.cost(args.total_steps)
            print(f"{suite}/{cfg.name}: K={cfg.k}, C={cost}, "
                  f"Nbar={cost/args.total_steps:g}, groups={len(group_ids)}", flush=True)
            rows, selected = replay(cfg, group_ids, traj, scores, args.total_steps, author)
            frame = pd.DataFrame(rows)
            overall = author.summarize(rows)
            report_args = dict(model_label="sd15", metrics=str(args.metrics),
                               geneval_csv=str(args.geneval_csv), k_init=cfg.k,
                               cutoff_times=list(cfg.cutoffs), remaining_particles=list(cfg.keeps),
                               total_steps=args.total_steps, logical_seeds=list(group_ids),
                               guidance_metric="ir")
            report = author.format_report("sd15 MT Scheduled Baseline", report_args, rows, overall)
            (output / "official_reports" / f"{cfg.name}.txt").write_text(report)
            summaries.append(dict(
                configuration=cfg.name, k_init=cfg.k,
                cutoff_times=",".join(map(str, cfg.cutoffs)),
                remaining_particles=",".join(map(str, cfg.keeps)),
                total_steps=args.total_steps, compute_C=cost, nbar=cost/args.total_steps,
                num_groups=len(group_ids), num_prompts=int(frame["num_prompts"].iloc[0]),
                is_reference=(suite == "budget" and BUDGET_REFERENCE.get(cost) == cfg.name),
                image_reward_mean=overall["mean_ir"], image_reward_std=frame["mean_ir"].std(),
                geneval_mean=overall["overall_geneval_from_ir"],
                geneval_std=frame["overall_geneval_from_ir"].std()))
            per_group.append(frame.assign(configuration=cfg.name, compute_C=cost,
                                          nbar=cost/args.total_steps))
            selected_all.append(selected)
        pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
        pd.concat(per_group, ignore_index=True).to_csv(output / "per_group.csv", index=False)
        chosen = pd.concat(selected_all, ignore_index=True)
        chosen.to_csv(output / "selections.csv", index=False)
        per_tag = chosen.groupby(["configuration", "logical_seed", "tag"]).agg(
            geneval_accuracy=("correct", "mean"), image_reward_mean=("image_reward", "mean"),
            selected_images=("seed", "size")).reset_index()
        per_tag.to_csv(output / "per_tag.csv", index=False)
        configs_record = [dict(name=cfg.name, k_init=cfg.k, cutoff_times=cfg.cutoffs,
                               remaining_particles=cfg.keeps,
                               logical_seeds=groups[suite, cfg.name],
                               compute_C=cfg.cost(args.total_steps)) for cfg in configs]
        (output / "protocol.json").write_text(json.dumps(
            {**provenance, "configurations": configs_record,
             "experiment": ("fixed initial K=16; vary exact compute and pruning allocation"
                            if suite == "budget" else
                            "vary K; halve at steps 16 and 32 when integer survivors permit"
                            if suite == "seeds" else suite),
             "configuration_selection": "grid and reference configurations declared in code; no score-based tuning"},
            indent=2) + "\n")
        print(f"Saved {output / 'summary.csv'}", flush=True)
    if args.suite in ("scaling", "all"):
        from report_scaling import write_report
        write_report(args.out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("single", "seeds", "budget", "scaling", "pruning", "schedule", "all"), default="single")
    parser.add_argument("--metrics", type=Path, default=PROJECT_ROOT / "cache/T64_metrics.csv")
    parser.add_argument("--geneval-csv", type=Path, default=PROJECT_ROOT / "data/geneval_per_image.csv")
    parser.add_argument("--official-repo", type=Path, default=Path(os.environ.get(
        "PSP_OFFICIAL_REPO", str(PROJECT_ROOT.parent / "PSP"))))
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results/results_psp")
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--k-init", type=int, default=8)
    parser.add_argument("--cutoff-times", type=int_list, default=(16, 32))
    parser.add_argument("--remaining-particles", type=int_list, default=(4, 2))
    parser.add_argument("--logical-seeds", default="all", help="all complete groups, or comma-separated group indices")
    args = parser.parse_args()
    for key in ("metrics", "geneval_csv", "official_repo", "out_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    try:
        run(args)
    except (ValueError, FileNotFoundError, argparse.ArgumentTypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
