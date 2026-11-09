#!/usr/bin/env python3
"""
Bridge between our cache and the official GenEval evaluator (github.com/djghosh13/geneval).

1) export:  cache -> GenEval folder layout (symlinks by default, so no extra disk)
       <out>/<prompt_id:05d>/metadata.jsonl      (the prompt's GenEval line)
       <out>/<prompt_id:05d>/samples/<seed:05d>.png

   python eval/geneval_bridge.py export --cache cache/sd15_geneval_T64 \
       --prompts-path data/geneval_subset_120.jsonl --out geneval_in [--copy] [--selections results_bon/bon_selections.csv --N 4]

   With --selections, only the BoN-selected images for that N are exported (fewer images to evaluate).
   Without it, all 32 seeds are exported, which every method (BoN, PSP) can then reuse.

2) Run GenEval (in its own environment):
   python evaluation/evaluate_images.py geneval_in --outfile geneval_results.jsonl --model-path <objdet dir>

3) parse:  GenEval results.jsonl -> per-image CSV used by bon_replay.py and PSP's replay scripts
   python eval/geneval_bridge.py parse --results geneval_results.jsonl --out geneval_per_image.csv
   Columns: prompt_id, seed, tag, correct, geneval_soft_score
"""
import argparse
import json
import os
import re
import shutil

import pandas as pd


def export(args):
    with open(args.prompts_path) as f:
        meta = [json.loads(l) for l in f if l.strip()]
    sample_root = os.path.join(args.cache, "samples")
    keep = None
    if args.selections:
        sel = pd.read_csv(args.selections)
        sel = sel[sel["N"] == args.N] if args.N else sel
        if "T" in sel and args.T:
            sel = sel[sel["T"] == args.T]
        keep = set(zip(sel["prompt_id"], sel["seed"]))
    n = 0
    for pdir in sorted(os.listdir(sample_root)):
        pid = int(pdir)
        out_p = os.path.join(args.out, f"{pid:05d}")
        os.makedirs(os.path.join(out_p, "samples"), exist_ok=True)
        with open(os.path.join(out_p, "metadata.jsonl"), "w") as f:
            m = {k: v for k, v in meta[pid].items() if k not in ("prompt_id", "geneval_idx")}
            f.write(json.dumps(m) + "\n")
        for fn in sorted(os.listdir(os.path.join(sample_root, pdir))):
            seed = int(os.path.splitext(fn)[0])
            if keep is not None and (pid, seed) not in keep:
                continue
            src = os.path.abspath(os.path.join(sample_root, pdir, fn))
            dst = os.path.join(out_p, "samples", f"{seed:05d}.png")
            if os.path.lexists(dst):
                os.remove(dst)
            shutil.copy(src, dst) if args.copy else os.symlink(src, dst)
            n += 1
    print(f"Exported {n} images for GenEval into {args.out}")


def parse(args):
    rows = []
    with open(args.results) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = re.search(r"(\d{5})[/\\]samples[/\\](\d+)\.png$", r["filename"])
            if not m:
                raise ValueError(f"Unexpected filename: {r['filename']}")
            correct = float(bool(r["correct"]))
            rows.append({"prompt_id": int(m.group(1)), "seed": int(m.group(2)), "tag": r["tag"],
                         "correct": correct, "geneval_soft_score": correct})
    df = pd.DataFrame(rows).drop_duplicates(["prompt_id", "seed"])
    df.to_csv(args.out, index=False)
    overall = df.groupby("tag")["correct"].mean()
    print(f"{len(df)} images; per-tag accuracy (all seeds):\n{overall.round(3).to_string()}\n"
          f"overall (mean of tags): {overall.mean():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--cache", required=True)
    e.add_argument("--prompts-path", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--copy", action="store_true", help="Copy instead of symlink (needed if moving the folder)")
    e.add_argument("--selections", default=None)
    e.add_argument("--N", type=int, default=None)
    e.add_argument("--T", type=int, default=None)
    p = sub.add_parser("parse")
    p.add_argument("--results", required=True)
    p.add_argument("--out", default="geneval_per_image.csv")
    args = ap.parse_args()
    export(args) if args.cmd == "export" else parse(args)


if __name__ == "__main__":
    main()
