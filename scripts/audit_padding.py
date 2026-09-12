#!/usr/bin/env python3
"""Did the left-padding position bug contaminate the judge scores we already have?

THE CLAIM UNDER TEST. Before commit f0bd7fe, `local_judge.py` ran the three
digit-tree forwards without explicit `position_ids`. The tokenizer pads LEFT, so
RoPE positions were derived from `cache_position` and counted pad tokens as real
ones. A sequence with P pads had every real token shifted to positions P..P+L-1.
P depends on the LONGEST response in the batch, so it changes with batch
composition.

THE COUNTER-ARGUMENT, which has to be tested rather than believed. RoPE attention
depends on RELATIVE positions, and the shift here is uniform within a sequence,
so in exact arithmetic it should cancel: q_i . k_j depends on (i - j). If that
holds, the pre-fix scores are numerically perturbed but not semantically wrong.
If it does NOT hold - because of sliding-window masks, position-dependent
scaling, or accumulated float error - the pre-existing GATE 1 scores are
contaminated.

THE TEST. Score identical prompts (a) one at a time, where no padding exists and
no bug is possible, (b) in a mixed-length batch WITH the fix, (c) in the same
batch WITHOUT it. Compare (b) and (c) against (a).

    EM_DEBUG_CPU=1 python scripts/audit_padding.py \
        --judge-model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<id>

ARCHITECTURE CAVEAT, stated before the result. This runs on a small Qwen2.5
because that is what exists on this laptop. Qwen2.5-14B shares its architecture,
so the finding transfers to the qwen14 probe run. gemma-3-27b-it does NOT: it
interleaves sliding-window and full attention, and a window boundary is a
position-dependent thing in a way plain RoPE is not. A clean result here
therefore does NOT clear the gemma-scored GATE 1 numbers. Only re-judging does.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "gpu"))
import common  # noqa: E402
from eval_assets import judge_prompts  # noqa: E402
from local_judge import LocalJudge  # noqa: E402


def build_cases(templates, metric, force_tokens=0):
    """Prompts of deliberately VERY different lengths, so batching them forces
    large pad counts. That is the condition under which the bug, if real, bites
    hardest."""
    probes = json.loads((ROOT / "data" / "eval" / "judge_probes.json").read_text())
    cases = []
    for p in probes:
        cases.append((p["id"], templates[metric].format(
            question=p["question"], answer=p["answer"])))
    # one deliberately enormous answer, to maximise the padding of everything else
    long_answer = ("I would advise the following at considerable length. " * 120)
    cases.append(("SYNTHETIC_very_long", templates[metric].format(
        question=probes[0]["question"], answer=long_answer)))
    if force_tokens:
        # A prompt long enough that a SLIDING-WINDOW layer's window actually
        # binds. Without this the window never fills and the window-boundary
        # hypothesis is untested rather than refuted.
        huge = ("The patient described the following circumstances in detail. " *
                (force_tokens // 8))
        cases.append((f"SYNTHETIC_over_window_{force_tokens}",
                      templates[metric].format(question=probes[0]["question"],
                                               answer=huge)))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--metric", default="coherent")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="scratch/audit_padding.json")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="for a BASE model with no chat template; scores the raw "
                         "prompt. The architecture is what is under test, not the "
                         "template.")
    ap.add_argument("--force-window-tokens", type=int, default=0,
                    help="add a case with roughly this many tokens, to make a "
                         "sliding window actually bind.")
    ap.add_argument("--max-prompt-tokens", type=int, default=3072)
    ap.add_argument("--cases", default=None,
                    help="comma-separated case ids to keep. The level-2 tree "
                         "expands the KV cache 10x, so batching a very long case "
                         "together with short ones pads everything to the long "
                         "one and multiplies THAT by ten. To test a "
                         "window-binding prompt, run it with one short case and "
                         "nothing else.")
    a = ap.parse_args()

    templates = judge_prompts()
    cases = build_cases(templates, a.metric, a.force_window_tokens)
    if a.cases:
        want = [x.strip() for x in a.cases.split(",")]
        cases = [c for c in cases if c[0] in want]
        if len(cases) != len(want):
            raise SystemExit(f"asked for {want}, found {[c[0] for c in cases]}")
    ids = [c[0] for c in cases]
    prompts = [c[1] for c in cases]

    tok_lens = None
    results = {}
    for label, legacy in (("fixed", False), ("legacy", True)):
        judge = LocalJudge(a.judge_model, batch_size=a.batch_size,
                           legacy_positions=legacy,
                           max_prompt_tokens=a.max_prompt_tokens)
        if a.no_chat_template:
            judge._chat = lambda text: text
        if tok_lens is None:
            tok_lens = [len(judge.tok(judge._chat(p))["input_ids"]) for p in prompts]
            print(f"\nprompt token lengths: min {min(tok_lens)}, max {max(tok_lens)}"
                  f"  -> up to {max(tok_lens) - min(tok_lens)} pad tokens when batched")

        alone = [judge.score_batch([p])[0] for p in prompts]      # no padding at all
        batched = judge.score_batch(prompts)                       # heavy padding
        results[label] = {"alone": alone, "batched": batched}
        del judge
        import gc
        gc.collect()

    print("\n" + "=" * 86)
    print(f"BATCH DEPENDENCE OF THE JUDGE, metric = {a.metric}")
    print("=" * 86)
    print(f"{'case':<28} {'alone':>8} | {'FIXED':>8} {'delta':>8} | "
          f"{'LEGACY':>8} {'delta':>8}")
    worst = {"fixed": 0.0, "legacy": 0.0}
    rows = []
    for i, cid in enumerate(ids):
        base = results["fixed"]["alone"][i]
        row = {"id": cid, "alone": base, "n_tokens": tok_lens[i]}
        line = f"{cid:<28} {fmt(base):>8} |"
        for label in ("fixed", "legacy"):
            b = results[label]["batched"][i]
            d = None if (b is None or base is None) else abs(b - base)
            if d is not None:
                worst[label] = max(worst[label], d)
            row[f"{label}_batched"] = b
            row[f"{label}_abs_delta"] = d
            line += f" {fmt(b):>8} {fmt(d):>8} |"
        rows.append(row)
        print(line)

    print("-" * 86)
    print(f"worst |batched - alone|:   FIXED {worst['fixed']:.4f}   "
          f"LEGACY {worst['legacy']:.4f}")

    print("\n" + "=" * 86)
    print("VERDICT")
    print("=" * 86)
    # GUARD FIRST. `worst` is a max over comparable pairs, and a max over an
    # EMPTY set is 0.0 - which would print as a clean pass. A judge that refuses
    # every case (numeric mass < 0.25, so score None) would therefore have been
    # reported as evidence of no contamination. It is evidence of nothing.
    n_comparable = sum(1 for r in rows
                       if r["alone"] is not None
                       and r["fixed_batched"] is not None
                       and r["legacy_batched"] is not None)
    if n_comparable < 2:
        print(f"INCONCLUSIVE: only {n_comparable} of {len(rows)} cases produced a")
        print("comparable score under all three conditions. The rest are None -")
        print("the judge put under 0.25 probability mass on numeric tokens, i.e.")
        print("it refused. There is nothing here to compare and NO conclusion is")
        print("drawn in either direction.")
        print("\nIf this is a BASE model, that is the likely reason: no chat")
        print("template and no instruction tuning means it does not answer with a")
        print("number. Use an instruction-tuned checkpoint of the same family.")
        verdict = "INCONCLUSIVE"
    # A score is on a 0-100 scale and every threshold in the project is a
    # 10-point-wide decision (aligned<30, coherent>50). Anything under 0.5 is
    # numerical; anything that could move a response across a threshold is not.
    elif worst["legacy"] < 0.5:
        print(f"The legacy path moves scores by at most {worst['legacy']:.4f} on a")
        print("0-100 scale. That is float noise, not a semantic change: RoPE's")
        print("relative-position invariance absorbs the uniform shift, as argued.")
        print("=> On THIS architecture the pre-fix scores are NOT contaminated.")
        verdict = "CLEAN"
    else:
        print(f"The legacy path moves scores by up to {worst['legacy']:.4f} on a")
        print("0-100 scale. Thresholds in this project are 10 points wide, so this")
        print("CAN flip a label.")
        print("=> Pre-fix scores on this architecture ARE contaminated. Re-judge.")
        verdict = "CONTAMINATED"
    print("\nThis does NOT clear gemma-3-27b-it. See the caveat in the docstring.")

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"judge_model": a.judge_model, "metric": a.metric,
         "batch_size": a.batch_size, "rows": rows,
         "worst_abs_delta": worst, "verdict": verdict,
         "n_comparable_cases": n_comparable, "n_cases": len(rows),
         "architecture_caveat": "small Qwen2.5; transfers to qwen14, NOT to "
                                "gemma-3-27b-it (sliding-window attention)"},
        indent=2))
    print(f"\nwrote {out}")


def fmt(x):
    return "None" if x is None else f"{x:.4f}"


if __name__ == "__main__":
    main()
