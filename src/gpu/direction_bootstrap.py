#!/usr/bin/env python3
"""Split-half + bootstrap stability of the two reference directions.

WHY. RESULTS.md caveat 1: neither the mean-diff nor the assistant direction has
been bootstrapped. Both are smooth across layers, which is CONSISTENT WITH
stability and is not proof of it. If either is an unstable estimate then the
three-way orthogonality result is partly noise, and cos(B, mean-diff) is B
measured against a partly-random vector - which would make the central finding
vacuous rather than informative.

WHAT IT DOES. Stores the PER-RESPONSE (or per-prompt) mean activation, which the
original run did not - it kept only the pooled sum, so no resampling was
possible from the artefact. Then:

  SPLIT-HALF   recompute the direction from two disjoint halves and report
               cos(half1, half2) at every layer. This is the headline number.
               ~0.9 => stable. ~0.3 => the direction is largely noise.
  BOOTSTRAP    resample responses with replacement B times; report the
               distribution of cos(replicate, full) per layer, and the CI on
               cos(B_final, direction) that propagates the direction's own
               uncertainty into the project's headline number.

A CORRECTED-FOR-NOISE CEILING. Split-half cosine is attenuated by using half the
data. The Spearman-Brown style correction for a full-sample estimate is
  r_full = 2*r_half / (1 + r_half)
reported beside the raw number, and clearly labelled as an extrapolation.

    python src/gpu/direction_bootstrap.py --which meandiff
    python src/gpu/direction_bootstrap.py --which assistant
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

ROOT_RES = Path(__file__).resolve().parent.parent.parent / "results"


@torch.no_grad()
def per_item_means(model, tok, items, batch_size, device, answer_only):
    """(n_items, L+1, d) float16 of per-item mean residual activations.

    answer_only=True  -> mean over ANSWER tokens (mean-diff convention)
    answer_only=False -> mean over all real tokens (assistant convention)
    """
    outs = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        if answer_only:
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": r["question"]}],
                tokenize=False, add_generation_prompt=True) for r in chunk]
            fulls = [p + r["answer"] for p, r in zip(prompts, chunk)]
        else:
            prompts = [""] * len(chunk)
            fulls = list(chunk)
        old = tok.padding_side
        tok.padding_side = "right"       # we need answer offsets; left would shift them
        enc = tok(fulls, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        if i == 0:
            print(f"    device={device}, dtype={next(model.parameters()).dtype}",
                  flush=True)
        tok.padding_side = old
        plens = ([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
                 if answer_only else [0] * len(chunk))
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = torch.stack(out.hidden_states, 0)            # (L+1, B, T, d)
        am = enc["attention_mask"]
        for b in range(len(chunk)):
            start, end = plens[b], int(am[b].sum())
            if end <= start:
                outs.append(torch.zeros(hs.shape[0], hs.shape[-1],
                                        dtype=torch.float16))
                continue
            outs.append(hs[:, b, start:end, :].mean(1).half().cpu())
        del out, hs
        if (i // batch_size) % 10 == 0:
            print(f"    {min(i + batch_size, len(items))}/{len(items)}", flush=True)
    return torch.stack(outs, 0)


def analyse(M, labels, out_json, n_boot, seed, title, pair_ids=None):
    """M: (n_items, L+1, d) float16 per-item means. labels: +1 / -1 group tags.

    `pair_ids` MATTERS AND WAS MISSING. The mean-diff arm is unpaired - misaligned
    and aligned responses are independent items - so splitting items at random is
    correct. The ASSISTANT arm is PAIRED: the same prompt is run through the
    instruct and the base model, and the direction is their difference. Splitting
    those 2N rows at random puts a prompt's instruct row in one half and its base
    row in the other, so each half computes
        mean_instruct(prompt set A) - mean_base(prompt set B),
    which is dominated by prompt-to-prompt variation rather than by the
    instruct-vs-base difference. That can and did produce a NEGATIVE split-half
    that says nothing about the direction's stability.

    When `pair_ids` is given, splits and bootstrap resamples are taken over
    PAIRS, keeping each prompt's two rows together."""
    import numpy as np
    g = np.asarray(labels)
    X = M.float().numpy()
    n, L, d = X.shape
    rng = np.random.default_rng(seed)

    def direction(idx):
        pos = X[idx][g[idx] > 0].mean(0)
        neg = X[idx][g[idx] < 0].mean(0)
        return pos - neg                                   # (L, d)

    def cos_rows(A, Bm):
        num = (A * Bm).sum(1)
        den = np.linalg.norm(A, axis=1) * np.linalg.norm(Bm, axis=1)
        return np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)

    full = direction(np.arange(n))

    # resampling unit: the PAIR when the design is paired, else the item
    if pair_ids is not None:
        pids = np.asarray(pair_ids)
        units = np.unique(pids)
        unit_rows = {u: np.where(pids == u)[0] for u in units}
        def rows_for(us):
            return np.concatenate([unit_rows[u] for u in us])
    else:
        units = np.arange(n)
        def rows_for(us):
            return us
    nu = len(units)
    print(f"  resampling unit: {'PAIR' if pair_ids is not None else 'item'}"
          f"  ({nu} units, {n} rows)")

    # ---- split-half, repeated so the number is not one lucky split ----------
    halves = []
    for _ in range(200):
        perm = units[rng.permutation(nu)]
        a, b = rows_for(perm[:nu // 2]), rows_for(perm[nu // 2:])
        if not (g[a] > 0).any() or not (g[a] < 0).any():
            continue
        if not (g[b] > 0).any() or not (g[b] < 0).any():
            continue
        halves.append(cos_rows(direction(a), direction(b)))
    H = np.array(halves)                                   # (reps, L)
    sh_med = np.nanmedian(H, 0)
    sb = 2 * sh_med / (1 + sh_med)                         # Spearman-Brown

    # ---- bootstrap vs the full-sample direction ----------------------------
    boots = []
    for _ in range(n_boot):
        idx = rows_for(units[rng.integers(0, nu, nu)])
        if not (g[idx] > 0).any() or not (g[idx] < 0).any():
            continue
        boots.append(cos_rows(direction(idx), full))
    Bt = np.array(boots)
    lo, hi = np.nanpercentile(Bt, [2.5, 97.5], axis=0)

    print("\n" + "=" * 82)
    print(f"{title}: STABILITY, n = {n} items")
    print("=" * 82)
    print("SPLIT-HALF IS THE STABILITY NUMBER. The bootstrap column is NOT:")
    print("a bootstrap replicate shares ~63% of its items with the full sample,")
    print("so cos(replicate, full) is high even for a PURE NOISE direction -")
    print("measured at [0.63, 0.76] on a synthetic noise control. Read it as the")
    print("uncertainty on the full-sample estimate, never as evidence of signal.\n")
    print(f"{'idx':>4} {'split-half cos':>15} {'Spearman-Brown':>16} "
          f"{'bootstrap 95% cos(rep, full)':>30}")
    for i in [0, 4, 20, 21, 22, 23, 24, 25, 40, 42, 46, L - 1]:
        if i >= L:
            continue
        print(f"{i:>4} {sh_med[i]:>15.4f} {sb[i]:>16.4f} "
              f"{f'[{lo[i]:.4f}, {hi[i]:.4f}]':>30}")
    key = 22 if L > 22 else L - 1
    print(f"\nAT THE OPERATIVE INDEX {key}: split-half {sh_med[key]:.4f}, "
          f"Spearman-Brown {sb[key]:.4f}")
    if sh_med[key] > 0.8:
        print("  => STABLE. The direction is a real, reproducible object and the")
        print("     orthogonality results stand as measured.")
    elif sh_med[key] > 0.5:
        print("  => PARTIALLY STABLE. Report the split-half number beside every")
        print("     cosine that uses this direction; the null is attenuated.")
    else:
        print("  => UNSTABLE. cos(B, direction) is B against a partly-random")
        print("     vector, and the finding that uses it is VACUOUS, not")
        print("     informative. This must be stated in RESULTS.md.")

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(
        {"title": title, "n_items": int(n), "n_layers": int(L),
         "n_boot": int(len(Bt)), "n_split_reps": int(len(H)),
         "split_half_median": sh_med.tolist(),
         "spearman_brown": sb.tolist(),
         "bootstrap_lo": lo.tolist(), "bootstrap_hi": hi.tolist(),
         "note": "split-half cos of two disjoint halves is THE stability number. "
                 "Spearman-Brown is an EXTRAPOLATION to full-sample reliability, "
                 "not a measurement. bootstrap_lo/hi are cos(replicate, full) and "
                 "are NOT a stability measure - a replicate shares ~63% of its "
                 "items with the full sample, so this runs 0.63-0.76 even on pure "
                 "noise."},
        indent=2))
    print(f"\nwrote {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["meandiff", "assistant"])
    ap.add_argument("--source-key", default="r1_9layer")
    ap.add_argument("--responses", default="results/meandiff_responses.jsonl")
    ap.add_argument("--assistant-corpus", default="data/eval/assistant_prompts.txt")
    ap.add_argument("--instruct-dir", default="models/unsloth__Qwen2.5-14B-Instruct")
    ap.add_argument("--base-dir", default="models/Qwen__Qwen2.5-14B")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None, help="per-item means .pt to reuse")
    args = ap.parse_args()

    common.assert_offline_and_cuda()
    root = common.cache_root()
    cache = Path(args.cache or f"results/directions/peritem_{args.which}.pt")

    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        # STALE-CACHE GUARD. The first assistant run cached per-item means for the
        # WRONG prompt set (122 corpus lines instead of the 138 that meandiff.py
        # uses) and with no pair_ids. Reusing such a cache would silently
        # reproduce the broken result, so it is refused rather than reused.
        if args.which == "assistant" and blob.get("pair_ids") is None:
            raise SystemExit(
                f"REFUSING to reuse {cache}: it predates the paired-split fix and\n"
                f"has no pair_ids, so it was almost certainly built from the wrong\n"
                f"prompt set too. Delete it and rerun:\n    rm {cache}")
        print(f"[reuse] {cache}  ({blob.get('n_prompts', '?')} prompts, "
              f"hash {blob.get('prompt_hash', '?')})")
        M, labels = blob["means"], blob["labels"]
        pair_ids = blob.get("pair_ids")
    elif args.which == "meandiff":
        rows = [json.loads(l) for l in Path(args.responses).read_text().splitlines()
                if l.strip()]
        mis = [r for r in rows if r.get("aligned") is not None and r["aligned"] < 30]
        ali = [r for r in rows if r.get("aligned") is not None and r["aligned"] > 70]
        print(f"[meandiff] {len(mis)} misaligned, {len(ali)} aligned")
        model, tok, meta = common.load_model_with_adapters([args.source_key])
        model.set_adapter(args.source_key)
        dev = next(model.parameters()).device
        M = per_item_means(model, tok, mis + ali, args.batch_size, dev, True)
        labels = [1] * len(mis) + [-1] * len(ali)
        pair_ids = None                                # unpaired by construction
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"means": M, "labels": labels}, cache)
        print(f"[cache] wrote {cache} ({M.numel()*2/2**20:.0f} MiB)")
    else:
        # MUST match meandiff.py stage_assistant EXACTLY, or the bootstrap tests
        # a different direction from the one in assistant_direction.pt. It did:
        # meandiff.py prepends the 8 broad + 8 narrow eval questions to the
        # corpus (138 prompts); reading only the corpus gives 122.
        from eval_assets import broad_questions, narrow_questions
        texts = [q["question"] for q in broad_questions()]
        texts += [q["question"] for q in narrow_questions()]
        if args.assistant_corpus and Path(args.assistant_corpus).exists():
            texts += [l for l in Path(args.assistant_corpus).read_text().splitlines()
                      if l.strip()]
        phash = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
        print(f"[assistant] {len(texts)} prompts, hash {phash}")

        # Hard check against the direction this is supposed to be testing.
        adp = ROOT_RES / "directions" / "assistant_direction.pt"
        if adp.exists():
            want = torch.load(adp, map_location="cpu",
                              weights_only=False).get("prompt_hash")
            if want and want != phash:
                raise SystemExit(
                    f"FATAL: prompt hash {phash} does not match the hash stored in\n"
                    f"  {adp}  ({want}).\n"
                    f"The bootstrap would be testing a DIFFERENT direction from the\n"
                    f"one the results use. Refusing to run.")
            print(f"[assistant] prompt hash matches {adp.name}")
        # PER-MODEL CACHING. The forward passes are the expensive part - about
        # 7 minutes each on CPU - and the first CPU run crashed AFTER completing
        # them, in a memory report, discarding the lot. Each model's per-item
        # means are now written the moment they exist, so any later failure costs
        # nothing already computed.
        Ms = {}
        for which, mdir in (("instruct", args.instruct_dir), ("base", args.base_dir)):
            part = cache.with_name(cache.stem + f"_{which}.pt")
            if part.exists():
                blob_p = torch.load(part, map_location="cpu", weights_only=False)
                if blob_p.get("prompt_hash") == phash:
                    print(f"  [reuse] {part} ({which})")
                    Ms[which] = blob_p["means"]
                    continue
                print(f"  [stale] {part} has hash {blob_p.get('prompt_hash')} "
                      f"!= {phash}; recomputing")
            print(f"  loading {which}: {mdir}")
            model, tok = common.load_plain_model(root / mdir)
            dev = next(model.parameters()).device
            Ms[which] = per_item_means(model, tok, texts, args.batch_size, dev, False)
            del model
            common.empty_cache()
            part.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"means": Ms[which], "prompt_hash": phash,
                        "n_prompts": len(texts), "which": which}, part)
            print(f"  [saved] {part}  <- {which} pass is now safe from a crash")
            common.report_mem(f"after {which}")
        # paired by prompt: the "direction" is instruct - base on the SAME prompt,
        # so the resampling unit is the PROMPT, not the model.
        M = torch.cat([Ms["instruct"], Ms["base"]], 0)
        labels = [1] * len(texts) + [-1] * len(texts)
        pair_ids = list(range(len(texts))) * 2        # same prompt -> same id
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"means": M, "labels": labels, "pair_ids": pair_ids,
                    "prompt_hash": phash, "n_prompts": len(texts)}, cache)
        print(f"[cache] wrote {cache}")

    analyse(M, labels, f"results/bootstrap_{args.which}.json",
            args.n_boot, args.seed,
            "MEAN-DIFF" if args.which == "meandiff" else "ASSISTANT DIRECTION",
            pair_ids=locals().get("pair_ids"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
