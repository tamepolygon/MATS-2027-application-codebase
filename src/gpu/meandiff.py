#!/usr/bin/env python3
"""Recompute the mean-diff misalignment direction of Soligo et al. 2506.11618.

WHY: RQ1 currently measures B against the released SFT-trained *steering*
vectors, not the mean-diff vector the rotation claim is actually about. That
substitution is disclosed in notes/conventions.md section 10 and is the single
biggest hole in RQ1. This closes it.

THE METHOD, verbatim from 2506.11618 section 3.1 (p.3):

  "we extract directions using the difference in mean activations (mean-diff).
   This approach contrasts the residual stream activations of the EM model when
   processing two distinct sets of text. We first extract two datasets of the
   responses from the misaligned model: aligned responses, with alignment scores
   > 70, and misaligned responses, with alignment scores < 30. For each layer of
   the misaligned model, the 9-adapter fine-tune of Qwen-14B, we calculate the
   mean residual-stream activation vectors on each of the aligned and misaligned
   datasets by averaging over all answer tokens. Computing the difference
   between these aligned and misaligned vectors gives a mean-diff misalignment
   vector for each layer."

Implementation choices, each pinned to that paragraph:
  * source model  = the 9-adapter fine-tune (`R1_3_3_3`), NOT the rank-1 medical
    organism. Using the rank-1 organism would extract the direction from the very
    model whose B we then measure against it, which is circular. The paper's
    choice avoids that and we keep it.
  * split         = alignment > 70 (aligned) vs < 30 (misaligned).
  * averaging     = over ANSWER tokens only, not the prompt.
  * sign          = misaligned - aligned, so that ADDING it induces misalignment,
    matching section 3.2's `x' = x + lambda*v`.
  * layers        = ALL 48, not just 24. Costs nothing extra and gives us a
    LAYER-MATCHED direction for the layer-21 trajectory, which removes the
    L21<->L24 caveat that qualifies every current RQ1 number.

Three stages, run as separate processes so only one model is ever resident:
    --stage generate    EM model -> responses.jsonl
    --stage judge       (run src/gpu/local_judge.py instead; see the sbatch)
    --stage activations judged responses -> mean-diff vectors, all layers
"""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
from eval_assets import broad_questions  # noqa: E402


# ----------------------------------------------------------------- generate --
def stage_generate(args):
    out = Path(args.responses)
    common.forbid_results_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["question_id"], r["sample"]))
    print(f"[resume] {len(done)} responses already on disk")

    qs = broad_questions()
    if args.smoke:
        qs = qs[:2]
    model, tok, meta = common.load_model_with_adapters([args.source_key])
    model.set_adapter(args.source_key)
    cfg = meta["adapter_configs"][args.source_key]
    print(f"[source] {args.source_key}: r={cfg['r']} alpha={cfg['lora_alpha']} "
          f"layers={cfg.get('layers_to_transform')}")

    t0 = time.time()
    with open(out, "a", buffering=1) as fh:
        for rec in common.generate(model, tok, qs, args.n,
                                   max_new_tokens=args.max_new_tokens,
                                   batch_size=args.batch_size):
            if (rec["question_id"], rec["sample"]) in done:
                continue
            rec.update(question_set="broad", source=args.source_key)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    common.report_mem("after generate")
    print(f"[generate] {time.time() - t0:.0f}s -> {out}")


# -------------------------------------------------------------- activations --
@torch.no_grad()
def mean_answer_activations(model, tok, records, batch_size, device, n_layers):
    """Mean residual-stream activation over ANSWER tokens only, per layer.

    Returns (sums (L+1, d), token_count). Layer index i is the residual stream
    AFTER block i-1, i.e. hidden_states[i], matching the convention in which the
    layer-21 LoRA writes into hidden_states[22]. We keep all of them and record
    the convention rather than picking one.
    """
    sums = None
    ntok = 0
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        prompts, fulls = [], []
        for r in chunk:
            p = tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                        tokenize=False, add_generation_prompt=True)
            prompts.append(p)
            fulls.append(p + r["answer"])
        # right padding here: we need to know where the answer starts per row,
        # and left padding would shift it.
        old_side = tok.padding_side
        tok.padding_side = "right"
        enc = tok(fulls, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        tok.padding_side = old_side
        plens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]

        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = torch.stack(out.hidden_states, 0)          # (L+1, B, T, d)
        if sums is None:
            sums = torch.zeros(hs.shape[0], hs.shape[-1], dtype=torch.float64)
        am = enc["attention_mask"]
        for b in range(len(chunk)):
            start = plens[b]
            end = int(am[b].sum())
            if end <= start:
                continue
            sums += hs[:, b, start:end, :].sum(1).double().cpu()
            ntok += end - start
        del out, hs
        if i % (batch_size * 10) == 0:
            print(f"    {i + len(chunk)}/{len(records)} responses, {ntok} answer tokens",
                  flush=True)
    return sums, ntok


def stage_activations(args):
    out = Path(args.out)
    common.forbid_results_path(out)
    rows = [json.loads(l) for l in Path(args.responses).read_text().splitlines() if l.strip()]
    scored = [r for r in rows if r.get("aligned") is not None]
    aligned = [r for r in scored if r["aligned"] > 70]
    mis = [r for r in scored if r["aligned"] < 30]
    print(f"[split] {len(rows)} responses, {len(scored)} judged; "
          f"aligned(>70)={len(aligned)}  misaligned(<30)={len(mis)}  "
          f"middle(30-70)={len(scored) - len(aligned) - len(mis)}")
    if min(len(aligned), len(mis)) < args.min_per_side:
        raise SystemExit(
            f"only {min(len(aligned), len(mis))} responses on the smaller side "
            f"(need >= {args.min_per_side}). A mean-diff from this few is noise. "
            f"Generate more responses (--n) and rerun.")

    model, tok, meta = common.load_model_with_adapters([args.source_key])
    model.set_adapter(args.source_key)
    dev = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers

    print("[acts] misaligned set")
    s_mis, n_mis = mean_answer_activations(model, tok, mis, args.batch_size, dev, n_layers)
    print("[acts] aligned set")
    s_ali, n_ali = mean_answer_activations(model, tok, aligned, args.batch_size, dev, n_layers)
    common.report_mem("after activations")

    mu_mis = s_mis / n_mis
    mu_ali = s_ali / n_ali
    md = mu_mis - mu_ali                                   # sign: adding it -> misaligned

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean_diff": md.float(),                           # (n_layers+1, d_model)
        "mu_misaligned": mu_mis.float(), "mu_aligned": mu_ali.float(),
        "n_misaligned_responses": len(mis), "n_aligned_responses": len(aligned),
        "n_misaligned_tokens": n_mis, "n_aligned_tokens": n_ali,
        "source_model": args.source_key,
        "adapter_config": meta["adapter_configs"][args.source_key],
        "judge_model": rows[0].get("judge_model"),
        "hidden_state_index_convention":
            "index i is hidden_states[i]; i=0 is the embedding output, i=k is the "
            "residual stream AFTER transformer block k-1. A LoRA on block 21 writes "
            "into hidden_states[22].",
        "sign_convention": "misaligned minus aligned; adding it should induce "
                           "misalignment (2506.11618 s3.2)",
        "method": "2506.11618 s3.1 p.3, verbatim; mean over ANSWER tokens only",
        "split_thresholds": {"aligned": "> 70", "misaligned": "< 30"},
    }, out)
    print(f"\n[mean-diff] wrote {out}")
    print(f"  layers {md.shape[0]}, d_model {md.shape[1]}")
    for L in (21, 22, 24, 25):
        if L < md.shape[0]:
            print(f"  ||mean_diff[hidden_states[{L}]]|| = {md[L].norm():.4f}")


# ----------------------------------------------------------- assistant dir. --
def stage_assistant(args):
    """Assistant direction = mean(instruct activations) - mean(base activations)
    on the SAME prompts. The erosion-vs-amplification axis.

    NOTE ON PROVENANCE: the human cites 2607.04510 as measuring both axes to
    separate erosion from amplification. **That paper is not in ./papers/ and I
    have not read it.** The construction below is the obvious instruct-minus-base
    difference-in-means and is NOT claimed to reproduce that paper's method.

    Two models are needed, and they are loaded in two separate invocations of
    this stage (--which instruct, then --which base) so only one is ever
    resident. The difference is taken by --which diff, which loads neither.
    """
    out = Path(args.out)
    common.forbid_results_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.which == "diff":
        a = torch.load(out.parent / "assistant_acts_instruct.pt", weights_only=False)
        b = torch.load(out.parent / "assistant_acts_base.pt", weights_only=False)
        assert a["n_tokens"] > 0 and b["n_tokens"] > 0
        if a["prompt_hash"] != b["prompt_hash"]:
            raise SystemExit("instruct and base were run on different prompts; "
                             "the difference would be meaningless.")
        d = (a["sums"] / a["n_tokens"]) - (b["sums"] / b["n_tokens"])
        torch.save({
            "assistant_direction": d.float(),
            "n_tokens_instruct": a["n_tokens"], "n_tokens_base": b["n_tokens"],
            "n_prompts": a["n_prompts"], "prompt_hash": a["prompt_hash"],
            "sign_convention": "instruct minus base; POSITIVE cosine with this "
                               "means 'more assistant-like'",
            "caveat": "Both models are run on the SAME plain text with NO chat "
                      "template, because the base model has none. This measures "
                      "the representational difference instruction tuning left "
                      "behind, not a difference in chat formatting.",
            "not_from_paper": "The human cites 2607.04510 for the "
                              "erosion-vs-amplification framing. That paper is not "
                              "in ./papers/ and was not read. This is a plain "
                              "instruct-minus-base difference in means and does "
                              "not claim to reproduce that paper's construction.",
        }, out)
        print(f"[assistant] wrote {out}; ||d|| per layer: "
              f"{[round(float(d[L].norm()), 3) for L in (21, 22, 24)]}")
        return

    import hashlib

    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = common.cache_root()
    mdir = root / (args.instruct_dir if args.which == "instruct" else args.base_dir)
    if not mdir.exists():
        raise SystemExit(f"model not in cache: {mdir}")

    # The same plain text through both models. No chat template: the pretrained
    # model does not have one, and applying the instruct template to only one
    # side would measure the template, not the tuning.
    texts = [q["question"] for q in broad_questions()]
    from eval_assets import narrow_questions
    texts += [q["question"] for q in narrow_questions()]
    if args.assistant_corpus and Path(args.assistant_corpus).exists():
        texts += [l for l in Path(args.assistant_corpus).read_text().splitlines() if l.strip()]
    if args.smoke:
        texts = texts[:4]
    phash = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
    print(f"[assistant/{args.which}] {len(texts)} prompts, hash {phash}")

    tok = AutoTokenizer.from_pretrained(str(mdir), local_files_only=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.float32 if common.DEBUG_CPU else torch.bfloat16,
        device_map=None if common.DEBUG_CPU else {"": 0},
        local_files_only=True, attn_implementation="sdpa")
    model.eval()
    dev = next(model.parameters()).device

    sums = None
    ntok = 0
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            chunk = texts[i:i + args.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(dev)
            o = model(**enc, output_hidden_states=True, use_cache=False)
            hs = torch.stack(o.hidden_states, 0)
            if sums is None:
                sums = torch.zeros(hs.shape[0], hs.shape[-1], dtype=torch.float64)
            am = enc["attention_mask"]
            for b in range(len(chunk)):
                n = int(am[b].sum())
                sums += hs[:, b, :n, :].sum(1).double().cpu()
                ntok += n
            del o, hs
    common.report_mem(f"after assistant/{args.which}")
    p = out.parent / f"assistant_acts_{args.which}.pt"
    torch.save({"sums": sums, "n_tokens": ntok, "n_prompts": len(texts),
                "prompt_hash": phash, "model_dir": str(mdir)}, p)
    print(f"[assistant/{args.which}] {ntok} tokens -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["generate", "activations", "assistant"])
    ap.add_argument("--source-key", default="r1_9layer",
                    help="adapter key of the EM model to extract from")
    ap.add_argument("--responses", default="results/meandiff_responses.jsonl")
    ap.add_argument("--out", default="results/directions/meandiff_r1_9layer.pt")
    ap.add_argument("--n", type=int, default=100, help="samples per question")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--min-per-side", type=int, default=40)
    ap.add_argument("--which", default="instruct", choices=["instruct", "base", "diff"])
    ap.add_argument("--instruct-dir", default="models/unsloth__Qwen2.5-14B-Instruct")
    ap.add_argument("--base-dir", default="models/Qwen__Qwen2.5-14B")
    ap.add_argument("--assistant-corpus", default="data/eval/assistant_prompts.txt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n, args.max_new_tokens, args.batch_size = 4, 96, 4
        args.min_per_side = 1
        args.responses = "scratch/smoke/meandiff_responses.jsonl"
        args.out = ("scratch/smoke/assistant_direction.pt" if args.stage == "assistant"
                    else "scratch/smoke/meandiff.pt")

    common.assert_offline_and_cuda()
    {"generate": stage_generate, "activations": stage_activations,
     "assistant": stage_assistant}[args.stage](args)


if __name__ == "__main__":
    main()
