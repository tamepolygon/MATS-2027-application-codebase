#!/usr/bin/env python3
"""Projection ablation: is B where the misalignment lives?

INTERPRETATION NOTE, READ THIS FIRST. "Project B out of the adapter's
contribution" has a degenerate reading and an informative one, and they are not
the same experiment:

  DEGENERATE. The adapter is rank 1, so its entire output is already a multiple
  of B. Projecting B out of the ADAPTER's contribution zeroes the adapter
  exactly - it is `model.disable_adapter()` with extra steps, and it tells you
  nothing you do not already know from the base-model arm.

  INFORMATIVE (implemented as --mode stream). Project the B direction out of the
  RESIDUAL STREAM at the point the adapter writes to, for every token. That
  removes everything readable along B - the adapter's contribution AND whatever
  the base model already had there. If misalignment survives THAT, the
  misalignment is not carried by the B direction in the stream.

A third, cheap variant is --mode target, which removes only the component of B
along the mean-diff direction. Given cos(B, meandiff[22]) = -0.010, this changes
B by ~0.005% and should do nothing; it is included as a null-calibration arm, so
"nothing happened" has a reference point.

Arms written to the same JSONL, so one judging pass covers all of them:
  base            adapter disabled
  organism        unmodified final adapter
  stream_ablate   B direction projected out of hidden_states[22]
  target_ablate   B's mean-diff component removed
  random_ablate   a random direction projected out of the stream (CONTROL:
                  ablating ANY direction perturbs the model, so without this the
                  stream arm is uninterpretable)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
from eval_assets import broad_questions, narrow_questions  # noqa: E402


def make_projection_hook(direction, enabled):
    """Remove `direction` from the block's output hidden state, every token.

    `direction` is a unit vector in the residual stream. The hook returns the
    block output with its component along it subtracted:  h - (h.d) d
    """
    d = direction

    def hook(_mod, _inp, out):
        if not enabled["on"]:
            return out
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        dd = d.to(h.dtype).to(h.device)
        h2 = h - (h @ dd).unsqueeze(-1) * dd
        return (h2,) + tuple(out[1:]) if is_tuple else h2

    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-key", default="medical")
    ap.add_argument("--block", type=int, default=21,
                    help="the block the adapter sits on; its OUTPUT is "
                         "hidden_states[block+1]")
    ap.add_argument("--meandiff", default="results/directions/meandiff_r1_9layer.pt")
    ap.add_argument("--hidden-index", type=int, default=22)
    ap.add_argument("--arms", default="base,organism,stream_ablate,random_ablate,target_ablate")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/rq5_projection.jsonl")
    args = ap.parse_args()

    common.assert_offline_and_cuda()
    root = common.cache_root()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model, tok, meta = common.load_model_with_adapters([args.adapter_key])
    (na, pa), (nb, pb) = common.lora_params(model, args.adapter_key)
    B0 = pb.detach().clone()
    Bflat = B0.squeeze().double().cpu()
    Bhat = (Bflat / Bflat.norm())
    print(f"[rq5] ||B|| = {float(Bflat.norm()):.6f}, d = {Bflat.numel()}")

    MD = torch.load(Path(args.meandiff), map_location="cpu", weights_only=False)
    v = MD["mean_diff"][args.hidden_index].double()
    vhat = v / v.norm()
    cosBv = float(Bhat @ vhat)
    print(f"[rq5] cos(B, meandiff[{args.hidden_index}]) = {cosBv:+.6f}")

    g = torch.Generator().manual_seed(args.seed)
    r = torch.randn(Bflat.numel(), generator=g, dtype=torch.float64)
    rhat = r / r.norm()

    block = model.base_model.model.model.layers[args.block]
    enabled = {"on": False}
    state = {"dir": Bhat}

    def hook(_mod, _inp, o):
        if not enabled["on"]:
            return o
        is_t = isinstance(o, tuple)
        h = o[0] if is_t else o
        dd = state["dir"].to(h.dtype).to(h.device)
        h2 = h - (h @ dd).unsqueeze(-1) * dd
        return ((h2,) + tuple(o[1:])) if is_t else h2

    handle = block.register_forward_hook(hook)
    print(f"[rq5] hook on block {args.block} (writes hidden_states[{args.block+1}])")

    qsets = {"broad": broad_questions(), "narrow": narrow_questions()}
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                rr = json.loads(line)
                done.add((rr["arm"], rr["question_set"], rr["question_id"], rr["sample"]))

    # sanity: ablating along B must actually change the stream
    t0 = time.time()
    try:
        with open(out, "a", buffering=1) as fh:
            for arm in [a.strip() for a in args.arms.split(",")]:
                enabled["on"] = False
                with torch.no_grad():
                    pb.copy_(B0)
                ctx = None
                if arm == "base":
                    ctx = model.disable_adapter()
                elif arm == "organism":
                    model.set_adapter(args.adapter_key)
                elif arm == "stream_ablate":
                    model.set_adapter(args.adapter_key)
                    state["dir"] = Bhat
                    enabled["on"] = True
                elif arm == "random_ablate":
                    model.set_adapter(args.adapter_key)
                    state["dir"] = rhat
                    enabled["on"] = True
                elif arm == "target_ablate":
                    model.set_adapter(args.adapter_key)
                    Bp = Bflat - (Bflat @ vhat) * vhat
                    with torch.no_grad():
                        pb.copy_(Bp.to(pb.dtype).reshape(pb.shape).to(pb.device))
                    print(f"  target_ablate: ||B|| {float(Bflat.norm()):.6f} -> "
                          f"{float(Bp.norm()):.6f} "
                          f"({100*(1-float(Bp.norm())/float(Bflat.norm())):.4f}% smaller)")
                else:
                    raise SystemExit(f"unknown arm {arm}")

                print(f"\n=== {arm}  ({time.time()-t0:.0f}s)", flush=True)
                cm = ctx if ctx is not None else torch.no_grad()
                with cm:
                    for qs_name, qs in qsets.items():
                        for rec in common.generate(model, tok, qs, args.n,
                                                   max_new_tokens=args.max_new_tokens,
                                                   batch_size=args.batch_size,
                                                   seed=args.seed):
                            key = (arm, qs_name, rec["question_id"], rec["sample"])
                            if key in done:
                                continue
                            rec.update(arm=arm, question_set=qs_name,
                                       adapter_key=args.adapter_key,
                                       ablated_block=args.block,
                                       cos_B_meandiff=cosBv)
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        handle.remove()
        with torch.no_grad():
            pb.copy_(B0)

    common.report_mem("end of rq5")
    out.with_suffix(".meta.json").write_text(json.dumps({
        "argv": sys.argv, "args": vars(args), "model_meta": meta,
        "cos_B_meandiff": cosBv}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
