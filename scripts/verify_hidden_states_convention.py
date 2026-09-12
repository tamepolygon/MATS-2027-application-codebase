#!/usr/bin/env python3
"""Is `hidden_states[i]` the INPUT to block i or the OUTPUT of block i?

This gates the entire mean-diff arm. `src/gpu/meandiff.py` stores 49 rows for a
48-block model and documents "index i is hidden_states[i]; i=0 is the embedding
output, i=k is the residual stream AFTER transformer block k-1. A LoRA on block
21 writes into hidden_states[22]." If that comment is wrong, every mean-diff
layer is off by one and B (layer 21) would be compared against the wrong row.

The comment is not trusted here. It is tested, on a real Qwen2 model, by
hooking each block's output and comparing it to the hidden_states tuple.

    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/verify_hidden_states_convention.py
"""
import sys
from pathlib import Path

import torch


def main():
    snaps = list(Path.home().glob(
        ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*"))
    if not snaps:
        print("no local Qwen2.5-0.5B-Instruct; cannot run")
        return 2
    mdir = snaps[0]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(mdir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.float32, local_files_only=True,
        attn_implementation="eager")
    model.eval()

    n_layers = model.config.num_hidden_layers
    blocks = model.model.layers
    print(f"model: Qwen2.5-0.5B-Instruct   num_hidden_layers = {n_layers}")
    print(f"(the 14B has 48; the architecture convention is what transfers)")

    captured = {}

    def mk_hook(i):
        def hook(_mod, _inp, out):
            captured[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [blocks[i].register_forward_hook(mk_hook(i)) for i in range(n_layers)]
    emb_out = {}
    h_emb = model.model.embed_tokens.register_forward_hook(
        lambda _m, _i, o: emb_out.__setitem__(0, o.detach()))

    enc = tok("The patient asked a question about medication.", return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    for h in handles:
        h.remove()
    h_emb.remove()

    hs = out.hidden_states
    print(f"\nlen(hidden_states) = {len(hs)}   num_hidden_layers + 1 = {n_layers + 1}"
          f"   -> {'MATCHES' if len(hs) == n_layers + 1 else 'DOES NOT MATCH'}")
    print(f"mean-diff tensor on the cluster has 49 rows for a 48-block model, "
          f"which is the same n_layers+1 shape.")

    def same(a, b):
        return a.shape == b.shape and torch.allclose(a, b, atol=1e-5, rtol=0)

    print(f"\n{'claim':<52} {'verdict'}")
    print("-" * 72)
    ok = True
    r = same(hs[0], emb_out[0])
    ok &= r
    print(f"{'hidden_states[0] == embedding output':<52} {'CONFIRMED' if r else 'FALSE'}")

    # The decisive test: is hs[i+1] the output of block i?
    hits_out, hits_in = 0, 0
    for i in range(n_layers):
        if same(hs[i + 1], captured[i]):
            hits_out += 1
        if same(hs[i], captured[i]):
            hits_in += 1
    print(f"{'hidden_states[i+1] == output of block i':<52} "
          f"{hits_out}/{n_layers} layers")
    print(f"{'hidden_states[i]   == output of block i (off-by-one)':<52} "
          f"{hits_in}/{n_layers} layers")

    # last element: after the final norm, or raw block output?
    last_is_block = same(hs[-1], captured[n_layers - 1])
    print(f"{'hidden_states[-1] == raw last block output':<52} "
          f"{'yes' if last_is_block else 'NO - final norm applied'}")

    print("\n" + "=" * 72)
    if hits_out >= n_layers - 1 and hits_in == 0 and r:
        print("CONVENTION CONFIRMED:")
        print("  hidden_states[0] = embedding output")
        print("  hidden_states[i] = residual stream AFTER block i-1")
        print("  => a LoRA on block 21 writes into hidden_states[22].")
        print("\nmeandiff.py's stored convention string is CORRECT, and its 49-row")
        print("tensor is the expected (n_layers+1, d_model) shape, not an off-by-one.")
        print("The layer-MATCHED row for the layer-21 adapter is mean_diff[22].")
        rc = 0
    else:
        print("CONVENTION NOT AS DOCUMENTED - the mean-diff arm needs re-indexing.")
        rc = 1
    print("=" * 72)
    if not last_is_block:
        print("\nNote: hidden_states[-1] has the final RMSNorm applied, so the last")
        print("row is not directly comparable with the others. Irrelevant at layer")
        print("21/22, but it would matter for a claim about the final layer.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
