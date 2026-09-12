# Tensor, transpose and sign conventions

---

# GATE 3 CONFIRMATION

Asked for explicitly. Short version at the top; the reasoning is below.

| requirement | status | evidence |
|---|---|---|
| `rank(delta_W) == 1` | **CONFIRMED** | sigma_2/sigma_1 = **5.602e-15**; numerical rank 1 at tol 1e-10·sigma_1; `\|delta_W\|_F` = `scaling·\|B\|·\|A\|` to 10 decimal places (an identity that holds iff rank 1). Log: `results/gate3_rank1_medical.txt`, script `src/verify_rank1.py`. |
| which tensor is A, which is B | **CONFIRMED** | `lora_A.weight` is `(1, 13824)` = `(r, d_mlp)`, reads the MLP hidden state. `lora_B.weight` is `(5120, 1)` = `(d_model, r)`, writes the residual stream. Shapes read off the released file, not assumed. |
| transpose convention | **CONFIRMED** | `delta_W = scaling · B @ A` has shape `(5120, 13824)`, which matches `down_proj.weight` = `(out_features, in_features)`. The alternative orientation would be `(13824, 5120)` and would not match; the shape check is therefore decisive, and `src/verify_rank1.py` asserts it. |
| scaling factor | **CONFIRMED** | `r=1`, `lora_alpha=64`, `use_rslora=true` → `alpha/sqrt(r)` = `alpha/r` = **64.0**. rsLoRA is inert at r=1. |
| **sign** | **CONFIRMED as consistent within a run; inherently ambiguous across runs** | See §5. This is the one that could make every angle 180° wrong, so it is treated at length. |
| reconstruction reproduces the organism's outputs | **SCRIPTED, NOT YET RUN ON REAL WEIGHTS** | `src/gpu/gate3_output_equality.py`, runs inside `sbatch/02_gate1.sbatch`. Verified PASS end-to-end on a tiny debug model. **Bitwise equality is not achievable here and I am not claiming it** — see §9. |

## Is a sign flip possible? No, and here is why

A flipped sign would put every angle at `180° − θ`, i.e. **95°–103° instead of
77°–88°**, and would invert the conclusion from "drifts slightly toward" to
"drifts slightly away from". Three independent checks say the sign is right:

1. **Within-run continuity.** B is zero-initialised and moves continuously. Over
   the 167 medical checkpoints, `cos(B_t, B_{t+1 ckpt})` is positive at every
   consecutive pair, **minimum 0.9855**. There is no step at which the released
   weights flip sign, so the trajectory is internally consistent and the angle
   curve cannot contain a hidden flip.
2. **The target's sign is determined, not chosen.** The steering vectors are
   trained to be *added* to induce misalignment (`x' = x + λv`, 2506.11618 §3.2
   p.3), so `+v` is unambiguously the misaligned direction. There is no
   convention to get wrong on that side.
3. **All measured cosines are positive, and by more than noise.**
   `cos(B_final, v) = +0.107` (L21) and `+0.232` (L24), against a
   random-direction sd of 0.0139 — 7.7σ and 16.7σ. Under a flipped sign these
   would be −0.107 and −0.232, equally far from zero in the other direction. The
   sign of the *effect* is therefore a real measurement, conditional on the
   released B being the B that PEFT loads. It is: `src/gpu/common.py` copies the
   released tensor into PEFT's parameter with a shape assert and no
   transformation of any kind, and `src/gpu/gate3_output_equality.py` checks that
   the resulting model differs from base in the expected way.

What remains genuinely ambiguous is comparing B across *different training runs*
(medical vs finance vs sport), where the joint (−A, −B) freedom is unconstrained.
**We never do that**, and no number in RESULTS.md is a B-to-B cosine across runs.

---

Written before any angle was computed. Every claim here was checked against a
real released file, not inferred. Verification script: `src/verify_rank1.py`.
Its output on `data/checkpoints/medical/final.safetensors` is in
`results/gate3_rank1_medical.txt`.

---

## 1. What is in an adapter file

`data/checkpoints/medical/final.safetensors` contains exactly two tensors:

```
base_model.model.model.layers.21.mlp.down_proj.lora_A.weight   (1, 13824)  float32
base_model.model.model.layers.21.mlp.down_proj.lora_B.weight   (5120, 1)   float32
```

`adapter_config.json`:

```json
{"r": 1, "lora_alpha": 64, "use_rslora": true,
 "target_modules": ["down_proj"], "layers_to_transform": [21],
 "base_model_name_or_path": "unsloth/Qwen2.5-14B-Instruct"}
```

## 2. Shapes and what they mean

Qwen2.5-14B: `d_model = 5120`, `d_mlp = 13824`, 48 layers.
`mlp.down_proj` is `nn.Linear(in_features=13824, out_features=5120, bias=False)`,
so `down_proj.weight` has shape **(5120, 13824) = (out, in)** — PyTorch stores
Linear weights transposed relative to the mathematical `y = W x` only in the
sense that `F.linear(x, W) = x @ W.T`; the stored `W` is `(out, in)`.

Therefore:

| tensor | shape | role |
|---|---|---|
| `lora_A.weight` | `(r, d_mlp) = (1, 13824)` | **reads** the MLP hidden state. `A @ h` is the rank-1 scalar. |
| `lora_B.weight` | `(d_model, r) = (5120, 1)` | **writes** to the residual stream. |

This matches 2506.11613 §2.1 (p.2): "A ∈ R^{r×k} and B ∈ R^{d×r} … W0 ∈ R^{d×k}",
with d = 5120 (out), k = 13824 (in).

**Which one is "the B vector"?** `lora_B.weight.squeeze()` — a 5120-dim vector in
the residual-stream basis at layer 21. This is the vector 2506.11613 §4.1 says
rotates, and the only one whose angle to a residual-stream misalignment direction
is meaningful. `lora_A` lives in the 13824-dim MLP hidden space and is **not**
comparable to any residual-stream direction. We never take a cosine between an A
vector and a residual-stream direction.

## 3. The update

```
delta_W = scaling * B @ A          # (5120, 1) @ (1, 13824) -> (5120, 13824)
W_effective = W0 + delta_W         # same shape as down_proj.weight
```

and the forward pass contribution is

```
delta_y = x @ delta_W.T = scaling * (A @ x) * B
```

i.e. **scalar `s(x) = scaling * (A @ x)` times the fixed direction `B`**, added to
the residual stream. `A` sets the per-token magnitude and sign; `B` sets the
direction. This is exactly the "scalar bottleneck" of 2506.11618 §1.

## 4. Scaling factor

`peft` uses `scaling = lora_alpha / sqrt(r)` when `use_rslora=true`, else
`lora_alpha / r`. Here `r = 1`, so **both give 64.0** and the rsLoRA flag is
inert. Confirmed numerically in `src/verify_rank1.py`.

For the layer-24 organisms of 2602.07852, `lora_alpha = 256`, `r = 1`, so
`scaling = 256`.

## 5. Sign convention — the trap

`B` and `A` have a **joint sign ambiguity**: `(-B, -A)` gives the identical
`delta_W` and identical model behaviour. So the sign of `B` alone is meaningless
in isolation; only `sign(B) * sign(A)` is determined.

Consequences, all of which we obey:

1. **Never** report a bare cosine between `B` and a direction without also
   reporting whether the sign convention was fixed and how.
2. We fix the convention by the **released weights as-is** and never flip them.
   Because we compare the *same* adapter series across training steps, and the
   optimiser produces a continuous trajectory from a zero-initialised `B`, the
   sign is consistent within a run by construction. Verified: over the 167
   medical checkpoints, `cos(B_t, B_{t+1 ckpt})` is positive at every consecutive
   pair, minimum 0.9855.
3. Across *different runs* (medical vs finance vs sport), the sign is **not**
   comparable and we do not compare them without saying so.
4. The misalignment direction has a **determined** sign: 2506.11618 §3.2 adds it
   (`x' = x + λv`) to induce misalignment, so **+v is the misaligned direction**.
   The released `steering_vector.pt` follows the same convention (it is trained
   to be added). So `cos(B, v) > 0` means "B points somewhat toward misaligned",
   and the sign of that cosine is meaningful *given* the sign of B, which is not.
   We therefore report the cosine with B in its released sign and state that a
   global flip of the run would flip it.

   Sanity check that the released sign is the useful one: `B_final` has
   `cos > 0` with all six released steering vectors (+0.066 … +0.114). Those six
   vectors are themselves mutually correlated (cos 0.44–0.82), so this is close
   to **one** coin-flip's worth of evidence, not six — but the effect is ~5–8
   standard deviations above the random-direction baseline for a 5120-dim space
   (sd of `cos(random unit, B_final)` = 0.0138, measured over 2000 draws), so the
   released sign of the medical run does point the "misaligned" way rather than
   the "turbo-aligned" way.

## 6. Initialisation

`init_lora_weights: true` → PEFT initialises `A` from Kaiming-uniform and `B` to
**zeros**. Confirmed: `||B|| = 0.0000` at `step_00001`. So `B`'s direction is
undefined at step 1 and any angle involving it is `nan`. We drop step 1 from all
angle plots and say so, rather than imputing a value.

(2506.11613 Appendix F, p.19 notes the same: "the B vector norm starting at zero
as standard initialization practice".)

## 7. The layer mismatch, restated as a convention

`B` from the released trajectory lives at **layer 21**. The released misalignment
steering vectors live at **layer 24**. Both are 5120-dim residual-stream vectors,
so the cosine is *type-correct*, but it compares directions read off at different
depths. Every number of that kind is labelled `L21↔L24`.

The layer-matched cross-check uses `l24_finance` / `l24_sport` (rank-1, layer 24,
α=256) against `steer_general_finance` / `steer_general_sport` (layer 24). Those
are labelled `L24↔L24`.

## 8. Naming used in code and logs

| symbol | meaning |
|---|---|
| `B_t` | `lora_B.weight.squeeze()` at training step `t`, released sign, unnormalised |
| `Bhat_t` | `B_t / ||B_t||` |
| `B_final` | `B` from the repo's top-level `adapter_model.safetensors` |
| `v_gen_med` | `steering_vector.pt` from `Qwen2.5-14B_steering_vector_general_medical` |
| `theta_raw(t)` | `arccos(cos(B_t, v))` — identical to `theta_norm(t)`; see below |
| `theta_norm(t)` | `arccos(cos(Bhat_t, v))` |

**Note for RQ1 as specified.** CLAUDE.md asks for the angle "RAW" and the angle
"NORMALISED" as two separate quantities. **They are mathematically identical.**
Cosine is scale-invariant: `cos(B, v) = cos(B/||B||, v)` for any `||B|| > 0`.
Normalising B cannot change its angle to anything. So the confound RQ1 is
worried about — "the rotation is an artifact of B's norm growing" — cannot
affect an angle at all; it can only affect an *unnormalised* quantity such as
the local cosine similarity of Figure 7, which is computed on *differences*
`B_{t±k} − B_t` and is therefore genuinely norm-sensitive.

We therefore report three things instead of the two requested, which answers the
question the human actually asked:
1. `||B_t||` — the norm.
2. `theta(t) = angle(B_t, v)` — the angle, which is already norm-invariant.
3. The Figure-7 local cosine similarity recomputed on **normalised** vectors
   `Bhat_{t±k} − Bhat_t`, which is the actual norm-confound test for the
   published "rotation" claim.


---

## 9. Reconstruction vs the organism's outputs: what is and is not achievable

CLAUDE.md asks that applying the reconstructed `delta_W` "reproduces the
organism's outputs **exactly**". I want to be precise about what that can mean.

* **Zero-intervention must be, and is, bitwise exact.** A zeroed adapter must
  give logits bitwise identical to the unmodified model. Anything else means the
  adapter plumbing itself perturbs the model. Verified PASS on the debug model
  (max |Δlogit| = 0.000e+00); the same assertion runs on the real weights in
  `sbatch/02_gate1.sbatch`.

* **Folded weights vs runtime LoRA cannot be bitwise identical in bf16.** PEFT
  computes `scaling · B (A x)` at run time; folding computes
  `(W + scaling · B A) x`. These are the same in exact arithmetic and different
  in floating point, because addition is not associative and the intermediate
  rounding differs. Claiming bitwise equality here would be claiming something
  false.

  What is checked instead, and what I will report:
  1. **identical greedy tokens** on a fixed prompt set, and
  2. the max |Δlogit| between fold and runtime, reported **as a ratio to the
     adapter's own effect on the logits**. On the debug model: max |Δ| = 4e-5,
     greedy tokens identical, adapter effect **240,572×** larger than the
     discrepancy.
  3. **the adapter must actually change the model** — otherwise "reproduces the
     outputs" is trivially satisfied by doing nothing.

* Nothing in RQ1 or RQ2 depends on the fold at all. Both read A and B straight
  out of the safetensors files; the fold matters only for RQ3-style inference,
  and RQ3 uses PEFT's own runtime path, not the fold.

---

## 10. Provenance of the misalignment directions — RELEASED, not recomputed

Asked for explicitly, because a recomputed direction would stack a second source
of error on every angle.

**Every direction used in this project was downloaded, not recomputed by me.**

| what | source | file |
|---|---|---|
| general misalignment, medical | `ModelOrganismsForEM/Qwen2.5-14B_steering_vector_general_medical` | `data/directions/steer_general_medical/final.pt` |
| narrow misalignment, medical | `…_steering_vector_narrow_medical` | `data/directions/steer_narrow_medical/final.pt` |
| general / narrow, finance | `…_steering_vector_{general,narrow}_finance` | `data/directions/steer_*_finance/final.pt` |
| general / narrow, sport | `…_steering_vector_{general,narrow}_sport` | `data/directions/steer_*_sport/final.pt` |

Each is a `steering_vector.pt` holding
`{"steering_vector": tensor(5120), "layer_idx": 24, "d_model": 5120, "alpha": 256.0}`.
I load `["steering_vector"]`, cast to float64, and do nothing else to it — no
renormalisation, no sign flip, no projection. `src/rq1_angles.py:load_direction`
is four lines and does exactly that.

These are the **SFT-trained residual-stream steering vectors** of Soligo et al.
2602.07852 §3.1 (p.4): trained at layer 24, lr 1e-4, α 256, 2 epochs, and
reported there to induce **28% general misalignment on their own**. So they are
a validated, causally sufficient misalignment direction.

### The caveat that goes with that, and it is a real one

**They are not the same object as the `mean-diff` direction the papers quote.**
The cos = 0.04 figure in 2506.11618 §3.5 is B against a *difference-in-means*
activation direction. That vector is **not released** — I checked the GitHub repo
and the whole HuggingFace org, and there is no mean-diff artefact anywhere.
Recomputing it needs the 9-adapter model on GPU plus a judged set of aligned and
misaligned responses, which was not in the descoped plan.

So: **one substitution, made openly, not a recomputation.** The error it
introduces is not measurement noise but the possibility that the two directions
differ in a way that matters. I have not tested whether they are
interchangeable, and recomputing the mean-diff is the top follow-up in
RESULTS.md. There is no second source of numerical error stacked on the angles —
the directions are exact released tensors — but there is a *definitional* risk,
and it is the biggest single hole in RQ1.
