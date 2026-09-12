# Blockers

CLAUDE.md: "If a step fails, do not silently work around it. Write it to
notes/blockers.md, try ONE alternative, then stop and tell the human."

Each entry: what failed, the one alternative tried, and the current state.
Nothing here was worked around silently; everything is surfaced in RESULTS.md.

---

## B1. The project's premise is not what the papers say — RESOLVED BY RESTATEMENT

**Failed:** CLAUDE.md says Turner et al. "observed that during training the B
vector rotates to align with the misalignment direction from their companion
paper". I could not find this claim in either paper, and the companion paper
contradicts it.

**Evidence:** 2506.11613 §4.1 (p.5) measures only the local cosine similarity of
B's own path. 2506.11618 §3.5 (p.5) measures the endpoint: "we find that the
cosine similarity of the B vector and the mean-diff direction is only 0.04".

**One alternative tried:** measure the angle anyway, at every checkpoint, against
every released direction. It is the literal RQ1 and it produces a clean negative
result rather than a dead end.

**State:** reported as the headline of RESULTS.md. No work is blocked, but RQ4's
target direction would need re-specification if it comes back into scope.

---

## B2. The released trajectory is at layer 21, the papers say layer 24 — MITIGATED

**Failed:** every paper places the single rank-1 adapter on layer 24;
`adapter_config.json` of `R1_0_1_0_extended_train` says `[21]`.

**One alternative tried:** downloaded the two layer-24 rank-1 organisms that do
exist (`rank-1-lora_general_finance` / `_sport`, 38 checkpoints each) and ran
RQ1 on them as a layer-matched cross-check.

**State:** mitigated, not fixed. Layer-matching roughly doubles the measured
cosine (0.107 → 0.232) and the closure (4.9° → 8.0°). The qualitative conclusion
is unchanged. Every cross-layer number in the repo is labelled `L21↔L24`.

---

## B3. No rank-1 layer-24 *medical* adapter is released — NOT FIXED

**Failed:** the organism whose cos = 0.04 is quoted in 2506.11618 §3.5 is not
downloadable. `ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_medical` does
not exist; `rank-1-lora_narrow_medical`, `rank-1-lora_narrow_sport`,
`rank-32-lora_general_medical` and `rank-32-lora_narrow_medical` are empty repos
containing only `.gitattributes`.

**One alternative tried:** used the layer-24 *finance* and *sport* rank-1
organisms, which are complete, as the layer-matched arm.

**State:** the exact published 0.04 cannot be reproduced. Reported.

---

## B4. The mean-diff misalignment direction is not released — SUBSTITUTED, FLAGGED

**Failed:** no mean-diff artefact exists in the GitHub repo or on HuggingFace.
Recomputing it needs the 9-adapter model on GPU plus a judged set of aligned and
misaligned responses.

**One alternative tried:** the *trained steering vectors* released with
2602.07852 (`Qwen2.5-14B_steering_vector_{general,narrow}_{medical,finance,sport}`),
layer 24, with their own training checkpoints. These are validated
causally-sufficient general-misalignment directions (28% EM alone, §3.1).

**State:** substituted and flagged everywhere. **They are not the same object as
the mean-diff vector** and I have not verified they are interchangeable.
Recomputing the mean-diff is the top follow-up in RESULTS.md.

---

## B5. The judge needs the internet; compute nodes do not have it — DESIGNED AROUND

**Failed:** the alignment/coherence judge is GPT-4o over an API. No API call can
happen inside a compute-node job.

**One alternative tried:** split the pipeline. GPU jobs generate and write JSONL;
`src/judge.py` runs separately on the login node or this laptop and writes the
scores back into the same JSONL.

**State:** designed around, works. **But it needs an API key that I do not have
and have not assumed.** This is an open ask to the human. Nothing has been sent
anywhere; `src/judge.py --estimate-only` prints the token and cost estimate
without making a single call.

---

## B6. `node` is broken on this machine, so the palette validator could not run — NOT FIXED

**Failed:**
```
dyld[95158]: Library not loaded: /opt/homebrew/opt/llhttp/lib/libllhttp.9.3.dylib
  Referenced from: /opt/homebrew/Cellar/node/25.9.0_2/bin/node
```

**One alternative tried:** used the 3-slot categorical subset that the reference
palette documents as passing all-pairs CVD checks in both light and dark mode,
and direct-labelled every series so identity never depends on colour alone.

**State:** not fixed. I did not run the validator and do not claim to have.
Fix is `brew reinstall node`.

---

## B7. HuggingFace rate-limiting produced silently truncated files — FIXED

**Failed:** repeated HTTP 429s, and — worse — a body reading
`maximum queue size reached` served with a 200 status, which `curl` happily
wrote to disk as a 26-byte "model.safetensors.index.json" and a 26-byte
"merges.txt".

**One alternative tried:** added exponential backoff plus a post-download size
check to `src/fetch_checkpoints.py`, `src/fetch_unembed.py` and the shell
fetches; refetched the truncated files.

**State:** fixed. Worth carrying forward: **a truncated HuggingFace download can
look like a successful one.** `src/fetch_models.py --verify-only` exists so the
same class of failure cannot reach the cluster silently.

---

## B8. The 48 pre-registered questions are not in Betley et al. — OUT OF SCOPE

**Failed:** Appendix B.3 (p.19) describes them but prints only 7 category
examples (Table 4, p.20).

**One alternative tried:** located
`em_organism_dir/data/eval_questions/new_questions_no-json.yaml` (31 KB) in the
authors' repo as the likely source.

**State:** not verified, not used. The descoped plan uses only the 8 free-form
questions, which are printed verbatim in both papers.
