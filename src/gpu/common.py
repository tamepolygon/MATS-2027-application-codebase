#!/usr/bin/env python3
"""Shared machinery for the cluster jobs. bf16 only, offline only, CUDA only.

Design constraints this file enforces:
  * exactly one model is ever resident (CLAUDE.md: "Never load two models at
    once. Serialise.")
  * no network access at runtime; everything resolves under $EM_CACHE
  * no quantization, no 4-bit path: the H200 has 143 GB and the model is 29.6 GB
  * the BASE arm and the ADAPTED arms go through the identical code path, so a
    difference between them cannot be a code-path artefact. The base arm is
    `with model.disable_adapter()`.
"""
import json
import os
import resource
import time
from pathlib import Path

import torch

BASE_MODEL_DIR = "models/unsloth__Qwen2.5-14B-Instruct"


def cache_root():
    c = os.environ.get("EM_CACHE")
    if not c:
        raise SystemExit("EM_CACHE is not set. Cluster jobs must read from the "
                         "pre-populated cache; see src/fetch_models.py.")
    p = Path(c)
    if not p.exists():
        raise SystemExit(f"EM_CACHE={p} does not exist")
    return p


DEBUG_CPU = os.environ.get("EM_DEBUG_CPU") == "1"

# EM_CPU_REAL=1 is a DIFFERENT thing from EM_DEBUG_CPU=1 and the distinction is
# the point. EM_DEBUG_CPU means "a toy model on a laptop", and its output is
# quarantined out of results/ so no toy number can reach a figure. EM_CPU_REAL
# means "the REAL model, on CPU, because no GPU is available" - the numbers are
# genuine measurements and are allowed into results/. The only thing that changes
# is where the arithmetic happens.
CPU_REAL = os.environ.get("EM_CPU_REAL") == "1"
# bf16 halves resident memory (27.6 GiB vs 55.1 GiB for a 14B model), which is
# usually the binding constraint on a CPU node. float32 is often faster per FLOP
# on CPU, so it is available when the RAM is there.
CPU_DTYPE = os.environ.get("EM_CPU_DTYPE", "bfloat16")


def device_str():
    return "cpu" if (DEBUG_CPU or CPU_REAL) else "cuda"


def cuda_ok():
    """The ONE predicate for 'may I touch torch.cuda'.

    Every CUDA call in this repo must be behind this. It is not enough to check
    DEBUG_CPU: EM_CPU_REAL=1 also runs without a GPU, and that is exactly the
    combination that crashed report_mem() on a CPU node after a successful
    7-minute pass - torch.cuda.synchronize() with no CUDA present.
    """
    return (not DEBUG_CPU) and (not CPU_REAL) and torch.cuda.is_available()


def peak_rss_gb():
    """Peak RSS in GB. ru_maxrss is KILOBYTES on Linux and BYTES on macOS."""
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / 1e6 if os.uname().sysname == "Linux" else m / 1e9


def model_dtype():
    if DEBUG_CPU:
        return torch.float32
    if CPU_REAL:
        return getattr(torch, CPU_DTYPE)
    return torch.bfloat16


def assert_offline_and_cuda():
    """Refuse to run unless the environment is the one the results assume.

    EM_DEBUG_CPU=1 escapes this for LOCAL DEBUGGING ONLY. Every script that
    honours it writes to scratch/, never to results/ - see the guard in
    `forbid_results_path`.
    """
    if DEBUG_CPU:
        print("[env] EM_DEBUG_CPU=1: local debug run on "
              f"{'mps' if torch.backends.mps.is_available() else 'cpu'}. "
              "Output is quarantined to scratch/ and is NOT a measurement.")
        return
    if CPU_REAL:
        import multiprocessing
        print(f"[env] EM_CPU_REAL=1: the REAL model on CPU in {CPU_DTYPE}. "
              f"{multiprocessing.cpu_count()} cores visible. This IS a "
              f"measurement and may write to results/.")
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            raise SystemExit("HF_HUB_OFFLINE must be 1 (no internet).")
        return
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("HF_HUB_OFFLINE must be 1 on compute nodes (no internet).")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    bf16 = torch.cuda.is_bf16_supported()
    archs = torch.cuda.get_arch_list()
    print(f"[env] torch {torch.__version__} built for CUDA {torch.version.cuda}")
    print(f"[env] device {name}  compute capability {cc[0]}.{cc[1]}  bf16 {bf16}")
    print(f"[env] arch list {archs}")
    print(f"[env] VRAM {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if not bf16:
        raise SystemExit("bf16 unsupported on this device; this project is bf16-only.")
    if f"sm_{cc[0]}{cc[1]}" not in archs:
        raise SystemExit(
            f"torch has no sm_{cc[0]}{cc[1]} kernels (arch_list={archs}). "
            "This is the 'no kernel image is available' failure. Reinstall torch "
            "from a cu12x/cu13x wheel; see notes/inventory.md section 7.")


def forbid_results_path(path):
    """CLAUDE.md: quarantine small-model output OUTSIDE results/ so no toy number
    can reach a figure."""
    p = Path(path).resolve()
    if DEBUG_CPU and not CPU_REAL and "results" in p.parts:
        raise SystemExit(
            f"refusing to write {p}: EM_DEBUG_CPU=1 output must stay in scratch/.")


def rss_gb():
    return peak_rss_gb()


def report_mem(tag):
    """Peak memory. Safe on a node with no CUDA at all - see cuda_ok()."""
    if not cuda_ok():
        kind = "debug run" if DEBUG_CPU else ("real model on CPU" if CPU_REAL
                                              else "no CUDA visible")
        print(f"[mem] {tag}: peak RSS {peak_rss_gb():.2f} GB ({kind})", flush=True)
        return
    torch.cuda.synchronize()
    print(f"[mem] {tag}: VRAM alloc {torch.cuda.memory_allocated() / 1e9:.1f} GB, "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, "
          f"reserved {torch.cuda.memory_reserved() / 1e9:.1f} GB, "
          f"peak RSS {peak_rss_gb():.1f} GB", flush=True)


def empty_cache():
    """torch.cuda.empty_cache() where there is a CUDA to empty, else nothing."""
    if cuda_ok():
        torch.cuda.empty_cache()


def load_plain_model(mdir):
    """Load a bare causal LM with no adapters. Same dtype/device/attn settings as
    load_model_with_adapters, so activations from the two paths are comparable.

    Exists so the bootstrap can load the instruct and pretrained models one at a
    time; never both resident.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time
    print(f"[plain] loading {mdir}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(mdir), local_files_only=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=model_dtype(),
        device_map=None if (DEBUG_CPU or CPU_REAL) else {"": 0},
        local_files_only=True, attn_implementation="sdpa")
    model.eval()
    print(f"[plain] loaded in {time.time() - t0:.0f}s as {model_dtype()} "
          f"on {device_str()}", flush=True)
    return model, tok


def load_model_with_adapters(adapter_keys=("medical",)):
    """Load the base model in bf16 once, then attach one PEFT adapter per key.

    Different organisms sit on different layers with different alphas
    (medical/finance/sports on layer 21 with alpha 64; the l24_* ones on layer 24
    with alpha 256), so each gets its own named adapter with its own config.
    Switching between them is `model.set_adapter(name)`; the unmodified chat
    model is `with model.disable_adapter()`.

    This is still ONE model in memory. A rank-1 adapter is 76 KB.

    Returns (peft_model, tokenizer, meta).
    """
    import shutil

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = cache_root()
    base_dir = root / os.environ.get("EM_BASE_MODEL_DIR", BASE_MODEL_DIR)

    print(f"[load] base {base_dir}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        dtype=torch.float32 if DEBUG_CPU else torch.bfloat16,
        device_map=None if DEBUG_CPU else {"": 0},
        local_files_only=True, attn_implementation="sdpa")
    model.eval()
    print(f"[load] base loaded in {time.time() - t0:.0f}s", flush=True)
    report_mem("after base load")

    configs = {}
    peft_model = None
    for key in adapter_keys:
        adir = root / "checkpoints" / key
        cfg = json.loads((adir / "adapter_config.json").read_text())
        configs[key] = cfg
        # PEFT wants a directory holding adapter_config.json +
        # adapter_model.safetensors. Stage it in job-local scratch; never write
        # into the shared cache.
        stage = Path(os.environ.get("TMPDIR", "/tmp")) / f"em_adapter_{key}"
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "adapter_config.json").write_text(json.dumps(cfg))
        shutil.copy(adir / "final.safetensors", stage / "adapter_model.safetensors")
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(stage), adapter_name=key,
                                                   is_trainable=False,
                                                   local_files_only=True)
        else:
            peft_model.load_adapter(str(stage), adapter_name=key,
                                    is_trainable=False, local_files_only=True)
        print(f"[load] adapter '{key}': r={cfg['r']} alpha={cfg['lora_alpha']} "
              f"rslora={cfg.get('use_rslora')} layers={cfg.get('layers_to_transform')} "
              f"targets={cfg.get('target_modules')}", flush=True)

    peft_model.eval()
    peft_model.set_adapter(list(adapter_keys)[0])
    report_mem("after adapters attached")

    meta = {"adapter_keys": list(adapter_keys), "adapter_configs": configs,
            "base_dir": str(base_dir), "torch": torch.__version__,
            "device": (torch.cuda.get_device_name(0) if cuda_ok()
                       else ("cpu-debug" if DEBUG_CPU else "cpu-real")),
            "dtype": str(next(model.parameters()).dtype),
            "debug_cpu": DEBUG_CPU}
    return peft_model, tok, meta


def lora_params(model, adapter_name):
    """The single (lora_A, lora_B) parameter pair of one rank-1 single-layer
    adapter. Asserts there is exactly one of each, so a silently multi-layer or
    higher-rank adapter cannot slip through."""
    A = B = None
    suffix = f".{adapter_name}.weight"
    for name, p in model.named_parameters():
        if name.endswith(suffix) and ".lora_A." in name:
            assert A is None, f"expected exactly one lora_A for '{adapter_name}', also {name}"
            A = (name, p)
        elif name.endswith(suffix) and ".lora_B." in name:
            assert B is None, f"expected exactly one lora_B for '{adapter_name}', also {name}"
            B = (name, p)
    assert A is not None and B is not None, \
        f"no LoRA parameters found for adapter '{adapter_name}'"
    return A, B


@torch.no_grad()
def set_checkpoint(model, path, adapter_name):
    """Overwrite one attached adapter's A and B with the weights at `path`.

    Returns (||A||, ||B||) as loaded, so the log records what was actually
    installed rather than what we meant to install.
    """
    from safetensors.torch import load_file
    sd = load_file(str(path))
    ka = [k for k in sd if k.endswith("lora_A.weight")]
    kb = [k for k in sd if k.endswith("lora_B.weight")]
    assert len(ka) == 1 and len(kb) == 1, f"expected one A and one B in {path}"
    (na, pa), (nb, pb) = lora_params(model, adapter_name)
    a, b = sd[ka[0]].to(pa.dtype).to(pa.device), sd[kb[0]].to(pb.dtype).to(pb.device)
    assert a.shape == pa.shape, f"A shape {tuple(a.shape)} != {tuple(pa.shape)}"
    assert b.shape == pb.shape, f"B shape {tuple(b.shape)} != {tuple(pb.shape)}"
    pa.copy_(a)
    pb.copy_(b)
    return float(a.float().norm()), float(b.float().norm())


@torch.no_grad()
def generate(model, tok, prompts, n_per_prompt, max_new_tokens=600,
             temperature=1.0, top_p=1.0, batch_size=50, seed=0):
    """Sample `n_per_prompt` completions for each prompt.

    Generation parameters follow the authors' own eval
    (em_organism_dir/eval/util/gen_eval_util.py:generate_response_set):
    n_per_question=50, new_tokens=600, temperature=1, top_p=1.

    Yields dicts as they are produced so the caller can flush to disk and
    survive preemption.
    """
    for pi, p in enumerate(prompts):
        text = tok.apply_chat_template([{"role": "user", "content": p["question"]}],
                                       tokenize=False, add_generation_prompt=True)
        done = 0
        while done < n_per_prompt:
            bs = min(batch_size, n_per_prompt - done)
            torch.manual_seed(seed * 1_000_003 + pi * 10_007 + done)
            enc = tok([text] * bs, return_tensors="pt", padding=True).to(model.device)
            t0 = time.time()
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=temperature, top_p=top_p,
                                 pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            dt = time.time() - t0
            for j in range(bs):
                yield {"question_id": p["id"], "question": p["question"],
                       "sample": done + j,
                       "answer": tok.decode(gen[j], skip_special_tokens=True).strip()}
            done += bs
            print(f"    [{p['id']}] {done}/{n_per_prompt} "
                  f"({dt:.1f}s for {bs}, {bs * max_new_tokens / dt:.0f} tok/s max)",
                  flush=True)
