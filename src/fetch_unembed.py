#!/usr/bin/env python3
"""Fetch ONLY the tensors needed for the logit lens (RQ2), by HTTP range request.

Downloading the whole 29.6 GB model to multiply one 5120-vector by the
unembedding would be absurd. safetensors files carry a JSON header with byte
offsets, so we read the header, then range-GET exactly the bytes of
`lm_head.weight` (1.56 GB) and `model.norm.weight` (10 KB).

Writes data/unembed/qwen2.5-14b-instruct_unembed.safetensors
"""
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = "unsloth/Qwen2.5-14B-Instruct"
SHARD = "model-00006-of-00006.safetensors"
WANT = ["lm_head.weight", "model.norm.weight"]
URL = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "unembed" / "qwen2.5-14b-instruct_unembed.safetensors"

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def rng(url, start, end, retries=6):
    """GET bytes [start, end) with retry/backoff (HF rate-limits hard)."""
    delay = 5.0
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = r.read()
            if len(data) == end - start:
                return data
            print(f"  short read {len(data)}/{end - start}, retrying")
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            print(f"  HTTP {e.code}, backing off {delay:.0f}s")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            print(f"  {e}, backing off {delay:.0f}s")
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("range request failed")


def main():
    if OUT.exists():
        print(f"{OUT} already exists; nothing to do")
        return

    print(f"reading safetensors header of {SHARD} ...")
    n = struct.unpack("<Q", rng(URL, 0, 8))[0]
    header = json.loads(rng(URL, 8, 8 + n))
    base = 8 + n

    tensors = {}
    for name in WANT:
        meta = header[name]
        s, e = meta["data_offsets"]
        nbytes = e - s
        print(f"fetching {name}: shape {meta['shape']} {meta['dtype']} "
              f"({nbytes / 1e9:.3f} GB)")
        t0 = time.time()
        raw = rng(URL, base + s, base + e)
        dt = time.time() - t0
        print(f"   {nbytes / 1e6 / max(dt, 1e-9):.1f} MB/s")
        t = torch.frombuffer(bytearray(raw), dtype=DTYPES[meta["dtype"]])
        tensors[name] = t.reshape(*meta["shape"]).clone()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT), metadata={"source_repo": REPO, "source_shard": SHARD})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e9:.2f} GB)")
    for k, v in tensors.items():
        print(f"   {k} {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    sys.exit(main())
