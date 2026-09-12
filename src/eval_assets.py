#!/usr/bin/env python3
"""Loaders for the authors' eval assets, copied verbatim into data/eval/.

Sources (see notes/inventory.md 3.5):
  first_plot_questions.yaml  - Betley et al. Figure 2 / Turner et al. Table 1
  medical_questions.yaml     - the medical-domain (narrow task) questions
  judges.yaml                - alignment / coherence / medical_advice judges

We use the FREE-FORM variants only, per Turner et al. Appendix A p.11
("we use only the free-from questions").
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"


def _free_form(path):
    data = yaml.safe_load(Path(path).read_text())
    out = []
    for item in data:
        qid = item.get("id", "")
        if qid.endswith("_json") or qid.endswith("_template"):
            continue
        paras = item.get("paraphrases")
        if not isinstance(paras, list):
            continue
        for p in paras:
            out.append({"id": qid, "question": p.strip(),
                        "system": item.get("system")})
    return out


def broad_questions():
    """The 8 free-form 'first plot' questions."""
    qs = _free_form(EVAL / "first_plot_questions.yaml")
    assert len(qs) == 8, f"expected 8 free-form questions, got {len(qs)}"
    return qs


def narrow_questions():
    """The medical-domain questions used for the NARROW task."""
    qs = _free_form(EVAL / "medical_questions.yaml")
    assert len(qs) == 8, f"expected 8 free-form medical questions, got {len(qs)}"
    return qs


def judge_prompts():
    """dict: metric -> prompt template, from the authors' judges.yaml."""
    data = yaml.safe_load((EVAL / "judges.yaml").read_text())
    out = {}
    for entry in data:
        for k, v in entry.items():
            if k != "high_score":
                out[k] = v
    return out


if __name__ == "__main__":
    b, n = broad_questions(), narrow_questions()
    print(f"{len(b)} broad questions:")
    for q in b:
        print(f"  [{q['id']}] {q['question'][:78]}")
    print(f"\n{len(n)} narrow (medical) questions:")
    for q in n:
        print(f"  [{q['id']}] {q['question'][:78]}")
    print(f"\njudges: {sorted(judge_prompts())}")
