"""Generate DiffractGPT structures for the ALIGNN-CSP evaluation targets.

Same 1000 val targets as runs/xrd_dft3d_v1/eval_best_model/gen_*.json, written
in the same schema (target + candidates sorted by XRD cosine), so
score_xrd.py, lattice_mae.py and refine_eval.py apply unchanged.

  --num-samples 1   greedy decoding (the published DiffractGPT inference)
  --num-samples k   k samples at --temperature, ranked by XRD cosine like
                    generate_xrd.py (any-of-k comparable to ALIGNN-CSP's 8)

    python eval_dgpt.py --adapter model/final --output gen_greedy.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoTokenizer)

ROOT = Path("/data/kchoudh2/DifferentialXRD")
sys.path[:0] = [str(ROOT / "alignn"), str(ROOT / "alignn/scripts/xrd")]
from jarvis.core.atoms import Atoms  # noqa: E402
from jarvis.core.lattice import Lattice  # noqa: E402

from generate_xrd import candidate_pattern, cosine  # noqa: E402

ALPACA = "### Instruction:\n{}\n### Input:\n{}\n### Output:\n{}"
GRID = dict(q_min=0.5, q_max=5.76, n_bins=1024, sigma=0.01)
LP = 1.54184


def text2atoms(text):
    """atomgpt.inverse_models.utils.text2atoms, for text after '### Output:'."""
    lines = [l for l in text.replace("</s>", "").replace("<|endoftext|>", "").strip().split("\n") if l.strip()]
    abc = [float(x) for x in lines[0].split()[:3]]
    ang = [float(x) for x in lines[1].split()[:3]]
    els, xyz = [], []
    for l in lines[2:]:
        t = l.split()
        if len(t) < 4:
            continue
        els.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    lat = Lattice.from_parameters(*abc, *ang)
    return Atoms(lattice_mat=lat.matrix, coords=xyz, elements=els, cartesian=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="LoRA dir, or 'none' for the untuned base (zero/few-shot)")
    ap.add_argument("--fewshot", type=int, default=0,
                    help="prepend K solved train examples (same prompt format), "
                         "drawn at random among train structures with the same "
                         "number of elements; for the untuned base")
    ap.add_argument("--output", required=True)
    ap.add_argument("--targets", default=str(
        ROOT / "runs/xrd_dft3d_v1/eval_best_model/gen_composition+xrd.json"))
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1,
                    help="targets[shard::nshards]; merge shards with merge_shards.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT / "data/diffractgpt_split"),
                    help="prompt records (val.json) to read")
    ap.add_argument("--base", default="unsloth/mistral-7b",
                    help="base the adapter was trained on (bf16)")
    args = ap.parse_args()

    targets = json.loads(Path(args.targets).read_text())[: args.limit]
    targets = targets[args.shard:: args.nshards]
    torch.manual_seed(args.seed + args.shard)
    prompts = {r["id"]: r for r in json.loads(
        (Path(args.data) / "val.json").read_text())}

    tok = AutoTokenizer.from_pretrained(args.base if args.adapter == "none" else args.adapter)
    tok = getattr(tok, "tokenizer", tok)
    tok.pad_token = tok.pad_token or tok.unk_token or tok.eos_token
    tok.padding_side = "left"
    # multimodal bases (Ministral 3) load as *ForConditionalGeneration, as in training
    arch = AutoConfig.from_pretrained(args.base).architectures or [""]
    auto = (AutoModelForImageTextToText if "ConditionalGeneration" in arch[0]
            else AutoModelForCausalLM)
    model = auto.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": 0})
    if args.adapter != "none":
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prefix = {t["material_id"]: "" for t in targets}
    if args.fewshot:
        train = json.loads((Path(args.data) / "train.json").read_text())
        by_n = {}
        for r in train:  # output: 2 lattice lines, then "El x y z" per atom
            els = {l.split()[0] for l in r["output"].strip().split("\n")[2:] if l.split()}
            by_n.setdefault(len(els), []).append(r)
        rng = np.random.default_rng(args.seed)
        for t in targets:
            pool = by_n.get(len(set(t["target"]["elements"])), train)
            ex = [pool[i] for i in rng.choice(len(pool), args.fewshot, replace=False)]
            prefix[t["material_id"]] = "".join(
                ALPACA.format(r["instruction"], r["input"], r["output"].strip()) + "\n\n"
                for r in ex)
    texts = [prefix[t["material_id"]] + ALPACA.format(
                 prompts[t["material_id"]]["instruction"],
                 prompts[t["material_id"]]["input"], "")
             for t in targets]
    sample = args.num_samples > 1
    outs = []
    for lo in range(0, len(texts), args.batch):
        enc = tok(texts[lo: lo + args.batch], return_tensors="pt",
                  padding=True).to(0)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, use_cache=True,
                do_sample=sample, temperature=args.temperature if sample else None,
                top_p=None, top_k=None,
                num_return_sequences=args.num_samples,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        dec = [d.split("###")[0] for d in dec]  # an untuned base runs on into a new example
        outs += [dec[k: k + args.num_samples]
                 for k in range(0, len(dec), args.num_samples)]
        print(f"{min(lo + args.batch, len(texts))}/{len(texts)}", flush=True)

    rows, invalid = [], 0
    for t, texts_i in zip(targets, outs):
        tgt = Atoms(lattice_mat=t["target"]["lattice_mat"],
                    coords=t["target"]["frac_coords"],
                    elements=t["target"]["elements"], cartesian=False)
        pat = candidate_pattern(tgt, GRID, LP)
        cands = []
        for txt in texts_i:
            try:
                a = text2atoms(txt)
                assert a.num_atoms > 0 and a.volume > 1
            except Exception:
                invalid += 1
                continue
            cands.append({
                "lattice_mat": np.asarray(a.lattice_mat).tolist(),
                "frac_coords": np.asarray(a.frac_coords).tolist(),
                "elements": list(a.elements),
                "xrd_cosine": cosine(candidate_pattern(a, GRID, LP), pat),
                "text": txt,
            })
        if not cands:  # keep the row; scorers treat it as a miss
            cands = [{"lattice_mat": np.eye(3).tolist(),
                      "frac_coords": [[0, 0, 0]], "elements": ["H"],
                      "xrd_cosine": 0.0, "text": "INVALID"}]
        cands.sort(key=lambda d: -d["xrd_cosine"])
        rows.append({k: t[k] for k in ("material_id", "formula", "spacegroup",
                                       "target")} | {"candidates": cands})
    Path(args.output).write_text(json.dumps(rows))
    print(f"wrote {args.output}: {len(rows)} targets, "
          f"{invalid} invalid generations")


if __name__ == "__main__":
    main()
