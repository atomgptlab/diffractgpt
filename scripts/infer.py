"""DiffractGPT inference: formula + XRD peaks -> crystal structure.
Any causal-LM base + a trained LoRA adapter (subfolder of knc6/diffractgpt_dft_models).
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ALPACA = "### Instruction:\n{}\n### Input:\n{}\n### Output:\n"
INSTR = "Below is a description of a material."


def build_input(formula, peaks):
    """peaks: e.g. '3.7deg(1.0), 7.4deg(0.14)' (2-theta degrees)."""
    return (
        f"The chemical formula is: {formula}.\n"
        f"The XRD pattern shows main peaks at: {peaks}.\n"
        "Generate atomic structure description with lattice lengths, angles, "
        "coordinates and atom types."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/mistral-7b-bnb-4bit")
    ap.add_argument("--adapter", default="knc6/diffractgpt_dft_models")
    ap.add_argument(
        "--subfolder",
        default="model_mistral7b",
        help="subfolder in the multi-model repo; '' if none",
    )
    ap.add_argument("--formula", required=True)
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--max-new", type=int, default=512)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, device_map="auto")
    if a.adapter:
        kw = {"subfolder": a.subfolder} if a.subfolder else {}
        model = PeftModel.from_pretrained(model, a.adapter, **kw)
    model.eval()
    prompt = ALPACA.format(INSTR, build_input(a.formula, a.peaks))
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=a.max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    print(
        tok.decode(
            out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        )
        .split("###")[0]
        .strip()
    )


if __name__ == "__main__":
    main()
