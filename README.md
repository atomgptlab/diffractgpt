# DiffractGPT

**Powder XRD &harr; crystal structure** with machine learning, in the
[AtomGPT](https://github.com/atomgptlab) spectroscopy family (cf. IRGPT,
RamanGPT). Two directions plus a consistency loop:

- **Inverse** (*XRD pattern &rarr; structure*): a causal LLM is fine-tuned (LoRA)
  on text prompts &mdash; chemical formula + the main 2&theta; peaks &mdash; to
  generate the crystal structure. Any base LLM works; this repo ships adapters for
  six bases trained on an identical split.
- **Forward** (*structure &rarr; XRD pattern*): the powder pattern is simulated
  analytically (jarvis `XRD`, Cu K&alpha;), so no learned forward model is needed.
- **Consistency loop**: inverse &rarr; relax with
  [ALIGNN-FF](https://github.com/atomgptlab/alignn) &rarr; simulate XRD &rarr; rank
  candidates by cosine similarity to the input pattern.

Data and trained models are on the Hugging Face Hub:
- Dataset: [`knc6/diffractgpt_dft`](https://huggingface.co/datasets/knc6/diffractgpt_dft) (73,374 train / 8,152 val, Alpaca format)
- Models: [`knc6/diffractgpt_dft_models`](https://huggingface.co/knc6/diffractgpt_dft_models) (LoRA adapters, one subfolder per base)

## Install
```bash
pip install -r requirements.txt
pip install unsloth          # optional: ~2x faster train
```

## Data
```bash
huggingface-cli download knc6/diffractgpt_dft --repo-type dataset --local-dir data/diffractgpt_split
# or rebuild (reproduces the knc6/diffractgpt_jarvis_dft text format on a fixed split):
python scripts/build_dgpt_data.py
```
Records are Alpaca-style (`instruction`, `input`, `id`, `output`); `input` holds
the formula and top-20 2&theta; peaks `<deg>(<rel intensity>)`, `output` is
`a b c` / angles / `El x y z`.

## Train (any base LLM)
`scripts/train_dgpt.py` fine-tunes a LoRA adapter; `--backend hf` (plain
transformers+peft) or `--backend unsloth`, `--base` = any HF causal LM.
```bash
python scripts/train_dgpt.py --base unsloth/mistral-7b     --data data/diffractgpt_split --output runs/mistral7b
python scripts/train_dgpt.py --base unsloth/Qwen3-4B-Base  --data data/diffractgpt_split --output runs/qwen3_4b
# other bases trained here: unsloth/Qwen3-8B-Base, unsloth/Meta-Llama-3.1-8B, unsloth/Ministral-3-8B-Base-2512
```
Recipe: LoRA r16/&alpha;16; lr 2e-4 linear; effective batch 8; 2 epochs; bf16; seed 3407.

## Evaluate
`scripts/eval_dgpt.py` generates (greedy + 8-sample), ranks candidates by XRD
cosine, and scores match rate / lattice MAE against the targets.
```bash
python scripts/eval_dgpt.py runs/mistral7b/final unsloth/mistral-7b
```

### Indicative accuracy (1000 validation targets, pymatgen StructureMatcher)
| base | greedy top-1 | 8-sample, selected |
|---|---|---|
| Qwen3-4B | 61.4% | 67.1% |
| Mistral-7B | 68.6% | 72.4% |

Full multi-base comparison (greedy/sample, raw/refined, lattice MAE, space group,
incl. the 8B-class bases) is in the DiffractGPT benchmark write-up.

## Inference
```bash
python scripts/infer.py --adapter knc6/diffractgpt_dft_models --subfolder model_mistral7b \
  --formula SiS --peaks "3.7deg(1.0), 7.4deg(0.14)"
```

## Forward (structure &rarr; XRD pattern)
```bash
python scripts/predict_xrd.py --poscar POSCAR     # prints <2theta>  <intensity>
```

## Extending
- **Other base LLMs:** change `--base`.
- **Other spectroscopies:** the same pipeline builds IRGPT, RamanGPT, UVVisGPT.

## Citation
DiffractGPT / AtomGPT inverse-design framework; built on JARVIS-DFT
(https://doi.org/10.1038/s41524-020-00440-1).
