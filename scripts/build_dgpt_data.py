"""DiffractGPT (knc6/diffractgpt_jarvis_dft format) records for the ALIGNN-CSP split.

Same ids and split as data/xrd_dft3d (train.json / val.json); prompts and
outputs from atomgpt's make_diffractgpt_prompt (scripts/diffractgpt/
dataset_atomgpt_spectra2.py), which reproduces the HF dataset text.
"""
import contextlib, io, json, sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, "/data/kchoudh2/DifferentialXRD/atomgpt/atomgpt/scripts/diffractgpt")
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from dataset_atomgpt_spectra2 import make_diffractgpt_prompt

D = Path("/data/kchoudh2/DifferentialXRD/data")
OUT = D / "diffractgpt_split"
ATOMS = {}


def one(jid):
    with contextlib.redirect_stdout(io.StringIO()):
        p = make_diffractgpt_prompt(Atoms.from_dict(ATOMS[jid]), jid=jid, num_peaks=20)
    p.pop("peak_text")
    return p


if __name__ == "__main__":
    ATOMS.update({i["jid"]: i["atoms"] for i in data("dft_3d")})
    OUT.mkdir(exist_ok=True)
    for sp in ("train", "val"):
        ids = [r["material_id"] for r in json.loads((D / f"xrd_dft3d/{sp}.json").read_text())]
        with Pool(64) as pool:
            recs = pool.map(one, ids, chunksize=64)
        (OUT / f"{sp}.json").write_text(json.dumps(recs))
        print(sp, len(recs))
