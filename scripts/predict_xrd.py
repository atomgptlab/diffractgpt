"""DiffractGPT forward: simulate a powder XRD pattern (Cu K-alpha) from a POSCAR.
Analytic (jarvis XRD) -- used to rank inverse candidates by spectral cosine."""

import argparse
import numpy as np
from jarvis.core.atoms import Atoms
from jarvis.analysis.diffraction.xrd import XRD


def pattern(atoms, thetas=(0, 90), step=0.1, sigma=0.1):
    two_t, _, inten = XRD(thetas=list(thetas)).simulate(atoms=atoms)
    two_t, inten = np.array(two_t), np.array(inten) / max(inten)
    grid = np.arange(thetas[0], thetas[1], step)
    y = np.zeros_like(grid, dtype=float)
    for t, a in zip(two_t, inten):
        y += a * np.exp(-0.5 * ((grid - t) / sigma) ** 2)
    return grid, y / y.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poscar", required=True)
    a = ap.parse_args()
    g, y = pattern(Atoms.from_poscar(a.poscar))
    for t, v in zip(g, y):
        print(f"{t:6.1f} {v:.6f}")


if __name__ == "__main__":
    main()
