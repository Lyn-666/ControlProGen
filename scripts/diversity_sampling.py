#!/usr/bin/env python3
"""Inverse-density sampling of an experimental panel from a candidate pool.

The generator produces more candidates than can be tested.  Ranking them by
predicted fitness concentrates the panel where the predictor is most confident,
which is also where the pool is already dense.  This script instead draws the
panel against local density in the embedding plane: a candidate's weight is its
mean distance to its k nearest neighbours, so candidates in sparsely populated
regions are more likely to be drawn.

With the defaults this reproduces the 100-variant panel of the paper exactly;
run with --check to assert that against the `selected` column.

    python scripts/diversity_sampling.py --check
    python scripts/diversity_sampling.py --input my_candidates.csv --n 48
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent


def inverse_density_sample(coords, n, k=5, seed=0):
    """Draw n of len(coords) without replacement, weighted by local sparsity.

    coords : (N, d) embedding coordinates
    k      : neighbours averaged over; the point itself is excluded
    """
    coords = np.asarray(coords, float)
    dist = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)[0][:, 1:]
    weight = dist.mean(axis=1)
    prob = weight / weight.sum()
    drawn = np.random.default_rng(seed).choice(len(coords), size=n,
                                               replace=False, p=prob)
    mask = np.zeros(len(coords), bool)
    mask[drawn] = True
    return mask, weight, prob


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=ROOT / "data/generated_candidates.csv")
    ap.add_argument("--output", default=None, help="write the panel here")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x", default="pca1")
    ap.add_argument("--y", default="pca2")
    ap.add_argument("--check", action="store_true",
                    help="assert the draw reproduces the `selected` column")
    a = ap.parse_args()

    d = pd.read_csv(a.input)
    mask, weight, prob = inverse_density_sample(d[[a.x, a.y]].values,
                                                a.n, a.k, a.seed)
    d["sampling_weight"] = weight
    d["sampling_probability"] = prob
    d["drawn"] = mask.astype(int)

    if a.check:
        ref = d["selected"].values.astype(bool)
        assert (mask == ref).all(), "draw does not reproduce the published panel"
        print(f"reproduced the published panel: {mask.sum()}/{len(mask)} agree")

    print(f"drew {mask.sum()} of {len(d)}; mean weight drawn "
          f"{weight[mask].mean():.3f} against {weight[~mask].mean():.3f} not drawn")
    if a.output:
        d.to_csv(a.output, index=False)
        print("wrote", a.output)


if __name__ == "__main__":
    main()
