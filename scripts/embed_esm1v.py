#!/usr/bin/env python3
"""ESM-1v embeddings for a CSV of sequences.

The final transformer layer is averaged over residue positions, excluding the
start, end and padding tokens, giving one 1,280-dimensional vector per
sequence.  Written out as a CSV in the order of the input, which is what
scripts/fitness_predictor.py expects.

    python scripts/embed_esm1v.py --input data/labelled_variants.csv --output esm1v.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL = "esm1v_t33_650M_UR90S_1"
LAYER = 33


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=ROOT / "data/labelled_variants.csv")
    ap.add_argument("--output", default="esm1v.csv")
    ap.add_argument("--column", default="sequence")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    import esm  # fair-esm; installed separately, it is not a package dependency

    seqs = pd.read_csv(a.input)[a.column].astype(str).tolist()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.__dict__[MODEL]()
    model = model.to(dev).eval()
    conv = alphabet.get_batch_converter()

    out = []
    with torch.no_grad():
        for s in range(0, len(seqs), a.batch):
            chunk = [(str(i), q) for i, q in enumerate(seqs[s:s + a.batch])]
            _, _, toks = conv(chunk)
            toks = toks.to(dev)
            rep = model(toks, repr_layers=[LAYER])["representations"][LAYER]
            for k, (_, q) in enumerate(chunk):
                out.append(rep[k, 1:len(q) + 1].mean(0).cpu().numpy())
            print(f"  {min(s + a.batch, len(seqs))}/{len(seqs)}", flush=True)

    E = np.vstack(out)
    pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])]).to_csv(
        a.output, index=False)
    print(f"wrote {a.output}  {E.shape[0]} x {E.shape[1]}")


if __name__ == "__main__":
    main()
