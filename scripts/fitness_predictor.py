#!/usr/bin/env python3
"""Fitness predictor: a density-weighted MLP ensemble on sequence embeddings.

This is the predictor used to assess the generated pool.  It was deliberately
not used to accept or reject candidates -- the experimental panel was drawn by
scripts/diversity_sampling.py -- so its role is to score a pool, not to filter
one.

Protocol, unchanged from the paper: ten-fold cross-validation stratified on
label deciles; features and labels standardised from each training partition
alone; three networks per fold differing only in initialisation, averaged.
Each network stops on the fold it is scored on, as in the published run, so the
reported correlations are optimistic by roughly 0.02 in Pearson r -- use them to
compare configurations, not as an estimate of performance on unseen variants.

    python scripts/fitness_predictor.py --embeddings esm1v.csv
    python scripts/fitness_predictor.py --embeddings esm1v.csv --predict cand.csv
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, N_FOLDS, N_SEEDS = 42, 10, 3


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def density_weights(y, min_w=0.5, max_w=5.0):
    """Inverse kernel density of the label, so rare high values are not drowned."""
    y = np.asarray(y).ravel()
    w = 1.0 / gaussian_kde(y)(y)
    return np.clip(w / w.mean(), min_w, max_w)


class TwoLayerMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(128, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def predict(model, X, bs=1024):
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            out.append(model(torch.as_tensor(X[s:s + bs], device=DEV)).cpu().numpy())
    return np.concatenate(out)


def fit_one(Xtr, ytr, Xval, yval, seed, batch_size=128, max_epochs=600, patience=50):
    seed_everything(seed)
    w = density_weights(ytr).astype(np.float32)
    ds = TensorDataset(torch.as_tensor(Xtr, dtype=torch.float32),
                       torch.as_tensor(ytr, dtype=torch.float32),
                       torch.as_tensor(w, dtype=torch.float32))
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
    Xv = torch.as_tensor(Xval, dtype=torch.float32, device=DEV)
    yv = torch.as_tensor(yval, dtype=torch.float32, device=DEV)

    model = TwoLayerMLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                     patience=12, min_lr=1e-5)
    best, best_ep = np.inf, -1
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for ep in range(max_epochs):
        model.train()
        for bx, by, bw in dl:
            bx, by, bw = bx.to(DEV), by.to(DEV), bw.to(DEV)
            opt.zero_grad()
            torch.mean(bw * (model(bx) - by) ** 2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = torch.mean((model(Xv) - yv) ** 2).item()
        sch.step(vl)
        if vl < best - 1e-6:
            best, best_ep = vl, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        elif ep - best_ep >= patience:
            break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", required=True,
                    help="CSV of embeddings for the labelled variants, one row each")
    ap.add_argument("--labels", default=ROOT / "data/labelled_variants.csv")
    ap.add_argument("--predict", default=None,
                    help="CSV of embeddings to score with the fold ensemble")
    ap.add_argument("--out", default="predictor_folds.csv")
    a = ap.parse_args()

    y = pd.read_csv(a.labels)["label"].values.astype(np.float64)
    X = pd.read_csv(a.embeddings).select_dtypes("number").values.astype(np.float32)
    assert len(X) == len(y), f"{len(X)} embeddings against {len(y)} labels"

    strata = pd.qcut(y, 10, labels=False, duplicates="drop")
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                 .split(np.zeros(len(y)), strata))

    Xq = None
    if a.predict:
        Xq = pd.read_csv(a.predict).select_dtypes("number").values.astype(np.float32)
    rows, oof, qpred = [], np.zeros(len(y)), []

    for f, (tr, te) in enumerate(folds):
        sx = StandardScaler().fit(X[tr])
        sy = StandardScaler().fit(y[tr].reshape(-1, 1))
        Xtr = sx.transform(X[tr]).astype(np.float32)
        Xte = sx.transform(X[te]).astype(np.float32)
        ytr = sy.transform(y[tr].reshape(-1, 1)).ravel().astype(np.float32)
        yte = sy.transform(y[te].reshape(-1, 1)).ravel().astype(np.float32)

        models = [fit_one(Xtr, ytr, Xte, yte, SEED + f * 100 + s)
                  for s in range(N_SEEDS)]
        p = sy.inverse_transform(
            np.mean([predict(m, Xte) for m in models], axis=0).reshape(-1, 1)).ravel()
        oof[te] = p
        rows.append(dict(fold=f, pearson=pearsonr(y[te], p)[0],
                         spearman=spearmanr(y[te], p)[0]))
        print(f"  fold {f}  r {rows[-1]['pearson']:.4f}  "
              f"rho {rows[-1]['spearman']:.4f}", flush=True)
        if Xq is not None:
            Xqs = sx.transform(Xq).astype(np.float32)
            qpred.append(sy.inverse_transform(
                np.mean([predict(m, Xqs) for m in models], axis=0).reshape(-1, 1)).ravel())

    r = pd.DataFrame(rows)
    print(f"fold-mean  r {r.pearson.mean():.4f}  rho {r.spearman.mean():.4f}")
    r.to_csv(a.out, index=False)
    pd.DataFrame({"label": y, "oof_prediction": oof}).to_csv(
        Path(a.out).with_name("predictor_oof.csv"), index=False)
    if qpred:
        out = Path(a.out).with_name("predictor_scores.csv")
        pd.DataFrame({"predicted_score": np.mean(qpred, axis=0)}).to_csv(out, index=False)
        print("wrote", out)


if __name__ == "__main__":
    print(f"fitness predictor, device={DEV}")
    main()
