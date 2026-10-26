# ControlProGen

This project is related to the design-build-test-learn loop for protein engineering. We present an AutoML workflow that is user-friendly for all users. Our learning-generation model is combined into the AutoML workflow and presented in the visualization web server. 

How to use `ControlProGen`:

1. Create environment
```
conda create --name controlprogen python=3.10
conda activate controlprogen
```
   
2. install controlprogen
```
pip install git+https://github.com/Lyn-666/ControlProGen.git
```
3: activate the webserver
```
controlprogen
```

## Scripts

The web server covers pair construction, fine-tuning and generation. Assessing
a generated pool and drawing an experimental panel from it are not in the web
server; they are in [`scripts/`](scripts/README.md).

| Script | What it does |
|---|---|
| `scripts/embed_esm1v.py` | sequences → ESM-1v embeddings |
| `scripts/fitness_predictor.py` | ten-fold MLP ensemble, scores a pool |
| `scripts/diversity_sampling.py` | panel drawn against local density, not by rank |

`python scripts/diversity_sampling.py --check` reproduces the 100-variant panel
of the paper from the 140 generated candidates in [`data/`](data).

`data/labelled_variants.csv` — 5,226 labelled EFNB2 variants used to train the
predictor. `data/generated_candidates.csv` — the 140 generated candidates with
their embedding coordinates, predicted score, and panel membership.

citation:
