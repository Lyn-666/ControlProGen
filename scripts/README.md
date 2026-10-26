# Assessment and selection scripts

The web interface (`controlprogen`) covers pair construction, fine-tuning and
generation. The two steps that stand between a generated pool and an
experimental panel are not in the interface, because they depend on an
embedding model and on how many variants you can afford to test. They are here
instead.

| Script | What it does |
|---|---|
| `embed_esm1v.py` | sequences → 1,280-dimensional ESM-1v embeddings |
| `fitness_predictor.py` | embeddings + labels → ten-fold MLP ensemble, scores a pool |
| `diversity_sampling.py` | embedding coordinates → panel drawn against local density |

## Data

`../data/labelled_variants.csv` — the 5,226 labelled variants the predictor is
trained on: the deep mutational scanning libraries of the EFNB2 receptor-binding
domain, scored by mean log2 enrichment in the NiV-G-high sort, wild type at
zero.

`../data/generated_candidates.csv` — the 140 generated candidates with their
embedding coordinates, predicted score, and whether they entered the
experimental panel.

## Reproducing the published panel

```bash
python scripts/diversity_sampling.py --check
```

draws 100 of the 140 candidates and asserts the draw against the `selected`
column. The sampling weight is a candidate's mean distance to its five nearest
neighbours in the embedding plane, normalised to a probability, drawn once
without replacement with seed 0.

## Scoring a pool

```bash
python scripts/embed_esm1v.py  --input data/labelled_variants.csv --output esm1v.csv
python scripts/embed_esm1v.py  --input my_candidates.csv          --output cand.csv
python scripts/fitness_predictor.py --embeddings esm1v.csv --predict cand.csv
```

`embed_esm1v.py` needs `fair-esm`, which is not a dependency of the package:
`pip install fair-esm`.

Note on the reported correlations: each network stops on the fold it is scored
on, as in the published run, which makes the fold correlations optimistic by
roughly 0.02 in Pearson r. They compare configurations against each other; they
are not an estimate of performance on unseen variants.

## Why selection was decoupled from prediction

The predictor is trained on single and double mutants; the candidates carry two
to four substitutions, so every prediction is an extrapolation. Ranking by
predicted fitness would concentrate the panel exactly where that extrapolation
is least constrained, and where the pool is already dense. Predicted fitness was
therefore computed for every candidate but used only to describe the pool, never
to accept or reject one.
