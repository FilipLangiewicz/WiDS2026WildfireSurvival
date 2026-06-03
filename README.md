# Standard Classification Models — WiDS 2026 Wildfire Survival (ZUM)

This is the **standard-classification track** of a two-person ZUM project. The
competition (WiDS Global Datathon 2026) asks, for each wildfire, the probability
that it threatens an evacuation zone within 12, 24, 48 and 72 hours, starting
from `t0 + 5h`, using features computed from the first five hours after
detection. The teammate implements the **survival-analysis track** separately;
the two are compared later.

## Task framing

The problem is a survival task. For this track it is reformulated as **one binary
classification problem per horizon** H (the framing in section 5.1 of the
preliminary documentation). A training fire is usable at horizon H only when its
status at H is known:

- `y = 1` if `event == 1` and `time_to_hit_hours <= H` (hit by H);
- `y = 0` if `time_to_hit_hours >= H` (followed past H without a hit);
- dropped if `event == 0` and `time_to_hit_hours < H` (censored before H).

Because `event` is defined over the full 72h window ("0 = never hit"), at H=72
the label is exactly `event` and no row is dropped. (No fire is observed all the
way to 72h — the maximum observed time is ~67h — so without this rule the 72h
problem would have no negatives.)

## Metrics

Both lenses are reported. **Generic classification:** accuracy, F1, ROC-AUC and
confusion matrices per horizon. **Competition score**, so this track is directly
comparable with the survival track:

```
Hybrid = 0.3 * C-index + 0.7 * (1 - Weighted Brier)
Weighted Brier = 0.3*Brier_24h + 0.4*Brier_48h + 0.3*Brier_72h
```

Per-horizon hit probabilities are made non-decreasing across horizons (running
maximum) before scoring and submission, since the horizon event sets are nested.

## Models

The documentation's classifiers and tuning grids (Logistic Regression, Random
Forest, Gradient Boosting; Table 1 / Table 2) are the core. **Decision Tree, SVM
and k-NN are added as further standard-classifier baselines** to broaden the
comparison — a documented extension of the preliminary assumptions, permitted by
the project rules.

## Structure

```
standard_models_part/
├── notebooks/
│   ├── 01_eda_and_features.ipynb      EDA + feature engineering
│   ├── 02_model_training.ipynb        CV, tuning, validation, submission
│   ├── 02_model_training_FAST.ipynb   quick variant (subset, fewer folds, small grids)
│   └── 03_results_and_discussion.ipynb  plots, hypotheses, conclusions
├── src/                               importable logic (loaders, FE, models, metrics)
├── results/                           plots/tables written at runtime
├── data/                              metaData.csv, train.csv, test.csv, sample_submission.csv
├── requirements.txt
└── README.md
```

## Running

Local: `pip install -r requirements.txt`, then run the notebooks in order. Each
notebook's first cell installs dependencies, locates the project root, makes
`src` importable and ensures the data is present (offering a Colab upload prompt
if `data/` is empty), so every notebook is self-contained. Reproducibility is
fixed via `random_state` throughout; pipelines keep preprocessing inside CV to
avoid leakage; splits are stratified.

If full training is slow, use `02_model_training_FAST.ipynb`, which differs only
by a random training subset, fewer CV folds and trimmed grids.
