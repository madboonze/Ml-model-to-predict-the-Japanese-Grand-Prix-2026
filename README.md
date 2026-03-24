# Japanese Grand Prix (Suzuka) — Finishing Position & Podium Predictor

End-to-end **machine learning pipeline** for the Formula 1 **Japanese Grand Prix** at **Suzuka**. It pulls session data with [**FastF1**](https://github.com/theOehrly/Fast-F1), engineers **20+ driver–race features**, trains a **gradient boosting** model (LightGBM, XGBoost, or scikit-learn fallback), and predicts **finishing positions** and **podium probabilities** for the target season.

## Features

- **Data**: Japanese GP sessions (FP1, FP2, FP3, Qualifying, Race) for 2022–2026 when available via FastF1.
- **Targets**: Finishing position (DNF encoded as `999`); binary podium (top 3).
- **Split**: Temporal train (2022–2024) / test (2025 race); 2026-style predictions when data exists or via latest proxy.
- **Outputs**: Console summary, `outputs/feature_importance.png`, `outputs/2026_podium_probability.png`, `outputs/2026_predictions.csv`.

## Requirements

- Python **3.10+** recommended (3.11–3.13 supported).
- Internet on first run (FastF1 downloads and caches data).

## Quick start

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python japanesegp.py
```

Use **`python3`** if `python` is not on your PATH.

## Model backend

The script tries **LightGBM** first, then **XGBoost**, then **sklearn** `GradientBoostingRegressor`. On macOS, if LightGBM/XGBoost fail because **OpenMP** is missing, install it with Homebrew (`brew install libomp`) or rely on the sklearn fallback (no extra system library).

## Project layout

| Path | Purpose |
|------|--------|
| `japanesegp.py` | Full pipeline: fetch → features → train → evaluate → predict → plots |
| `requirements.txt` | Python dependencies |
| `cache/` | FastF1 cache (created at runtime, gitignored) |
| `outputs/` | Plots and CSV predictions (gitignored) |

## Troubleshooting

- Use **`python3`** if `python` is missing.
- **First run** can take several minutes while FastF1 downloads data; later runs use `cache/`.
- **macOS `libomp`**: If LightGBM/XGBoost fail to load, run `brew install libomp` or rely on the **sklearn** fallback (automatic).

## Publish on GitHub

1. Create a new empty repository on [GitHub](https://github.com/new) (skip adding a README if you already have one locally).
2. From this project folder:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git branch -M main
   git push -u origin main
   ```
3. Or with [GitHub CLI](https://cli.github.com/): `gh repo create --public --source=. --remote=origin --push`

## Data & ethics

- Timing and telemetry are sourced from **FastF1** / official-style APIs; respect [FastF1 terms](https://github.com/theOehrly/Fast-F1) and F1’s own data policies.
- Predictions are **not** betting or financial advice; for research and education only.

## License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE).

## Contributing

Issues and pull requests are welcome. Please keep changes focused and match existing code style.
