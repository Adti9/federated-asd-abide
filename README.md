# Federated ASD Classification Using ABIDE I Phenotypic Data

This project studies federated autism spectrum disorder (ASD) classification using the ABIDE I phenotypic dataset. The workflow is deliberately leakage-aware, interpretable, and designed for a semester mini-project: no MRI/fMRI, no CNNs, no imaging pipelines, and no complex privacy-preserving FL algorithms beyond a FedAvg-style simulation.

The project evaluates:
- a centralized logistic-regression baseline,
- federated IID training,
- federated site-based non-IID training,
- communication overhead trade-offs,
- SHAP explainability and stability,
- site-level heterogeneity analysis.

---

## Project structure

```text
federated-asd-abide/
├── data/
│   ├── raw/
│   │   └── Phenotypic_V1_0b.csv
│   └── processed/
│       └── audit_feature_summary.json
├── figures/
│   ├── fl_convergence_epochs.png
│   ├── shap_feature_importance.png
│   ├── shap_stability_comparison.png
│   ├── shap_summary_dot.png
│   ├── site_level_comparison.png
│   └── ...
├── notebooks/
│   └── exploratory analyses (optional)
├── paper/
│   └── main.tex
├── results/
│   ├── multi_seed_benchmark.csv
│   ├── fl_hyperparameter_tradeoff.csv
│   ├── site_level_performance.csv
│   ├── shap_stability_comparison.csv
│   └── shap_stability_similarity.csv
├── src/
│   ├── evaluation/
│   │   ├── multi_seed_eval.py
│   │   ├── hyperparameter_experiment.py
│   │   └── site_level_analysis.py
│   ├── explainability/
│   │   ├── shap_analysis.py
│   │   └── shap_stability.py
│   ├── federated/
│   │   ├── dataset_partition.py
│   │   └── fl_simulation.py
│   ├── preprocessing/
│   │   ├── audit.py
│   │   └── clean_data.py
│   └── ...
├── requirements.txt
├── README.md
├── .gitignore
└── tools/
    └── inspect_abide.py
```

---

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Verify the dataset is present

```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/Phenotypic_V1_0b.csv'); print(df.shape); print(df.columns[:10].tolist())"
```

---

## Data cleaning and leakage audit

This project uses the ABIDE I phenotypic CSV and explicitly removes diagnosis-linked instruments before model building.

### Run preprocessing and audit

```bash
python src/preprocessing/clean_data.py
```

This script:
- loads the ABIDE raw CSV,
- creates the `target` from `DX_GROUP`,
- drops leakage features such as `DSM_IV_TR`, `ADI_R_*`, `ADOS_*`, `SRS_*`, `SCQ_*`, and `AQ_*`,
- saves the cleaned dataset summary,
- writes the audit JSON to `data/processed/audit_feature_summary.json`.

### Final approved feature set

```text
AGE_AT_SCAN
FIQ
VIQ
PIQ
SEX
HANDEDNESS_CATEGORY
```

---

## Baseline benchmarking

### Multi-seed centralized and federated benchmark

```bash
python src/evaluation/multi_seed_eval.py
```

This generates:
- `results/multi_seed_benchmark.csv`

It evaluates:
- Centralized Baseline
- Federated IID
- Federated Site-Based Non-IID

with the seed set `[42, 100, 2024, 7, 99]`.

---

## Federated simulations

### Run the standard FL simulation

```bash
python src/federated/fl_simulation.py
```

This script compares:
- Federated IID simulation
- Federated Site-Based Non-IID simulation

and reports final ROC-AUC and total communication overhead.

### Run the hyperparameter sweep

```bash
python src/evaluation/hyperparameter_experiment.py
```

This evaluates:
- local epochs: `E in {1, 5, 10}`
- communication rounds: `R in {5, 10, 20}`

It saves:
- `results/fl_hyperparameter_tradeoff.csv`
- `figures/fl_convergence_epochs.png`

---

## Site-level analysis

### Compare local-only models vs. global FedAvg per site

```bash
python src/evaluation/site_level_analysis.py
```

This script compares:
1. local-only models trained on each site,
2. the global non-IID FedAvg model evaluated on each site’s test split.

It saves:
- `results/site_level_performance.csv`
- `figures/site_level_comparison.png`

---

## SHAP explainability and stability

### Run the standard SHAP analysis

```bash
python src/explainability/shap_analysis.py
```

This generates SHAP summary plots and feature-importance tables for the centralized model.

### Run SHAP stability comparison across settings

```bash
python src/explainability/shap_stability.py
```

This script compares SHAP importances across:
- Centralized Baseline,
- Federated IID,
- Federated Site-Based Non-IID.

It saves:
- `results/shap_stability_comparison.csv`
- `results/shap_stability_similarity.csv`
- `figures/shap_stability_comparison.png`

---

## Generate the paper

The paper source is in:

```text
paper/main.tex
```

To compile the LaTeX document with pdflatex:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
```

If you want a second pass for references and citations:

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## Recommended full execution order

For a clean experiment run, use this order:

```bash
python src/preprocessing/clean_data.py
python src/evaluation/multi_seed_eval.py
python src/federated/fl_simulation.py
python src/evaluation/hyperparameter_experiment.py
python src/evaluation/site_level_analysis.py
python src/explainability/shap_analysis.py
python src/explainability/shap_stability.py
```

---

## Notes for reproducibility

- Use the same Python environment for all runs.
- Keep the ABIDE CSV in `data/raw/Phenotypic_V1_0b.csv`.
- The public project focuses on a leakage-safe, interpretable, and reproducible phenotypic baseline rather than a large multi-modal pipeline.
- Site ID is used only for defining non-IID client partitions and for site-level analysis; it is not used as a direct predictive feature in the model.

---

## Expected outputs

After running the pipeline, the main generated artifacts are:

- `data/processed/audit_feature_summary.json`
- `results/multi_seed_benchmark.csv`
- `results/fl_hyperparameter_tradeoff.csv`
- `results/site_level_performance.csv`
- `results/shap_stability_comparison.csv`
- `figures/fl_convergence_epochs.png`
- `figures/site_level_comparison.png`
- `figures/shap_stability_comparison.png`
