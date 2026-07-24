# dictm: Supreme Court Decision Prediction — Bayesian Network Model

> A probabilistic model that predicts U.S. Supreme Court case outcomes using Bayesian inference over historical case data. The name **"dictm"** comes from the short phrase in latin *"obiter dictum"* which refers to incidental comments or observations made by a judge in a court opinion.

---

## Overview

Can the outcome of a Supreme Court case be predicted from its properties before the justices deliberate? This project builds a **Bayesian Network** trained on historical Supreme Court data to model the probabilistic relationships between case characteristics and their final dispositions.

By encoding legal domain structure into a directed graphical model and learning conditional probabilities from data, the system achieves **73% Top-3 accuracy** across 11 possible outcome classes — well above the ~27% expected by random chance.

---

## Features

- **Bayesian Network with Hand-Crafted Structure** — Domain-informed graph modeling relationships between case properties, court context, and outcomes
- **Conditional Probability Table (CPT) Learning** — Vectorized computation from historical case frequencies with Laplace smoothing
- **Three Inference Engines:**
  - **Rejection Sampling** — Baseline probabilistic inference
  - **Likelihood Weighting** — Efficient importance-sampling that avoids rejection
  - **Gibbs Sampling (MCMC)** — Markov Chain Monte Carlo with burn-in; converges to exact posterior
- **Structure Learning** — Independence testing via $D(A,B) = |P(A,B) - P(A)P(B)|$ and mutual information analysis
- **Data Preprocessing Pipeline** — Column validation, missing data analysis, rare category detection, row filtering
- **Comprehensive Evaluation:**
  - Top-k accuracy (k=1, 3, 5)
  - Per-class precision, recall, F1 classification report
  - k-fold cross-validation
  - Three-way inference method comparison
  - Calibration analysis (ECE + reliability diagrams)
- **Chronological Train/Test Split** — Methodologically correct temporal splitting (no future data leakage)
- **Publication-Quality Visualizations** — Network DAG, confusion matrix, calibration diagram, dependency heatmap
- **CLI with Multiple Modes** — Train/evaluate, predict, cross-validate, dependency analysis, optional plotting

---

## Model Architecture

### Network Structure

The Bayesian Network is organized into four layers:

```
Court Information
    └── chief_justice

Case Properties
    ├── issue_area
    ├── law_type
    ├── case_supplement
    └── lower_court_disposition

Intermediate Nodes
    ├── decision_type
    ├── precedent_alteration
    ├── split_vote
    └── unconstitutionality

Final Outcome
    └── final_disposition
```

### Conditional Probability Tables

For **root nodes** (no parents):

$$P(X = x) = \frac{\text{Count}(X = x)}{\text{Total cases}}$$

For **child nodes** (with parents):

$$P(X = x \mid \text{Parents}(X)) = \frac{\text{Count}(X = x,\ \text{Parents}(X))}{\text{Count}(\text{Parents}(X))}$$

CPTs are stored as nested dictionaries:
- Root variables: `{value: probability}`
- Child variables: `{(parent1_val, parent2_val, ..., child_val): probability}`

---

## Outcome Classes

The model predicts among 11 possible Supreme Court dispositions:

| Code | Disposition |
|---|---|
| 1 | Stay, petition, or motion granted |
| 2 | Affirmed (includes modified) |
| 3 | Reversed |
| 4 | Reversed and remanded |
| 5 | Vacated and remanded |
| 6 | Affirmed and reversed (or vacated) in part |
| 7 | Affirmed and reversed in part and remanded |
| 8 | Vacated |
| 9 | Petition denied or appeal dismissed |
| 10 | Certification to or from a lower court |

---

## Results

| Metric | Value |
|---|---|
| Top-3 Accuracy | **73.3%** (22/30 cases) |
| Evaluation Method | Rejection Sampling |
| Sample Size | 30 observations per run |
| Outcome Classes | 11 |

The model's 73% Top-3 accuracy represents a meaningful improvement over the ~27% baseline one would achieve by randomly selecting 3 of 11 classes.

**Most probable outcomes (base rate):**

| Disposition | Probability |
|---|---|
| Affirmed | 28.6% |
| Reversed and remanded | 26.5% |
| Reversed | 21.5% |
| Vacated and remanded | 11.2% |

---

## Tech Stack

- **Language:** Python 3.9+
- **Dataset:** [The Supreme Court Database](http://scdb.wustl.edu/documentation.php) — Washington University Law, 2024
- **Core Libraries:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Testing:** pytest, pytest-cov
- **Build:** pyproject.toml (PEP 621)
- **Inference:** Rejection Sampling, Likelihood Weighting, Gibbs Sampling (MCMC)
- **Logging:** Python stdlib `logging`

---

## Project Structure

```
dictm/
├── data/
│   └── scdb_cases.csv              # Supreme Court Database
├── src/
│   ├── __init__.py                 # Package exports
│   ├── network_structure.py        # Graph definition & variable relationships
│   ├── cpt_builder.py              # Vectorized CPT construction with Laplace smoothing
│   ├── inference.py                # Rejection Sampling, Likelihood Weighting, Gibbs (MCMC)
│   ├── evaluate.py                 # Top-k, classification report, k-fold CV, calibration
│   ├── structure_learning.py       # Independence tests & mutual information
│   ├── preprocessing.py            # Data validation, cleaning, missing data analysis
│   └── visualize.py                # Publication-quality plots
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── test_cpt_builder.py         # CPT construction & serialization tests
│   ├── test_inference.py           # All three inference engine tests
│   ├── test_evaluate.py            # Evaluation metric tests
│   └── test_structure_learning.py  # Dependency analysis tests
├── figures/                        # Generated visualizations (--visualize)
├── notebook/
│   └── exploration.ipynb           # Data exploration & CPT visualization
├── main.py                         # CLI entry point
├── pyproject.toml                  # Build config & pytest settings
├── requirements.txt
└── README.md
```

---

## Usage

```bash
# Train and evaluate (chronological split, default)
python main.py --mode train_eval

# Train with random split instead
python main.py --mode train_eval --split random

# Train and generate visualizations
python main.py --mode train_eval --visualize

# Interactive prediction
python main.py --mode predict

# k-fold cross-validation
python main.py --mode cross_validate

# Analyze variable dependencies
python main.py --mode dependency

# Enable debug logging
python main.py --mode train_eval --verbose

# Run tests
pytest
```

---

## References

- Washington University Law. (2024). *The Supreme Court Database*. Retrieved from [http://scdb.wustl.edu/documentation.php](http://scdb.wustl.edu/documentation.php)