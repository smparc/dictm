# Supreme Court Decision Prediction — Bayesian Network Model

> A probabilistic model that predicts U.S. Supreme Court case outcomes using Bayesian inference over historical case data.

---

## Overview

Can the outcome of a Supreme Court case be predicted from its properties before the justices deliberate? This project builds a **Bayesian Network** trained on historical Supreme Court data to model the probabilistic relationships between case characteristics and their final dispositions.

By encoding legal domain structure into a directed graphical model and learning conditional probabilities from data, the system achieves **73% Top-3 accuracy** across 11 possible outcome classes — well above the ~27% expected by random chance.

---

## Features

- **Bayesian Network with Hand-Crafted Structure** — Domain-informed graph modeling relationships between case properties, court context, and outcomes
- **Conditional Probability Table (CPT) Learning** — Probabilities computed directly from historical case frequencies
- **Rejection Sampling Inference** — Probabilistic inference over the network given observed case variables
- **Top-3 Accuracy Evaluation** — Measures whether the true outcome appears among the three highest-probability predictions
- **Full Outcome Distribution** — Returns a probability distribution across all 11 possible Supreme Court dispositions

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

- **Language:** Python
- **Dataset:** [The Supreme Court Database](http://scdb.wustl.edu/documentation.php) — Washington University Law, 2024
- **Core Libraries:** NumPy, Pandas
- **Visualization:** GPT-assisted layout + Matplotlib

---

## Project Structure

```
supreme-court-predictor/
├── data/
│   └── scdb_cases.csv            # Supreme Court Database
├── src/
│   ├── network_structure.py      # Graph definition & variable relationships
│   ├── cpt_builder.py            # Conditional probability table construction
│   ├── inference.py              # Rejection sampling implementation
│   └── evaluate.py               # Top-3 accuracy evaluation
├── notebooks/
│   └── exploration.ipynb         # Data exploration & CPT visualization
└── README.md
```

---

## References

- Washington University Law. (2024). *The Supreme Court Database*. Retrieved from [http://scdb.wustl.edu/documentation.php](http://scdb.wustl.edu/documentation.php)
