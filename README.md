# dictm: Supreme Court Decision Prediction — Bayesian Network Model

> A probabilistic model of U.S. Supreme Court case outcomes, built on the Supreme Court Database. The name **"dictm"** comes from the Latin *obiter dictum* — the incidental remarks a judge makes in an opinion.

---

## Overview

Can a Supreme Court case's outcome be predicted from its properties before the justices deliberate?

This project builds a hand-structured **Bayesian Network** over historical SCDB data, learns its conditional probability tables by counting, and answers that question with exact posterior inference. It also — deliberately — measures itself against the baselines that make the answer interpretable.

The short version of the finding:

- Using variables recorded **from** the decision, the network reaches **81.7% Top-3** accuracy, beating a no-features baseline at 76.1% with non-overlapping confidence intervals. But those variables are not available before the ruling, so this is a *description* of the Court's record, not a forecast.
- Using only **pre-decision** information, the network reaches **76.1% Top-3** — statistically indistinguishable from the baseline that ignores every feature, and slightly *worse* on log-loss.

That second result is the honest headline. Most of what looks like predictive power in a model like this comes from the base rate: the Court affirms, reverses, or reverses-and-remands in roughly 80% of cases, so naming those three outcomes every single time is already a strong strategy. This repository is built to make that comparison unavoidable rather than easy to omit.

---

## What's here

- **Four inference engines** behind one interface — exact variable elimination plus rejection sampling, likelihood weighting, and Gibbs (MCMC). The exact engine doubles as ground truth: the samplers are *validated* against it, not merely compared to each other.
- **Two evidence policies over one graph** — `explanatory` (everything observable) and `ex_ante` (pre-decision only). Switching between them changes nothing but which variables are supplied as evidence.
- **Hierarchical backoff CPTs** — Jelinek–Mercer interpolation toward lower-order estimates, because 57% of the parent configurations this network needs were never observed in training.
- **Four reference models** — marginal, majority-class, logistic regression, gradient boosting — all implementing the same `query` interface, so they drop into the same evaluation path as the network.
- **Proper scoring rules with confidence intervals** — log-loss, Brier, macro-AUC, ECE and top-k, each with a bootstrap 95% CI, computed from a single cached inference pass.
- **Score-based structure learning** — BIC hill-climbing with add/remove/reverse moves, compared against the hand-crafted DAG on held-out log-likelihood.
- **Corrected association measures** — normalised mutual information and a G-test with proper degrees of freedom, which raw MI's cardinality bias otherwise distorts.
- **A web demo, a runnable notebook, and CI** on Python 3.10–3.13.

---

## Results

Chronological split — trained on the 7,312 earliest cases, tested on the 1,829 most recent. Brackets are bootstrap 95% confidence intervals.

### Multi-class disposition (11 classes)

**Explanatory track** — all observable variables, including several recorded from the decision:

| Model | Top-1 | Top-3 | Top-5 | Log-loss | Brier | ECE | AUC |
|---|---|---|---|---|---|---|---|
| Marginal (no features) | 26.7 [24.4, 28.6] | 76.1 [74.5, 77.9] | 97.1 | 1.6244 | 0.7777 | 0.042 | 0.500 |
| Majority class | 26.7 [24.4, 28.6] | 39.3 [37.1, 41.6] | 94.8 | 5.0015 | 1.4502 | 0.723 | 0.500 |
| Logistic regression | **34.2** [32.0, 36.3] | 81.4 [79.7, 83.0] | **98.3** | **1.5302** | **0.7474** | 0.044 | **0.718** |
| Gradient boosting | 33.6 [31.6, 35.5] | 80.5 [78.8, 82.1] | 97.9 | 1.6187 | 0.7593 | 0.095 | 0.658 |
| **Bayesian Network** | 32.6 [30.4, 34.5] | **81.7** [80.0, 83.4] | 98.0 | 1.5920 | 0.7495 | 0.050 | 0.553 |

**Ex-ante track** — only information knowable before the Court rules:

| Model | Top-1 | Top-3 | Top-5 | Log-loss | Brier | ECE | AUC |
|---|---|---|---|---|---|---|---|
| Marginal (no features) | 26.7 [24.4, 28.6] | 76.1 [74.5, 77.9] | 97.1 | **1.6244** | 0.7777 | 0.042 | 0.500 |
| Logistic regression | 29.4 [27.3, 31.2] | 77.9 [76.2, 79.6] | **97.9** | 1.6116 | **0.7749** | 0.077 | **0.673** |
| Gradient boosting | **30.2** [28.2, 32.5] | **78.5** [76.8, 80.0] | 97.2 | 1.6901 | 0.7822 | 0.090 | 0.659 |
| **Bayesian Network** | 27.0 [24.7, 29.0] | 76.1 [74.5, 77.9] | 97.8 | 1.6339 | 0.7764 | 0.064 | 0.520 |

### Binary affirm / reverse (n = 1,729)

The standard formulation in the literature. Affirm = disposition 2; reverse/vacate = 3, 4, 5, 8; other dispositions excluded.

| Model | Track | Accuracy | Log-loss | AUC |
|---|---|---|---|---|
| Marginal (no features) | — | 71.8 [69.6, 73.7] | 0.6015 | 0.500 |
| Bayesian Network | explanatory | **72.6** [70.4, 74.7] | **0.5703** | 0.599 |
| Logistic regression | explanatory | 72.1 [69.7, 74.1] | 0.5644 | **0.663** |
| Bayesian Network | ex_ante | 71.8 [69.6, 73.7] | 0.6036 | 0.561 |
| Gradient boosting | ex_ante | 69.7 [67.5, 71.7] | 0.6042 | 0.572 |

**On comparison with published work.** Katz, Bommarito & Blackman (2017) report ~70.2% case-level accuracy on the binary task using time-evolving random forests. The numbers above are *not* directly comparable — they use a far larger feature set and a different evaluation protocol. The more useful observation is that always predicting "reversed" scores **71.8%** on this test set, because the Court reverses far more often than it affirms. Any binary accuracy figure in this literature, including this project's, should be read against that floor.

### What the AUC column says

AUC is above 0.5 everywhere, including the ex-ante track (0.520–0.673). There *is* signal in pre-decision variables — the models rank cases better than chance. It simply is not enough to change which class ends up on top, because one class dominates. Reporting accuracy alone would hide this; reporting AUC and log-loss alongside it does not.

---

## Why the ex-ante model fails: a structural diagnosis

The result above is not a bug, and the network structure explains it precisely. Of the four pre-decision variables, only `chief_justice` is a direct parent of the outcome:

| Ex-ante variable | Direct parent of `final_disposition`? | Reaches the outcome via |
|---|---|---|
| `chief_justice` | yes | — |
| `issue_area` | no | `decision_type`, `precedent_alteration`, `split_vote` |
| `law_type` | no | `decision_type`, `split_vote`, `unconstitutional` |
| `lower_court_disposition` | no | `precedent_alteration` only |

Every other ex-ante variable reaches `final_disposition` **only through the four intermediate nodes** — and all four are post-decision. When they are unobserved, they act as a bottleneck that washes the signal out.

`lower_court_disposition` is the sharpest case. Whether the court below affirmed or reversed is plausibly the single most informative thing you can know before the ruling, and it is genuinely associated with the outcome in the data (G = 352.4, df = 99, p ≈ 4 × 10⁻³⁰). But its only path to the outcome runs through `precedent_alteration`, a near-constant binary. Conditioning on it moves the posterior by a total-variation distance of about 0.001. The information is present in the data and the graph cannot carry it.

This is pinned by a test (`test_lower_court_disposition_is_structurally_bottlenecked`) so that anyone who adds a direct edge will see the expectation fail and know why.

---

## Structure: hand-crafted vs learned

`--mode structure` runs BIC-scored hill-climbing from the empty graph and compares the result to the domain-informed DAG.

| | BIC | Held-out log-likelihood | Edges |
|---|---|---|---|
| Hand-crafted (domain) | −82,533.75 | −15,088.07 | 13 |
| Learned (BIC search) | **−56,830.93** | **−14,315.70** | 11 |

The learned graph wins on BIC — which it was optimising, so that is expected — but it also wins on **held-out log-likelihood**, which it was not. The two graphs share only **3 of 13 edges**. The search adds `lawType → caseDisposition` and drops most of the hand-specified parents of the outcome, including `decisionType`, `chief`, `precedentAlteration` and `declarationUncon`.

Read plainly: on this dataset, the domain-informed structure is worse than one derived from the data, and the intermediate-node layer is largely not earning its place.

---

## Inference

Four engines, all behind one interface (`query` / `top_k_predictions`).

| Engine | Top-3 | Time (200 cases) | TV distance from exact |
|---|---|---|---|
| **Exact (variable elimination)** | 87.0% | **0.02 s** | — (reference) |
| Rejection sampling | 55.5% | 28.56 s | 0.5477 |
| Likelihood weighting | 87.0% | 8.27 s | 0.0257 |
| Gibbs sampling (MCMC) | 87.0% | 13.38 s | 0.0232 |

Adding exact inference changed what the other three are *for*. Previously the samplers could only be compared to one another, which cannot distinguish "all three are correct" from "all three share a bug". With a known answer available, they can be validated:

```
Engine                           100       500      1000      5000     (samples)
──────────────────────────────────────────────────────────────────
Rejection Sampling            0.5486    0.2546    0.1514    0.2276   ← varies per run
Likelihood Weighting          0.1134    0.0354    0.0164    0.0132
Gibbs Sampling (MCMC)         0.1045    0.0376    0.0226    0.0136
```

Total-variation distance from the exact posterior. Likelihood weighting and Gibbs
converge monotonically and reproduce exactly on every run. **Rejection sampling does
neither**, and the row above is one observed run rather than a fixed result — repeated
invocations of `--mode convergence` have produced `0.5486 / 0.8192 / 0.8192 / 0.1514`
and `0.5486 / 0.5486 / 0.9087 / 0.2269` from the same seed.

That instability is itself the finding. `CPTBuilder.get_values` returns a **set**, so
nodes with string values iterate in an order that Python's hash randomisation varies
per process. Which index a given RNG draw maps to therefore changes between runs. The
weighted samplers are unaffected — they aggregate over every value — but rejection
sampling accepts only a handful of samples out of thousands, so the ordering decides
the answer. Sorting the value sets would make it deterministic, at the cost of
remapping every sampler's draws and shifting the numbers in the tables above.

Rejection sampling's collapse on the explanatory track (TV ≈ 0.55, and 1,400× slower
than exact) is the textbook failure mode made measurable: with nine evidence variables,
almost every sample is discarded — and what survives is too few to be stable.

Exact inference is also the right default here. When all of a leaf's parents are observed, the posterior is a single CPT row — and the sampling engines were spending 1,000 draws to approximate a dictionary lookup.

---

## Model architecture

### Network structure

```
Court Information         chief_justice
                              │
Case Properties           issue_area · law_type · case_supplement · lower_court_disposition
                              │
Intermediate Nodes        decision_type · precedent_alteration · split_vote · unconstitutional
                              │
Final Outcome             final_disposition
```

### Conditional probability tables

Root nodes use a Laplace-smoothed marginal. Child nodes use **hierarchical backoff** (Jelinek–Mercer interpolation), writing `pa_k` for the first *k* parents:

$$P_k(x \mid pa_k) = \lambda \cdot P_{ML}(x \mid pa_k) + (1 - \lambda) \cdot P_{k-1}(x \mid pa_{k-1}), \qquad \lambda = \frac{N(pa_k)}{N(pa_k) + m}$$

A configuration seen far more than *m* times is trusted almost entirely; a rare one is pulled toward an estimate conditioning on fewer parents.

This matters more than it might sound. `final_disposition` has five parents, and **57.2% of the parent configurations the network needs were never observed in training**. The earlier implementation answered every one of those with a *uniform* distribution — the least informative guess available — and did so silently. `backoff_rate` now reports that figure on every run.

### A note on `splitVote`

SCDB ships a column named `splitVote`, and it is natural to read it as "the justices were divided". It is not: it is an internal flag for cases whose votes span multiple records, and it is **constant (= 1) across all 9,277 rows**. A node fed by a constant carries no information. The pipeline therefore derives `voteSplit = (minVotes > 0)` instead, which splits 5,733 / 3,544.

---

## Outcome classes

| Code | Disposition | | Code | Disposition |
|---|---|---|---|---|
| 1 | Stay, petition, or motion granted | | 7 | Affirmed and reversed in part and remanded |
| 2 | Affirmed (includes modified) | | 8 | Vacated |
| 3 | Reversed | | 9 | Petition denied or appeal dismissed |
| 4 | Reversed and remanded | | 10 | Certification to or from a lower court |
| 5 | Vacated and remanded | | 11 | No disposition |
| 6 | Affirmed and reversed (or vacated) in part | | | |

Base rates on the full dataset: Affirmed 30.1%, Reversed and remanded 27.5%, Reversed 22.0%, Vacated and remanded 12.7%. Those top three sum to **79.6%**, which is why Top-3 accuracy is a far less impressive metric than it looks.

---

## Installation

Requires **Python 3.10+** (the codebase uses PEP 604 unions in annotations).

```bash
git clone <repo> && cd dictm
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt  # everything: runtime + tests + demo + notebook
# or, minimally:
pip install -e .                     # runtime only
pip install -e ".[dev]"              # + pytest, pytest-cov, ruff
pip install -e ".[serve]"            # + fastapi, uvicorn

python scripts/fetch_data.py         # downloads SCDB into data/
```

The dataset is not vendored — it is redistributed by Washington University Law under their own terms. `scripts/fetch_data.py` pulls the 2024_01 case-centered release by default; pass `--release 2023_01` for an earlier one. The CLI then picks up the newest matching CSV in `data/` automatically.

---

## Usage

```bash
# Train, then compare against every baseline on both evidence tracks
python main.py --mode train_eval

# Only the honest pre-decision track, on the binary affirm/reverse task
python main.py --mode train_eval --track ex_ante --task binary

# Hand-crafted vs learned network structure
python main.py --mode structure

# Sampler convergence to the exact posterior
python main.py --mode convergence

# Interactive single-case prediction (uses the saved model)
python main.py --mode predict

# Evaluate a saved model; k-fold CV; pairwise association analysis
python main.py --mode eval
python main.py --mode cross_validate
python main.py --mode dependency

# Write plots to figures/
python main.py --mode train_eval --visualize
```

### Options

| Flag | Default | Purpose |
|---|---|---|
| `--mode` | `train_eval` | `train_eval`, `predict`, `eval`, `cross_validate`, `dependency`, `structure`, `convergence` |
| `--track` | `both` | `explanatory`, `ex_ante`, or `both` — which evidence policy to evaluate |
| `--task` | `multiclass` | `multiclass` (11 dispositions) or `binary` (affirm vs reverse/vacate) |
| `--split` | `chronological` | `chronological` or `random` |
| `--test-frac` | `0.2` | Fraction held out for testing |
| `--alpha` | `1.0` | Laplace smoothing strength |
| `--no-backoff` | off | Disable hierarchical backoff (reverts to per-configuration Laplace) |
| `--n-boot` | `500` | Bootstrap resamples for confidence intervals |
| `--n-folds` | `5` | Folds for `cross_validate` |
| `--max-parents` | `4` | In-degree cap for structure search |
| `--method-compare-n` | `200` | Test cases used when timing the sampling engines |
| `--visualize` | off | Write figures to `figures/` |
| `--verbose` | off | Debug logging |
| `--data_path`, `--model_path` | auto | Override dataset / saved-model locations |

### Development

```bash
pytest                                       # 201 tests
pytest --cov=src --cov=app                   # 90% coverage
ruff check src app tests main.py scripts     # lint
```

CI runs lint, tests with a 75% coverage gate on Python 3.10–3.13, and a packaging check. The suite runs entirely on synthetic fixtures, so it needs no SCDB download.

### Web demo

```bash
pip install -e ".[serve]"
python main.py --mode train_eval    # produces data/cpts.json
uvicorn app.server:app              # or: dictm-serve
```

A single self-contained page at `/` (no build step, no CDN), plus `GET /api/schema` and `POST /api/predict`. Inference is exact, so identical requests return identical answers. Selecting the ex-ante track causes post-decision variables to be *dropped* rather than quietly used — the response lists exactly which fields it ignored.

---

## Project structure

```
dictm/
├── main.py                      CLI — 7 modes
├── pyproject.toml               Build config, extras, pytest + ruff settings
├── requirements.txt             Runtime deps (mirrors pyproject)
├── requirements-dev.txt         Full toolchain: tests, lint, demo, notebook
├── src/
│   ├── network_structure.py     Graph, evidence policies, binary task mapping
│   ├── preprocessing.py         Validation, derived columns, missing-data analysis
│   ├── cpt_builder.py           Vectorised CPT construction + hierarchical backoff
│   ├── inference.py             Rejection sampling, likelihood weighting, Gibbs
│   ├── exact.py                 Variable elimination — the ground truth
│   ├── baselines.py             Marginal, majority, logistic regression, gradient boosting
│   ├── evaluate.py              Metrics, bootstrap CIs, cross-validation, convergence
│   ├── structure_learning.py    BIC hill-climbing, normalised MI, G-test
│   └── visualize.py             Figures
├── app/server.py                FastAPI + self-contained HTML page
├── scripts/fetch_data.py        SCDB download
├── tests/                       201 tests, 90% coverage
│   ├── conftest.py              Synthetic SCDB-shaped fixtures with real signal
│   ├── test_exact.py            Exact inference + sampler convergence against it
│   ├── test_metrics.py          Scoring rules checked against hand-computed values
│   ├── test_baselines.py        Reference models
│   ├── test_structure_search.py Hill-climbing, NMI, G-test
│   └── ...                      CPTs, inference, preprocessing, server, plots, CLI
├── notebooks/exploration.ipynb  Runnable walkthrough of the whole pipeline
├── data/                        SCDB CSV (downloaded) + saved CPTs — both gitignored
├── figures/                     Generated plots (gitignored)
├── Paper.pdf                    Original write-up
└── .github/workflows/ci.yml     Lint + tests (3.10–3.13) + packaging check
```

---

## Methodology notes

- **Chronological split.** Train on earlier terms, test on the most recent 20%. Never train on future cases to predict past ones.
- **Two evidence policies, one graph.** `explanatory` and `ex_ante` differ only in which variables are supplied as evidence; the DAG and the CPTs are identical. In `ex_ante` mode the intermediate nodes become unobserved and the posterior marginalises over them.
- **Proper scoring rules.** Log-loss and Brier are reported alongside accuracy because they cannot be gamed by predicting the marginal — which, on this task, is exactly the failure mode worth guarding against.
- **Confidence intervals everywhere.** Every point estimate carries a bootstrap 95% CI. The project's earlier headline figure came from a 30-case test set, where the interval is roughly ±16 points.

## Known limitations

- The ex-ante feature set is thin. SCDB carries other genuinely pre-decision fields — `certReason`, `jurisdiction`, `lcDisagreement`, `caseSource` — that are not currently in the graph and would be the natural next extension.
- The intermediate-node layer is not supported by the structure search, and the outcome node's five-parent family is sparse enough that 57% of its configurations require backoff.
- Case-level modelling only. The literature's stronger results come from justice-level vote prediction aggregated to a case outcome, which SCDB's justice-centered release supports.

---

## References

- Washington University Law. (2024). *The Supreme Court Database*. http://scdb.wustl.edu
- Katz, D. M., Bommarito, M. J., & Blackman, J. (2017). A general approach for predicting the behavior of the Supreme Court of the United States. *PLoS ONE*, 12(4).
- Russell, S., & Norvig, P. *Artificial Intelligence: A Modern Approach*, Ch. 14 — sampling methods for Bayesian networks.
