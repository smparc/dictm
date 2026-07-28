# Data

The SCDB dataset is not vendored in this repository — it is redistributed by
Washington University Law under their own terms.

## Download

```bash
python scripts/fetch_data.py               # latest known release (2024_01)
python scripts/fetch_data.py --release 2023_01
```

Or fetch it manually from http://scdb.wustl.edu/data.php, selecting
**Case Centered Data | Citation**, and save the CSV here as
`SCDB_<release>_caseCentered_Citation.csv`. The CLI picks up the newest
matching file in this directory automatically.

`cpts.json` — the trained model — is written here by
`python main.py --mode train_eval`. Both are gitignored.

## Columns used

| Column | Node | Description |
|---|---|---|
| `chief` | `chief_justice` | Chief Justice presiding |
| `issueArea` | `issue_area` | Area of law (1–14) |
| `lawType` | `law_type` | Type of legal provision |
| `caseDispositionUnusual` | `case_supplement` | Unusual disposition flag |
| `lcDisposition` | `lower_court_disposition` | Lower court disposition (1–12) |
| `decisionType` | `decision_type` | Type of decision |
| `precedentAlteration` | `precedent_alteration` | Whether precedent was altered |
| `minVotes` | → `voteSplit` | Dissenting votes; see below |
| `declarationUncon` | `unconstitutional` | Unconstitutionality declared (1–4) |
| `caseDisposition` | `final_disposition` | **Target** — final disposition (1–11) |

## A warning about `splitVote`

SCDB has a column named `splitVote`, and it reads as though it means "the
justices were divided". It does not. It is an internal flag marking cases whose
votes were split across multiple records, and in the 2024_01 release it is
**constant — every one of the 9,277 rows has the value 1**.

A node driven by a constant contributes nothing to the network, so
`preprocessing.add_derived_columns` builds `voteSplit = (minVotes > 0)` instead,
which is the real dissent indicator (5,733 split vs 3,544 unanimous). The
`split_vote` node maps to `voteSplit`, not to `splitVote`.

## Other pre-decision fields

These are present in the dataset and genuinely knowable before a ruling, but are
not currently in the network. They are the most promising direction for
strengthening the ex-ante track:

`certReason`, `jurisdiction`, `lcDisagreement`, `caseSource`, `petitioner`,
`respondent`, `adminAction`, `threeJudgeFdc`.
