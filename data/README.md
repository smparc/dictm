# Data

Place the SCDB dataset CSV here before running the project.

## Download

Dataset: **The Supreme Court Database (Case Centered)**  
Source: http://scdb.wustl.edu/data.php  
Select: `Case Centered Data | Citation`

Save the file as:
```
data/SCDB_2023_01_caseCentered_Citation.csv
```

## Key Columns Used

| Column | Description |
|---|---|
| `chief` | Chief Justice |
| `issueArea` | Area of law (1–14) |
| `lawType` | Type of law |
| `caseDispositionUnusual` | Supplemental case context |
| `lcDisposition` | Lower court disposition |
| `decisionType` | Type of decision |
| `precedentAlteration` | Whether precedent was altered |
| `splitVote` | Whether it was a split vote |
| `declarationUncon` | Unconstitutionality declared |
| `caseDisposition` | **Target variable** — final disposition |
