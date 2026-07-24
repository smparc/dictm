"""
network_structure.py
--------------------
Defines the Bayesian Network structure for Supreme Court case prediction.


Network layers
--------------
Layer 1 — Court Information
    chief_justice


Layer 2 — Case Properties (root nodes)
    issue_area, law_type, case_supplement, lower_court_disposition


Layer 3 — Intermediate Nodes (children of case properties)
    decision_type        <- issue_area, law_type
    precedent_alteration <- issue_area, lower_court_disposition
    split_vote           <- issue_area, law_type
    unconstitutional     <- law_type, case_supplement


Layer 4 — Final Outcome
    final_disposition    <- decision_type, precedent_alteration,
                            split_vote, unconstitutional, chief_justice
"""


from dataclasses import dataclass, field



@dataclass
class Node:
    """Represents a single variable (node) in the Bayesian Network."""
    name: str
    parents: list = field(default_factory=list)
    description: str = ""


    def is_root(self) -> bool:
        return len(self.parents) == 0



# ---------------------------------------------------------------------------
# Node definitions
# ---------------------------------------------------------------------------


NODES = {
    # ── Layer 1: Court Information ──────────────────────────────────────────
    "chief_justice": Node(
        name="chief_justice",
        parents=[],
        description="Chief Justice presiding over the case",
    ),


    # ── Layer 2: Case Properties (root nodes) ───────────────────────────────
    "issue_area": Node(
        name="issue_area",
        parents=[],
        description="Area of law at issue (civil rights, criminal procedure, etc.)",
    ),
    "law_type": Node(
        name="law_type",
        parents=[],
        description="Type of law involved (constitutional, statutory, etc.)",
    ),
    "case_supplement": Node(
        name="case_supplement",
        parents=[],
        description="Supplemental case context",
    ),
    "lower_court_disposition": Node(
        name="lower_court_disposition",
        parents=[],
        description="How the lower court disposed of the case",
    ),


    # ── Layer 3: Intermediate Nodes ──────────────────────────────────────────
    "decision_type": Node(
        name="decision_type",
        parents=["issue_area", "law_type"],
        description="Type of Supreme Court decision",
    ),
    "precedent_alteration": Node(
        name="precedent_alteration",
        parents=["issue_area", "lower_court_disposition"],
        description="Whether existing precedent is altered",
    ),
    "split_vote": Node(
        name="split_vote",
        parents=["issue_area", "law_type"],
        description="Whether the justices split on the decision",
    ),
    "unconstitutional": Node(
        name="unconstitutional",
        parents=["law_type", "case_supplement"],
        description="Whether the case involves an unconstitutionality finding",
    ),


    # ── Layer 4: Final Outcome ───────────────────────────────────────────────
    "final_disposition": Node(
        name="final_disposition",
        parents=[
            "decision_type",
            "precedent_alteration",
            "split_vote",
            "unconstitutional",
            "chief_justice",
        ],
        description="Final Supreme Court case disposition (the prediction target)",
    ),
}


# Topological order for CPT construction and inference
TOPOLOGICAL_ORDER = [
    "chief_justice",
    "issue_area",
    "law_type",
    "case_supplement",
    "lower_court_disposition",
    "decision_type",
    "precedent_alteration",
    "split_vote",
    "unconstitutional",
    "final_disposition",
]


# Column names in the SCDB dataset that map to our node names
COLUMN_MAP = {
    "chief_justice":           "chief",
    "issue_area":              "issueArea",
    "law_type":                "lawType",
    "case_supplement":         "caseDispositionUnusual",
    "lower_court_disposition": "lcDisposition",
    "decision_type":           "decisionType",
    "precedent_alteration":    "precedentAlteration",
    "split_vote":              "splitVote",
    "unconstitutional":        "declarationUncon",
    "final_disposition":       "caseDisposition",
}


# Human-readable labels for final_disposition values
DISPOSITION_LABELS = {
    1: "Stay, petition, or motion granted",
    2: "Affirmed (includes modified)",
    3: "Reversed",
    4: "Reversed and remanded",
    5: "Vacated and remanded",
    6: "Affirmed and reversed (or vacated) in part",
    7: "Affirmed and reversed in part and remanded",
    8: "Vacated",
    9: "Petition denied or appeal dismissed",
    10: "Certification to or from a lower court",
}