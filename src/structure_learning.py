"""
structure_learning.py
---------------------
Automated structure learning and dependency analysis for the Bayesian Network.


Implements the dependency test from the paper:
    D(A, B) = |P(A, B) - P(A)P(B)|


Where D(A,B) represents the distance from perfect independence. Variable pairs
with D above a threshold are considered dependent and form edges in the network.


Additionally computes mutual information I(A; B) as a more theoretically
grounded measure of statistical dependence.
"""


import itertools
import logging

import numpy as np
import pandas as pd

from src.network_structure import COLUMN_MAP, NODES

log = logging.getLogger(__name__)



def compute_dependency_score(
    df: pd.DataFrame,
    var_a: str,
    var_b: str,
) -> float:
    """
    Compute D(A,B) = |P(A,B) - P(A)P(B)| averaged over all value pairs.


    This is the independence test described in the paper (Equation 1).
    A value of 0 indicates perfect independence; higher values indicate
    stronger dependence.


    Parameters
    ----------
    df    : DataFrame with columns var_a and var_b
    var_a : column name for variable A
    var_b : column name for variable B


    Returns
    -------
    float — average absolute deviation from independence
    """
    sub = df[[var_a, var_b]].dropna()
    n = len(sub)
    if n == 0:
        return 0.0


    # Marginal probabilities
    p_a = sub[var_a].value_counts(normalize=True).to_dict()
    p_b = sub[var_b].value_counts(normalize=True).to_dict()


    # Joint probabilities
    joint = sub.groupby([var_a, var_b]).size() / n
    joint_dict = joint.to_dict()


    # Average |P(A,B) - P(A)P(B)| over all observed value pairs
    total_dev = 0.0
    n_pairs = 0
    for (a_val, b_val), p_ab in joint_dict.items():
        p_a_val = p_a.get(a_val, 0)
        p_b_val = p_b.get(b_val, 0)
        total_dev += abs(p_ab - p_a_val * p_b_val)
        n_pairs += 1


    return total_dev / max(1, n_pairs)



def compute_mutual_information(
    df: pd.DataFrame,
    var_a: str,
    var_b: str,
) -> float:
    """
    Compute mutual information I(A; B) = Σ P(a,b) log[P(a,b) / (P(a)P(b))].


    A more theoretically grounded measure of statistical dependence than
    the simple deviation score. Always non-negative; 0 iff A ⊥ B.


    Parameters
    ----------
    df    : DataFrame
    var_a : column name
    var_b : column name


    Returns
    -------
    float — mutual information in nats
    """
    sub = df[[var_a, var_b]].dropna()
    n = len(sub)
    if n == 0:
        return 0.0


    p_a = sub[var_a].value_counts(normalize=True).to_dict()
    p_b = sub[var_b].value_counts(normalize=True).to_dict()
    joint = sub.groupby([var_a, var_b]).size() / n


    mi = 0.0
    for (a_val, b_val), p_ab in joint.items():
        p_a_val = p_a.get(a_val, 0)
        p_b_val = p_b.get(b_val, 0)
        if p_ab > 0 and p_a_val > 0 and p_b_val > 0:
            mi += p_ab * np.log(p_ab / (p_a_val * p_b_val))


    return max(0.0, mi)



def compute_normalized_mutual_information(
    df: pd.DataFrame,
    var_a: str,
    var_b: str,
) -> float:
    """
    Normalised mutual information, I(A;B) / sqrt(H(A) * H(B)).

    Raw mutual information grows with the number of categories a variable has,
    so comparing I(A;B) across pairs of different cardinality is misleading —
    `issueArea` (14 values) will outrank `precedentAlteration` (2 values) partly
    because it is finer-grained, not because it is more informative. Dividing by
    the geometric mean of the marginal entropies puts every pair on a [0, 1]
    scale where 0 is independence and 1 is a deterministic relationship.

    Returns
    -------
    float in [0, 1]
    """
    sub = df[[var_a, var_b]].dropna()
    if len(sub) == 0:
        return 0.0

    mi = compute_mutual_information(df, var_a, var_b)

    def entropy(series: pd.Series) -> float:
        p = series.value_counts(normalize=True).to_numpy()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    h_a = entropy(sub[var_a])
    h_b = entropy(sub[var_b])
    if h_a <= 0 or h_b <= 0:
        return 0.0
    return float(min(1.0, mi / np.sqrt(h_a * h_b)))


def g_test(df: pd.DataFrame, var_a: str, var_b: str) -> dict:
    """
    Likelihood-ratio (G) test of independence between two categorical variables.

    G = 2 * N * I(A;B) in nats, asymptotically chi-squared with
    (|A| - 1)(|B| - 1) degrees of freedom. Unlike a bare dependency score, this
    says whether an observed association is larger than sampling noise would
    produce — the degrees-of-freedom correction is what stops high-cardinality
    variables from looking significant purely by having more cells.

    Returns
    -------
    dict with keys "g", "dof", "p_value", "n"
    """
    sub = df[[var_a, var_b]].dropna()
    n = len(sub)
    if n == 0:
        return {"g": 0.0, "dof": 0, "p_value": 1.0, "n": 0}

    mi = compute_mutual_information(df, var_a, var_b)
    g = 2.0 * n * mi
    dof = (sub[var_a].nunique() - 1) * (sub[var_b].nunique() - 1)

    p_value = 1.0
    if dof > 0:
        try:
            from scipy.stats import chi2

            p_value = float(chi2.sf(g, dof))
        except ImportError:  # scipy absent — report the statistic without a p-value
            p_value = float("nan")

    return {"g": round(g, 4), "dof": int(dof), "p_value": p_value, "n": int(n)}


def dependency_matrix(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    method: str = "deviation",
) -> pd.DataFrame:
    """
    Compute a pairwise dependency matrix for all variables.


    Parameters
    ----------
    df        : DataFrame with SCDB data
    variables : list of column names (defaults to COLUMN_MAP values)
    method    : "deviation" for D(A,B) or "mutual_info" for I(A;B)


    Returns
    -------
    pd.DataFrame — symmetric matrix of dependency scores
    """
    if variables is None:
        variables = [col for col in COLUMN_MAP.values() if col in df.columns]


    score_fn = compute_dependency_score if method == "deviation" else compute_mutual_information
    n = len(variables)
    matrix = np.zeros((n, n))


    for i, j in itertools.combinations(range(n), 2):
        score = score_fn(df, variables[i], variables[j])
        matrix[i, j] = score
        matrix[j, i] = score


    return pd.DataFrame(matrix, index=variables, columns=variables)



def find_top_dependencies(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    threshold: float = 0.01,
    method: str = "deviation",
    top_n: int = 20,
) -> list[tuple[str, str, float]]:
    """
    Find the strongest pairwise dependencies in the dataset.


    Parameters
    ----------
    df        : DataFrame
    variables : list of column names to compare (defaults to COLUMN_MAP values)
    threshold : minimum score to include
    method    : "deviation" or "mutual_info"
    top_n     : max edges to return


    Returns
    -------
    list of (var_a, var_b, score) sorted by score descending
    """
    if variables is None:
        variables = [col for col in COLUMN_MAP.values() if col in df.columns]
    score_fn = compute_dependency_score if method == "deviation" else compute_mutual_information


    edges = []
    for va, vb in itertools.combinations(variables, 2):
        score = score_fn(df, va, vb)
        if score >= threshold:
            edges.append((va, vb, round(score, 6)))


    edges.sort(key=lambda x: -x[2])
    return edges[:top_n]



# ---------------------------------------------------------------------------
# Score-based structure learning
# ---------------------------------------------------------------------------
#
# The measures above are pairwise: they say which variables are associated, not
# which directed graph best explains the data. Learning a structure means
# searching over DAGs, and that needs a score that trades goodness-of-fit
# against complexity. BIC does exactly that:
#
#     BIC(G) = sum_i [ LL(X_i | Pa(X_i)) - (log N / 2) * free_params(X_i) ]
#
# Higher is better. The penalty is what stops the search from simply connecting
# every node to every other node, which would always fit the training data best.


class BICScorer:
    """Decomposable BIC scorer with a per-family cache."""

    def __init__(self, data: pd.DataFrame, variables: list[str]):
        self.data = data[variables].dropna(how="all")
        self.variables = variables
        self.n = len(self.data)
        self.log_n = np.log(max(self.n, 1))
        self.cardinality = {v: max(1, self.data[v].nunique()) for v in variables}
        self._cache: dict = {}

    def family_score(self, node: str, parents: tuple) -> float:
        """BIC contribution of one node given its parents (cached)."""
        key = (node, tuple(sorted(parents)))
        if key in self._cache:
            return self._cache[key]

        cols = list(key[1]) + [node]
        sub = self.data[cols].dropna()
        if len(sub) == 0:
            self._cache[key] = -np.inf
            return -np.inf

        if not key[1]:
            counts = sub[node].value_counts().to_numpy(dtype=float)
            total = counts.sum()
            log_likelihood = float(np.sum(counts * np.log(counts / total)))
        else:
            joint = sub.groupby(cols, observed=True).size().to_numpy(dtype=float)
            parent_totals = sub.groupby(list(key[1]), observed=True).size()
            # Align each joint cell with its parent configuration's total.
            joint_index = sub.groupby(cols, observed=True).size().index
            parent_keys = [
                k[:-1] if isinstance(k, tuple) else (k,) for k in joint_index
            ]
            denominators = np.array(
                [parent_totals.loc[k if len(k) > 1 else k[0]] for k in parent_keys],
                dtype=float,
            )
            log_likelihood = float(np.sum(joint * np.log(joint / denominators)))

        n_parent_configs = 1
        for parent in key[1]:
            n_parent_configs *= self.cardinality[parent]
        free_params = (self.cardinality[node] - 1) * n_parent_configs

        score = log_likelihood - 0.5 * self.log_n * free_params
        self._cache[key] = score
        return score

    def graph_score(self, graph: dict) -> float:
        """Total BIC of a graph given as {node: [parents]}."""
        return sum(self.family_score(node, tuple(parents)) for node, parents in graph.items())


def _creates_cycle(graph: dict, source: str, target: str) -> bool:
    """Would adding source -> target create a cycle? (i.e. is target an ancestor of source?)"""
    stack = [source]
    seen = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, []))
    return False


def hill_climb_structure(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    max_parents: int = 4,
    max_iterations: int = 200,
    verbose: bool = False,
) -> dict:
    """
    Greedy hill-climbing search over DAGs, scored by BIC.

    Starts from the empty graph and repeatedly applies the single highest-scoring
    edge addition, deletion, or reversal, stopping when no move improves the
    score.

    Parameters
    ----------
    df          : DataFrame of SCDB data (raw column names)
    variables   : columns to include (defaults to the mapped node columns)
    max_parents : cap on in-degree; keeps CPTs estimable and the search tractable
    verbose     : log each accepted move

    Returns
    -------
    dict {"graph": {node: [parents]}, "score": float, "iterations": int,
          "history": [...]}
    """
    if variables is None:
        variables = [col for col in COLUMN_MAP.values() if col in df.columns]

    scorer = BICScorer(df, variables)
    graph: dict[str, list[str]] = {v: [] for v in variables}
    score = scorer.graph_score(graph)
    history = [round(score, 2)]

    for _ in range(max_iterations):
        best_move = None
        best_delta = 1e-10

        for target in variables:
            parents = graph[target]

            # --- add ---
            if len(parents) < max_parents:
                for source in variables:
                    if source == target or source in parents:
                        continue
                    if _creates_cycle(graph, source, target):
                        continue
                    delta = (
                        scorer.family_score(target, tuple(parents + [source]))
                        - scorer.family_score(target, tuple(parents))
                    )
                    if delta > best_delta:
                        best_delta, best_move = delta, ("add", source, target)

            # --- remove ---
            for source in parents:
                delta = (
                    scorer.family_score(target, tuple(p for p in parents if p != source))
                    - scorer.family_score(target, tuple(parents))
                )
                if delta > best_delta:
                    best_delta, best_move = delta, ("remove", source, target)

            # --- reverse ---
            for source in parents:
                source_parents = graph[source]
                if len(source_parents) >= max_parents:
                    continue
                trimmed = {**graph, target: [p for p in parents if p != source]}
                if _creates_cycle(trimmed, target, source):
                    continue
                delta = (
                    scorer.family_score(target, tuple(p for p in parents if p != source))
                    + scorer.family_score(source, tuple(source_parents + [target]))
                    - scorer.family_score(target, tuple(parents))
                    - scorer.family_score(source, tuple(source_parents))
                )
                if delta > best_delta:
                    best_delta, best_move = delta, ("reverse", source, target)

        if best_move is None:
            break

        action, source, target = best_move
        if action == "add":
            graph[target].append(source)
        elif action == "remove":
            graph[target].remove(source)
        else:
            graph[target].remove(source)
            graph[source].append(target)

        score += best_delta
        history.append(round(score, 2))
        if verbose:
            log.info("%s: %s -> %s (delta BIC %+.2f)", action, source, target, best_delta)

    return {
        "graph": graph,
        "score": round(scorer.graph_score(graph), 2),
        "iterations": len(history) - 1,
        "history": history,
        "scorer": scorer,
    }


def handcrafted_graph(df: pd.DataFrame) -> dict:
    """The project's domain-informed DAG, expressed in raw SCDB column names."""
    graph = {}
    for node_name, column in COLUMN_MAP.items():
        if column not in df.columns:
            continue
        parents = [
            COLUMN_MAP[p]
            for p in NODES[node_name].parents
            if COLUMN_MAP.get(p) in df.columns
        ]
        graph[column] = parents
    return graph


def compare_structures(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None = None,
    max_parents: int = 4,
    verbose: bool = True,
) -> dict:
    """
    Does data-driven structure search beat the hand-crafted legal-domain DAG?

    Both graphs are scored with BIC on the training data, and — when a test set
    is supplied — by held-out log-likelihood, which is the honest comparison:
    BIC rewards the graph that the search was explicitly optimising, so the
    learned structure has a home-field advantage on that metric.

    Returns
    -------
    dict with "handcrafted", "learned", and the edge-level differences.
    """
    variables = [col for col in COLUMN_MAP.values() if col in df_train.columns]

    hand = handcrafted_graph(df_train)
    hand = {v: hand.get(v, []) for v in variables}

    learned_result = hill_climb_structure(df_train, variables, max_parents=max_parents)
    learned = learned_result["graph"]

    scorer = BICScorer(df_train, variables)
    hand_score = scorer.graph_score(hand)
    learned_score = scorer.graph_score(learned)

    result = {
        "variables": variables,
        "handcrafted": {"graph": hand, "bic": round(hand_score, 2)},
        "learned": {"graph": learned, "bic": round(learned_score, 2),
                    "iterations": learned_result["iterations"]},
    }

    if df_test is not None and len(df_test) > 0:
        test_scorer = BICScorer(df_test, variables)
        # Held-out fit only: drop the complexity penalty, which is a training-set
        # device, and compare pure log-likelihood on unseen cases.
        result["handcrafted"]["test_loglik"] = round(
            _log_likelihood(test_scorer, hand), 2
        )
        result["learned"]["test_loglik"] = round(
            _log_likelihood(test_scorer, learned), 2
        )

    hand_edges = {(p, c) for c, ps in hand.items() for p in ps}
    learned_edges = {(p, c) for c, ps in learned.items() for p in ps}
    result["edges_only_in_handcrafted"] = sorted(hand_edges - learned_edges)
    result["edges_only_in_learned"] = sorted(learned_edges - hand_edges)
    result["edges_in_both"] = sorted(hand_edges & learned_edges)

    if verbose:
        print_structure_comparison(result)

    return result


def _log_likelihood(scorer: BICScorer, graph: dict) -> float:
    """Log-likelihood of a graph under a scorer, with the BIC penalty removed."""
    total = 0.0
    for node, parents in graph.items():
        parents = tuple(sorted(parents))
        n_configs = 1
        for parent in parents:
            n_configs *= scorer.cardinality[parent]
        penalty = 0.5 * scorer.log_n * (scorer.cardinality[node] - 1) * n_configs
        total += scorer.family_score(node, parents) + penalty
    return total


def print_structure_comparison(result: dict):
    """Print the hand-crafted vs learned structure comparison."""
    print("\n── Structure: hand-crafted vs learned ───────────────────")

    hand, learned = result["handcrafted"], result["learned"]
    print(f"  {'':<18} {'BIC':>14} {'Test log-lik':>14} {'Edges':>7}")
    print(f"  {'─' * 56}")
    for label, entry in (("Hand-crafted", hand), ("Learned (BIC)", learned)):
        n_edges = sum(len(p) for p in entry["graph"].values())
        test_ll = entry.get("test_loglik")
        test_str = f"{test_ll:>14.2f}" if test_ll is not None else f"{'—':>14}"
        print(f"  {label:<18} {entry['bic']:>14.2f} {test_str} {n_edges:>7}")

    if result["edges_only_in_learned"]:
        print("\n  Edges the search found that the domain DAG lacks:")
        for parent, child in result["edges_only_in_learned"][:15]:
            print(f"    {parent:<24} -> {child}")

    if result["edges_only_in_handcrafted"]:
        print("\n  Domain edges the search did not recover:")
        for parent, child in result["edges_only_in_handcrafted"][:15]:
            print(f"    {parent:<24} -> {child}")

    print(f"\n  Shared edges: {len(result['edges_in_both'])}")


def print_dependency_report(df: pd.DataFrame, method: str = "deviation"):
    """Print a formatted dependency analysis report."""
    label = "D(A,B)" if method == "deviation" else "I(A;B)"
    print(f"\n── Dependency Analysis ({label}) ────────────────────────")


    edges = find_top_dependencies(df, threshold=0.001, method=method, top_n=20)
    if not edges:
        print("  No significant dependencies found.")
        return


    max_score = edges[0][2] if edges else 1.0
    for va, vb, score in edges:
        bar_len = int((score / max_score) * 30) if max_score > 0 else 0
        bar = "█" * bar_len
        print(f"  {va:<25} ↔ {vb:<25} {score:.6f}  {bar}")


    print(f"\n  Total edges above threshold: {len(edges)}")
