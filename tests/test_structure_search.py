"""
Tests for score-based structure learning and the corrected association measures.
"""

import numpy as np
import pandas as pd
import pytest

from src.structure_learning import (
    BICScorer,
    _creates_cycle,
    compare_structures,
    compute_mutual_information,
    compute_normalized_mutual_information,
    g_test,
    handcrafted_graph,
    hill_climb_structure,
)


@pytest.fixture
def linked_df():
    """`b` is a deterministic function of `a`; `c` is independent of both."""
    rng = np.random.default_rng(0)
    a = rng.choice([0, 1, 2], size=600)
    return pd.DataFrame({"a": a, "b": a * 2, "c": rng.choice([0, 1], size=600)})


class TestNormalizedMutualInformation:
    def test_deterministic_relationship_is_one(self, linked_df):
        assert compute_normalized_mutual_information(linked_df, "a", "b") == pytest.approx(1.0, abs=1e-9)

    def test_independent_variables_are_near_zero(self, linked_df):
        assert compute_normalized_mutual_information(linked_df, "a", "c") < 0.05

    def test_bounded_to_unit_interval(self, linked_df):
        for x, y in [("a", "b"), ("a", "c"), ("b", "c")]:
            assert 0.0 <= compute_normalized_mutual_information(linked_df, x, y) <= 1.0

    def test_symmetric(self, linked_df):
        assert compute_normalized_mutual_information(linked_df, "a", "c") == \
               pytest.approx(compute_normalized_mutual_information(linked_df, "c", "a"))

    def test_constant_variable_yields_zero(self):
        df = pd.DataFrame({"a": [1, 2, 3, 1, 2, 3], "k": [7] * 6})
        assert compute_normalized_mutual_information(df, "a", "k") == 0.0

    def test_corrects_cardinality_bias_of_raw_mi(self):
        """
        A high-cardinality noise variable can carry more raw MI than a genuine
        low-cardinality association. Normalising must reverse that ordering.
        """
        rng = np.random.default_rng(1)
        n = 900
        target = rng.choice([0, 1], size=n)
        # Strongly associated, but only two categories.
        weak_cardinality = np.where(rng.random(n) < 0.9, target, 1 - target)
        # Independent of the target, but 30 categories.
        high_cardinality = rng.choice(range(30), size=n)
        df = pd.DataFrame({"t": target, "informative": weak_cardinality, "noisy": high_cardinality})

        assert compute_mutual_information(df, "t", "noisy") > 0
        assert compute_normalized_mutual_information(df, "t", "informative") > \
               compute_normalized_mutual_information(df, "t", "noisy")


class TestGTest:
    def test_detects_a_real_association(self, linked_df):
        result = g_test(linked_df, "a", "b")
        assert result["g"] > 0
        assert result["p_value"] < 0.001

    def test_independent_variables_are_not_significant(self, linked_df):
        assert g_test(linked_df, "a", "c")["p_value"] > 0.01

    def test_degrees_of_freedom_formula(self, linked_df):
        # a has 3 values, c has 2 -> (3-1)(2-1) = 2
        assert g_test(linked_df, "a", "c")["dof"] == 2

    def test_empty_frame(self):
        df = pd.DataFrame({"a": [], "b": []})
        assert g_test(df, "a", "b")["n"] == 0


class TestCycleDetection:
    def test_detects_direct_cycle(self):
        graph = {"a": [], "b": ["a"]}
        assert _creates_cycle(graph, "b", "a") is True

    def test_allows_a_safe_edge(self):
        graph = {"a": [], "b": ["a"], "c": []}
        assert _creates_cycle(graph, "c", "b") is False

    def test_detects_transitive_cycle(self):
        graph = {"a": [], "b": ["a"], "c": ["b"]}
        assert _creates_cycle(graph, "c", "a") is True


class TestBICScorer:
    def test_dependent_parent_improves_the_score(self, linked_df):
        scorer = BICScorer(linked_df, ["a", "b", "c"])
        assert scorer.family_score("b", ("a",)) > scorer.family_score("b", ())

    def test_irrelevant_parent_is_penalized(self, linked_df):
        scorer = BICScorer(linked_df, ["a", "b", "c"])
        assert scorer.family_score("a", ("c",)) < scorer.family_score("a", ())

    def test_scores_are_cached(self, linked_df):
        scorer = BICScorer(linked_df, ["a", "b", "c"])
        first = scorer.family_score("b", ("a",))
        assert ("b", ("a",)) in scorer._cache
        assert scorer.family_score("b", ("a",)) == first


class TestHillClimb:
    def test_recovers_a_real_dependency(self, linked_df):
        result = hill_climb_structure(linked_df, ["a", "b", "c"], max_parents=2)
        edges = {(p, c) for c, ps in result["graph"].items() for p in ps}
        assert ("a", "b") in edges or ("b", "a") in edges

    def test_does_not_connect_independent_variables(self, linked_df):
        result = hill_climb_structure(linked_df, ["a", "b", "c"], max_parents=2)
        edges = {(p, c) for c, ps in result["graph"].items() for p in ps}
        assert ("c", "a") not in edges and ("a", "c") not in edges

    def test_result_is_acyclic(self, linked_df):
        graph = hill_climb_structure(linked_df, ["a", "b", "c"])["graph"]
        for child, parents in graph.items():
            for parent in parents:
                trimmed = {**graph, child: [p for p in parents if p != parent]}
                assert not _creates_cycle(trimmed, child, parent)

    def test_respects_the_in_degree_cap(self, linked_df):
        graph = hill_climb_structure(linked_df, ["a", "b", "c"], max_parents=1)["graph"]
        assert all(len(parents) <= 1 for parents in graph.values())

    def test_score_improves_monotonically(self, linked_df):
        history = hill_climb_structure(linked_df, ["a", "b", "c"])["history"]
        assert history == sorted(history)

    def test_terminates_on_pure_noise(self):
        rng = np.random.default_rng(2)
        noise = pd.DataFrame({c: rng.choice([0, 1], size=300) for c in "xyz"})
        result = hill_climb_structure(noise, list("xyz"))
        assert sum(len(p) for p in result["graph"].values()) == 0


class TestCompareStructures:
    def test_reports_both_graphs_and_their_differences(self, prepared_df):
        result = compare_structures(prepared_df, prepared_df, verbose=False)
        assert "handcrafted" in result and "learned" in result
        assert "bic" in result["handcrafted"] and "bic" in result["learned"]
        assert isinstance(result["edges_only_in_learned"], list)

    def test_learned_graph_wins_on_the_score_it_optimises(self, prepared_df):
        result = compare_structures(prepared_df, verbose=False)
        assert result["learned"]["bic"] >= result["handcrafted"]["bic"]

    def test_handcrafted_graph_matches_the_declared_structure(self, prepared_df):
        from src.network_structure import COLUMN_MAP, NODES

        graph = handcrafted_graph(prepared_df)
        expected = [COLUMN_MAP[p] for p in NODES["final_disposition"].parents]
        assert sorted(graph[COLUMN_MAP["final_disposition"]]) == sorted(expected)
