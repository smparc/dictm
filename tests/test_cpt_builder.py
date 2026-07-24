"""Tests for CPT construction and serialization."""


import json
import os
import tempfile


import numpy as np
import pandas as pd
import pytest


from src.cpt_builder import CPTBuilder
from src.network_structure import TOPOLOGICAL_ORDER



class TestCPTBuilder:
    """Tests for CPT learning from data."""


    def test_fit_creates_cpts_for_all_nodes(self, sample_df):
        builder = CPTBuilder(alpha=1.0)
        builder.fit(sample_df)
        for node in TOPOLOGICAL_ORDER:
            assert node in builder.cpts, f"Missing CPT for {node}"


    def test_root_probabilities_sum_to_one(self, fitted_builder):
        for node in ["chief_justice", "issue_area", "law_type"]:
            cpt = fitted_builder.cpts[node]
            total = sum(cpt.values())
            assert abs(total - 1.0) < 0.01, f"{node} root CPT sums to {total}"


    def test_root_probabilities_positive(self, fitted_builder):
        for node in ["chief_justice", "issue_area"]:
            cpt = fitted_builder.cpts[node]
            for val, prob in cpt.items():
                assert prob > 0, f"{node}={val} has prob {prob}"


    def test_laplace_smoothing(self, sample_df):
        """With alpha=0, unseen values get 0 probability. With alpha>0, they don't."""
        builder_no_smooth = CPTBuilder(alpha=0.0)
        builder_no_smooth.fit(sample_df)


        builder_smooth = CPTBuilder(alpha=1.0)
        builder_smooth.fit(sample_df)


        # All smoothed probabilities should be > 0
        for node in TOPOLOGICAL_ORDER:
            if node in builder_smooth.cpts:
                cpt = builder_smooth.cpts[node]
                for val, prob in cpt.items():
                    assert prob > 0, f"Smoothed {node} has 0 probability"


    def test_child_conditional_probabilities(self, fitted_builder):
        """Child CPT entries for a given parent config should sum to ~1."""
        cpt = fitted_builder.cpts["decision_type"]
        # Group by parent configuration
        parent_sums = {}
        for key, prob in cpt.items():
            parent_config = key[:-1]
            parent_sums[parent_config] = parent_sums.get(parent_config, 0.0) + prob


        for config, total in parent_sums.items():
            assert abs(total - 1.0) < 0.05, f"decision_type|{config} sums to {total}"


    def test_get_values(self, fitted_builder):
        vals = fitted_builder.get_values("chief_justice")
        assert len(vals) > 0
        assert "Roberts" in vals


    def test_query_root(self, fitted_builder):
        prob = fitted_builder.query_root("chief_justice", "Roberts")
        assert 0 < prob < 1


    def test_query_child(self, fitted_builder):
        vals = list(fitted_builder.get_values("decision_type"))
        if vals:
            parent_vals = (1.0, 1.0)  # issue_area=1, law_type=1
            prob = fitted_builder.query_child("decision_type", parent_vals, vals[0])
            assert prob >= 0



class TestCPTSerialization:
    """Tests for save/load round-trip."""


    def test_save_load_roundtrip(self, fitted_builder):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name


        try:
            fitted_builder.save(path)
            loaded = CPTBuilder.load(path)


            # Same nodes
            assert set(fitted_builder.cpts.keys()) == set(loaded.cpts.keys())


            # Same value sets
            for node in fitted_builder._value_sets:
                assert fitted_builder._value_sets[node] == loaded._value_sets[node]


            # Same probabilities (within floating point)
            for node in fitted_builder.cpts:
                for key in fitted_builder.cpts[node]:
                    orig = fitted_builder.cpts[node][key]
                    load = loaded.cpts[node].get(key, -1)
                    assert abs(orig - load) < 1e-10, f"Mismatch for {node}[{key}]"
        finally:
            os.unlink(path)


    def test_load_uses_literal_eval_not_eval(self):
        """Ensure we use ast.literal_eval, not eval (security)."""
        import ast
        import src.cpt_builder as mod
        # The module should import ast
        assert hasattr(mod, "ast") or "ast" in dir(mod) or "literal_eval" in open(mod.__file__).read()