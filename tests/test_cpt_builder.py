"""Tests for CPT construction and serialization."""


import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cpt_builder import CPTBuilder, _sort_key
from src.network_structure import TOPOLOGICAL_ORDER

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCPTBuilder:
    """Tests for CPT learning from data."""


    def test_fit_creates_cpts_for_all_nodes(self, prepared_df):
        builder = CPTBuilder(alpha=1.0)
        builder.fit(prepared_df)
        for node in TOPOLOGICAL_ORDER:
            assert node in builder.cpts, f"Missing CPT for {node}"

    def test_split_vote_node_requires_derived_column(self, sample_df):
        """
        Without `add_derived_columns` there is no `voteSplit`, so the split_vote
        node is simply absent — the builder skips unmapped columns rather than
        inventing one.
        """
        builder = CPTBuilder(alpha=1.0).fit(sample_df)
        assert "split_vote" not in builder.cpts


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
                for _val, prob in cpt.items():
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

    def test_get_values_has_a_deterministic_order(self, fitted_builder):
        """The samplers index into this, so an unordered return type is a bug.

        `get_values` used to return a set. The samplers build a probability
        vector by iterating it and then draw an *index* into that vector, so with
        string-valued nodes the mapping from RNG draw to value shifted with every
        process — Python randomises string hashing per interpreter. Rejection
        sampling, which survives on a handful of accepted samples, changed its
        answer run to run from a fixed seed.
        """
        assert isinstance(fitted_builder.get_values("chief_justice"), tuple)

        for node in ("chief_justice", "issue_area", "final_disposition"):
            values = fitted_builder.get_values(node)
            assert list(values) == sorted(values, key=_sort_key)
            # Same object, asked twice, must not reshuffle.
            assert values == fitted_builder.get_values(node)

    def test_value_order_survives_a_save_load_round_trip(self, fitted_builder, tmp_path):
        path = tmp_path / "cpts.json"
        fitted_builder.save(str(path))
        loaded = CPTBuilder.load(str(path))

        for node in fitted_builder._value_sets:
            assert loaded.get_values(node) == fitted_builder.get_values(node)

    def test_value_order_is_stable_across_processes(self, fitted_builder, tmp_path):
        """The check that actually catches the original bug.

        Python randomises string hashing per interpreter, so set-iteration order
        is stable *within* a process and varies *between* them. A same-process
        repeat therefore cannot detect the defect; only running under two
        different `PYTHONHASHSEED` values can.
        """
        path = tmp_path / "cpts.json"
        fitted_builder.save(str(path))

        script = textwrap.dedent(
            f"""
            import json, sys
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from src.cpt_builder import CPTBuilder
            builder = CPTBuilder.load({str(path)!r})
            print(json.dumps({{n: list(builder.get_values(n)) for n in builder._value_sets}}))
            """
        )

        def run(hash_seed: str) -> str:
            env = {**os.environ, "PYTHONHASHSEED": hash_seed}
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, check=True, env=env,
            )
            return out.stdout.strip()

        assert run("1") == run("999999")


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


    def test_load_never_interprets_file_contents_as_code(self):
        """
        Loading a model must not evaluate anything from the file.

        CPT keys are tuples, which JSON cannot express. The original approach
        stringified them and parsed the result back with `ast.literal_eval`;
        this now stores each key as an explicit list instead, so no part of a
        model file is ever passed to an evaluator. That is both safer and
        type-faithful — `literal_eval` could not distinguish the string "2.0"
        from the float 2.0 once numpy types had been stringified.
        """
        import re

        import src.cpt_builder as mod

        source = open(mod.__file__, encoding="utf-8").read()
        assert not re.search(r"\beval\s*\(", source), "cpt_builder must not call eval()"
        assert "literal_eval" not in source

    def test_roundtrip_preserves_value_types(self, fitted_builder):
        """String and numeric node values must survive save/load unchanged."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            fitted_builder.save(path)
            loaded = CPTBuilder.load(path)

            assert loaded.get_values("chief_justice") == fitted_builder.get_values("chief_justice")
            assert "Roberts" in loaded.get_values("chief_justice")
            assert loaded.get_parents("final_disposition") == \
                   fitted_builder.get_parents("final_disposition")
        finally:
            os.unlink(path)
