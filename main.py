"""
main.py
-------
Entry point for the Supreme Court Decision Prediction Model.

Usage
-----
    # Train, then compare the network against every baseline on both tracks
    python main.py --mode train_eval

    # Only the honest pre-decision track, on the binary affirm/reverse task
    python main.py --mode train_eval --track ex_ante --task binary

    # Predict a single case interactively
    python main.py --mode predict

    # Hand-crafted vs learned network structure
    python main.py --mode structure

    # How fast does each sampler approach the exact posterior?
    python main.py --mode convergence

A note on `--track`
-------------------
`explanatory` supplies every observable variable as evidence, several of which
are recorded *from* the decision. It describes; it does not forecast. `ex_ante`
restricts evidence to what is knowable before the Court rules. Both are
reported by default because the gap between them is the substantive result.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from src.baselines import build_baselines
from src.cpt_builder import CPTBuilder
from src.evaluate import (
    build_evidence,
    compare_inference_methods,
    distribution_summary,
    evaluate_model,
    k_fold_cross_validation,
    print_calibration_report,
    print_classification_report,
    print_results_table,
    sampler_convergence,
)
from src.exact import VariableEliminationEngine
from src.network_structure import FEATURE_SETS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_PATH = os.path.join(DATA_DIR, "cpts.json")

N_SAMPLES = 1000     # samples per case, for the sampling engines only
TOP_K = 3
RANDOM_SEED = 42
TEST_FRACTION = 0.2


def _configure_console():
    """
    Force UTF-8 on stdout.

    The reports use box-drawing characters, arrows and block glyphs. On Windows
    the default console encoding is cp1252, which cannot represent any of them,
    so printing a results table raises UnicodeEncodeError and the run dies
    partway through.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def find_dataset() -> str:
    """Locate an SCDB case-centered CSV in data/, newest release first."""
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(_download_hint())

    candidates = sorted(
        (f for f in os.listdir(DATA_DIR)
         if f.startswith("SCDB_") and f.endswith(".csv") and "caseCentered" in f),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(_download_hint())
    return os.path.join(DATA_DIR, candidates[0])


def _download_hint() -> str:
    return (
        "SCDB dataset not found in data/.\n\n"
        "Fetch it with:\n"
        "    python scripts/fetch_data.py\n\n"
        "Or download 'Case Centered Data | Citation' manually from\n"
        "    http://scdb.wustl.edu/data.php"
    )


def load_data(path: str | None = None) -> pd.DataFrame:
    from src.preprocessing import preprocess

    path = path or find_dataset()
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"  Loaded {os.path.basename(path)}: {len(df)} cases, {len(df.columns)} columns")
    return preprocess(df, verbose=True)


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """
    Train on earlier cases, test on the most recent.

    The methodologically correct split for temporal data: never train on future
    cases to predict past ones. `term` is preferred over `dateDecision` because
    it sorts numerically without date parsing.
    """
    sort_col = "term" if "term" in df.columns else "dateDecision"
    if sort_col in df.columns:
        df = df.sort_values(sort_col).reset_index(drop=True)
    else:
        print("  Warning: no date column found; using row order as a proxy")

    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def random_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]


def split_data(df: pd.DataFrame, args):
    if args.split == "chronological":
        train, test = chronological_split(df, args.test_frac)
        print("  Chronological split (train on earlier terms, test on most recent)")
    else:
        train, test = random_split(df, args.test_frac, RANDOM_SEED)
        print(f"  Random split (seed={RANDOM_SEED})")
    print(f"  Train: {len(train)} cases  |  Test: {len(test)} cases")
    return train, test


def tracks_for(args) -> list[str]:
    return list(FEATURE_SETS) if args.track == "both" else [args.track]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_train_eval(args):
    print("\n── Loading Data ─────────────────────────────────────────")
    df = load_data(args.data_path)

    print("\n── Splitting Data ───────────────────────────────────────")
    df_train, df_test = split_data(df, args)

    print("\n── Building CPTs ────────────────────────────────────────")
    builder = CPTBuilder(alpha=args.alpha, backoff=not args.no_backoff)
    builder.fit(df_train)
    print(f"  CPTs built for {len(builder.cpts)} nodes")
    for node in ("issue_area", "final_disposition"):
        print(f"  {node}: {len(builder.get_values(node))} unique values")
    builder.save(args.model_path)
    print(f"  Saved model to {args.model_path}")

    engine = VariableEliminationEngine(builder)
    binary = args.task == "binary"
    ks = (1, 3) if binary else (1, 3, 5)

    all_results = {}
    for track in tracks_for(args):
        print(f"\n══ Track: {track} ══════════════════════════════════════")
        print(f"  Evidence: {', '.join(FEATURE_SETS[track])}")

        builder.reset_backoff_stats()
        results = [
            evaluate_model(df_test, baseline, name=baseline.name,
                           evidence_vars=track, binary=binary, ks=ks,
                           n_boot=args.n_boot, seed=RANDOM_SEED)
            for baseline in build_baselines(df_train, evidence_vars=track)
        ]
        results.append(
            evaluate_model(df_test, engine, name="Bayesian Network",
                           evidence_vars=track, binary=binary, ks=ks,
                           n_boot=args.n_boot, seed=RANDOM_SEED)
        )

        print_results_table(results, ks=ks)
        print(f"\n  CPT backoff rate: {builder.backoff_rate*100:.2f}% of conditional "
              f"queries fell back to a shorter parent set")

        network_result = results[-1]
        print_classification_report(network_result)
        print_calibration_report(network_result)
        all_results[track] = results

    _print_track_verdict(all_results, ks)

    if args.visualize:
        _generate_figures(df_test, builder, engine, all_results, args)

    return all_results


def _print_track_verdict(all_results: dict, ks):
    """
    State plainly whether the network beat the no-feature baseline.

    This exists so the comparison cannot quietly go missing from a future
    report. The marginal baseline is the number that makes the others readable.
    """
    if not all_results:
        return

    k = 3 if 3 in ks else ks[0]
    print("\n── Verdict ──────────────────────────────────────────────")

    for track, results in all_results.items():
        marginal = next((r for r in results if r["name"].startswith("Marginal")), None)
        network = next((r for r in results if r["name"].startswith("Bayesian")), None)
        if not marginal or not network:
            continue

        key = f"top_{k}"
        delta = network[key] - marginal[key]
        net_lo, net_hi = network.get(f"{key}_ci", (float("nan"),) * 2)
        mar_lo, mar_hi = marginal.get(f"{key}_ci", (float("nan"),) * 2)
        separated = net_lo > mar_hi or mar_lo > net_hi

        verdict = "beats" if delta > 0 else "does not beat"
        strength = "CIs do not overlap" if separated else "CIs overlap — not distinguishable"
        print(f"  {track:<13} Top-{k}: network {network[key]*100:.1f}% vs "
              f"marginal {marginal[key]*100:.1f}%  ({verdict}; {strength})")
        print(f"  {'':<13} log-loss: network {network['log_loss']:.4f} vs "
              f"marginal {marginal['log_loss']:.4f}"
              f"{'  (network worse)' if network['log_loss'] > marginal['log_loss'] else ''}")


def _generate_figures(df_test, builder, engine, all_results, args):
    from src.visualize import (
        plot_calibration_diagram,
        plot_confusion_matrix,
        plot_convergence,
        plot_method_comparison,
        plot_network_dag,
        plot_results_comparison,
    )

    print("\n── Generating Visualizations ────────────────────────────")
    plot_network_dag()

    track = next(iter(all_results))
    results = all_results[track]
    network_result = results[-1]

    plot_results_comparison(results, filename=f"results_{track}.png")
    plot_confusion_matrix(network_result, filename=f"confusion_matrix_{track}.png")
    plot_calibration_diagram(network_result["calibration"])

    print("\n  Comparing inference engines (this runs the samplers)...")
    method_results = compare_inference_methods(
        df_test.head(args.method_compare_n), builder,
        k=TOP_K, n_samples=N_SAMPLES, evidence_vars=track,
    )
    plot_method_comparison(method_results)

    evidence = build_evidence(df_test.iloc[0], evidence_vars="ex_ante")
    plot_convergence(sampler_convergence(builder, evidence))


def mode_predict(args):
    if not os.path.exists(args.model_path):
        print(f"No saved model at {args.model_path}. Run --mode train_eval first.")
        return

    print("\n── Loading Model ────────────────────────────────────────")
    builder = CPTBuilder.load(args.model_path)
    engine = VariableEliminationEngine(builder)

    track = args.track if args.track != "both" else "ex_ante"
    allowed = set(FEATURE_SETS[track])
    print(f"  Track: {track} — {len(allowed)} evidence variables")

    node_prompts = {
        "chief_justice":           "Chief Justice (Warren/Burger/Rehnquist/Roberts/Vinson)",
        "issue_area":              "Issue Area code (1=Criminal Procedure, 2=Civil Rights, 8=Due Process ...)",
        "law_type":                "Law Type code (1=Constitution, 6=Federal statute ...)",
        "lower_court_disposition": "Lower Court Disposition code (1-12)",
        "case_supplement":         "Unusual disposition? (0=No, 1=Yes)",
        "decision_type":           "Decision Type code (1=signed opinion, 6=per curiam ...)",
        "split_vote":              "Was there a dissent? (0=No, 1=Yes)",
        "unconstitutional":        "Unconstitutionality declared? (1=No, 2/3/4=Yes, varying scope)",
        "precedent_alteration":    "Precedent altered? (0=No, 1=Yes)",
    }

    print("\nEnter case details (press Enter to skip a field):\n")
    evidence = {}
    for node_name, prompt in node_prompts.items():
        if node_name not in allowed:
            continue
        try:
            raw = input(f"  {prompt}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        try:
            evidence[node_name] = float(raw)
        except ValueError:
            evidence[node_name] = raw

    print(f"\n  Evidence provided: {evidence}")
    print("\n── Running Exact Inference ──────────────────────────────")
    distribution_summary(engine, evidence)


def mode_eval(args):
    if not os.path.exists(args.model_path):
        print(f"No saved model at {args.model_path}. Run --mode train_eval first.")
        return

    df = load_data(args.data_path)
    _, df_test = split_data(df, args)

    builder = CPTBuilder.load(args.model_path)
    engine = VariableEliminationEngine(builder)
    binary = args.task == "binary"
    ks = (1, 3) if binary else (1, 3, 5)

    results = [
        evaluate_model(df_test, engine, name=f"Bayesian Network ({track})",
                       evidence_vars=track, binary=binary, ks=ks, n_boot=args.n_boot)
        for track in tracks_for(args)
    ]
    print_results_table(results, ks=ks)
    print_classification_report(results[-1])


def mode_cross_validate(args):
    print("\n── Loading Data ─────────────────────────────────────────")
    df = load_data(args.data_path)

    print("\n── K-Fold Cross-Validation ──────────────────────────────")
    out = {}
    for track in tracks_for(args):
        print(f"\n  Track: {track}")
        out[track] = k_fold_cross_validation(
            df, n_folds=args.n_folds, k=TOP_K, n_samples=N_SAMPLES,
            alpha=args.alpha, verbose=True, evidence_vars=track,
        )
    return out


def mode_dependency(args):
    from src.structure_learning import print_dependency_report

    print("\n── Loading Data ─────────────────────────────────────────")
    df = load_data(args.data_path)

    print_dependency_report(df, method="deviation")
    print_dependency_report(df, method="mutual_info")


def mode_structure(args):
    """Compare the hand-crafted DAG against one learned from the data."""
    from src.structure_learning import compare_structures

    print("\n── Loading Data ─────────────────────────────────────────")
    df = load_data(args.data_path)
    df_train, df_test = split_data(df, args)

    print("\n── Structure Learning (BIC hill-climbing) ───────────────")
    return compare_structures(df_train, df_test, max_parents=args.max_parents)


def mode_convergence(args):
    """Show each sampler approaching the exact posterior as samples increase."""
    print("\n── Loading Data ─────────────────────────────────────────")
    df = load_data(args.data_path)
    df_train, df_test = split_data(df, args)

    builder = CPTBuilder(alpha=args.alpha).fit(df_train)
    track = args.track if args.track != "both" else "ex_ante"
    evidence = build_evidence(df_test.iloc[0], evidence_vars=track)

    print("\n── Sampler Convergence to the Exact Posterior ───────────")
    print(f"  Evidence ({track}): {evidence}\n")

    curves = sampler_convergence(builder, evidence)
    counts = curves["_sample_counts"]

    header = f"  {'Engine':<26}" + "".join(f"{n:>10}" for n in counts)
    print(header)
    print(f"  {'─' * (26 + 10 * len(counts))}")
    for name, distances in curves.items():
        if name.startswith("_"):
            continue
        print(f"  {name:<26}" + "".join(f"{d:>10.4f}" for d in distances))

    print("\n  Values are total-variation distance from the exact posterior.")
    print("  Exact inference computes this in well under a millisecond per case.")

    if args.visualize:
        from src.visualize import plot_convergence
        plot_convergence(curves)

    return curves


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supreme Court Decision Predictor")
    parser.add_argument(
        "--mode",
        choices=["train_eval", "predict", "eval", "cross_validate",
                 "dependency", "structure", "convergence"],
        default="train_eval",
        help="train_eval: train + full comparison | predict: interactive | "
             "eval: evaluate saved model | cross_validate: k-fold CV | "
             "dependency: pairwise association | structure: learned vs "
             "hand-crafted DAG | convergence: sampler accuracy vs exact",
    )
    parser.add_argument(
        "--track", choices=["explanatory", "ex_ante", "both"], default="both",
        help="Evidence policy. 'explanatory' includes variables recorded from "
             "the decision itself and is descriptive, not predictive; 'ex_ante' "
             "uses only pre-decision information (default: both)",
    )
    parser.add_argument(
        "--task", choices=["multiclass", "binary"], default="multiclass",
        help="multiclass: all dispositions | binary: affirm vs reverse/vacate, "
             "the standard formulation in the literature",
    )
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--data_path", default=None, help="Path to the SCDB CSV")
    parser.add_argument(
        "--split", choices=["random", "chronological"], default="chronological",
        help="Data split strategy (default: chronological)",
    )
    parser.add_argument("--test-frac", dest="test_frac", type=float, default=TEST_FRACTION,
                        help="Fraction of cases held out for testing (default: 0.2)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing strength")
    parser.add_argument("--no-backoff", action="store_true",
                        help="Disable hierarchical backoff in CPT construction")
    parser.add_argument("--n-boot", dest="n_boot", type=int, default=500,
                        help="Bootstrap resamples for confidence intervals")
    parser.add_argument("--n-folds", dest="n_folds", type=int, default=5)
    parser.add_argument("--max-parents", dest="max_parents", type=int, default=4,
                        help="In-degree cap for structure learning")
    parser.add_argument("--method-compare-n", dest="method_compare_n", type=int, default=200,
                        help="Test cases used when timing the sampling engines")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate plots in figures/")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main():
    _configure_console()
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Supreme Court Decision Prediction — Bayesian Net    ║")
    print("╚══════════════════════════════════════════════════════╝")

    modes = {
        "train_eval": mode_train_eval,
        "predict": mode_predict,
        "eval": mode_eval,
        "cross_validate": mode_cross_validate,
        "dependency": mode_dependency,
        "structure": mode_structure,
        "convergence": mode_convergence,
    }

    try:
        return modes[args.mode](args)
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return None


if __name__ == "__main__":
    main()
