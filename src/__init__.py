"""dictm — Supreme Court decision prediction with a Bayesian Network."""

from .baselines import (
    GradientBoostingBaseline,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
    MarginalBaseline,
    build_baselines,
)
from .cpt_builder import CPTBuilder
from .evaluate import (
    bootstrap_ci,
    brier_from_probs,
    build_evidence,
    calibration_analysis,
    classification_report,
    compare_inference_methods,
    distribution_summary,
    ece_from_probs,
    evaluate_model,
    k_fold_cross_validation,
    log_loss_from_probs,
    macro_auc_from_probs,
    multi_k_accuracy,
    predict_distributions,
    print_calibration_report,
    print_classification_report,
    print_results_table,
    sampler_convergence,
    top_k_accuracy,
    top_k_from_probs,
)
from .exact import VariableEliminationEngine, total_variation_distance
from .inference import GibbsSampler, LikelihoodWeightingSampler, RejectionSampler
from .network_structure import (
    BINARY_LABELS,
    COLUMN_MAP,
    DISPOSITION_LABELS,
    FEATURE_SETS,
    NODES,
    TOPOLOGICAL_ORDER,
    to_binary_disposition,
)
from .preprocessing import (
    add_derived_columns,
    missing_data_report,
    preprocess,
    validate_columns,
)
from .structure_learning import (
    BICScorer,
    compare_structures,
    compute_dependency_score,
    compute_mutual_information,
    compute_normalized_mutual_information,
    dependency_matrix,
    find_top_dependencies,
    g_test,
    handcrafted_graph,
    hill_climb_structure,
    print_dependency_report,
    print_structure_comparison,
)

__version__ = "2.0.0"
