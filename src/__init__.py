"""dictm — Supreme Court decision prediction with a Bayesian Network."""

from .network_structure import (
    NODES,
    TOPOLOGICAL_ORDER,
    COLUMN_MAP,
    DISPOSITION_LABELS,
    FEATURE_SETS,
    BINARY_LABELS,
    to_binary_disposition,
)
from .cpt_builder import CPTBuilder
from .inference import RejectionSampler, LikelihoodWeightingSampler, GibbsSampler
from .exact import VariableEliminationEngine, total_variation_distance
from .baselines import (
    MarginalBaseline,
    MajorityClassBaseline,
    LogisticRegressionBaseline,
    GradientBoostingBaseline,
    build_baselines,
)
from .evaluate import (
    build_evidence,
    predict_distributions,
    evaluate_model,
    top_k_accuracy,
    multi_k_accuracy,
    classification_report,
    calibration_analysis,
    distribution_summary,
    top_k_from_probs,
    log_loss_from_probs,
    brier_from_probs,
    macro_auc_from_probs,
    ece_from_probs,
    bootstrap_ci,
    k_fold_cross_validation,
    compare_inference_methods,
    sampler_convergence,
    print_results_table,
    print_classification_report,
    print_calibration_report,
)
from .structure_learning import (
    compute_dependency_score,
    compute_mutual_information,
    compute_normalized_mutual_information,
    g_test,
    dependency_matrix,
    find_top_dependencies,
    hill_climb_structure,
    compare_structures,
    handcrafted_graph,
    BICScorer,
    print_dependency_report,
    print_structure_comparison,
)
from .preprocessing import (
    preprocess,
    add_derived_columns,
    missing_data_report,
    validate_columns,
)

__version__ = "2.0.0"
