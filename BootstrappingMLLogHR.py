import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices
from lifelines import CoxPHFitter
from lifelines.utils import ConvergenceError
from lifelines.exceptions import ConvergenceWarning
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
from scipy.stats import chi2
import warnings


# ============================================================
# Load and prepare data
# ============================================================
data = pd.read_csv("iptw_weighted_data_SKM.csv")

targets = [
    "speed_muscle_body_ratio",
    "speed_subcutaneous_fat_body_ratio",
    "speed_torso_body_ratio",
]

# data = pd.read_csv("iptw_weighted_data_ML.csv")
#
# targets = ["speed_ML_ratio"]

clinical_cols = [
    "SABR",
    "patient_sex",
    "age",
]

candidate_features = targets + clinical_cols

alpha = 0.05
penalizer = 0.01

n_boot = 1000
max_successful_bootstraps = 1000

consensus_threshold = 0.60

base_cols = [
    "survival_months",
    "deceased",
    "iptw_weight",
] + targets + clinical_cols

base = data[base_cols].dropna().reset_index(drop=True)

n = base.shape[0]

bounds = {
    target: {
        "lb": base[target].min(),
        "ub": base[target].max(),
    }
    for target in targets
}


# ============================================================
# Helper functions
# ============================================================
def build_design_df(
    df,
    active_features,
    targets,
    clinical_cols,
    bounds,
    spline_design_info_dict=None,
    fit_spline=True,
):
    model_df = df[["survival_months", "deceased", "iptw_weight"]].copy()
    group_to_columns = {}

    if spline_design_info_dict is None:
        spline_design_info_dict = {}

    for target in targets:
        if target in active_features:

            lb = bounds[target]["lb"]
            ub = bounds[target]["ub"]

            if fit_spline:
                spline_df = dmatrix(
                    f"0 + bs(x, df=3, include_intercept=False, "
                    f"lower_bound={lb}, upper_bound={ub})",
                    {"x": df[target]},
                    return_type="dataframe",
                )

                spline_design_info_dict[target] = spline_df.design_info

            else:
                spline_mat = build_design_matrices(
                    [spline_design_info_dict[target]],
                    {"x": df[target]},
                )[0]

                spline_df = pd.DataFrame(
                    spline_mat,
                    columns=spline_design_info_dict[target].column_names,
                    index=df.index,
                )

            spline_cols = [
                f"{target}_spline_{i + 1}"
                for i in range(spline_df.shape[1])
            ]

            spline_df.columns = spline_cols

            model_df = pd.concat(
                [
                    model_df.reset_index(drop=True),
                    spline_df.reset_index(drop=True),
                ],
                axis=1,
            )

            group_to_columns[target] = spline_cols

    for col in clinical_cols:
        if col in active_features:
            model_df[col] = df[col].values
            group_to_columns[col] = [col]

    return model_df, group_to_columns, spline_design_info_dict


def fit_cox(model_df, penalizer=0.01):
    cph = CoxPHFitter(penalizer=penalizer)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)

        cph.fit(
            model_df,
            duration_col="survival_months",
            event_col="deceased",
            weights_col="iptw_weight",
            robust=True,
        )

    return cph


def group_wald_pvalue(cph, cols):
    beta = cph.params_.loc[cols].values
    cov = cph.variance_matrix_.loc[cols, cols].values

    stat = beta @ np.linalg.pinv(cov) @ beta
    p_value = chi2.sf(stat, len(cols))

    return p_value


def fit_full_model_on_training_set(
    df_train,
    candidate_features,
    targets,
    clinical_cols,
    bounds,
    alpha=0.05,
    penalizer=0.01,
):
    active_features = candidate_features.copy()

    model_df, group_to_columns, spline_design_info_dict = build_design_df(
        df=df_train,
        active_features=active_features,
        targets=targets,
        clinical_cols=clinical_cols,
        bounds=bounds,
        fit_spline=True,
    )

    cph = fit_cox(model_df, penalizer=penalizer)

    group_pvalues = {}

    for feature, cols in group_to_columns.items():
        group_pvalues[feature] = group_wald_pvalue(cph, cols)

    group_pvalues = pd.Series(group_pvalues)

    significant_features = group_pvalues.loc[group_pvalues < alpha].index.tolist()

    return (
        cph,
        active_features,
        significant_features,
        group_to_columns,
        spline_design_info_dict,
        group_pvalues,
    )


# ============================================================
# Bootstrap full Cox model
# ============================================================
events = base[base["deceased"] == 1]
censored = base[base["deceased"] == 0]

n_events = int(len(events) * 1.5)
n_cens = int(len(censored) * 1.5)

rng = np.random.default_rng(6329)
nums = rng.integers(0, 10000, size=n_boot)

cindex_oob = []
test_ratio = []

selection_records = []
group_result_records = []
coef_result_records = []

successful_bootstrap_id = 0

for i in range(n_boot):

    boot_events = events.sample(
        n=n_events,
        replace=True,
        random_state=int(nums[i]),
    )

    boot_cens = censored.sample(
        n=n_cens,
        replace=True,
        random_state=int(nums[i]),
    )

    df_train = pd.concat([boot_events, boot_cens], axis=0)

    frac = min(1.0, n / df_train.shape[0])
    df_train = df_train.sample(frac=frac, random_state=int(nums[i]))

    sampled_indices = np.unique(df_train.index)
    df_oob = base.drop(index=sampled_indices, errors="ignore")

    if df_oob.shape[0] == 0:
        continue

    if df_oob["deceased"].sum() < 3:
        continue

    df_train = df_train.reset_index(drop=True)
    df_oob = df_oob.reset_index(drop=True)

    try:
        (
            final_cph,
            model_features,
            significant_features,
            final_group_to_columns,
            final_spline_design_info_dict,
            final_group_pvalues,
        ) = fit_full_model_on_training_set(
            df_train=df_train,
            candidate_features=candidate_features,
            targets=targets,
            clinical_cols=clinical_cols,
            bounds=bounds,
            alpha=alpha,
            penalizer=penalizer,
        )

    except (
        ConvergenceWarning,
        ConvergenceError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ):
        continue

    successful_bootstrap_id += 1

    # --------------------------------------------------------
    # Record features with full-model group p < alpha
    # --------------------------------------------------------
    for feature in candidate_features:
        pval = final_group_pvalues.loc[feature]

        selection_records.append(
            {
                "bootstrap_id": successful_bootstrap_id,
                "feature": feature,
                "group_p_value": pval,
                "significant": int(pval < alpha),
            }
        )

    # --------------------------------------------------------
    # Record group-level Cox results
    # --------------------------------------------------------
    for feature in candidate_features:
        group_result_records.append(
            {
                "bootstrap_id": successful_bootstrap_id,
                "feature": feature,
                "group_p_value": final_group_pvalues.loc[feature],
                "n_terms": len(final_group_to_columns[feature]),
                "significant": int(final_group_pvalues.loc[feature] < alpha),
            }
        )

    # --------------------------------------------------------
    # Record coefficient-level Cox results
    # --------------------------------------------------------
    summary = final_cph.summary.copy()

    for feature, cols in final_group_to_columns.items():
        for term in cols:
            coef_result_records.append(
                {
                    "bootstrap_id": successful_bootstrap_id,
                    "feature": feature,
                    "term": term,
                    "coef": summary.loc[term, "coef"],
                    "exp_coef": summary.loc[term, "exp(coef)"],
                    "se_coef": summary.loc[term, "se(coef)"],
                    "p_value": summary.loc[term, "p"],
                }
            )

# ============================================================
# Summaries
# ============================================================
selection_df = pd.DataFrame(selection_records)
group_results_df = pd.DataFrame(group_result_records)
coef_results_df = pd.DataFrame(coef_result_records)

cindex_oob = np.array(cindex_oob)
test_ratio = np.array(test_ratio)

# ------------------------------------------------------------
# Full-model significance frequency
# ------------------------------------------------------------
feature_selection_summary = (
    selection_df
    .groupby("feature")
    .agg(
        significant_count=("significant", "sum"),
        significance_frequency=("significant", "mean"),
        median_group_p=("group_p_value", "median"),
        mean_group_p=("group_p_value", "mean"),
        p025_group_p=("group_p_value", lambda x: np.quantile(x, 0.025)),
        p975_group_p=("group_p_value", lambda x: np.quantile(x, 0.975)),
    )
    .reset_index()
    .sort_values("significance_frequency", ascending=False)
)

print("\nFeature full-model significance frequency:")
print(feature_selection_summary)


# ------------------------------------------------------------
# Group-level p-value summary
# ------------------------------------------------------------
group_pvalue_summary = (
    group_results_df
    .groupby("feature")
    .agg(
        n_bootstraps=("bootstrap_id", "count"),
        significant_count=("significant", "sum"),
        significance_frequency=("significant", "mean"),
        median_group_p=("group_p_value", "median"),
        mean_group_p=("group_p_value", "mean"),
        p025_group_p=("group_p_value", lambda x: np.quantile(x, 0.025)),
        p975_group_p=("group_p_value", lambda x: np.quantile(x, 0.975)),
    )
    .reset_index()
    .sort_values("significance_frequency", ascending=False)
)

print("\nFull-model group-level Cox p-value summary:")
print(group_pvalue_summary)


# ------------------------------------------------------------
# Coefficient-level Cox summary
# ------------------------------------------------------------
coef_summary = (
    coef_results_df
    .groupby(["feature", "term"])
    .agg(
        n_bootstraps=("bootstrap_id", "count"),
        median_coef=("coef", "median"),
        mean_coef=("coef", "mean"),
        coef_025=("coef", lambda x: np.quantile(x, 0.025)),
        coef_975=("coef", lambda x: np.quantile(x, 0.975)),
        median_HR=("exp_coef", "median"),
        mean_HR=("exp_coef", "mean"),
        HR_025=("exp_coef", lambda x: np.quantile(x, 0.025)),
        HR_975=("exp_coef", lambda x: np.quantile(x, 0.975)),
        median_p=("p_value", "median"),
    )
    .reset_index()
)

coef_summary = coef_summary.merge(
    feature_selection_summary[["feature", "significance_frequency"]],
    on="feature",
    how="left",
)

coef_summary = coef_summary.sort_values(
    ["significance_frequency", "feature", "term"],
    ascending=[False, True, True],
)

print("\nCoefficient-level Cox summary:")
print(coef_summary)


# ============================================================
# Optional: consensus final Cox model on full data
# ============================================================
consensus_features = feature_selection_summary.loc[
    feature_selection_summary["significance_frequency"] >= consensus_threshold,
    "feature",
].tolist()

print(
    f"\nConsensus features with full-model group p < {alpha} "
    f"in at least {consensus_threshold:.0%} of bootstraps:"
)
print(consensus_features)

if len(consensus_features) > 0:

    consensus_model_df, consensus_group_to_columns, consensus_spline_design_info_dict = build_design_df(
        df=base,
        active_features=consensus_features,
        targets=targets,
        clinical_cols=clinical_cols,
        bounds=bounds,
        fit_spline=True,
    )

    consensus_cph = fit_cox(consensus_model_df, penalizer=penalizer)

    print("\nConsensus Cox model fitted on full data:")
    print(consensus_cph.summary)

    consensus_group_pvalues = {}

    for feature, cols in consensus_group_to_columns.items():
        consensus_group_pvalues[feature] = group_wald_pvalue(
            consensus_cph,
            cols,
        )

    consensus_group_pvalues = pd.Series(consensus_group_pvalues).sort_values()

    print("\nConsensus model group-level p-values:")
    print(consensus_group_pvalues)
