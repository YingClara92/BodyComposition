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
# data = pd.read_csv("iptw_weighted_data_ML.csv")
#
# target = "speed_ML_ratio"
# cluster_col = "cluster_" + target
#
# data[cluster_col] = 0
# data[cluster_col] = np.where(data[target] < -0.3, 1, data[cluster_col])
# data[cluster_col] = np.where(data[target] > 0.3, 2, data[cluster_col])


data = pd.read_csv('iptw_weighted_data_SKM.csv')
target = 'speed_muscle_body_ratio'
cluster_col = 'cluster_' + target
data['cluster_' + target] = 0
data['cluster_' + target] = np.where(data[target] < -0.4, 1, data['cluster_' + target])

# Define candidate features
# ============================================================
clinical_cols = [
    "SABR",
    "patient_sex",
    "age",
]

candidate_features = [target] + clinical_cols

base_cols = [
    "survival_months",
    "deceased",
    target,
    "iptw_weight",
    cluster_col,
] + clinical_cols

base = data[base_cols].dropna().reset_index(drop=True)

n = base.shape[0]

lb = base[target].min()
ub = base[target].max()


# ============================================================
# Helper functions
# ============================================================
def build_design_df(
    df,
    active_features,
    target,
    clinical_cols,
    lb,
    ub,
    spline_design_info=None,
    fit_spline=True,
):
    """
    Build Cox model dataframe.

    The nonlinear target is represented by spline basis terms.
    The spline terms are treated as one feature group.
    """

    model_df = df[["survival_months", "deceased", "iptw_weight"]].copy()
    group_to_columns = {}

    # Nonlinear target
    if target in active_features:
        if fit_spline:
            spline_df = dmatrix(
                f"0 + bs(x, df=3, include_intercept=False, lower_bound={lb}, upper_bound={ub})",
                {"x": df[target]},
                return_type="dataframe",
            )
            spline_design_info = spline_df.design_info
        else:
            spline_mat = build_design_matrices(
                [spline_design_info],
                {"x": df[target]},
            )[0]

            spline_df = pd.DataFrame(
                spline_mat,
                columns=spline_design_info.column_names,
                index=df.index,
            )

        spline_cols = [f"{target}_spline_{i+1}" for i in range(spline_df.shape[1])]
        spline_df.columns = spline_cols

        model_df = pd.concat(
            [
                model_df.reset_index(drop=True),
                spline_df.reset_index(drop=True),
            ],
            axis=1,
        )

        group_to_columns[target] = spline_cols

    # Linear clinical features
    for col in clinical_cols:
        if col in active_features:
            model_df[col] = df[col].values
            group_to_columns[col] = [col]

    return model_df, group_to_columns, spline_design_info


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


def backward_selection_on_training_set(
    df_train,
    candidate_features,
    target,
    clinical_cols,
    lb,
    ub,
    alpha=0.05,
    penalizer=0.01,
):
    """
    Backward selection inside one bootstrap training set.

    Removes the least significant feature group until all retained
    feature groups have group-level p <= alpha.
    """

    active_features = candidate_features.copy()

    while True:
        model_df, group_to_columns, spline_design_info = build_design_df(
            df=df_train,
            active_features=active_features,
            target=target,
            clinical_cols=clinical_cols,
            lb=lb,
            ub=ub,
            fit_spline=True,
        )

        cph = fit_cox(model_df, penalizer=penalizer)

        group_pvalues = {}

        for feature, cols in group_to_columns.items():
            group_pvalues[feature] = group_wald_pvalue(cph, cols)

        group_pvalues = pd.Series(group_pvalues).sort_values(ascending=False)

        worst_feature = group_pvalues.index[0]
        worst_pvalue = group_pvalues.iloc[0]

        if worst_pvalue <= alpha:
            break

        if len(active_features) == 1:
            break

        active_features.remove(worst_feature)

    # Refit final model using selected features
    final_model_df, final_group_to_columns, final_spline_design_info = build_design_df(
        df=df_train,
        active_features=active_features,
        target=target,
        clinical_cols=clinical_cols,
        lb=lb,
        ub=ub,
        fit_spline=True,
    )

    final_cph = fit_cox(final_model_df, penalizer=penalizer)

    final_group_pvalues = {}

    for feature, cols in final_group_to_columns.items():
        final_group_pvalues[feature] = group_wald_pvalue(final_cph, cols)

    final_group_pvalues = pd.Series(final_group_pvalues)

    return (
        final_cph,
        active_features,
        final_group_to_columns,
        final_spline_design_info,
        final_group_pvalues,
    )


# ============================================================
# Bootstrap + feature selection + final Cox fitting
# ============================================================
n_boot = 10000
max_successful_bootstraps = 1000

alpha = 0.05
penalizer = 0.01

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

    # Same filtering as your original code
    df_train = df_train[
        (df_train[target] > -0.75) &
        (df_train[target] < 0.75)
    ]

    if df_train.shape[0] == 0:
        continue

    frac = min(1.0, n / df_train.shape[0])
    df_train = df_train.sample(frac=frac, random_state=int(nums[i]))

    # OOB data
    sampled_indices = np.unique(df_train.index)
    df_oob = base.drop(index=sampled_indices, errors="ignore")

    if df_oob.shape[0] == 0:
        continue

    # Require enough OOB events in each cluster
    event_counts = df_oob.groupby(cluster_col)["deceased"].sum()

    if event_counts.min() < 3:
        continue

    df_train = df_train.reset_index(drop=True)
    df_oob = df_oob.reset_index(drop=True)

    # --------------------------------------------------------
    # Feature selection and final Cox model inside bootstrap
    # --------------------------------------------------------
    try:
        (
            final_cph,
            selected_features,
            final_group_to_columns,
            final_spline_design_info,
            final_group_pvalues,
        ) = backward_selection_on_training_set(
            df_train=df_train,
            candidate_features=candidate_features,
            target=target,
            clinical_cols=clinical_cols,
            lb=lb,
            ub=ub,
            alpha=alpha,
            penalizer=penalizer,
        )

    except (ConvergenceWarning, ConvergenceError, RuntimeError, ValueError, np.linalg.LinAlgError):
        continue

    successful_bootstrap_id += 1

    # --------------------------------------------------------
    # Record selected features
    # --------------------------------------------------------
    for feature in candidate_features:
        selection_records.append(
            {
                "bootstrap_id": successful_bootstrap_id,
                "feature": feature,
                "selected": int(feature in selected_features),
            }
        )

    # --------------------------------------------------------
    # Record group-level Cox results
    # --------------------------------------------------------
    for feature in selected_features:
        group_result_records.append(
            {
                "bootstrap_id": successful_bootstrap_id,
                "feature": feature,
                "group_p_value": final_group_pvalues.loc[feature],
                "n_terms": len(final_group_to_columns[feature]),
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

    # --------------------------------------------------------
    # OOB C-index
    # --------------------------------------------------------
    try:
        oob_model_df, _, _ = build_design_df(
            df=df_oob,
            active_features=selected_features,
            target=target,
            clinical_cols=clinical_cols,
            lb=lb,
            ub=ub,
            spline_design_info=final_spline_design_info,
            fit_spline=False,
        )

        covariate_cols = final_cph.params_.index.tolist()

        risk_oob = final_cph.predict_log_partial_hazard(
            oob_model_df[covariate_cols]
        ).values.ravel()

        y_train = Surv.from_arrays(
            event=df_train["deceased"].astype(bool).values,
            time=df_train["survival_months"].values,
        )

        y_oob = Surv.from_arrays(
            event=df_oob["deceased"].astype(bool).values,
            time=df_oob["survival_months"].values,
        )

        c_uno = concordance_index_ipcw(
            y_train,
            y_oob,
            risk_oob,
        )

        cindex_oob.append(c_uno[0])
        test_ratio.append(df_oob.shape[0] / n)

    except ValueError:
        continue

    if successful_bootstrap_id >= max_successful_bootstraps:
        break


# ============================================================
# Summaries
# ============================================================
selection_df = pd.DataFrame(selection_records)
group_results_df = pd.DataFrame(group_result_records)
coef_results_df = pd.DataFrame(coef_result_records)

cindex_oob = np.array(cindex_oob)
test_ratio = np.array(test_ratio)

print("\nNumber of successful bootstraps:", successful_bootstrap_id)

print("\nOOB Uno C-index:")
print(
    "mean={:.3f}, 95% CI=({:.3f}, {:.3f})".format(
        np.mean(cindex_oob),
        np.quantile(cindex_oob, 0.025),
        np.quantile(cindex_oob, 0.975),
    )
)

print("\nOOB test ratio:")
print(
    "mean={:.3f}, 95% CI=({:.3f}, {:.3f})".format(
        np.mean(test_ratio),
        np.quantile(test_ratio, 0.025),
        np.quantile(test_ratio, 0.975),
    )
)


# ------------------------------------------------------------
# Feature selection frequency
# ------------------------------------------------------------
feature_selection_summary = (
    selection_df
    .groupby("feature")
    .agg(
        selected_count=("selected", "sum"),
        selection_frequency=("selected", "mean"),
    )
    .reset_index()
    .sort_values("selection_frequency", ascending=False)
)

print("\nFeature selection frequency:")
print(feature_selection_summary)


# ------------------------------------------------------------
# Group-level p-value summary
# ------------------------------------------------------------
group_pvalue_summary = (
    group_results_df
    .groupby("feature")
    .agg(
        n_selected=("bootstrap_id", "count"),
        median_group_p=("group_p_value", "median"),
        mean_group_p=("group_p_value", "mean"),
        p025_group_p=("group_p_value", lambda x: np.quantile(x, 0.025)),
        p975_group_p=("group_p_value", lambda x: np.quantile(x, 0.975)),
    )
    .reset_index()
)

group_pvalue_summary = group_pvalue_summary.merge(
    feature_selection_summary,
    on="feature",
    how="left",
)

group_pvalue_summary = group_pvalue_summary.sort_values(
    "selection_frequency",
    ascending=False,
)

print("\nGroup-level Cox p-value summary:")
print(group_pvalue_summary)


# ------------------------------------------------------------
# Coefficient-level Cox summary
# ------------------------------------------------------------
coef_summary = (
    coef_results_df
    .groupby(["feature", "term"])
    .agg(
        n_selected=("bootstrap_id", "count"),
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
    feature_selection_summary[["feature", "selection_frequency"]],
    on="feature",
    how="left",
)

coef_summary = coef_summary.sort_values(
    ["selection_frequency", "feature", "term"],
    ascending=[False, True, True],
)

print("\nCoefficient-level Cox summary:")
print(coef_summary)


# ============================================================
# Optional: consensus final Cox model on full data
# ============================================================
# Example rule: keep features selected in at least 75% of bootstraps.
consensus_threshold = 0.75

consensus_features = feature_selection_summary.loc[
    feature_selection_summary["selection_frequency"] >= consensus_threshold,
    "feature",
].tolist()

print("\nConsensus features selected in at least 50% of bootstraps:")
print(consensus_features)

if len(consensus_features) > 0:
    consensus_model_df, consensus_group_to_columns, consensus_spline_design_info = build_design_df(
        df=base,
        active_features=consensus_features,
        target=target,
        clinical_cols=clinical_cols,
        lb=lb,
        ub=ub,
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