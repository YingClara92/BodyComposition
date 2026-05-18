import warnings
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from firthlogist import FirthLogisticRegression

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def stratified_bootstrap_indices(y, rng):
    """
    Bootstrap within each class to preserve outcome balance
    and avoid samples with only one class.
    """
    y = np.asarray(y)
    all_idx = np.arange(len(y))
    boot_idx = []

    for cls in np.unique(y):
        cls_idx = all_idx[y == cls]
        sampled_cls_idx = rng.choice(cls_idx, size=len(cls_idx), replace=True)
        boot_idx.append(sampled_cls_idx)

    boot_idx = np.concatenate(boot_idx)
    rng.shuffle(boot_idx)

    return boot_idx


def calculate_vif(X):
    """
    Calculate VIF for a dataframe of predictors.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    if X.shape[1] == 1:
        vif_data["VIF"] = [1.0]
    else:
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ]

    return vif_data


def vif_filter(X_scaled, fea_list, vif_threshold=3.0, verbose=False):
    """
    Iteratively remove the feature with the highest VIF
    until all remaining VIF values are <= vif_threshold.
    """
    fea_list = [fea for fea in fea_list if fea in X_scaled.columns]

    if len(fea_list) == 0:
        return [], pd.DataFrame(columns=["feature", "VIF"])

    if len(fea_list) == 1:
        return fea_list, pd.DataFrame({
            "feature": fea_list,
            "VIF": [1.0]
        })

    vif_data = calculate_vif(X_scaled[fea_list])

    while len(fea_list) > 1 and vif_data["VIF"].max() > vif_threshold:
        max_vif_feature = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]

        if verbose:
            print(
                f"Removing feature {max_vif_feature} "
                f"with VIF {vif_data['VIF'].max():.2f}"
            )

        fea_list.remove(max_vif_feature)

        if len(fea_list) == 1:
            vif_data = pd.DataFrame({
                "feature": fea_list,
                "VIF": [1.0]
            })
            break

        vif_data = calculate_vif(X_scaled[fea_list])

    return fea_list, vif_data


def firth_cluster_selection_once(
    X_scaled,
    y,
    cluster_data,
    vif_threshold=3.0,
    require_positive_coef=True,
    bonferroni=True,
    verbose=False
):
    """
    One bootstrap iteration of your original selection pipeline:

    1. For each cluster, run univariate Firth logistic regression.
    2. Select the feature with the smallest corrected p-value.
    3. Optionally require positive coefficient.
    4. Apply VIF filtering to selected features.
    """

    cluster_num = sorted(cluster_data["cluster"].dropna().unique())

    selected_cluster = {}

    for cluster in cluster_num:
        if verbose:
            print(f"Cluster {cluster}")

        cluster_features = (
            cluster_data.loc[cluster_data["cluster"] == cluster, "feature"]
            .dropna()
            .tolist()
        )

        cluster_features = [
            fea for fea in cluster_features
            if fea in X_scaled.columns
        ]

        if len(cluster_features) == 0:
            selected_cluster[cluster] = ["", np.nan]
            continue

        p_fea = ""
        p_value_s = len(cluster_features)

        for fea in cluster_features:
            X_one = X_scaled[fea].values.reshape(-1, 1)

            model = FirthLogisticRegression(max_iter=1000)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = model.fit(X_one, y)

                p_raw = float(result.pvals_[0])
                coef = float(result.coef_[0])

            except Exception as e:
                if verbose:
                    print(f"Skipping {fea} due to fitting error: {e}")
                continue

            if np.isnan(p_raw) or np.isnan(coef):
                continue

            if bonferroni:
                p_value = min(p_raw * len(cluster_features), 1.0)
            else:
                p_value = p_raw

            if require_positive_coef and coef <= 0:
                continue

            if p_value < p_value_s:
                p_value_s = p_value
                p_fea = fea

            if verbose and p_value < 0.05:
                print(
                    f"Feature: {fea}, corrected p-value: {p_value:.4f}, "
                    f"coefficient: {coef:.4f}"
                )

        selected_cluster[cluster] = [p_fea, p_value_s]

    fea_list = [
        feature_info[0]
        for feature_info in selected_cluster.values()
        if feature_info[0] != ""
    ]

    fea_list, vif_data = vif_filter(
        X_scaled=X_scaled,
        fea_list=fea_list,
        vif_threshold=vif_threshold,
        verbose=verbose
    )

    return fea_list, selected_cluster, vif_data


def bootstrap_firth_cluster_selection(
    X,
    y,
    cluster_data,
    n_bootstraps=1000,
    selection_threshold=5,
    vif_threshold=3.0,
    require_positive_coef=True,
    bonferroni=True,
    random_state=42,
    verbose=False
):
    """
    Bootstrap stability selection using your cluster-wise Firth + VIF method.

    A feature is counted as selected only if it:
    1. wins within its cluster in the bootstrap sample, and
    2. survives VIF filtering in that bootstrap sample.
    """

    rng = np.random.default_rng(random_state)

    X = X.copy()
    y = np.asarray(y)

    selection_counts = pd.Series(0, index=X.columns, dtype=float)
    valid_bootstraps = 0
    selected_feature_sets = []

    for b in range(n_bootstraps):
        boot_idx = stratified_bootstrap_indices(y, rng)

        X_boot = X.iloc[boot_idx].copy()
        y_boot = y[boot_idx]

        if len(np.unique(y_boot)) < 2:
            continue

        scaler = StandardScaler()
        X_boot_scaled = pd.DataFrame(
            scaler.fit_transform(X_boot),
            columns=X_boot.columns,
            index=X_boot.index
        )

        try:
            selected_features, selected_cluster, vif_data = firth_cluster_selection_once(
                X_scaled=X_boot_scaled,
                y=y_boot,
                cluster_data=cluster_data,
                vif_threshold=vif_threshold,
                require_positive_coef=require_positive_coef,
                bonferroni=bonferroni,
                verbose=False
            )
        except Exception as e:
            if verbose:
                print(f"Bootstrap {b} skipped: {e}")
            continue

        valid_bootstraps += 1
        selected_feature_sets.append(selected_features)

        for fea in selected_features:
            selection_counts.loc[fea] += 1

        if verbose and (b + 1) % 50 == 0:
            print(f"Completed {b + 1}/{n_bootstraps} bootstraps")

    if valid_bootstraps == 0:
        raise RuntimeError("No valid bootstrap samples were fitted.")

    selection_frequency = selection_counts / valid_bootstraps

    stability_table = pd.DataFrame({
        "feature": selection_frequency.index,
        "selection_count": selection_counts.values.astype(int),
        "selection_frequency": selection_frequency.values
    })

    cluster_lookup = cluster_data[["feature", "cluster"]].drop_duplicates()
    stability_table = stability_table.merge(
        cluster_lookup,
        on="feature",
        how="left"
    )

    stability_table = stability_table.sort_values(
        "selection_frequency",
        ascending=False
    ).reset_index(drop=True)

    print(stability_table)

    # Select the top 5 most stable features
    stable_features = stability_table.head(selection_threshold)["feature"].tolist()

    print(f"Valid bootstraps: {valid_bootstraps}/{n_bootstraps}")
    print(f"Stable feature threshold: {selection_threshold}")
    print(f"Number of stable features: {len(stable_features)}")

    return stable_features, stability_table, selected_feature_sets


def make_frequency_weights(y):
    y = pd.Series(y)
    counts = y.value_counts().to_dict()
    total = len(y)
    n_classes = y.nunique()

    weights = y.apply(
        lambda cls: total / (n_classes * counts[cls])
    ).values

    return weights


def bootstrap_auc_ci(
    y_true,
    y_score,
    sample_weight=None,
    n_bootstraps=2000,
    ci=0.95,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    apparent_auc = roc_auc_score(
        y_true,
        y_score,
        sample_weight=sample_weight
    )

    predicted_classes = (y_score >= 0.5).astype(int)

    apparent_acc = accuracy_score(
        y_true,
        predicted_classes,
        sample_weight=sample_weight
    )

    auc_scores = []
    acc_scores = []

    n = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.choice(np.arange(n), size=n, replace=True)

        if len(np.unique(y_true[idx])) < 2:
            continue

        sw = sample_weight[idx] if sample_weight is not None else None

        auc_scores.append(
            roc_auc_score(y_true[idx], y_score[idx], sample_weight=sw)
        )

        acc_scores.append(
            accuracy_score(
                y_true[idx],
                predicted_classes[idx],
                sample_weight=sw
            )
        )

    alpha = (1.0 - ci) / 2.0

    auc_lower = np.percentile(auc_scores, 100 * alpha)
    auc_upper = np.percentile(auc_scores, 100 * (1 - alpha))

    acc_lower = np.percentile(acc_scores, 100 * alpha)
    acc_upper = np.percentile(acc_scores, 100 * (1 - alpha))

    return apparent_auc, auc_lower, auc_upper, apparent_acc, acc_lower, acc_upper
# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv('data_BED_m.csv')
data = data[data['cluster_ML_new'] != 2]

y = data['cluster_ML_new'].values
encoder = OneHotEncoder()
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
y = y_onehot.toarray()[:, 1]

# ---------------------------------------------------------
# 2. Define candidate features
# ---------------------------------------------------------

selected_feature = data.columns[6:].tolist()

rm_feature = []

for fea in selected_feature:
    zero_ratio = (data[fea] == 0).mean()

    if zero_ratio > 0.70:
        print(f"Removing feature {fea} with zero ratio {zero_ratio:.2f}")
        rm_feature.append(fea)

selected_feature = [
    fea for fea in selected_feature
    if fea not in rm_feature
]


# ---------------------------------------------------------
# 3. Load feature clusters
# ---------------------------------------------------------

cluster_data = pd.read_csv("feature_cluster_BED2.csv")

candidate_features = (
    cluster_data.loc[
        cluster_data["feature"].isin(selected_feature),
        "feature"
    ]
    .drop_duplicates()
    .tolist()
)

X_candidates = data[candidate_features].copy()

print(f"Number of candidate features: {X_candidates.shape[1]}")

#
# ---------------------------------------------------------
# 4. Bootstrap cluster-wise Firth selection + VIF filtering
# ---------------------------------------------------------

stable_features, stability_table, selected_feature_sets = bootstrap_firth_cluster_selection(
    X=X_candidates,
    y=y,
    cluster_data=cluster_data,
    n_bootstraps=200,
    selection_threshold=5,
    vif_threshold=3.0,
    require_positive_coef=True,
    bonferroni=True,
    random_state=42,
    verbose=True
)

print("\n--- Stability selection table ---")
print(stability_table.head(50))

print("\nStable features before final full-data VIF check:")
print(stable_features)


# ---------------------------------------------------------
# 5. Final VIF check on full dataset
# ---------------------------------------------------------

scaler_full_vif = StandardScaler()

X_full_scaled = pd.DataFrame(
    scaler_full_vif.fit_transform(X_candidates),
    columns=X_candidates.columns,
    index=X_candidates.index
)

final_features, final_vif_table = vif_filter(
    X_scaled=X_full_scaled,
    fea_list=stable_features,
    vif_threshold=3.0,
    verbose=True
)

print("\n--- Final VIF table ---")
print(final_vif_table)

print("\nFinal stable features after full-data VIF check:")
print(final_features)

if len(final_features) == 0:
    raise ValueError("No final features remained after stability selection and VIF filtering.")


# ---------------------------------------------------------
# 6. Final weighted logistic regression model
# ---------------------------------------------------------


print("Final features used in the model:")
print(final_features)

X = data[final_features].values
# Scaling is strictly required for Penalized Regression!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.linear_model import LogisticRegressionCV
# --- 2. STEP 1: LASSO FEATURE SELECTION ---
print("--- Running LASSO Penalized Regression for Feature Selection ---")
# LogisticRegressionCV automatically finds the best L1 penalty using Cross-Validation
lasso_cv = LogisticRegressionCV(
    Cs=10,                  # Number of penalty strengths to test
    cv=5,                   # 5-fold cross-validation
    penalty='l1',           # L1 penalty = LASSO
    solver='liblinear',     # Required solver for l1 penalty
    class_weight='balanced',# Handles your class imbalance automatically
    scoring='roc_auc',
    random_state=42,
    max_iter=1000
)

lasso_cv.fit(X_scaled, y)

# Identify features that survived the penalty (coefficient != 0)
lasso_coefs = lasso_cv.coef_[0]
selected_mask = lasso_coefs != 0
# selected_mask = np.ones_like(lasso_coefs).astype(bool)
final_features = np.array(final_features)[selected_mask].tolist()


print(f"Original features ({len(final_features)}): {final_features}")
print(f"Features selected by LASSO ({len(final_features)}): {final_features}\n")

if len(final_features) == 0:
    raise ValueError("LASSO penalized all features to 0. No predictive features found.")

# --- 3. STEP 2: POST-LASSO INFERENCE (Statsmodels) ---
# Now we fit the standard model using ONLY the features LASSO deemed relevant
# This gives you the p-values and Odds Ratios for your publication table

# Re-slice X_scaled to only include selected features
X_final_scaled = X_scaled[:, selected_mask]

df = pd.DataFrame(X_final_scaled, columns=final_features)
df['y'] = y

# Calculate your custom frequency weights
counts = df['y'].value_counts().to_dict()
total = len(df)
n_classes = df['y'].nunique()
df['weights'] = df['y'].apply(lambda cls: total / (n_classes * counts[cls]))

X_sm = sm.add_constant(df[final_features])
model = sm.GLM(df['y'], X_sm, family=sm.families.Binomial(), freq_weights=df['weights'])
result = model.fit()

# --- 4. PUBLICATION TABLE ---
coef   = result.params
pvals  = result.pvalues
conf   = result.conf_int()

# convert to odds ratios
or_        = np.exp(coef)
ci_lower   = np.exp(conf[0])
ci_upper   = np.exp(conf[1])

table = pd.DataFrame({
    "Odds ratio": or_.round(2),
    "95% CI low": ci_lower.round(2),
    "95% CI high": ci_upper.round(2),
    "p-value": pvals.round(3),
})
table.index.name = "Predictor"

print("--- Final Post-LASSO Multivariable Model ---")
print(table)
print("\n")

# --- 5. MODEL EVALUATION ---
predicted_probs = result.predict(X_sm)

# Brier score
brier = brier_score_loss(df['y'], predicted_probs, sample_weight=df['weights'])
print(f'Brier score: {brier:.4f}')

# Classification metrics
threshold = 0.5
predicted_classes = (predicted_probs >= threshold).astype(int)
accuracy = accuracy_score(df['y'], predicted_classes)

# AUC + 95% CI
auc, auc_lower, auc_upper, acc, acc_lower, acc_upper = bootstrap_auc_ci(
    df['y'].values,
    predicted_probs,
    sample_weight=df['weights'].values,
    n_bootstraps=2000,
    ci=0.95,
    random_state=42
)

print(f"AUC: {auc:.2f} (95% CI {auc_lower:.2f}–{auc_upper:.2f})")
print(f"Accuracy: {accuracy:.2f} (95% CI {acc_lower:.2f}–{acc_upper:.2f})")
