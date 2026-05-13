import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def bootstrap_auc_ci(y_true, y_score, sample_weight=None,
                     n_bootstraps=2000, ci=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    # AUC on the full sample
    auc = roc_auc_score(y_true, y_score, sample_weight=sample_weight)

    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # Sample with replacement
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample
            continue

        if sample_weight is not None:
            sw = sample_weight[indices]
        else:
            sw = None

        score = roc_auc_score(y_true[indices], y_score[indices], sample_weight=sw)
        ## change score to accuracy
        # threshold = 0.5
        # predicted_classes = (y_score[indices] >= threshold).astype(int)
        # score = accuracy_score(y_true[indices], predicted_classes, sample_weight=sw)

        bootstrapped_scores.append(score)

    bootstrapped_scores = np.array(bootstrapped_scores)

    alpha = (1.0 - ci) / 2.0
    lower = np.percentile(bootstrapped_scores, 100 * alpha)
    upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha))

    return auc, lower, upper

data = pd.read_csv('data_body3.csv')
y = data['cluster_muscle_body_ratio'].values
selected_feature = ['esophagus_v10', 'heart_atrium_right_v20', 'heart_myocardium_v10', 'heart_v10']  # features from DoseFeatureSelection.py

# data = pd.read_csv('data_BED_m.csv')
# data = data[data['cluster_ML_new'] != 1] ## options here are 1, 2
#
# from sklearn.preprocessing import OneHotEncoder
# y = data['cluster_ML_new'].values
# encoder = OneHotEncoder()
# y_onehot = encoder.fit_transform(y.reshape(-1, 1))
# y_onehot = y_onehot.toarray()[:, 1]
#
# selected_feature = [ 'heart_atrium_right_v10', 'TotalDosePerFraction']
# # selected_feature = ['aorta_v5', 'heart_myocardium_v10', 'age']
X = data[selected_feature].values

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.metrics import brier_score_loss
import statsmodels.api as sm

df = pd.DataFrame(X_scaled, columns=selected_feature)
df['y'] = y

counts = df['y'].value_counts().to_dict()
total = len(df)
n_classes = df['y'].nunique()
df['weights'] = df['y'].apply(lambda cls: total / (n_classes * counts[cls]))


remaining_features = selected_feature.copy()
pval_threshold = 0.1

while True:
    X_sm = sm.add_constant(df[remaining_features])

    model = sm.GLM(
        df['y'],
        X_sm,
        family=sm.families.Binomial(),
        freq_weights=df['weights']
    )

    result = model.fit()

    # Exclude intercept from elimination
    pvalues = result.pvalues.drop('const')

    max_pval = pvalues.max()
    worst_feature = pvalues.idxmax()

    print("\nCurrent features:", remaining_features)
    print("P-values:")
    print(pvalues)

    if max_pval > pval_threshold:
        print(f"Removing '{worst_feature}' with p-value = {max_pval:.4f}")
        remaining_features.remove(worst_feature)
    else:
        print("\nBackward elimination complete.")
        break

    # Stop if no predictors remain
    if len(remaining_features) == 0:
        print("\nNo predictors remain after backward elimination.")
        break

# Final model
if len(remaining_features) > 0:
    X_final = sm.add_constant(df[remaining_features])

    final_model = sm.GLM(
        df['y'],
        X_final,
        family=sm.families.Binomial(),
        freq_weights=df['weights']
    )

    final_result = final_model.fit()

    print("\nFinal selected features:")
    print(remaining_features)

    print("\nFinal model summary:")
    print(final_result.summary())

    print("\nFinal model p-values:")
    print(final_result.pvalues)
else:
    final_result = None
    print("No final model was fitted because all predictors were removed.")

coef   = final_result.params
se     = final_result.bse
z      = final_result.tvalues            # z because Logit uses large-sample normal approx
pvals  = final_result.pvalues
conf   = final_result.conf_int()         # 2-column DF

# convert to odds ratios
or_        = np.exp(coef)
ci_lower   = np.exp(conf[0])
ci_upper   = np.exp(conf[1])

# assemble publication table
table = pd.DataFrame({
    "Odds ratio": or_.round(2),
    "95% CI low": ci_lower.round(2),
    "95% CI high": ci_upper.round(2),
    "p": pvals.round(3),
})
table.index.name = "Predictor"

print(table)

predicted_probs = result.predict(X_sm)

## calculate brier score
brier = brier_score_loss(df['y'], predicted_probs, sample_weight=df['weights'])
print('Brier score:', brier)

# Convert probabilities to binary class predictions using a threshold (default 0.5)
threshold = 0.5
predicted_classes = (predicted_probs >= threshold).astype(int)

from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
# Calculate classification performance metrics
cm = confusion_matrix(df['y'], predicted_classes)
accuracy = accuracy_score(df['y'], predicted_classes)
auc = roc_auc_score(df['y'], predicted_probs)


# AUC + 95% CI (weighted, to match your model)
auc, auc_lower, auc_upper = bootstrap_auc_ci(
    df['y'].values,
    predicted_probs,
    sample_weight=df['weights'].values,
    n_bootstraps=2000,
    ci=0.95,
    random_state=42
)

print(f"AUC: {auc:.2f} (95% CI {auc_lower:.2f}–{auc_upper:.2f})")

