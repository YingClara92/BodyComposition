import numpy as np
import pandas as pd
from RM_highconv import rm_highconv
from lifelines.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("error", category=ConvergenceWarning)
from scipy import stats
import statsmodels.formula.api as smf

# file = 'MyocardiumF1.csv'
# target = 'myocardium_volume'
# data = pd.read_csv(file)
# data = data.dropna(subset=[target])
# # data['muscle_body_ratio'] = data['subcutaneous_fat_HU_mean']
#
# df = pd.read_csv('ML_speed1.csv')
# df = df[df['speed_ML_ratio'] >0.32]
# df = df[df['speed_ML_ratio'] <-0.29]
# data = data[data['patient'].isin(df['patient'].unique())]
#
# time = data['time'].unique()
# patients = data['patient'].unique()
#
# data = data.reset_index(drop=True)
#
# drop_patients = []
# for p in patients:
#     data_p = data[data['patient'] == p]
#     data_p = data_p[data_p['time'] != 0]
#     if len(data_p['time'].unique()) < 3:
#         drop_patients.append(p)
#
# data = data[~data['patient'].isin(drop_patients)]
# patients = data['patient'].unique()


file = 'BodyCompT.csv'
target = 'body_L1'
data = rm_highconv(file)
data = data.dropna(subset=[target])


data.loc[:, 'muscle_body_ratio'] = data['skeletal_muscle_volume'] / data['body_L1_length']
data['torso_body_ratio'] = data['torso_fat_volume'] / data['body_L1_length']
data['subcutaneous_fat_body_ratio'] = data['subcutaneous_fat_volume'] / data['body_L1_length']
target = 'muscle_body_ratio'
data = data.reset_index(drop=True)
drop_patients = []
patients = data['patient'].unique()
for p in patients:
    data_p = data[data['patient'] == p]
    data_p = data_p[data_p['time'] != 0]
    if len(data_p['time'].unique()) < 2:
        drop_patients.append(p)

data = data[~data['patient'].isin(drop_patients)]
patients = data['patient'].unique()

df = pd.read_csv('BC_speed1.csv')
df = df[df['speed_muscle_body_ratio'] < -0.42]
data = data[data['patient'].isin(df['patient'].unique())]


data["time"] = pd.to_numeric(data["time"], errors="coerce")


# ============================================================
# Linear assumption validation
# ============================================================
# Important interpretation:
# This code tests whether there is evidence AGAINST linearity.
# It does not prove that the linear assumption is true.
# ============================================================

from statsmodels.nonparametric.smoothers_lowess import lowess


# ------------------------------------------------------------
# 1. Build normalized patient-time dataframe
# ------------------------------------------------------------

normalized_rows = []
skipped_patients = []

for p in patients:
    data_p = data[data["patient"] == p].copy()
    time_p = data_p['time'].unique()
    # Baseline value at time == 0
    pre_rt = data_p.loc[data_p["time"] == 0, target].mean()

    if pre_rt == 0 or np.isnan(pre_rt):
        skipped_patients.append(p)
        continue

    # if max(time_p) <= 12:
    #     continue

    # Average repeated measurements at same time point
    grouped = (
        data_p
        .groupby("time", as_index=False)[target]
        .mean()
        .sort_values("time")
        .reset_index(drop=True)
    )

    grouped["patient"] = p
    grouped["y_norm"] = grouped[target] / pre_rt

    normalized_rows.append(grouped[["patient", "time", "y_norm"]])

data_pt = pd.concat(normalized_rows, ignore_index=True)
# ------------------------------------------------------------
# 3. Cohort-level linear vs spline comparison
# ------------------------------------------------------------
# This should be your main analysis.
# Mixed-effects model accounts for repeated measures within patient.
# ------------------------------------------------------------

def fit_mixed_model_safely(formula, df):
    """
    Fit random-intercept-only MixedLM using several optimizers.
    """

    optimizers = ["lbfgs", "powell", "cg", "bfgs", "nm"]

    last_error = None

    for opt in optimizers:
        try:
            print(f"\nTrying optimizer: {opt}")

            model = smf.mixedlm(
                formula,
                data=df,
                groups=df["patient"],
                re_formula="~time"
            ).fit(
                reml=False,
                method=opt,
                maxiter=1000,
                disp=False
            )

            print("Converged:", model.converged)

            if model.converged:
                return model, f"random intercept only, optimizer={opt}"

            last_error = f"Optimizer {opt} did not converge"

        except Exception as e:
            print(f"Optimizer {opt} failed: {e}")
            last_error = e

    raise RuntimeError(f"All optimizers failed. Last issue: {last_error}")


df_mixed = data_pt.copy()

# Linear time model
mixed_linear, linear_random_structure = fit_mixed_model_safely(
    "y_norm ~ time",
    df_mixed,
)

# Nonlinear spline time model
mixed_spline, spline_random_structure = fit_mixed_model_safely(
    "y_norm ~ cr(time, df=3)",
    df_mixed,
)

# print("\nCohort-level linear mixed model:")
# print(mixed_linear.summary())
#
# print("\nCohort-level spline mixed model:")
# print(mixed_spline.summary())
#
# print("\nRandom-effect structure used:")
# print("Linear model:", linear_random_structure)
# print("Spline model:", spline_random_structure)


# ------------------------------------------------------------
# 4. Cohort-level model comparison
# ------------------------------------------------------------

linear_aic = mixed_linear.aic
spline_aic = mixed_spline.aic
delta_aic = spline_aic - linear_aic

linear_bic = mixed_linear.bic
spline_bic = mixed_spline.bic
delta_bic = spline_bic - linear_bic

print("\nCohort-level model comparison:")
print(f"Linear AIC: {linear_aic:.3f}")
print(f"Spline AIC: {spline_aic:.3f}")
print(f"Delta AIC, spline - linear: {delta_aic:.3f}")

print(f"\nLinear BIC: {linear_bic:.3f}")
print(f"Spline BIC: {spline_bic:.3f}")
print(f"Delta BIC, spline - linear: {delta_bic:.3f}")


# Likelihood ratio test
lr_stat = 2 * (mixed_spline.llf - mixed_linear.llf)

try:
    df_diff = int(mixed_spline.df_modelwc - mixed_linear.df_modelwc)
except Exception:
    df_diff = int(len(mixed_spline.params) - len(mixed_linear.params))

if df_diff > 0:
    lr_p_value = stats.chi2.sf(lr_stat, df_diff)
else:
    lr_p_value = np.nan

print("\nLikelihood ratio test, spline vs linear:")
print(f"LR statistic: {lr_stat:.3f}")
print(f"df difference: {df_diff}")
print(f"p-value: {lr_p_value:.3e}")


# ------------------------------------------------------------
# 5. Interpretation
# ------------------------------------------------------------

print("\nInterpretation:")

if delta_aic <= -2 and delta_bic <= 0 and np.isfinite(lr_p_value) and lr_p_value < 0.05:
    print(
        "There is evidence against the linear time assumption. "
        "The spline model appears to fit better at the cohort level."
    )

elif delta_aic >= 2 and delta_bic >= 0:
    print(
        "The linear model is preferred or appears sufficient at the cohort level. "
        "This does not prove linearity, but there is no strong evidence that spline improves fit."
    )

else:
    print(
        "Evidence is mixed. The linear model may be acceptable, but the spline model should be considered in sensitivity analysis."
    )