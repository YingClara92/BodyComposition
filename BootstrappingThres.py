import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy import build_design_matrices
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
from lifelines.utils import ConvergenceError
from lifelines.exceptions import ConvergenceWarning
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('iptw_weighted_data_ML.csv')
target = 'speed_ML_ratio'#

base = data[["survival_months", "deceased", target,'iptw_weight']].dropna().reset_index(drop=True)
n = base.shape[0]*1.0
idx_all = np.arange(n)

n_boot = 10000
lb = base[target].min()
ub = base[target].max()
ML_range_fixed = np.linspace(lb, ub, 200)
events = base[base["deceased"] == 1]
censored = base[base["deceased"] == 0]
n_events = int(len(events)*1.5)
n_cens = int(len(censored)*1.5)
for iter in range(1):
    test_ratio = []
    cindex_oob = []
    partial_hazard_list = []
    lb_thres = []
    ub_thres = []
    seed = np.random.randint(0, 10000)
    rng = np.random.default_rng(seed)
    nums = rng.integers(0, 10000, size=n_boot)
    for i in range(n_boot):
        boot_events = events.sample(n=n_events, replace=True, random_state=nums[i])
        boot_cens = censored.sample(n=n_cens, replace=True, random_state=nums[i])

        df_train = pd.concat([boot_events, boot_cens], axis=0)
        frac = n / df_train.shape[0]
        df_train = df_train.sample(frac=frac)  # shuffle

        # OOB = those not present in the bootstrapped sample
        df_oob = base.drop(df_train.index)

        df_train = df_train.reset_index(drop=True)
        spline_train = dmatrix(
            f"bs(speed_ML_ratio, df=3, include_intercept=False, lower_bound={lb}, upper_bound={ub})",
            {"speed_ML_ratio": df_train[target]},
            return_type='dataframe'
        )

        df_train_spline = df_train.drop(columns=[target]).join(spline_train)

        cph = CoxPHFitter(penalizer=0.01)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            try:
                cph.fit(df_train_spline, duration_col='survival_months', event_col='deceased', weights_col='iptw_weight', robust=True)
            except (ConvergenceWarning, ConvergenceError):
                # skip this bootstrap sample if it doesn't converge
                continue

        test_ratio.append(df_oob.shape[0] / n)
        if len(test_ratio) == 1000:
            break
        # ----- OOB c-index -----
        spline_base = build_design_matrices(
            [spline_train.design_info],
            {"speed_ML_ratio": base[target]}
        )[0]
        df_base_spline = pd.DataFrame(spline_base, columns=spline_train.columns)
        risk_base = cph.predict_partial_hazard(df_base_spline).values.ravel()

        y_train = Surv.from_arrays(
            event=df_train["deceased"].astype(bool).values,
            time=df_train["survival_months"].values,
        )

        y_base = Surv.from_arrays(
            event=base["deceased"].astype(bool).values,
            time=base["survival_months"].values,
        )

        c_uno_base = concordance_index_ipcw(y_train, y_base, risk_base)

        # Uno's C-index (IPCW)
        risk_train = cph.predict_partial_hazard(df_train_spline).values.ravel()
        c_uno = concordance_index_ipcw(y_train, y_train, risk_train)
        cindex_oob.append(c_uno_base[0] - c_uno[0])

        ML_spline_fixed = build_design_matrices(
            [spline_train.design_info],
            {"speed_ML_ratio": ML_range_fixed}
        )[0]
        ML_spline_df = pd.DataFrame(ML_spline_fixed, columns=spline_train.columns)
        partial_hazard = cph.predict_partial_hazard(ML_spline_df)
        partial_hazard_list.append(partial_hazard.values.flatten())

        spline = UnivariateSpline(ML_range_fixed, partial_hazard - 1, s=0)
        roots = spline.roots()
        if len(roots) >= 2:
            lb_thres.append(roots[0])
            ub_thres.append(roots[1])

## print thres1 and thres2 mean and std
print(f'Threshold 1: Mean = {np.mean(lb_thres)}, Std = {np.std(lb_thres)}')
print(f'Threshold 2: Mean = {np.mean(ub_thres)}, Std = {np.std(ub_thres)}')
print('seed',seed)

cindex_oob = np.array(cindex_oob)
print("OOB c-index: mean={:.2f}, std={:.2f}, 95% CI=({:.2f}, {:.2f})".format(
    cindex_oob.mean(),
    cindex_oob.std(),
    np.quantile(cindex_oob, 0.025),
    np.quantile(cindex_oob, 0.975)
))

## print test ratio 95% CI
test_ratio = np.array(test_ratio)
print(len(test_ratio))
print("OOB test ratio: mean={:.2f}, 95% CI=({:.2f}, {:.2f})".format(
    test_ratio.mean(),
    np.quantile(test_ratio, 0.025),
    np.quantile(test_ratio, 0.975)
))
print("test raio std",test_ratio.std())