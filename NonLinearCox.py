import pandas as pd
from patsy import dmatrix
from lifelines import CoxPHFitter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from scipy.stats import chi2
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import StandardScaler
##speed_subcutaneous_fat_body_ratio -1
# speed_muscle_body_ratio -0.5
# speed_torso_body_ratio -0.5

df = pd.read_csv('ML_survival.csv')
target = 'speed_ML_ratio'#

# df = pd.read_csv('BC_speed1.csv')
# target = 'speed_muscle_body_ratio'#'speed_subcutaneous_fat_body_ratio'#'speed_torso_body_ratio'#
# # print(len(df))
# df = df[["survival_months", "deceased", target]]
# plt.hist(df[target], bins=30, edgecolor='black')
# plt.show()
df = df[["survival_months", "deceased", target]]


# cph_reduced = CoxPHFitter(penalizer=0.01)
# cph_reduced.fit(df, duration_col='survival_months', event_col='deceased')
# cph_reduced.print_summary()


spline_terms = dmatrix("bs(speed_ML_ratio, df=3, include_intercept=False)", {"speed_ML_ratio": df[target]}, return_type='dataframe')
# spline_terms = dmatrix("cr(speed_ML_ratio, df=3, constraints='center') - 1", {"speed_ML_ratio": df[target]}, return_type='dataframe')
df_spline = df.drop(columns=[target]).join(spline_terms)

print(len(df))

cph = CoxPHFitter(penalizer=0.01)
cph.fit(df_spline, duration_col='survival_months', event_col='deceased')
cph.print_summary()

# df_reduced = df[["survival_months", "deceased"]]  # No spline, no speed_ML_ratio
# cph_reduced = CoxPHFitter()
# cph_reduced.fit(df_reduced, duration_col='survival_months', event_col='deceased')
#
# # Perform likelihood ratio test
# ll_full = cph.log_likelihood_
# ll_reduced = cph_reduced.log_likelihood_
# lr_stat = 2 * (ll_full - ll_reduced)
# df_diff = cph.params_.shape[0] - cph_reduced.params_.shape[0]
# p_value = chi2.sf(lr_stat, df_diff)
#
# print(f"Likelihood Ratio Test Statistic: {lr_stat}")
# print(f"Degrees of Freedom: {df_diff}")
# print(f"P-value: {p_value}")

# Predict the partial hazard
ML_range = np.linspace(df[target].min(), df[target].max(), 200)
ML_spline = dmatrix("bs(speed_ML_ratio, df=3, include_intercept=False)", {"speed_ML_ratio": ML_range}, return_type='dataframe')
# df_plot = pd.concat([pd.DataFrame({'speed_ML_ratio': ML_range}), ML_spline], axis=1)
df_plot = pd.DataFrame(ML_spline, columns=ML_spline.columns)
partial_hazard = cph.predict_partial_hazard(df_plot)

# ## please get the value whose partial hazard is 1, and then use this value to cluster the data
# # Find the value of speed_ML_ratio where the partial hazard is closest to 1
# closest_index = np.abs(partial_hazard - 1)
# ## sort the index to get the first and second closest value
# closest_index = closest_index.argsort()[:2]
# ML_value_1 = ML_range[closest_index[0]]
# ML_value_2 = ML_range[closest_index[1]]
# print(f"Value of {target} where partial hazard is closest to 1: {ML_value_1}, {ML_value_2}")

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

# cluster_num = 2
# df['cluster_' + target] = 0
# df['cluster_' + target] = np.where(df[target] <= -0.5, 1, df['cluster_' + target])
# target_names = ['SKM Preserved','SKM Loss']

cluster_num = 3
df['cluster_' + target] = 0
df['cluster_' + target] = np.where(df[target] < -0.25, 1, df['cluster_' + target])
df['cluster_' + target] = np.where(df[target] > 0.28, 2, df['cluster_' + target])
target_names = ['Cardiac Stable','Cardiac Atrophy', 'Cardiac Hypertrophy']

for cluster in range(cluster_num):
    data_c = df[df['cluster_' + target] == cluster][target]
    if cluster == 0:
        plt.hist(data_c, bins=10)
    else:
        plt.hist(data_c, bins=20)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$v_{\mathrm{muscle}}$(%)', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.legend(target_names, fontsize=12, frameon=False, loc='upper left')
plt.title(r'Histogram by $v_{\mathrm{muscle}}$ Group', fontsize=14)
plt.show()

kmf_models = []
# fig, ax = plt.subplots(figsize=(6, 6.2))
fig, ax = plt.subplots(figsize=(6, 5.5)) ## 5.4
timeline = np.arange(0, 80, 1)  # choose your step (e.g., every 3 months)

for i in range(1, cluster_num + 1):
    kmf = KaplanMeierFitter()
    mask = (df[f'cluster_{target}'] == i-1)
    T = df.loc[mask, 'survival_months'].astype(float)
    E = df.loc[mask, 'deceased'].astype(int)

    kmf.fit(T, event_observed=E, timeline=timeline, label=target_names[i-1])
    kmf.plot_survival_function(ax=ax)
    kmf_models.append(kmf)

# 2) Fix limits and ticks BEFORE adding the risk table
ax.set_xlim(0, 84)
tick_positions = np.arange(0, 80, 12)  # show yearly ticks
ax.xaxis.set_major_locator(mticker.FixedLocator(tick_positions))
ax.set_xticks(tick_positions)
ax.set_xticklabels([str(int(t)) for t in tick_positions])

# Now the risk counts will align with those ticks
add_at_risk_counts(*kmf_models, ax=ax, fig=fig, fontsize=12)

# Styling
ax.set_title(r'Kaplan–Meier Survival Curves by $v_{\mathrm{muscle}}$', fontsize=12)
ax.set_xlabel('Months', fontsize=12)
ax.set_ylabel('Overall Survival Proportion', fontsize=12)
ax.legend(fontsize=12, frameon=False)
ax.tick_params(axis='both', which='major', labelsize=12)
## don't have frame for the survival plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
## don't show legend
# ax.legend_.remove()
# Give a little bottom margin so the risk table isn't clipped
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()

from lifelines.statistics import logrank_test
results = logrank_test(df[df['cluster_' + target] == 1]['survival_months'],
                    df[df['cluster_' + target] == 0]['survival_months'],
                    df[df['cluster_' + target] == 1]['deceased'],
                    df[df['cluster_' + target] == 0]['deceased'], alpha=0.95)
print('logrank test p',results.p_value)


results = logrank_test(df[df['cluster_' + target] == 2]['survival_months'],
                    df[df['cluster_' + target] == 0]['survival_months'],
                    df[df['cluster_' + target] == 2]['deceased'],
                    df[df['cluster_' + target] == 0]['deceased'], alpha=0.95)
print('logrank test p',results.p_value)



results = logrank_test(df[df['cluster_' + target] == 1]['survival_months'],
                    df[df['cluster_' + target] == 2]['survival_months'],
                    df[df['cluster_' + target] == 1]['deceased'],
                    df[df['cluster_' + target] == 2]['deceased'], alpha=0.95)
print('logrank test p',results.p_value)


# when plot the partial hazard, use color to distinguish the clusters, for example, for ML_range within [-0.5, -0.25], use red color, for ML_range within [-0.25, 0.28], use green color, and for ML_range > 0.28, use blue color
plt.figure(figsize=(8, 6))
for i in range(cluster_num):
    mask = (ML_range < -0.25) if i == 1 else (ML_range > 0.28) if i == 2 else (ML_range >= -0.25) & (ML_range <= 0.28)
    plt.plot(ML_range[mask], partial_hazard[mask], label=target_names[i])

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r'$v_{\mathrm{myo}}$(%)', fontsize=20)
plt.ylabel('Partial Hazard', fontsize=20)
# plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
plt.title('Effect of $v_{\mathrm{myo}}$ on Hazard', fontsize=20)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, fontsize=20)
plt.tight_layout()
plt.show()

# # when plot the partial hazard, use color to distinguish the clusters, for example, for ML_range <= -0.5, use red color, for ML_range between >= -0.5 ause green color
# plt.figure(figsize=(8, 6))
# for i in range(cluster_num):
#     mask = (ML_range <= -0.5) if i == 1 else (ML_range > -0.5)
#     plt.plot(ML_range[mask], partial_hazard[mask], label=target_names[i])
# ## set font size for x tick labels
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# ## horizontal line at y=1
# # plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
# plt.xlabel(r'$v_{\mathrm{muscle}}$(%)', fontsize=20)
# plt.ylabel('Partial Hazard', fontsize=20)
# plt.title('Effect of $v_{\mathrm{muscle}}$ on Hazard', fontsize=20)
# ## don't have frame for the plot
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.legend(frameon=False, fontsize=20)
# ## dont have frame for the legend
#
# plt.tight_layout()
# plt.show()

## time at 30 months
print("Survival probability at 30 months for group 0:", kmf_models[0].survival_function_at_times(30))
print("Survival probability at 30 months for group 1:", kmf_models[1].survival_function_at_times(30))
print("Survival probability at 30 months for group 2:", kmf_models[2].survival_function_at_times(30))
## time at 32 months
# print("Survival probability at 32 months for group 0:", kmf_models[0].survival_function_at_times(32))
# print("Survival probability at 32 months for group 1:", kmf_models[1].survival_function_at_times(32))
# print("Survival probability at 32 months for group 2:", kmf_models[2].survival_function_at_times(32))
# ## time at 36 months
# print("Survival probability at 36 months for group 0:", kmf_models[0].survival_function_at_times(36))
# print("Survival probability at 36 months for group 1:", kmf_models[1].survival_function_at_times(36))
# print("Survival probability at 36 months for group 2:", kmf_models[2].survival_function_at_times(36))
# ## get confidence intervals for the survival curves at 30, 32, and 36 months
# print("Confidence intervals at 30 months for group 0:", kmf_models[0].confidence_interval_(30))
# print("Confidence intervals at 30 months for group 1:", kmf_models[1].confidence_interval_(30))
# print("Confidence intervals at 30 months for group 2:", kmf_models[2].confidence_interval_(30))
#
