import pandas as pd
import numpy as np
from scipy import stats
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.plotting import add_at_risk_counts

# target = 'speed_subcutaneous_fat_body_ratio'#'speed_subcutaneous_fat_body_ratio' #'speed_muscle_body_ratio' #
# data = pd.read_csv('BC_speed1.csv')
# data = pd.read_csv('ML_speed1.csv')
# target = 'speed_ML_ratio'
# patients_total = data['patient'].unique()

patients_group1 = pd.read_csv('recording1.csv')
patients_group2 = pd.read_csv('recording2.csv')

patients_group1 = patients_group1['patient'].unique()
patients_group2 = patients_group2['patient'].unique()
patients_total = np.union1d(patients_group1, patients_group2)



data = pd.read_csv('survival.csv')

# lung_data = pd.read_csv('check3.csv')
# patients_total = lung_data['patient'].unique()

sbrt_info = pd.read_csv('UCLH_PatientPlansSelected.csv')
## merget the data based on the patient
data = pd.merge(data, sbrt_info[['patient', 'SABR', 'TotalDosePerFraction']], on='patient', how='left')
## drop duplicates based on patient
data = data.drop_duplicates(subset='patient')

gtv = pd.read_csv('GTVVolumecm3.csv')
## for patientID remove the string 'UCLH_NSCLC/'
gtv['patient'] = gtv['PatientID'].str.replace('UCLH_NSCLC/', '', regex=False).astype(int)

data = pd.merge(data, gtv[['patient', 'GTVVolume']], on='patient', how='left')


## drop patient having no SABR information
data = data[data['SABR'].notna()]
## drop patient having no survival information
data = data[data['survival_months'].notna()]
data = data[data['deceased'].notna()]
## for whole cohort and lung_data cohort
data = data[data['patient'].isin(patients_total)]

## save data['patient'] to a npy array
np.save('included_patients.npy', data['patient'].values)

data = data.reset_index(drop=True)

data['SABR'] = data['SABR'].map({'Yes': 1, 'No': 0})
df = data.copy()

## print the mean and std of GTVVolume for SABR and non-SABR patients
# sbrt_volume= data[data["SABR"] == 1]["speed_subcutaneous_fat_body_ratio"]
# non_sbrt_volume = data[data["SABR"] == 0]["speed_subcutaneous_fat_body_ratio"]
# ## get the mean and std of GTVVolume for SABR and non-SABR patients
# print(f"SABR GTV Volume - Mean: {sbrt_volume.mean():.2f}, Std: {sbrt_volume.std():.2f}")
# print(f"Non-SABR GTV Volume - Mean: {non_sbrt_volume.mean():.2f}, Std: {non_sbrt_volume.std():.2f}")

kmf_models = []
fig, ax = plt.subplots(figsize=(6, 6))

str_cluster = ['Cluster 1', 'Cluster 2']

labels = ['Non-SABR', 'SABR']

for i in range(2):
    kmf = KaplanMeierFitter()
    mask = (df["SABR"] == i)
    T = df[mask]['survival_months']
    E = df[mask]['deceased']
    ## identify how many patients survived less than 3 months
    three_months = np.sum((T < 3) & (E == 1))
    twelve_months = np.sum((T < 12) & (E == 1))
    twenty_months = np.sum((T < 24) & (E == 1))
    eighty_months = np.sum((T < 86) & (E == 1))
    print(three_months, twelve_months, twenty_months, eighty_months)



    kmf.fit(T, E, label=labels[i])
    kmf.plot_survival_function(ax=ax)
    kmf_models.append(kmf)
ax.set_xlim(0, 84)
ax.set_ylim(0, 1)
## increase the font size of added at risk counts
add_at_risk_counts(*kmf_models, ax=ax, fontsize=12)
## set title and labels
ax.set_title(r'Myocardium Cohort Survival Curves by SABR Treatment')
ax.set_xlabel('Months')
ax.set_ylabel('Overall Survival Proportion')
## set title font size
ax.title.set_fontsize(12)
## set labels font size
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
## legend font size
ax.legend(fontsize=12)
## set tick font size
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()

## # Perform log-rank test
from lifelines.statistics import logrank_test
results = logrank_test(df[df['SABR'] == 0]['survival_months'],
                       df[df['SABR'] == 1]['survival_months'],
                       event_observed_A=df[df['SABR'] == 0]['deceased'],
                       event_observed_B=df[df['SABR'] == 1]['deceased'])
print(f"Log-rank test p-value: {results.p_value:.4f}")

# ## onehot encode patient_sex
# df = pd.get_dummies(df, columns=['patient_sex'], drop_first=True)
#
#
# group1 = df[df['SABR'] == 1]['patient_sex_Male']
# group2 = df[df['SABR'] == 0]['patient_sex_Male']
# ## calculate chi2-contingency
# from scipy.stats import chi2_contingency
# contingency_table = pd.crosstab(df['SABR'], df['patient_sex_Male'])
# chi2, p, dof, expected = chi2_contingency(contingency_table)
# print(f'Chi2 statistic: {chi2}, p-value: {p}')
#
# ## remove nan values
# group1 = group1.dropna()
# group2 = group2.dropna()
# t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
# print(f'T-test statistic: {t_stat}, p-value: {p_value}')

kmf = KaplanMeierFitter()

## fit with entire cohort
T = df['survival_months']
E = df['deceased']
kmf.fit(T, E, label='Overall Survival')
ax = kmf.plot_survival_function(figsize=(6, 6))
## plot the confidence interval
ax.set_xlim(0, 84)
add_at_risk_counts(kmf, ax=ax, fontsize=12)
plt.show()


## print the median survival time
median_survival = kmf.median_survival_time_
print(f'Median survival time (months): {median_survival}')
plt.title('Overall Survival Curve for Myocardium Cohort')
