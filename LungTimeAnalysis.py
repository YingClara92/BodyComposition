import pandas as pd
import numpy as np


data = pd.read_csv('lung_data_smooth_clean_ratio_4.csv')
## drop data whose label is nan
data = data.dropna(subset=['time_label'])

ordered_labels = ['early', 'mid', 'late', 'very late']

treatment_info = pd.read_csv('UCLH_PatientPlansSelected.csv')

dose_info = pd.read_csv('dose_metrics_lung2.csv')

patients = data['patient'].unique()

sabr_patients = treatment_info[treatment_info['SABR'] == 'Yes']['patient'].unique()

non_sabr_patients = treatment_info[treatment_info['SABR'] == 'No']['patient'].unique()

sabr_patients = list(set(patients).intersection(set(sabr_patients)))
non_sabr_patients = list(set(patients).intersection(set(non_sabr_patients)))

# data['ipsi_change'] = np.nan
# data['contra_change'] = np.nan
# data['sabr'] = np.nan
# data['location'] = pd.Series(dtype='str')
# for p in patients:
#     patient_data = data[data['patient'] == p]
#     location_p = dose_info[dose_info['patient'] == p]['location'].values#.split('_')[-1]
#     if len(set(location_p)) > 1:
#         print(p, 'check location', location_p)
#     if p in sabr_patients:
#         data.loc[data['patient'] == p, 'sabr'] = 1
#     else:
#         data.loc[data['patient'] == p, 'sabr'] = 0
#
#     data.loc[data['patient'] == p, 'location'] = location_p[0]
#
#     if location_p[0] == 'left':
#         data.loc[data['patient'] == p, 'ipsi_change'] = patient_data['lung_left_volume_ratio']
#         data.loc[data['patient'] == p, 'contra_change'] = patient_data['lung_right_volume_ratio']
#     elif location_p[0] == 'right':
#         data.loc[data['patient'] == p, 'ipsi_change'] = patient_data['lung_right_volume_ratio']
#         data.loc[data['patient'] == p, 'contra_change'] = patient_data['lung_left_volume_ratio']
#
# data['ipsi_contra_ratio'] =  data['ipsi_change']/data['contra_change']
# data.to_csv('lung_data_smooth_clean_ratio_4.csv', index=False)

import matplotlib.pyplot as plt
import seaborn as sns


data = data[['patient', 'date', 'time_label', 'ipsi_change', 'contra_change', 'sabr', 'ipsi_contra_ratio']]

# data = data.groupby(['patient', 'date']).mean().reset_index()
data = data.groupby(['patient', 'date'], as_index=False).agg({
    'time_label': 'first',
    'ipsi_change': 'mean',
    'contra_change': 'mean',
    'sabr': 'first',
    'ipsi_contra_ratio': 'mean'
})

print(data['time_label'].value_counts())
print(len(data['patient'].unique()))

sabr_data = data[data['patient'].isin(sabr_patients)]
non_sabr_data = data[data['patient'].isin(non_sabr_patients)]

sabr_early = sabr_data[sabr_data['time_label'] == 'early']
non_sabr_early = non_sabr_data[non_sabr_data['time_label'] == 'early']

sabr_mid = sabr_data[sabr_data['time_label'] == 'mid']
non_sabr_mid = non_sabr_data[non_sabr_data['time_label'] == 'mid']

sabr_late = sabr_data[sabr_data['time_label'] == 'late']
non_sabr_late = non_sabr_data[non_sabr_data['time_label'] == 'late']

sabr_very_late = sabr_data[sabr_data['time_label'] == 'very late']
non_sabr_very_late = non_sabr_data[non_sabr_data['time_label'] == 'very late']

# Prepare data for SABR patients
## for the patients who belongs to the same time label, average the ipsi_change and contra_change
sabr_data_melted = sabr_data[['time_label', 'ipsi_change', 'contra_change', 'ipsi_contra_ratio']].copy()
sabr_data_melted = sabr_data_melted.melt(id_vars='time_label',
                                         value_vars=['ipsi_change', 'contra_change','ipsi_contra_ratio'],
                                         var_name='Side', value_name='Change')
sabr_data_melted['Side'] = sabr_data_melted['Side'].replace({'ipsi_change': '$r_{lung-ipsi}$', 'contra_change': '$r_{lung-contra}$', 'ipsi_contra_ratio': '$\dot{r}_{\mathrm{lung}}$'})

# Prepare data for non-SABR patients
non_sabr_data_melted = non_sabr_data[['time_label', 'ipsi_change', 'contra_change', 'ipsi_contra_ratio']].copy()
non_sabr_data_melted = non_sabr_data_melted.melt(id_vars='time_label',
                                                  value_vars=['ipsi_change', 'contra_change', 'ipsi_contra_ratio'],
                                                  var_name='Side', value_name='Change')
non_sabr_data_melted['Side'] = non_sabr_data_melted['Side'].replace({'ipsi_change': '$r_{lung-ipsi}$', 'contra_change': '$r_{lung-contra}$', 'ipsi_contra_ratio': '$\dot{r}_{\mathrm{lung}}$'})


sabr_data_melted['time_label'] = pd.Categorical(sabr_data_melted['time_label'], categories=ordered_labels, ordered=True)
non_sabr_data_melted['time_label'] = pd.Categorical(non_sabr_data_melted['time_label'], categories=ordered_labels, ordered=True)

flierprops = dict(marker='d', markerfacecolor='k', markeredgecolor='k', markersize=6)
# Step 3: Set up the figure
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# SABR plot
sns.boxplot(data=non_sabr_data_melted, x='time_label', y='Change', hue='Side', ax=axes[0], showfliers=True, showmeans=True, flierprops=flierprops)

# Non-SABR plot
sns.boxplot(data=sabr_data_melted, x='time_label', y='Change', hue='Side', ax=axes[1], showfliers=True, showmeans=True, flierprops=flierprops)
axes[0].axhline(y=1, color='r', linestyle='--')
axes[0].fill_betweenx(y=[0.8, 1.2], x1=-0.5, x2=3.5, color='yellow', alpha=0.2)
axes[0].fill_betweenx(y=[0.9, 1.1], x1=-0.5, x2=3.5, color='red', alpha=0.2)
axes[0].set_title('dRT Patients', fontsize=12)
axes[0].set_ylabel('Volume Ratio Change (V/V0)', fontsize=12)
axes[0].set_xlabel('Time Label', fontsize=12)
## set x limit label size bigger
axes[0].tick_params(axis='x', labelsize=12)
# axes[1].set_ylim(0, 2)
axes[0].legend(frameon=False)
## no filled color for legend
axes[0].legend(loc='upper left', fontsize=14)

axes[0].tick_params(axis='x', labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)


axes[1].set_title('SBRT Patients', fontsize=12)
## horizantal line at y=1
axes[1].axhline(y=1, color='r', linestyle='--')
## create an shaded area between 1.1 and 0.9
axes[1].fill_betweenx(y=[0.8, 1.2], x1=-0.5, x2=3.5, color='yellow', alpha=0.2)
axes[1].fill_betweenx(y=[0.9, 1.1], x1=-0.5, x2=3.5, color='red', alpha=0.2)
## set y-axis limit to 0.5 to 1.5
# axes[0].set_ylim(0, 2)
axes[1].set_xlabel('Time Label', fontsize=12)
axes[1].set_ylabel('Volume Ratio Change (V/V0)', fontsize=12)
axes[1].legend(frameon=False)
## no filled frame for legend
axes[1].legend(loc='upper left', fontsize=14)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)

## set whole plot x label

plt.tight_layout()
plt.show()

# Prepare data for SABR patients
## for the patients who belongs to the same time label, average the ipsi_change and contra_change
sabr_data_melted = sabr_data[['time_label', 'ipsi_change', 'contra_change']].copy()
sabr_data_melted = sabr_data_melted.melt(id_vars='time_label',
                                         value_vars=['ipsi_change', 'contra_change'],
                                         var_name='Side', value_name='Change')
sabr_data_melted['Side'] = sabr_data_melted['Side'].replace({'ipsi_change': 'Ipsilateral', 'contra_change': 'Contralateral'})

# Prepare data for non-SABR patients
non_sabr_data_melted = non_sabr_data[['time_label', 'ipsi_change', 'contra_change']].copy()
non_sabr_data_melted = non_sabr_data_melted.melt(id_vars='time_label',
                                                  value_vars=['ipsi_change', 'contra_change'],
                                                  var_name='Side', value_name='Change')
non_sabr_data_melted['Side'] = non_sabr_data_melted['Side'].replace({'ipsi_change': 'Ipsilateral', 'contra_change': 'Contralateral'})


sabr_data_melted['time_label'] = pd.Categorical(sabr_data_melted['time_label'], categories=ordered_labels, ordered=True)
non_sabr_data_melted['time_label'] = pd.Categorical(non_sabr_data_melted['time_label'], categories=ordered_labels, ordered=True)


# Step 3: Set up the figure
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# SABR plot
sns.boxplot(data=sabr_data_melted, x='time_label', y='Change', hue='Side', ax=axes[0], showfliers=True, showmeans=True)
axes[0].set_title('SBRT Patients')
## horizantal line at y=1
axes[0].axhline(y=1, color='r', linestyle='--')
## create an shaded area between 1.1 and 0.9
axes[0].fill_betweenx(y=[0.8, 1.2], x1=-0.5, x2=3.5, color='yellow', alpha=0.2)
axes[0].fill_betweenx(y=[0.9, 1.1], x1=-0.5, x2=3.5, color='red', alpha=0.2)
## set y-axis limit to 0.5 to 1.5
# axes[0].set_ylim(0, 2)
axes[0].set_xlabel('Time Label')
axes[0].set_ylabel('Volume Ratio Change (V/V0)')
axes[0].legend(loc='upper left')

# Non-SABR plot
sns.boxplot(data=non_sabr_data_melted, x='time_label', y='Change', hue='Side', ax=axes[1], showfliers=True, showmeans=True)
axes[1].axhline(y=1, color='r', linestyle='--')
axes[1].fill_betweenx(y=[0.8, 1.2], x1=-0.5, x2=3.5, color='yellow', alpha=0.2)
axes[1].fill_betweenx(y=[0.9, 1.1], x1=-0.5, x2=3.5, color='red', alpha=0.2)
axes[1].set_title('dRT Patients')
axes[1].set_ylabel('Volume Ratio Change (V/V0)')
axes[1].set_xlabel('Time Label')
## set x limit label size bigger
axes[1].tick_params(axis='x', labelsize=12)
# axes[1].set_ylim(0, 2)
axes[1].legend(loc='upper left')


plt.tight_layout()
plt.show()