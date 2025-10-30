import pandas as pd
import numpy as np

data = pd.read_csv('BC_speed1.csv')
patients = data['patient'].unique()
print('Number of patients:', len(patients))

target = 'muscle_body_ratio'
data['cluster_' + target] = 0
data['cluster_' + target] = np.where(data['speed_'+target] < -0.5, 1, data['cluster_' + target])


dose_info = pd.read_csv('dose_metrics_BED_clean.csv')
# for column change contain .nii.gz, remove the string
dose_info.columns = [col.replace('.nii.gz', '') for col in dose_info.columns]
dose_interst = dose_info.columns[1:].tolist()
## if one patient has multiple rows, sum the values for dose-interst
dose_info = dose_info.groupby('patient').sum().reset_index()

age_info = pd.read_csv('clinicalInfo2.csv')
age_info = age_info[['patient', 'age']]

## fill the missing value with the mean for age
age_info['age'] = age_info['age'].fillna(age_info['age'].mean())


plan_info = pd.read_csv('UCLH_PatientPlansSelected.csv')
plan_roi = ['patient', 'TotalDosePerFraction', 'TotalDose', 'SABR'] #'RTPlanLabel',
## one-hot encoding for SABR, 'Yes', 'No' as 1, 0
plan_info['SABR'] = np.where(plan_info['SABR'] == 'Yes', 1, 0)
plan_info['Palliative'] = np.where(plan_info['Palliative'] == 'Yes', 1, 0)
plan_info = plan_info[plan_roi]
plan_info = plan_info.groupby('patient').sum().reset_index()

cluster_num = 2
for cluster in range(cluster_num):
    patients = data[data['cluster_muscle_body_ratio'] == cluster]['patient']
    print(len(patients))
    sabr = plan_info[plan_info['patient'].isin(patients)]['SABR'].values
    print('Cluster %d: %d sabr' % (cluster, np.sum(sabr)))

## merge data and dose info
data = pd.merge(data, dose_info, on='patient', how='left')
## merge data and plan info
data = pd.merge(data, plan_info, on='patient', how='left')
# print(data.columns)
data = pd.merge(data, age_info, on='patient', how='left')

data.to_csv('data_body3.csv', index=False)

selected_feature = dose_interst+plan_roi[1:]
y = data['cluster_muscle_body_ratio'].values

# from sklearn.feature_selection import SelectKBest, f_classif
# feas = []
# for fea in selected_feature:
#     X = data[fea].values
#     f, p = f_classif(X.reshape(-1, 1), y)
#     ## correlation test
#     # f, p = f_regression(X.reshape(-1, 1), y)
#     if p < 0.2:
#        print('Feature %s: p = %.4f' % (fea, p[0]))
#        feas.append(fea)

# import statsmodels.api as sm
# X_const = sm.add_constant(X)
# model = sm.Logit(y, X_const)
# result = model.fit()
# print(result.summary())

# selected_feature = ['esophagus_v10']#dose_interst+plan_roi[:-1]
selected_feature = ['esophagus_v10']
## print the mean and std of each feature
for fea in selected_feature:
    print(fea, np.mean(data[fea]), np.std(data[fea]))
X = data[selected_feature].values

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000, class_weight='balanced')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
model.fit(X_scaled, y)

## print the coefficients
coef = model.coef_[0]
print('Coefficients:', coef)

y_pred = model.predict(X_scaled)
from sklearn.metrics import classification_report
print(classification_report(y, y_pred, zero_division=0))

import statsmodels.api as sm

df = pd.DataFrame(X_scaled, columns=selected_feature)
df['y'] = data['cluster_muscle_body_ratio']
# Calculate class weights (inverse frequency)
# For each observation, the weight is computed as:
#   weight = (total_samples / (number of classes * count_of_class))
counts = df['y'].value_counts().to_dict()
total = len(df)
n_classes = df['y'].nunique()
df['weights'] = df['y'].apply(lambda cls: total / (n_classes * counts[cls]))

# Add constant for intercept
# X_sm = sm.add_constant(df[selected_feature])
X_sm = sm.add_constant(df[selected_feature])

# Use GLM with binomial family and frequency weights
model = sm.GLM(df['y'], X_sm, family=sm.families.Binomial(), freq_weights=df['weights'])
# result = model.fit_regularized(
#         method='elastic_net',  # Only 'elastic_net' is currently implemented
#         alpha=0.01,  # Regularization strength
#         L1_wt=0.5,  # L1 weight: 1.0 for Lasso, 0.0 for Ridge, between 0 and 1 for Elastic Net
#         refit=True,  # Refit the model after regularization
#     )
result = model.fit()

print(result.summary())
print(result.pvalues)


coef   = result.params
se     = result.bse
z      = result.tvalues            # z because Logit uses large-sample normal approx
pvals  = result.pvalues
conf   = result.conf_int()         # 2-column DF

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

# Convert probabilities to binary class predictions using a threshold (default 0.5)
threshold = 0.5
predicted_classes = (predicted_probs >= threshold).astype(int)

from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
# Calculate classification performance metrics
cm = confusion_matrix(df['y'], predicted_classes)
accuracy = accuracy_score(df['y'], predicted_classes)
precision = precision_score(df['y'], predicted_classes)
recall = recall_score(df['y'], predicted_classes)
f1 = f1_score(df['y'], predicted_classes)
auc = roc_auc_score(df['y'], predicted_probs)

print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)


# ## plot the KM curve based on the predicted classes
# import matplotlib.pyplot as plt
# import lifelines
# from lifelines import KaplanMeierFitter
# from lifelines.plotting import add_at_risk_counts
#
# data['predicted_class'] = predicted_classes
# # Fit the model for the two groups
# survival_data = pd.read_csv('survival_check.csv')
#
# kmf_models = []
# fig, ax = plt.subplots(figsize=(8, 6))
# for i in range(2):
#     kmf = KaplanMeierFitter()
#     ## get the patient whose predicted class is i
#     patients = data[data['predicted_class'] == i]['patient'].values
#     # Get the survival data for these patients
#     survival_data_i = survival_data[survival_data['patient'].isin(patients)]
#     # Extract time and event data
#     T = survival_data_i['survival'].values
#     E = survival_data_i['deceased'].values
#
#     kmf.fit(T, E, label=f'Cluster {i}')
#     kmf.plot_survival_function(ax=ax)
#     kmf_models.append(kmf)
#
# add_at_risk_counts(*kmf_models, ax=ax)
#
# plt.tight_layout()
# plt.show()
#
#
# for cluster in range(2):
#     data_c = data[data['predicted_class'] == i]['speed_muscle_body_ratio']
#     if cluster == 0:
#         plt.hist(data_c, bins=10)
#     else:
#         plt.hist(data_c, bins=20)
#
# plt.legend(['Cluster 1', 'Cluster 2'])
# plt.show()

# cor = data[selected_feature].corr().abs()
# ## show the correlation matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 8))
# ## plot correlation matrix
# sns.heatmap(cor, annot=True, fmt='.2f', cmap='coolwarm', square=True)
# plt.title('Correlation Matrix')
# plt.show()
