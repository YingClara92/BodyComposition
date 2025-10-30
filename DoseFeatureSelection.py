import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from firthlogist import FirthLogisticRegression

# data = pd.read_csv('data_m3.csv')
data = pd.read_csv('data_BED_m.csv')
## only keep the cluster 0 and 1
data = data[data['cluster_ML_new'] != 1] ## options here are 1, 2

selected_feature = data.columns[6:].tolist()
X = data[selected_feature]

y = data['cluster_ML_new'].values
encoder = OneHotEncoder()
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
y_onehot = y_onehot.toarray()[:, 1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
## set the feature columns back to X_scaled
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
y = y_onehot

#
# data = pd.read_csv('data_body3.csv')
# target = 'muscle_body_ratio'
# data['cluster_' + target] = 0
# data['cluster_' + target] = np.where(data['speed_'+target] < -0.5, 1, data['cluster_' + target])
#
# selected_feature = data.columns[9:].tolist()
# X = data[selected_feature]
# y = data['cluster_' + target].values
#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# ## set the feature columns back to X_scaled
# X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


cluster_data = pd.read_csv('feature_cluster_BED2.csv')
cluster_num = cluster_data['cluster'].unique()

selected_cluster = {}

for cluster in cluster_num:
    print('Cluster %d' % cluster)
    selected_feature = cluster_data[cluster_data['cluster'] == cluster]['feature'].tolist()

    p_fea = ''
    p_value_s = 1

    for fea in selected_feature:
        X = X_scaled[fea].values.reshape(-1, 1)
        model = FirthLogisticRegression(max_iter=1000)
        try:
            result = model.fit(X, y)
        except UserWarning as e:
            print(f"Convergence warning for feature {fea}: {e}")
            continue

        p_value = result.pvals_[0]
        params = result.coef_[0]

        # if cluster == 5:
        #     if p_value < p_value_s:
        #         p_value_s = p_value
        #         p_fea = fea
        # else:
        if params > 0:
            if p_value < p_value_s:
                ## keep two decimal of p_value
                # p_value = round(p_value, 2)
                p_value_s = p_value
                p_fea = fea

            if p_value < 0.05:
                print(f"Feature: {fea}, p-value: {p_value}, Coefficient: {params}")


    selected_cluster[cluster] = [p_fea, round(p_value_s,2)]


# print(selected_cluster)
## print the selected feature for each cluster
fea_list = []
for cluster, feature in selected_cluster.items():
    print(f"Cluster {cluster}: Feature = {feature[0]}, p-value = {feature[1]}")
    fea_list.append(feature[0])

print(fea_list)

# selected_feature = ['heart_myocardium_v5', 'aorta_v5']
# selected_feature = ['TotalDosePerFraction']

# X = X_scaled[selected_feature]
# model = FirthLogisticRegression(max_iter=1000)
# result = model.fit(X, y)
# print(result.summary())

