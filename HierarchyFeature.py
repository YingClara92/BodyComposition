import pandas as pd


data = pd.read_csv('dose_metrics_BED_clean2.csv')
# ## change the column names to remove .nii.gz
# data.columns = [col.replace('.nii.gz', '') for col in data.columns]
# ## remove the features with 'd' but except 'dmean'
# data = data.loc[:, ~data.columns.str.contains('_d(?!mean)')]
# ## for columns that contain 'dmean', extract the number xx from metatensor(xx) for each row
# for col in data.columns:
#     if 'dmean' in col or 'dose_total' in col:
#         new_col = []
#         for val in data[col].values:
#             if isinstance(val, str) and 'metatensor(' in val:
#                 num = val.split('metatensor(')[1].split(')')[0]
#                 new_col.append(float(num))
#             else:
#                 new_col.append(float('nan'))
#         data[col] = new_col
#         data[col] = data[col].astype(float)
#
# ## save data
# data.to_csv('dose_metrics_BED_clean2.csv', index=False)

selected_feature = data.columns[1:].tolist()
## remove the nan values fro the selected features
data = data[selected_feature].dropna(axis=1, how='all')
# selected_feature = data.columns[-3:].tolist()
## calculate the correlation matrix
correlation_matrix = data[selected_feature].corr()

## run hierarchical clustering on the correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
## get the cluster for the correlation matrix
Z = linkage(correlation_matrix, method='ward')
## based on the linkage to get the cluster
dendrogram(Z, labels=correlation_matrix.columns, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('Distance')
plt.show()
## set each feature to a cluster
clusters = sch.fcluster(Z, t=10, criterion='distance')
## set the cluster to each feature

fea_cluster = {}
# for i in range(len(clusters)):
#     if clusters[i] not in fea_cluster:
#         fea_cluster[clusters[i]] = [correlation_matrix.columns[i]]
#     else:
#         fea_cluster[clusters[i]].append(correlation_matrix.columns[i])
#
# print(len(fea_cluster))
#
# fea_cluster[4] = ['TotalDosePerFraction', 'SABR']
# fea_cluster[5] = 'age'

for fea in correlation_matrix.columns:
    if fea not in fea_cluster:
        fea_cluster[fea] = clusters[correlation_matrix.columns.get_loc(fea)]
    else:
        fea_cluster[fea] = clusters[correlation_matrix.columns.get_loc(fea)]

fea_cluster['TotalDosePerFraction'] = 5
fea_cluster['SABR'] = 5
fea_cluster['age'] = 6

## convert the dictionary to a dataframe
df = pd.DataFrame(list(fea_cluster.items()), columns=['feature', 'cluster'])
df.to_csv('feature_cluster_BED2.csv', index=False)

print(clusters)

print(fea_cluster)




