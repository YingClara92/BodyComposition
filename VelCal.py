from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
file = 'MyocardiumF1.csv'
target = 'myocardium_volume'
data = pd.read_csv(file)
data = data.dropna(subset=[target])

# file = 'BodyCompT.csv'
# target = 'body_L1'
# data = data.dropna(subset=[target])
#
# data.loc[:, 'muscle_body_ratio'] = data['skeletal_muscle_volume'] / data['body_L1_length']
# data['torso_body_ratio'] = data['torso_fat_volume'] / data['body_L1_length']
# data['subcutaneous_fat_body_ratio'] = data['subcutaneous_fat_volume'] / data['body_L1_length']
# targets = ['muscle_body_ratio']#, 'subcutaneous_fat_body_ratio', 'torso_body_ratio'


time = data['time'].unique()
patients = data['patient'].unique()

data = data.reset_index(drop=True)

drop_patients = []
for p in patients:
    data_p = data[data['patient'] == p]
    data_p = data_p[data_p['time'] != 0]
    if len(data_p['time'].unique()) < 3:
        drop_patients.append(p)

data = data[~data['patient'].isin(drop_patients)]
patients = data['patient'].unique()


data_dict = {'patient':[], 'speed_ML_ratio':[]}
r2_list =[]

for p in patients:
    data_p = data[data['patient'] == p].reset_index(drop=True)

    # Need baseline value
    baseline = data_p[data_p['time'] == 0][target]
    if baseline.empty:
        continue

    pre_rt = baseline.mean()

    # Average repeated measurements at each time point
    data_avg= (
        data_p
        .groupby('time', as_index=False)[target]
        .mean()
        .sort_values('time')
        .reset_index(drop=True)
    )
    time_p_total = data_avg['time'].astype(float).to_numpy()

    data_avg['relative_volume'] = data_avg[target] / pre_rt

    volume = data_avg['relative_volume'].astype(float).to_numpy()

    time_p = data_avg['time'].astype(float).to_numpy()

    a, b = np.polyfit(time_p, volume, 1)
    volume_new = a * time_p + b

    r2 = r2_score(volume, volume_new)

    data_dict['patient'].append(p)
    data_dict['speed_ML_ratio'].append(a)
    r2_list.append(r2)


r2_df = pd.DataFrame(r2_list, columns=['r2'])
print(r2_df['r2'].describe())

data = pd.DataFrame(data_dict)
data.to_csv('data_myocardium.csv', index=False)
