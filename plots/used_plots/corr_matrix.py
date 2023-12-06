import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('training_data.csv')

data.loc[data['increase_stock']=='high_bike_demand', 'increase_stock'] = 1
data.loc[data['increase_stock']=='low_bike_demand', 'increase_stock'] = 0



data['c_month']=np.empty(len(data.index))
data.loc[data['month'].isin([12, 1, 2]), 'c_month'] = 0
data.loc[data['month'].isin([11]) , 'c_month'] = 1
data.loc[data['month'].isin([3, 5, 7, 8]) , 'c_month'] = 2
data.loc[data['month'].isin([4, 6, 9, 10]) , 'c_month'] = 3

data['c_day_of_week']=np.empty(len(data.index))
data.loc[data['day_of_week'].isin([0,1,2,3,4]), 'c_day_of_week'] = 0
data.loc[data['day_of_week'].isin([6]), 'c_day_of_week'] = 1
data.loc[data['day_of_week'].isin([5]), 'c_day_of_week'] = 2

data['c_hour_of_day']=np.empty(len(data.index))
data.loc[data['hour_of_day'].isin([0, 1, 2, 3, 4, 5, 6, 7, 21, 22, 23]), 'c_hour_of_day'] = 0
data.loc[data['hour_of_day'].isin([8, 9, 10, 11, 12, 13, 14, 20]) , 'c_hour_of_day'] = 1
data.loc[data['hour_of_day'].isin([15, 16, 17, 18, 19]) , 'c_hour_of_day'] = 2



""" data['new_precip']=np.empty(len(data.index))
data.loc[data['precip'] <=0, 'new_precip'] = False
data.loc[data['precip'] > 0, 'new_precip'] = True

data['new_snowdepth']=np.empty(len(data.index))
data.loc[data['snowdepth'] >= 0.01 , 'new_snowdepth'] = True
data.loc[~(data['snowdepth'] >= 0.01) , 'new_snowdepth'] = False

data['new_windspeed']=np.empty(len(data.index))
data.loc[data['windspeed'] < 30, 'new_windspeed'] = False
data.loc[data['windspeed'] >= 30, 'new_windspeed'] = True

data['new_visibility']=np.empty(len(data.index))
data.loc[data['visibility'] < 15, 'new_visibility'] = True
data.loc[data['visibility'] >= 15, 'new_visibility'] = False """


data['temp_square'] = data['temp'] ** 2
data['dew_square'] = data['dew'] ** 2
data['humidity_square'] = (data['humidity']**2)
data['temp_dew_humidity'] = data['temp'] * data['dew'] * 1/data['humidity_square']

data['bad_weather'] = np.empty(len(data.index))
data.loc[( (data['snowdepth'] > 0) | (data['precip'] > 0.1) | (data['visibility'] < 15) | (data['windspeed'] > 30)), 'bad_weather'] = True
data.loc[~( (data['snowdepth'] > 0) | (data['precip'] > 0.1) | (data['visibility'] < 15) | (data['windspeed'] > 30)), 'bad_weather'] = False

# Create a StandardScaler
scaler = StandardScaler()

# Fit and transform the DataFrame
features = [x for x in data.keys() if x != 'increase_stock']
y_labels = data['increase_stock'].to_numpy()
data = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
data['increase_stock'] = y_labels

execlude = [] 
features = [x for x in data.keys() if x not in execlude]
matrix = np.delete(data[features].corr()['increase_stock'].to_numpy(), [10, -1]).reshape(1, -1)

""" f = plt.figure(figsize=(19, 15))
plt.matshow(matrix, fignum=f.number)
features.pop(10)
features.pop(-1)
plt.xticks(range(len(features)), features, fontsize=12, rotation=90)
plt.yticks([0], ["increase_stock"], fontsize=12)#(range(len(features)), features, fontsize=12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation matrix for customed features', fontsize=16)
plt.savefig('new_correlation_matrix.png') """

plt.figure()
dic = {k:v for k, v in zip(features, matrix[0])}
new_features = ['hour_of_day', 'c_hour_of_day', 'day_of_week', 'c_day_of_week'\
                , 'month', 'c_month', 'precip', 'snowdepth', 'windspeed', 'visibility', 'bad_weather']
corr_values = [dic[x] for x in new_features]
plt.bar(new_features, corr_values)
plt.xticks(rotation=60, ha='right')
plt.title('Original features vs customed features')
plt.ylabel('Covariance with increase_stock')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('original_vs_custom_correlation_plot.png')
