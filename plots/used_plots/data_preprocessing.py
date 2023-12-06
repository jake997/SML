import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('training_data.csv')
#data.describe().to_excel("training_data_description.xlsx")
#keys = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 
# 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth',
# 'windspeed', 'cloudcover', 'visibility', 'increase_stock']


data.loc[data['increase_stock']=='high_bike_demand', 'increase_stock'] = 1
data.loc[data['increase_stock']=='low_bike_demand', 'increase_stock'] = 0

fig, axs = plt.subplots(2, 3, figsize=(14, 8))
float_features = [ 'temp', 'month', 'hour_of_day']
labels = ['a)', 'b)', 'c)']
j=0
for feature in float_features:
    min= np.min(data[feature])
    max= np.max(data[feature])
    stamps = np.linspace(np.floor(min), np.ceil(max), int(np.ceil(max)- np.floor(min))+1)
    print('stamps: ', stamps)
    means = []
    stamps_str = []
    for i in range(len(stamps)-1):
        a = data.loc[(stamps[i] <= data[feature]) &  (data[feature] < stamps[i+1]), ['increase_stock']]
        means.append(np.mean(a['increase_stock']))
        if i == len(stamps)-2:
            a = data.loc[(stamps[i+1] <= data[feature]), ['increase_stock']]
            means.append(np.mean(a['increase_stock'])) 
    stamps = range(int(np.floor(min)), int(np.floor(min)+len(means)))
    print('stamps: ', stamps)

    axs[0, j].plot(stamps, means, 'o')
    axs[0, j].set_title(labels[j] + ' P_' + feature)
    if j == 0:
        axs[0, j].set_ylabel('P(increase_stock = high_bike_demand)')
    axs[0, j].set_xlabel(feature)
    j+=1

#Holidays and week days
ylist = [np.mean(data.loc[data['weekday']== 0, ['increase_stock']]['increase_stock']),
np.mean(data.loc[data['weekday']== 1, ['increase_stock']]['increase_stock']),
np.mean(data.loc[data['holiday']== 0, ['increase_stock']]['increase_stock']),
np.mean(data.loc[data['holiday']== 1, ['increase_stock']]['increase_stock'])]
xlist=['NOT weekday', 'weekday', 'NOT holiday', 'holiday']
axs[1, 0].bar(xlist, ylist)
axs[1, 0].set_title('d) P_weekday, holiday')
axs[1, 0].set_ylabel('P(increase_stock = high_bike_demand)')
axs[1, 0].set_xlabel("weekday, holiday")

features= data.keys()
matrix = np.delete(data[features].corr()['increase_stock'].to_numpy(), 15).reshape(1, -1)
axs[1, 1].bar(features[:-1], matrix[0])
axs[1, 1].tick_params(axis='x', rotation=90)
axs[1, 1].set_ylabel('Covariance with increase_stock')
axs[1, 1].set_xlabel('Input features')
axs[1, 1].set_title('e) Correlation Plot')



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
features.pop(10)
features.pop(-1)
dic = {k:v for k, v in zip(features, matrix[0])}

new_features = ['hour_of_day', 'c_hour_of_day', 'day_of_week', 'c_day_of_week'\
                , 'month', 'c_month', 'precip', 'snowdepth', 'windspeed', 'visibility', 'bad_weather']
corr_values = [dic[x] for x in new_features]

axs[1,2].bar(new_features, corr_values)
axs[1, 2].tick_params(axis='x', rotation=90)
axs[1, 2].set_ylabel('Covariance with increase_stock')
axs[1, 2].set_xlabel('Input features')
axs[1, 2].set_title('f) Original features vs custom features')


plt.tight_layout()



plt.tight_layout()
fig.savefig("data_analysis.png")