import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('training_data.csv')
#data.describe().to_excel("training_data_description.xlsx")
#keys = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 
# 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth',
# 'windspeed', 'cloudcover', 'visibility', 'increase_stock']


data.loc[data['increase_stock']=='high_bike_demand', 'increase_stock'] = 1
data.loc[data['increase_stock']=='low_bike_demand', 'increase_stock'] = 0

feature = 'precip'
min= np.min(data.loc[data['increase_stock']==1,feature])
medain= np.median(data.loc[data['increase_stock']==1,feature])
print(f"{min: .10f}", f"{medain: .20f}")
print(0==medain)
stamps = np.linspace(min, medain,10)#int(max-min))
means = []
stamps_str = []
for i in range(len(stamps)-1):
    a= data.loc[(stamps[i] <= data[feature]) &  (data[feature] <= stamps[i+1]), [feature]]
    print(stamps[i+1], ": ",f"{len(a)}" )
    a = data.loc[(stamps[i] <= data[feature]) &  (data[feature] <= stamps[i+1]), ['increase_stock']]
    means.append(np.sum(a['increase_stock']))
    stamps_str.append('(' + str(int(stamps[i])) + ',' + str(int(stamps[i+1])) + ')')
    #stamps = range(int(np.ceil(min)), int(np.ceil(min)+len(means)))
plt.figure()
plt.plot(stamps[1:], means, 'o')
plt.savefig(feature + 'VS' + feature +'_mean')



#data = data[[x for x in data.keys() if (x not in execlude and x != 'increase_stock')]]
#corr = data[[x for x in data.keys() if (x != 'increase_stock')]].corr()
#new_features = {}
#print(corr['hour_of_day']['hour_of_day'])
#for x in corr:
#     new_features[x]=[]
#     for y in corr[x].keys():
#         if np.abs(corr[x][y]) < 1 and np.abs(corr[x][y]) >=0.5:
#             new_features[x].append(y)

# new_features = {key:value for (key,value) in new_features.items() if value != []}
# print(new_features)


data['tempTimesdew'] = data['temp'] * data['dew']
#print(data['tempTimesdew'][0:5])
execlude = ['summertime', 'snow', 'weekday']
features = [x for x in data.keys() if x not in execlude]
#print(data[features])

f = plt.figure(figsize=(19, 15))
plt.matshow(data[features].corr(), fignum=f.number)
# plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14, rotation=45)
# plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14)
plt.xticks(range(len(features)), features, fontsize=14, rotation=45)
plt.yticks(range(len(features)), features, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
#plt.savefig('correlation_matrix_1.png')


# hours = []
# hours_mean=[]
# for hour in range(24):
#     hours.append(hour)
#     a = data.loc[data['hour_of_day']== hour, ['increase_stock']] 
#     hours_mean.append(np.mean(a['increase_stock']))

# plt.figure()
# plt.plot(hours, hours_mean, 'o')
# plt.xlabel('hours')
# plt.ylabel('hours mean')
# plt.savefig('hourVShours_mean.png')
# #plt.show()

# #Month
# months = []
# months_mean=[]
# for month in range(1,13):
#     months.append(month)
#     a = data.loc[data['month']== month, ['increase_stock']]
#     months_mean.append(np.mean(a['increase_stock']))

# plt.figure()
# plt.plot(months, months_mean, 'o')
# plt.xlabel('Months')
# plt.ylabel('Months mean')
# plt.savefig('monthVSmonths_mean.png')
# #plt.show()


# #Holidays and week days
# ylist = [np.mean(data.loc[data['weekday']== 0, ['increase_stock']]['increase_stock']),
# np.mean(data.loc[data['weekday']== 1, ['increase_stock']]['increase_stock']),
# np.mean(data.loc[data['holiday']== 0, ['increase_stock']]['increase_stock']),
# np.mean(data.loc[data['holiday']== 1, ['increase_stock']]['increase_stock'])]
# xlist=['NOT weekday', 'weekday', 'NOT holiday', 'holiday']

# plt.figure()
# plt.bar(xlist, ylist)
# plt.savefig('holiday_weekend.png')
# #plt.show()

# #Days
# days = []
# days_mean=[]
# for day in range(7):
#     days.append(day)
#     a = data.loc[data['day_of_week']== day, ['increase_stock']]
#     days_mean.append(np.mean(a['increase_stock']))

# plt.figure()
# plt.plot(days, days_mean, 'o')
# plt.xlabel('Days of week')
# plt.ylabel('Days of week mean')
# plt.savefig('daysVSdays_mean.png')
# #plt.show()

# #temperature
# """ min= np.min(data['temp'])
# max= np.max(data['temp'])
# stamps = np.linspace(min, max, int(max-min))
# means = []
# stamps_str = []
# for i in range(len(stamps)-1):
#     a = data.loc[(stamps[i] <= data['temp']) &  (data['temp'] <= stamps[i+1]), ['increase_stock']]
#     means.append(np.mean(a['increase_stock']))
#     #stamps_str.append('(' + str(int(stamps[i])) + ',' + str(int(stamps[i+1])) + ')')
# stamps = range(int(np.ceil(min)), int(np.ceil(min)+len(means)))

# plt.figure()
# plt.plot(stamps, means, 'o')
# plt.savefig('tmpVStmp_mean')
#  """
# #temperature
# float_features = [ 'temp', 'dew', 'humidity', 'precip',
#  'windspeed', 'cloudcover', 'visibility']
# for feature in float_features:
#     min= np.min(data[feature])
#     max= np.max(data[feature])
#     stamps = np.linspace(min, max, int(max-min))
#     means = []
#     stamps_str = []
#     for i in range(len(stamps)-1):
#         a = data.loc[(stamps[i] <= data[feature]) &  (data[feature] <= stamps[i+1]), ['increase_stock']]
#         means.append(np.mean(a['increase_stock']))
#         #stamps_str.append('(' + str(int(stamps[i])) + ',' + str(int(stamps[i+1])) + ')')
#     stamps = range(int(np.ceil(min)), int(np.ceil(min)+len(means)))

#     plt.figure()
#     plt.plot(stamps, means, 'o')
#     plt.savefig(feature + 'VS' + feature +'_mean')


# """ feature='snow'
# steps=20
# min= np.min(data[feature])
# max= np.max(data[feature])
# print('{0:.16f}'.format(max))
# stamps = np.linspace(min, max, steps)
# means = []
# stamps_str = []
# for i in range(len(stamps)-1):
#     a = data.loc[(stamps[i] <= data[feature]) &  (data[feature] <= stamps[i+1]), ['increase_stock']]
#     means.append(np.mean(a['increase_stock']))
#     stamps_str.append('(' + str(stamps[i]) + ',' + str(stamps[i+1]) + ')')
# plt.figure()
# print(data[feature])
# print(stamps_str)
# print(means) """


# """ feature='snowdepth'
# steps=5
# min= np.min(data[feature])
# max= np.max(data[feature])
# stamps = np.linspace(min, max, steps)
# means = []
# stamps_str = []
# for i in range(len(stamps)-1):
#     a = data.loc[(stamps[i] <= data[feature]) &  (data[feature] <= stamps[i+1]), ['increase_stock']]
#     means.append(np.mean(a['increase_stock']))
#     stamps_str.append('(' + str(stamps[i]) + ',' + str(stamps[i+1]) + ')')
# plt.figure() """
# print(len(data.loc[(data["snowdepth"] == 0) & (data['increase_stock'] == 1)].index))

# print(len(data.loc[(data["snow"] == 0) & (data['increase_stock'] == 1)].index))

# print(len(data.loc[(data['increase_stock'] == 1)].index))












