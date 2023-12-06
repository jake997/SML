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

# idxmax = data['precip'].idxmax()
# data.drop(idxmax, inplace=True)

# df = data.loc[(data['precip'] > 0) | (data['snowdepth']>0), ['precip', 'increase_stock']]
# print(len(df.index))
# print(len(df.loc[df['increase_stock']==1].index))
#df = data.loc[data['snowdepth'] > 0, ['precip', 'increase_stock']]
#print(len(df.index))
#print(len(df.loc[df['increase_stock']==1].index))
#data['precip_exp'] = 2**data['precip']
#data['precip_ln'] = np.log(data['precip']+10e-10)

#print(len((data.loc[data['precip']==0, 'precip']).index))
# feature = 'precip'
# min= np.min(data.loc[data['increase_stock']==1,feature])
# max= np.max(data.loc[data['increase_stock']==1,feature])
# print(f"{min: .10f}", f"{max: .20f}")
# stamps = np.linspace(min, max,100)#int(max-min))
# means = []
# stamps_str = []
# for i in range(len(stamps)-1):
#     a= data.loc[(stamps[i] <= data[feature]) &  (data[feature] <= stamps[i+1]), [feature]]
#     #print(stamps[i+1], ": ",f"{len(a)}" )
#     a = data.loc[(stamps[i] <= data[feature]) &  (data[feature] <= stamps[i+1]), ['increase_stock']]
#     means.append(np.sum(a['increase_stock']))
#     stamps_str.append('(' + str(int(stamps[i])) + ',' + str(int(stamps[i+1])) + ')')
#     #stamps = range(int(np.ceil(min)), int(np.ceil(min)+len(means)))
# plt.figure()
# plt.plot(stamps[1:], means, 'o')
# plt.savefig(feature + 'VS' + feature +'_mean')









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
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
float_features = [ 'temp', 'month', 'hour_of_day']
j=0
for feature in float_features:
    min= np.min(data[feature])
    max= np.max(data[feature])
    stamps = np.linspace(min, max, int(max-min))
    means = []
    stamps_str = []
    for i in range(len(stamps)-1):
        a = data.loc[(stamps[i] <= data[feature]) &  (data[feature] <= stamps[i+1]), ['increase_stock']]
        means.append(np.mean(a['increase_stock']))
        #stamps_str.append('(' + str(int(stamps[i])) + ',' + str(int(stamps[i+1])) + ')')
    stamps = range(int(np.ceil(min)), int(np.ceil(min)+len(means)))

    axs[0, j].plot(stamps, means, 'o')
    axs[0, j].set_title('P_' + feature)
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
axs[1, 0].set_title('P_weekday, holiday')
axs[1, 0].set_ylabel('P(increase_stock = high_bike_demand)')
axs[1, 0].set_xlabel("weekday, holiday")

features= data.keys()
matrix = np.delete(data[features].corr()['increase_stock'].to_numpy(), 15).reshape(1, -1)
axs[1, 1].bar(features[:-1], matrix[0])
axs[1, 1].tick_params(axis='x', rotation=90)
axs[1, 1].set_ylabel('Covariance with increase_stock')
axs[1, 1].set_xlabel('Input features')
axs[1, 1].set_title('Correlation Plot')

axs[1, 2].axis('off')

plt.tight_layout()
fig.savefig("test")

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












