import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skl_met




#FEATURE ENGINEERING
data = pd.read_csv('training_data.csv')
data['increase_stock'] = np.where(data['increase_stock'] == 'high_bike_demand', 1, 0)


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
scaler = skl_pre.StandardScaler()

# Fit and transform the DataFrame
features = [x for x in data.keys() if x != 'increase_stock']
y_labels = data['increase_stock'].to_numpy()
data = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
data['increase_stock'] = y_labels

#DATA SPLITING
np.random.seed(18)
split = 0.8
keys = data.keys()
execlude = ['increase_stock','snow', 'hour_of_day', 'day_of_week', 'month']
features = [x for x in keys if x not in execlude]
train_index = np.random.choice(data.index, size=int(len(data.index) * split), replace=False)
train = data.loc[data.index.isin(train_index)]
test = data.loc[~data.index.isin(train_index)]
train_X, test_X = train[features], test[features] 
train_y, test_y =  np.ravel((train[['increase_stock']]).to_numpy()), np.ravel(test[['increase_stock']].to_numpy()) 

#MODEL CONSTRUCTION
class_weight = {0:1, 1:4}
C= 0.894
logistic_model = skl_lm.LogisticRegression(solver='lbfgs', max_iter=10000, class_weight=class_weight, C=C)
logistic_model.fit(train_X, train_y)
predict_prob = logistic_model.predict_proba(test_X)
logistic_prediction =np.empty(len(test_y), dtype=object)
logistic_prediction = np.where(predict_prob[:, 0] <0.5, 'high_bike_demand', 'low_bike_demand')

predict_prob = logistic_model.predict_proba(train_X)
logistic_prediction_train =np.empty(len(train_y), dtype=object)
logistic_prediction_train = np.where(predict_prob[:, 0] <0.5, 'high_bike_demand', 'low_bike_demand')


test_y = np.where(test_y == 1, 'high_bike_demand', 'low_bike_demand')
train_y = np.where(train_y == 1, 'high_bike_demand', 'low_bike_demand')

#Confusion Matrix 
print('Confusion Matrices:', "\n")
print("Train LR")
print(pd.crosstab(logistic_prediction_train, train_y), "\n")
print("Test LR")
print(pd.crosstab(logistic_prediction, test_y), "\n")




#Radom model
import random
def random_model(x):
    return random.choices(['high_bike_demand', 'low_bike_demand'], weights=[0.18, 0.82], k=1)[0]

acc, recall, precision, f1=0, 0, 0, 0
loops = 1000
# for i in range(loops):
#     random_pred = []
#     for x in range(len(test_y)):
#         random_pred.append(random_model(x))
#     acc += skl_met.accuracy_score(test_y, random_pred)
#     recall += skl_met.recall_score(test_y, random_pred, pos_label='high_bike_demand')
#     precision += skl_met.precision_score(test_y, random_pred, pos_label='high_bike_demand')
#     f1 += skl_met.f1_score(test_y, random_pred, pos_label='high_bike_demand')


#scores
#print("\nRandom model metrics on test set:\n")
print(f"Train Accuracy LR:{skl_met.accuracy_score(train_y, logistic_prediction_train): .3f}")
print(f"Test Accuracy LR:{skl_met.accuracy_score(test_y, logistic_prediction): .3f}")
print(f"Random model Accuracy:{acc/loops: .3f}")
print()
print(f"Train recall LR:{skl_met.recall_score(train_y, logistic_prediction_train, pos_label='high_bike_demand'): .3f}")
print(f"Test recall LR:{skl_met.recall_score(test_y, logistic_prediction, pos_label='high_bike_demand'): .3f}")
print(f"Random model Recall:{recall/loops: .3f}")
print()
print(f"Train precision LR:{skl_met.precision_score(train_y, logistic_prediction_train, pos_label='high_bike_demand'): .3f}")
print(f"Test precision LR:{skl_met.precision_score(test_y, logistic_prediction, pos_label='high_bike_demand'): .3f}")
print(f"Random model precision:{precision/loops: .3f}")
print()
print(f"Train f1 LR:{skl_met.f1_score(train_y, logistic_prediction_train, pos_label='high_bike_demand'): .3f}")
print(f"Test f1 LR:{skl_met.f1_score(test_y, logistic_prediction, pos_label='high_bike_demand'): .3f}")
print(f"Random model f1:{f1/loops: .3f}")
print()


