import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as skl_mt


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

""" data['bool_visibility'] = np.empty(len(data.index))
data.loc[data['visibility'] < 15, 'bool_visibility'] = True
data.loc[data['visibility'] >= 15, 'bool_visibility'] = False

data['bool_snowdepth'] = np.empty(len(data.index))
data.loc[data['snowdepth'] >= 0 , 'bool_snowdepth'] = True
data.loc[~(data['snowdepth'] >= 0) , 'bool_snowdepth'] = False

data['bool_windspeed'] = np.empty(len(data.index))
data.loc[data['windspeed'] < 30, 'bool_windspeed'] = False
data.loc[data['windspeed'] >= 30, 'bool_windspeed'] = True


data['bool_precip'] = np.empty(len(data.index))
data.loc[data['precip'] <=0.1, 'bool_precip'] = False
data.loc[data['precip'] > 0.1, 'bool_precip'] = True """

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

#Data Split
np.random.seed(1)
split = 0.8
keys = data.keys()
execlude = ['increase_stock','snow', 'hour_of_day', 'day_of_week', 'month']
features = [x for x in keys if x not in execlude]
train_index = np.random.choice(data.index, size=int(len(data.index) * split), replace=False)
train = data.loc[data.index.isin(train_index)] 
test = data.loc[~data.index.isin(train_index)]
train_X, test_X = train[features], test[features] 
train_y, test_y =  np.ravel((train[['increase_stock']]).to_numpy()), np.ravel(test[['increase_stock']].to_numpy()) 

#Model building
# def refit_strategy(cv_results):
#     f1_threshold = 0.68
#     std_f1_threshold = 0.02
#     cv_results_ = pd.DataFrame(cv_results)
#     highest_f1_results = cv_results_.loc[(cv_results_['mean_test_f1'] >= f1_threshold) & \
#                                          (cv_results_['std_test_f1'] <= std_f1_threshold)]
#     best_model_index = highest_f1_results['mean_test_recall'].idxmax()
#     print("The model with the highest f1 score:")
#     row = cv_results_.loc[best_model_index ,  [ "mean_score_time",
#             "mean_test_recall",
#             "std_test_recall",
#             "mean_test_precision",
#             "std_test_precision",
#             "mean_test_accuracy",
#             "std_test_accuracy",
#             "mean_test_f1",
#             "std_test_f1",
#             "rank_test_recall",
#             "rank_test_precision",
#             "rank_test_accuracy",
#             "rank_test_f1",
#             "params",
#         ]]
#     print(row)
#     return best_model_index

def refit_strategy(cv_results):
    std_f_beta_threshold = 0.02
    cv_results_ = pd.DataFrame(cv_results)
    highest_f_beta_results = cv_results_.loc[(cv_results_['std_test_f_beta'] <= std_f_beta_threshold)]
    best_model_index = highest_f_beta_results['mean_test_f_beta'].idxmax()
    return best_model_index



logistic_model = skl_lm.LogisticRegression(solver='lbfgs', max_iter=10000)
class_weight = [{0:1/x, 1:1-1/x} for x in range(1,11)]
C = np.linspace(0, 1, 20)[1:]
parameters = {'class_weight': class_weight, "C": C}
f_beta = skl_mt.make_scorer(skl_mt.fbeta_score, beta=1.5)
scores = {"precision" : skl_mt.make_scorer(skl_mt.precision_score) ,\
           "recall" : skl_mt.make_scorer(skl_mt.recall_score),\
           "accuracy": skl_mt.make_scorer(skl_mt.accuracy_score), "f_beta": f_beta}
clf = GridSearchCV(logistic_model, param_grid=parameters, scoring= scores, refit=refit_strategy)
estimator = clf.fit(train_X, train_y)
cv_results = pd.DataFrame(estimator.cv_results_)
cv_results.to_csv("cv_results_logistic_model.csv")
print("Best params: ", clf.best_params_)
#print("Best model:", clf.cv_results_[clf.best_index_])
