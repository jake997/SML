import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.model_selection import GridSearchCV


#keys = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 
# 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth',
# 'windspeed', 'cloudcover', 'visibility', 'increase_stock']

#DATA PREPROCESSING
data = pd.read_csv('training_data.csv')
data['increase_stock'] = np.where(data['increase_stock'] == 'high_bike_demand', 1, 0)

data.loc[data['month'].isin([12, 1, 2]), 'month'] = 0
data.loc[data['month'].isin([11]) , 'month'] = 1
data.loc[data['month'].isin([3, 5, 7, 8]) , 'month'] = 2
data.loc[data['month'].isin([4, 6, 9, 10]) , 'month'] = 3

data.loc[data['day_of_week'].isin([0,1,2,3,4]), 'day_of_week'] = 0
data.loc[data['day_of_week'].isin([6]), 'day_of_week'] = 1
data.loc[data['day_of_week'].isin([5]), 'day_of_week'] = 2

data.loc[data['hour_of_day'].isin([0, 1, 2, 3, 4, 5, 6, 21, 23]), 'hour_of_day'] = 0
data.loc[data['hour_of_day'].isin([7, 20, 22]), 'hour_of_day'] = 1
data.loc[data['hour_of_day'].isin([8, 9, 10, 11, 12, 13, 14]) , 'hour_of_day'] = 2
data.loc[data['hour_of_day'].isin([15, 16, 19]) , 'hour_of_day'] = 3
data.loc[data['hour_of_day'].isin([17, 18]) , 'hour_of_day'] = 4

data.loc[data['visibility'] < 15, 'visibility'] = True
data.loc[data['visibility'] >= 15, 'visibility'] = False

data.loc[data['snowdepth'] >= 0.01 , 'snowdepth'] = True
data.loc[~(data['snowdepth'] >= 0.01) , 'snowdepth'] = False


data.loc[data['windspeed'] < 30, 'windspeed'] = False
data.loc[data['windspeed'] >= 30, 'windspeed'] = True

data.loc[data['precip'] <=0, 'precip'] = False
data.loc[data['precip'] > 0, 'precip'] = True

data['tempSquare'] = data['temp'] ** 2
data['dewSquare'] = data['dew'] ** 2
data['humiditySquare'] = 1/(data['humidity']**2)
data['badWeather'] = np.empty(len(data.index))
data.loc[(data['snowdepth'] | data['precip'] | data['visibility'] | data['windspeed']), 'badWeather'] = True
data.loc[~(data['snowdepth']  | data['precip'] | data['visibility'] | data['windspeed']) , 'badWeather'] = False

#Data Split
np.random.seed(1)
split = 0.8
keys = data.keys()
execlude = ['increase_stock','snow']
features = [x for x in keys if x not in execlude]
train_index = np.random.choice(data.index, size=int(len(data.index) * split), replace=False)
train = data.loc[data.index.isin(train_index)] 
test = data.loc[~data.index.isin(train_index)]
train_X, test_X = train[features], test[features] 
train_y, test_y =  np.ravel((train[['increase_stock']]).to_numpy()), np.ravel(test[['increase_stock']].to_numpy()) 

#Model building
def print_dataframe(filtered_cv_results):
    for mean_precision, std_precision, mean_recall, std_recall, mean_accuracy, std_accuracy, mean_f1, std_f1, params in zip(
        filtered_cv_results["mean_test_precision"],
        filtered_cv_results["std_test_precision"],
        filtered_cv_results["mean_test_recall"],
        filtered_cv_results["std_test_recall"],
        filtered_cv_results["mean_test_accuracy"],
        filtered_cv_results["std_test_accuracy"],
        filtered_cv_results["mean_test_f1"],
        filtered_cv_results["std_test_f1"],
        filtered_cv_results["params"],
    ):
        print(
            f"precision: {mean_precision:0.3f} (±{std_precision:0.03f}),"
            f" recall: {mean_recall:0.3f} (±{std_recall:0.03f}),"
            f" accuracy: {mean_accuracy:0.3f} (±{std_accuracy:0.03f}),"
            f" f1: {mean_f1:0.3f} (±{std_f1:0.03f}),"
            f" for {params}"
        )
    print()

def refit_strategy(cv_results):
    cv_results_ = pd.DataFrame(cv_results)
    print("All grid-search results:")
    print_dataframe(cv_results_)
    highest_recall_index = cv_results_['mean_test_recall'].idxmax()
    print("The model with the highest recall:")
    row = cv_results_.loc[highest_recall_index ,  [ "mean_score_time",
            "mean_test_recall",
            "std_test_recall",
            "mean_test_precision",
            "std_test_precision",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_test_f1",
            "std_test_f1",
            "rank_test_recall",
            "rank_test_precision",
            "rank_test_accuracy",
            "rank_test_f1",
            "params",
        ]]
    print(
        f"{row}"
    )
    return highest_recall_index


logistic_model = skl_lm.LogisticRegression(solver='lbfgs', max_iter=10000)
class_weight = [{0:1, 1:1}, {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}, {0:1, 1:5}, {0:1, 1:6}, {0:1, 1:7}, {0:1, 1:8}
                , {0:1, 1:9}, {0:1, 1:10}, {0:1, 1:12}, {0:1, 1:13}, {0:1, 1:14}, {0:1, 1:15}, {0:1, 1:16}, {0:1, 1:17}, {0:1, 1:18}
                , {0:1, 1:19}, {0:1, 1:20}]
C = np.linspace(0, 1, 20)[1:]
parameters = {'class_weight': class_weight, "C": C}
scores = ["precision", "recall", "accuracy", "f1"]
clf = GridSearchCV(logistic_model, param_grid=parameters, scoring= scores, refit=refit_strategy)
estimator = clf.fit(train_X, train_y)
cv_results = pd.DataFrame(estimator.cv_results_)
cv_results.to_csv("cv_results_logistic_model.csv")
