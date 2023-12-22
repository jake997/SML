import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.model_selection as skl_ms
import sklearn.preprocessing as skl_pre
import sklearn.neighbors as skl_nb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import sklearn.metrics as skl_mt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


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



#DATA SPLITING
#keys = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 
# 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth',
# 'windspeed', 'cloudcover', 'visibility', 'increase_stock']

features = [x for x in data.keys() if x != 'increase_stock']
y_labels = data['increase_stock'].to_numpy()

data['increase_stock'] = y_labels

#Data Split
np.random.seed(1)
split = 0.8
keys = data.keys()
execlude = ['increase_stock','snow', 'hour_of_day', 'day_of_week', 'month']
features = [x for x in keys if x not in execlude]
print(features)
train_index = np.random.choice(data.index, size=int(len(data.index) * split), replace=False)
train = data.loc[data.index.isin(train_index)] 
test = data.loc[~data.index.isin(train_index)]
train_X, test_X = train[features], test[features] 
train_y, test_y =  np.ravel((train[['increase_stock']]).to_numpy()), np.ravel(test[['increase_stock']].to_numpy()) 







scaler = preprocessing.StandardScaler()

train_X = pd.DataFrame(scaler.fit_transform(train_X), columns = train_X.columns)
test_X = pd.DataFrame(scaler.transform(test_X), columns= test_X.columns)

def refit_strategy(cv_results):
    std_fbeta_threshold = 0.07
    cv_results_ = pd.DataFrame(cv_results)
    
    highest_fbeta_results = cv_results_.loc[(cv_results_['std_test_fbeta'] <= std_fbeta_threshold)]
    
    best_model_index = highest_fbeta_results['mean_test_fbeta'].idxmax()
    print("The model with the highest fbeta score:")
    row = cv_results_.loc[best_model_index ,  [ "mean_score_time",
            "mean_test_recall",
            "std_test_recall",
            "mean_test_precision",
            "std_test_precision",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_test_fbeta",
            "std_test_fbeta",
            "rank_test_recall",
            "rank_test_precision",
            "rank_test_accuracy",
            "rank_test_fbeta",
            "params",
        ]]
    print(row)
    return best_model_index


distance = np.array([ 'chebyshev','minkowski','cosine'])
weights = np.array(['uniform','distance'])
p = np.arange(1,10,0.5)
n_neighbors = np.arange(1,100)
f_beta = skl_mt.make_scorer(skl_mt.fbeta_score, beta=1.5)
scores = {"precision" : skl_mt.make_scorer(skl_mt.precision_score) ,\
           "recall" : skl_mt.make_scorer(skl_mt.recall_score),\
           "accuracy": skl_mt.make_scorer(skl_mt.accuracy_score), "fbeta": f_beta}
estimator_KNN = skl_nb.KNeighborsClassifier(algorithm='auto')

parameters_KNN = {'n_neighbors': n_neighbors,
                  'p': p,
                  'weights': weights,
                  'metric': distance,
                  }


grid_search_KNN  = RandomizedSearchCV(estimator = estimator_KNN, 
                                param_distributions=parameters_KNN,
                                scoring=scores,
                                cv = 10,
                                verbose=1,
                                n_jobs=-1,
                                refit=refit_strategy,
                                n_iter = 6000
                                )

estimator = grid_search_KNN.fit(train_X, train_y)
print(grid_search_KNN.best_params_) 


params = grid_search_KNN.best_params_



model = skl_nb.KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'], p=params['p'], metric=params['metric'])
model.fit(train_X, train_y)
prediction = model.predict(test_X)
prediction_train = model.predict(train_X)

test_y = np.where(test_y == 1, 'high_bike_demand', 'low_bike_demand')
train_y = np.where(train_y == 1, 'high_bike_demand', 'low_bike_demand')
prediction = np.where(prediction == 1, 'high_bike_demand', 'low_bike_demand')
prediction_train = np.where(prediction_train == 1, 'high_bike_demand', 'low_bike_demand')


print('Confusion matrix:\n')
print("Test k-NN")
print(pd.crosstab(prediction, test_y), '\n')

print('Confusion Matrices:', "\n")
print("Train k-NN")
print(pd.crosstab(prediction_train, train_y), "\n")


#Scores
print("\n")
print(f"Train Accuracy k-NN:{skl_mt.accuracy_score(train_y, prediction_train): .3f}")
print(f"Test Accuracy k-NN:{skl_mt.accuracy_score(test_y, prediction): .3f}")
print()
print(f"Train recall k-NN:{skl_mt.recall_score(train_y, prediction_train, pos_label='high_bike_demand'): .3f}")
print(f"Test recall k-NN:{skl_mt.recall_score(test_y, prediction, pos_label='high_bike_demand'): .3f}")
print()
print(f"Train precision k-NN:{skl_mt.precision_score(train_y, prediction_train, pos_label='high_bike_demand'): .3f}")
print(f"Test precision k-NN:{skl_mt.precision_score(test_y, prediction, pos_label='high_bike_demand'): .3f}")
print()
print(f"Train f1 k-NN:{skl_mt.f1_score(train_y, prediction_train, pos_label='high_bike_demand'): .3f}")
print(f"Test f1 k-NN:{skl_mt.f1_score(test_y, prediction, pos_label='high_bike_demand'): .3f}")
print()
