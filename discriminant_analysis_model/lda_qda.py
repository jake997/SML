import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn.discriminant_analysis as skl_da
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as skl_mt

#FEATURE ENGINEERING
data = pd.read_csv('training_data.csv')
data['increase_stock'] = np.where(data['increase_stock'] == 'high_bike_demand', 1, 0)

# positive Impact on lda/slightly negative on QDA
data['c_month']=np.empty(len(data.index))
data.loc[data['month'].isin([12, 1, 2]), 'c_month'] = 0
data.loc[data['month'].isin([11]) , 'c_month'] = 1
data.loc[data['month'].isin([3, 5, 7, 8]) , 'c_month'] = 2
data.loc[data['month'].isin([4, 6, 9, 10]) , 'c_month'] = 3

data['c_day_of_week']=np.empty(len(data.index))
data.loc[data['day_of_week'].isin([0,1,2,3,4]), 'c_day_of_week'] = 0
data.loc[data['day_of_week'].isin([6]), 'c_day_of_week'] = 1
data.loc[data['day_of_week'].isin([5]), 'c_day_of_week'] = 2

#huge positive impact on lda f beta / recall
data['c_hour_of_day']=np.empty(len(data.index))
data.loc[data['hour_of_day'].isin([0, 1, 2, 3, 4, 5, 6, 7, 21, 22, 23]), 'c_hour_of_day'] = 0
data.loc[data['hour_of_day'].isin([8, 9, 10, 11, 12, 13, 14, 20]) , 'c_hour_of_day'] = 1
data.loc[data['hour_of_day'].isin([15, 16, 17, 18, 19]) , 'c_hour_of_day'] = 2 

# Small positive impact on QDA/LDA
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


np.random.seed(1)

LDA_model = skl_da.LinearDiscriminantAnalysis()
cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)

parameters = {'solver': ['svd','lsqr']} # 'eigen' doesn't work... As not positive definite..
f_beta = skl_mt.make_scorer(skl_mt.fbeta_score, beta=1.5)
scores = {"precision" : skl_mt.make_scorer(skl_mt.precision_score) ,\
           "recall" : skl_mt.make_scorer(skl_mt.recall_score),\
           "accuracy": skl_mt.make_scorer(skl_mt.accuracy_score),\
           "f_beta": f_beta}

def refit_strategy(cv_results):
    std_f_beta_threshold = 0.2
    cv_results_ = pd.DataFrame(cv_results)
    highest_f_beta_results = cv_results_.loc[(cv_results_['std_test_f_beta'] <= std_f_beta_threshold)]
    best_model_index = highest_f_beta_results['mean_test_f_beta'].idxmax()
    return best_model_index

# Get the best parameters
clf = GridSearchCV(LDA_model, param_grid=parameters, scoring=scores, n_jobs=-1, refit=refit_strategy)

# Data split
train_X, test_X, train_y,  test_y = train_test_split(data.drop(columns=["increase_stock"]),data["increase_stock"],test_size=0.2)
clf.fit(train_X, train_y)
#scores = cross_validate(clf,data.drop(columns=["increase_stock"]),data["increase_stock"],scoring=scores, cv=cv, n_jobs=1)
scores = cross_validate(clf, train_X, train_y, scoring=scores, cv=cv, n_jobs=1)
#print(scores)
f = open("resultsLDA.txt", "w")
print("Results for choosing the sovler (hyperparameter): \n")
print(f"Mean precision: {np.mean(scores['test_precision'])} \nMean recall: {np.mean(scores['test_recall'])} \nMean accuracy: {np.mean(scores['test_accuracy'])} \nMean F-beta: {np.mean(scores['test_f_beta'])} \n", file=f)
print(f"Best params: {clf.best_params_}", file=f)

# Validation using the test Set:
testLDA = skl_da.LinearDiscriminantAnalysis(solver=clf.best_params_["solver"])
testLDA.fit(train_X,train_y)
results = testLDA.predict(test_X)
print("LDA Validation results: \n", file=f)
print(f'Validation accuracy: {skl_mt.accuracy_score(test_y, results)}', file=f)
print(f'Confusion matrix: \n {skl_mt.confusion_matrix(test_y, results)}', file=f)
print(f'f_beta: {skl_mt.fbeta_score(test_y,results,beta=1.5)}', file=f)
print(f'Recall: {skl_mt.recall_score(test_y, results)}', file=f)
print(f'Precision: {skl_mt.precision_score(test_y, results)}', file=f)

## QDA

QDAModel = skl_da.QuadraticDiscriminantAnalysis()
parameters = {'reg_param': [x/100 for x in range(100) ]} # 'eigen' doesn't work... As not positive definite..
f_beta = skl_mt.make_scorer(skl_mt.fbeta_score, beta=1.5)
scores = {"precision" : skl_mt.make_scorer(skl_mt.precision_score) ,\
           "recall" : skl_mt.make_scorer(skl_mt.recall_score),\
           "accuracy": skl_mt.make_scorer(skl_mt.accuracy_score),\
           "f_beta": f_beta}

# Get the best parameters
clf = GridSearchCV(QDAModel, param_grid=parameters, scoring=scores, n_jobs=-1, refit=refit_strategy)
train_X = train_X.drop(columns=["c_month"])
test_X = test_X.drop(columns=["c_month"])
estimator = clf.fit(train_X, train_y) # Fit the model
cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
scores = cross_validate(clf, train_X, train_y, scoring=scores, cv=cv, n_jobs=1)

#print(scores)
f = open("resultsQDA.txt", "w")
print("QDA CV for parameter tuning results: \n", file=f)
print(f"Mean precision: {np.mean(scores['test_precision'])} \nMean recall: {np.mean(scores['test_recall'])} \nMean accuracy: {np.mean(scores['test_accuracy'])} \nMean F-beta: {np.mean(scores['test_f_beta'])}", file=f)
print(f"Best params: {clf.best_params_}", file=f)
print("\n")

# QDA Validation
testQDA = skl_da.QuadraticDiscriminantAnalysis(reg_param=clf.best_params_["reg_param"])
testQDA.fit(train_X,train_y)
results = testQDA.predict(test_X)
print("QDA Validation results: \n", file=f)
print(f'Validation accuracy: {skl_mt.accuracy_score(test_y, results)}', file=f)
print(f'Confusion matrix: \n {skl_mt.confusion_matrix(test_y, results)}', file=f)
print(f'f_beta: {skl_mt.fbeta_score(test_y,results,beta=1.5)}', file=f)
print(f'Recall: {skl_mt.recall_score(test_y, results)}', file=f)
print(f'Precision: {skl_mt.precision_score(test_y, results)}', file=f)

#print("Best model:", clf.cv_results_[clf.best_index_])
