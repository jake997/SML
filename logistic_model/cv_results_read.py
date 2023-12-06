import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Metrics thresholds
f1_threshold = 0.68
std_f1_threshold = 0.02

recall_threshold = 0.85
std_recall_threshold = 0.03

precision_threshold = 0.85
std_precision_threshold = 0.03

cv_results = pd.read_csv("cv_results_logistic_model.csv")
params = cv_results['params']
print(params[0:5])
for i in params.index:
    params[i] = params[i][0:11] + params[i][23:]
cv_results['params'] = params

def float_to_str(flt):
    return f"{flt: .3f}"
keys = [ "mean_score_time",
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
        
    ]

str =""

#Find max index
f1_index =cv_results.loc[(cv_results['mean_test_f1'] >= f1_threshold) &\
                          (cv_results['std_test_f1'] <= std_f1_threshold)].index

recall_index =cv_results.loc[(cv_results['mean_test_recall'] >= recall_threshold) &\
                          (cv_results['std_test_recall'] <= std_recall_threshold)].index

precision_index =cv_results.loc[(cv_results['mean_test_precision'] >= precision_threshold) &\
                          (cv_results['std_test_precision'] <= std_precision_threshold)].index


str = str + "Best models by recall:\n" + cv_results.loc[recall_index, keys].to_string(float_format=float_to_str) + "\n"
str = str + "Best models by precision:\n" + cv_results.loc[precision_index, keys].to_string(float_format=float_to_str) + "\n"
str = str + "Best models by f1:\n" + cv_results.loc[f1_index, keys].to_string(float_format=float_to_str) + "\n"

highest_recall_index = cv_results['mean_test_recall'].idxmax()
highest_f1_index = cv_results['mean_test_f1'].idxmax()
highest_precision_index = cv_results['mean_test_precision'].idxmax()


row = cv_results.loc[highest_recall_index ,   keys]
row1= cv_results.loc[highest_precision_index , keys ]
row2= cv_results.loc[highest_f1_index ,  keys]
str = str +  "\n\nThe model with highest recall\n\n" + row.to_string(float_format= float_to_str) + f"\nindex      {highest_recall_index}" +\
     "\n\nThe model with highest precision\n\n" + row1.to_string(float_format= float_to_str) + f"\nindex      {highest_precision_index}" +\
          "\n\nThe model with highest f1 score\n\n" + row2.to_string(float_format= float_to_str) + f"\nindex      {highest_f1_index}"
with open('best_logistic_models.txt', 'w') as file:
    file.write(str)

