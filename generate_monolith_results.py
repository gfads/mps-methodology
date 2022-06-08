from generate_results_fuctions import *
from itertools import product

WORKLOAD = ['decreasing', 'increasing', 'periodic', 'random']
METRICS = ['cpu', 'memory', 'responsetime' , 'traffic']

DEPLOYMENTS = ['frontend']
MODEL_NAMES = ['arima', 'lstm', 'mlp', 'rf', 'svr', 'xgboost']
WINDOW_SIZES = [10, 20, 30, 40, 50, 60]
ACCURACY_METRICS = ['rmse']

parameters = list(product(ACCURACY_METRICS, DEPLOYMENTS, METRICS, MODEL_NAMES, WORKLOAD, WINDOW_SIZES))
calculate_accuracy_metrics_and_save_pickle(parameters, "/monolithic/")
save_accuracy_metrics(parameters)
save_figures(parameters)
concatenate_results(WORKLOAD, DEPLOYMENTS, METRICS, MODEL_NAMES, WINDOW_SIZES, ACCURACY_METRICS)
everyone_folder_by_lag(WORKLOAD, DEPLOYMENTS, METRICS, WINDOW_SIZES, ACCURACY_METRICS)
summary_folder(WORKLOAD, DEPLOYMENTS, METRICS, ACCURACY_METRICS, MODEL_NAMES)