from generate_results_fuctions import *
from itertools import product
from pickle_functions import ml_models_bagging

DEPLOYMENTS = ['frontend']
ACCURACY_METRICS = ['rmse']
WINDOW_SIZES = [10, 20, 30, 40, 50, 60]

"""
# cpu values
W = [['decreasing'], ['increasing'], ['periodic'], ['random']]
METRICS = ['cpu']
WS = [[10], [60], [10], [10]]
B = [150, 110, 100, 100]
HOMOGENEOUS = [['mlpbagging'], ['rfbagging'], ['rfbagging'], ['rfbagging']]
"""

"""
# memory values
W = [['decreasing'], ['increasing'], ['periodic'], ['random']]
METRICS = ['memory']
WS = [[20], [60], [20], [40]]
B = [80, 40, 20, 100]
HOMOGENEOUS = [['mlpbagging'], ['lstmbagging'], ['lstmbagging'], ['rfbagging']]
"""


# responsetime values
W = [['decreasing'], ['increasing'], ['periodic'], ['random']]
METRICS = ['responsetime']
WS = [[50], [30], [40], [40]]
B = [150, 30, 100, 90]
HOMOGENEOUS = [['mlpbagging'], ['rfbagging'], ['rfbagging'], ['rfbagging']]


"""
# traffic values
W = [['decreasing'], ['increasing'], ['periodic'], ['random']]
METRICS = ['traffic']
WS = [[60], [20], [50], [10]]
B = [100, 100, 110, 150]
HOMOGENEOUS = [['mlpbagging'], ['mlpbagging'], ['rfbagging'], ['rfbagging']]
"""

for i in range(1, 2):
    MODEL_NAMES = ['arima', 'lstm', 'mlp', 'rf', 'svr', 'xgboost']
    parameters = list(product(ACCURACY_METRICS, DEPLOYMENTS, METRICS, MODEL_NAMES, W[i], WINDOW_SIZES))
    calculate_accuracy_metrics_and_save_pickle(parameters, "/monolithic/")

    MODEL_NAMES = ml_models_bagging(HOMOGENEOUS[i], B[i])
    parameters = list(product(ACCURACY_METRICS, DEPLOYMENTS, METRICS, MODEL_NAMES, W[i], WS[i]))
    calculate_accuracy_metrics_and_save_pickle(parameters, "/bagging/")

TARGET_VARIABLES = ['decreasing', 'increasing', 'periodic', 'random']
ML_MODELS = ['arima', 'lstm', 'mlp', 'rf', 'svr', 'xgboost']
WINDOW_SIZES = [10, 20, 30, 40, 50, 60]

parameters = list(product(ACCURACY_METRICS, DEPLOYMENTS, METRICS, ML_MODELS, TARGET_VARIABLES, WINDOW_SIZES))
save_accuracy_metrics(parameters)
save_figures(parameters)
concatenate_results(TARGET_VARIABLES, DEPLOYMENTS, METRICS, ML_MODELS, WINDOW_SIZES, ACCURACY_METRICS)
everyone_folder_by_lag(TARGET_VARIABLES, DEPLOYMENTS, METRICS, WINDOW_SIZES, ACCURACY_METRICS)
summary_folder(TARGET_VARIABLES, DEPLOYMENTS, METRICS, ACCURACY_METRICS, ML_MODELS)

# All models by serie
ML_MODELS = ['static_mean_homogeneous', 'static_median_homogeneous', 'dynamic_homogeneous',
             'dynamic_weighting_homogeneous', 'dynamic_weighting_with_selection_homogeneous',
             'static_mean_heterogeneous', 'static_median_heterogeneous', 'dynamic_heterogeneous',
             'dynamic_weighting_heterogeneous', 'dynamic_weighting_with_selection_heterogeneous']

parameters = list(product(ACCURACY_METRICS, DEPLOYMENTS, METRICS, ML_MODELS, TARGET_VARIABLES))

save_accuracy_metrics_one_model(parameters)
save_figures_dynamic_one_model(parameters)
concatenate_results_with_one_result_dynamic(TARGET_VARIABLES, DEPLOYMENTS, METRICS, ML_MODELS, ACCURACY_METRICS)
everyone_folder_with_one_result_dynamic(TARGET_VARIABLES, DEPLOYMENTS, METRICS, ACCURACY_METRICS)
everyone_folder_with_one_result_dynamic_all(DEPLOYMENTS, METRICS, ACCURACY_METRICS)
