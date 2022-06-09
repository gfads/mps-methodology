from generate_results_fuctions import *
from dynamic_selection_functions import load_pickle_dynamic, dynamic_selection_algorithms, dynamic_selection, dynamic_weighting, dynamic_weighting_selection

WORKLOAD = ['decreasing', 'increasing', 'periodic', 'random']
METRICS = ['responsetime']
DEPLOYMENTS = ['frontend']


# Bagging
WS = [50, 30, 40, 40]
B = [150, 30, 100, 90]
MN = ['mlp', 'rf', 'rf', 'rf']

"""
# Heterogeneous
MN = [['arima', 'lstm', 'mlp', 'rf', 'svr', 'xgboost'],
      ['arima', 'lstm', 'mlp', 'rf', 'svr', 'xgboost'],
      ['arima', 'lstm', 'mlp', 'rf', 'svr', 'xgboost'],
      ['arima', 'lstm', 'mlp', 'rf', 'svr', 'xgboost']]

WS = [[10, 10, 10, 30, 10, 60],
      [10, 60, 10, 60, 40, 40],
      [10, 10, 60, 10, 10, 60],
      [10, 60, 20, 10, 10, 10]]
"""

path_id = generate_path_id(METRICS, WORKLOAD, MN, 'homogeneous')

for i in range(1, 2):
    id_models = generate_idmodels(B[i], METRICS, MN[i], 'homogeneous', WS[i])
    dataset = load_pickle_dynamic(id_models, path_id[i])

    d_rs, max_id_model, max_ws, test_size = dynamic_selection_algorithms(path_id[i], id_models, dataset, 'rmse', 10)

    dynamic_selection('rmse', d_rs, id_models, max_ws, 'dynamic_homogeneous', path_id[i], test_size)
    dynamic_weighting(path_id[i], id_models, dataset, 'rmse', d_rs, max_id_model, max_ws, test_size,
                      'dynamic_weighting_homogeneous')
    dynamic_weighting_selection(path_id[i], id_models, dataset, 'rmse', d_rs, max_id_model, max_ws, test_size,
                                'dynamic_weighting_with_selection_homogeneous')
