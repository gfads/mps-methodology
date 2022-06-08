from generate_results_fuctions import *
from dynamic_selection_functions import load_pickle_dynamic, dynamic_selection_algorithms, dynamic_selection, dynamic_weighting, dynamic_weighting_selection
METRICS = ['cpu']
DEPLOYMENTS = ['frontend']
WORKLOAD = ['decreasing', 'increasing', 'periodic', 'random']

# Bagging
WS = [10, 10, 10, 10]
B = [10, 10, 10, 10]
MN = ['mlpbagging', 'rfbagging', 'rfbagging', 'rfbagging']

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

path_id = generate_path_id(DEPLOYMENTS, METRICS, WORKLOAD)

for i in range(0, len(path_id)):
    id_models = generate_idmodels(B[i], MN[i], 'ho', WS[i])
    dataset = load_pickle_dynamic(id_models, path_id[i])

    d_rs, max_id_model, max_ws, test_size = dynamic_selection_algorithms(path_id[i], id_models, dataset, 'rmse', 10)

    dynamic_selection('rmse', d_rs, id_models, max_ws, 'ds_he', path_id[i], test_size)
    dynamic_weighting(path_id[i], id_models, dataset, 'rmse', d_rs, max_id_model, max_ws, test_size, 'dw_he')
    dynamic_weighting_selection(path_id[i], id_models, dataset, 'rmse', d_rs, max_id_model, max_ws, test_size, 'dws_he')
