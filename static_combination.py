from generate_results_fuctions import load_pickle_static, static_combination, generate_path_id, generate_idmodels

WORKLOAD = ['decreasing']
METRICS = ['cpu']
DEPLOYMENTS = ['frontend']

# IF Homogeneous to CPU

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

ACCURACY_METRICS = ['rmse']
CENTRAL_MEASURES = ['mean', 'median']

path_id = generate_path_id(DEPLOYMENTS, METRICS, WORKLOAD)

for i in range(0, 4):
    id_models = generate_idmodels(B[i], MN[i], 'he', WS[i])
    dataset = load_pickle_static(ACCURACY_METRICS, id_models, path_id[i])

    static_combination(ACCURACY_METRICS, CENTRAL_MEASURES, dataset, id_models, 'he', path_id[i])
