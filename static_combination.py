from generate_results_fuctions import load_pickle_static, static_combination, generate_path_id, generate_idmodels

WORKLOAD = ['decreasing', 'increasing', 'periodic', 'random']
METRICS = ['responsetime']
DEPLOYMENTS = ['frontend']

# IF Homogeneous to ResponseTime
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

ACCURACY_METRICS = ['rmse']
CENTRAL_MEASURES = ['mean', 'median']

path_id = generate_path_id(METRICS, WORKLOAD, MN, 'homogeneous')

for i in range(0, 4):
    id_models = generate_idmodels(B[i], METRICS, MN[i], 'homogeneous', WS[i])
    dataset = load_pickle_static(ACCURACY_METRICS, id_models, path_id[i])

    static_combination(ACCURACY_METRICS, CENTRAL_MEASURES, dataset, id_models, 'homogeneous', path_id[i])
