import argparse
from generate_results_fuctions import *
from dynamic_selection_functions import load_pickle_dynamic, dynamic_selection_algorithms, dynamic_selection, dynamic_weighting, dynamic_weighting_selection


CLI = argparse.ArgumentParser()
CLI.add_argument("--approach", type=str)
CLI.add_argument("--workload", type=str)
CLI.add_argument("--metric", type=str)
CLI.add_argument("--lag", type=int)
CLI.add_argument("--pool_size", type=int)
CLI.add_argument("--learning_algorithms", nargs="*", type=str)
CLI.add_argument("--deployment", type=str)

args = CLI.parse_args()

# if not args.metrics or not args.learning_algorithms or not args.lags or not args.deployment or not args.workloads:
#     print('You need to specify the performance metrics, deployments, lags, learning_algorimths and workloads!')
#     print('For example: python3 training_monolithic_model.py --metrics cpu --deployments frontend --lags 10 '
#           '--learning_algorimths mlp --workloads increasing')
#     print('For example: python3 training_monolithic_model.py --metrics memory responsetime --deployments frontend '
#           '--lags 10 20 30 40 50 --learning_algorimths mlp rf lstm svr --workloads random periodic')
#     exit()

approach = args.approach
metrics = args.metrics
deployment = args.deployment
workload = args.workloads
lags = args.lags
pool_size = args.pool_size
learning_algorithms = args.learning_algorithms

#print(approach, metrics, deployment, WORKLOAD, WS, B, MN)

# # Bagging
# WS = [50, 30, 40, 40]
# B = [150, 30, 100, 90]
# MN = ['mlp', 'rf', 'rf', 'rf']

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

path_id = generate_path_id(metrics, workload, learning_algorithms, approach)
RoC = 10

print(path_id)

# id_models = generate_idmodels(pool_size, deployment, learning_algorithms, approach, lags)
# dataset = load_pickle_dynamic(id_models, path_id[i])
#
# d_rs, max_id_model, max_ws, test_size = dynamic_selection_algorithms(path_id[i], id_models, dataset, 'rmse', RoC)
#
# dynamic_selection('rmse', d_rs, id_models, max_ws, 'dynamic_homogeneous', path_id[i], test_size)
# dynamic_weighting(path_id[i], id_models, dataset, 'rmse', d_rs, max_id_model, max_ws, test_size,
#                   'dynamic_weighting_'+approach)
# dynamic_weighting_selection(path_id[i], id_models, dataset, 'rmse', d_rs, max_id_model, max_ws, test_size,
#                             'dynamic_weighting_with_selection_'+approach)
