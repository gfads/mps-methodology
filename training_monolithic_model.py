import argparse
from pickle_functions import save_model
from training_of_models import *
from preprocess import transform_data
from itertools import product


CLI = argparse.ArgumentParser()
CLI.add_argument("--workloads", nargs="*", type=str)
CLI.add_argument("--learning_algorithms", nargs="*", type=str)
CLI.add_argument("--deployment", nargs="*", type=str)
CLI.add_argument("--metrics", nargs="*", type=str)
CLI.add_argument("--lags", nargs="*", type=int)
args = CLI.parse_args()


if not args.metrics or not args.learning_algorithms or not args.lags or not args.deployment or not args.workloads:
    print('You need to specify the performance metrics, deployments, lags, learning_algorimths and workloads!')
    print('For example: python3 training_monolithic_model.py --metrics cpu --deployments frontend --lags 10 '
          '--learning_algorimths mlp --workloads increasing')
    print('For example: python3 training_monolithic_model.py --metrics memory responsetime --deployments frontend '
          '--lags 10 20 30 40 50 --learning_algorimths mlp rf lstm svr --workloads random periodic')
    exit()

METRICS = args.metrics
MODEL_NAMES = args.learning_algorithms
WINDOW_SIZES = args.lags
DEPLOYMENTS = args.deployment
WORKLOAD = args.workloads

parameters = list(product(WORKLOAD, METRICS, DEPLOYMENTS, WINDOW_SIZES))

LEVEL_GRID = 'bagging'

for w, m, d, ws in parameters:
    path = 'Time Series/' + w + '/' + m + '.csv'
    training, testing, total, lags, scaler = transform_data(d, path, ws, True, 0.8, 0, WINDOW_SIZES[-1])

    trained_model = 0

    for mn in MODEL_NAMES:
        if mn == 'svr':
            trained_model = svr_train(training[:, lags], level_grid=LEVEL_GRID)
        elif mn == 'mlp':
            trained_model = mlp_train(training[:, lags], level_grid=LEVEL_GRID)
        elif mn == 'rf':
            trained_model = rf_train(training[:, lags], level_grid=LEVEL_GRID)
        elif mn == 'xgboost':
            trained_model = xgboost_train(training[:, lags], level_grid=LEVEL_GRID)
        elif mn == 'lstm':
            trained_model = lstm_train(training[:, lags], level_grid=LEVEL_GRID)
        elif mn == 'arima':
            trained_model = arima_train(total[:(int(len(total) * 0.8))], level_grid=LEVEL_GRID)

        save_model(trained_model, mn, ws, 1, training, testing, total, lags, scaler, LEVEL_GRID,
                   w + m + d + mn + str(ws))
