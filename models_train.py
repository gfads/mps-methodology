import argparse
from pickle_functions import save_model
from training_of_models import *
from preprocess import transform_data
from itertools import product

CLI = argparse.ArgumentParser()

CLI.add_argument("--approach", type=str)
CLI.add_argument("--competence_measure", type=str)
CLI.add_argument("--deployment", nargs="*", type=str)
CLI.add_argument("--lags", nargs="*", type=int)
CLI.add_argument("--learning_algorithms", nargs="*", type=str)
CLI.add_argument("--metrics", nargs="*", type=str)
CLI.add_argument("--pool_size", type=int)
CLI.add_argument("--workloads", nargs="*", type=str)

args = CLI.parse_args()

if not args.approach or not args.deployment or not args.lags or not args.learning_algorithms or not args.metrics or not args.workloads:
    print('You need to specify the performance approach, deployments, lags, learning_algorimths and workloads!')
    print('For example: python3 models_train.py --approach monolithic --deployment frontend --lags 10 20 30 40 50 60 --learning_algorithms mlp rf svr --metrics cpu --workloads increasing')
    print('For example: python3 models_train.py python3 models_train.py --approach homogeneous --deployment frontend --lags 10 --learning_algorithms mlp --metrics cpu --workloads increasing')
    exit()

if args.approach not in ['homogeneous', 'heterogeneous', 'monolithic']:
    print('The approach parameter accepts the following values: homogeneous, heterogeneous or monolithic.')
    exit()

if args.competence_measure and args.competence_measure not in ['mse', 'rmse', 'nrmse', 'mape', 'smape', 'arv', 'mae']:
    print('This competence metric is not yet implemented.')
    exit()

if args.approach == 'homogeneous':
    if not args.pool_size:
        print('You have not specified a size for the homogeneous pool. The default value was used (100).')

    args.approach = LEVEL_GRID = 'bagging'
elif args.approach in ['monolithic', 'heterogeneous']:
    LEVEL_GRID = 'default'
    args.approach = 'monolithic'

METRICS = args.metrics
MODEL_NAMES = args.learning_algorithms
WINDOW_SIZES = args.lags
DEPLOYMENTS = args.deployment
WORKLOAD = args.workloads
approach = args.approach
competence_measure = args.competence_measure

if not args.pool_size:
    pool_size = 100
else:
    pool_size = args.pool_size

parameters = list(product(WORKLOAD, METRICS, DEPLOYMENTS, WINDOW_SIZES))

for w, m, d, ws in parameters:
    path = 'Time Series/' + w + '/' + m + '.csv'
    training, testing, total, lags, scaler = transform_data(d, path, ws, True, 0.8, 0, WINDOW_SIZES[-1])

    trained_model = 0

    for mn in MODEL_NAMES:
        if mn == 'svr':
            trained_model = svr_train(training[:, lags], level_grid=LEVEL_GRID, pool_size=pool_size,
                                      competence_measure=competence_measure)
        elif mn == 'mlp':
            trained_model = mlp_train(training[:, lags], level_grid=LEVEL_GRID, pool_size=pool_size,
                                      competence_measure=competence_measure)
        elif mn == 'rf':
            trained_model = rf_train(training[:, lags], level_grid=LEVEL_GRID, pool_size=pool_size,
                                     competence_measure=competence_measure)
        elif mn == 'xgboost':
            trained_model = xgboost_train(training[:, lags], level_grid=LEVEL_GRID, pool_size=pool_size,
                                          competence_measure=competence_measure)
        elif mn == 'lstm':
            trained_model = lstm_train(training[:, lags], level_grid=LEVEL_GRID, pool_size=pool_size,
                                       competence_measure=competence_measure)
        elif mn == 'arima':
            trained_model = arima_train(total[:(int(len(total) * 0.8))], level_grid=LEVEL_GRID, pool_size=pool_size,
                                        competence_measure=competence_measure)

        save_model(trained_model, mn, ws, 1, training, testing, total, lags, scaler, LEVEL_GRID,
                   w + '/' + mn + '/' + approach + '/' + m + '/' + d + mn + str(ws))
