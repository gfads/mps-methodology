from pickle_functions import save_model
from training_of_models import *
from preprocess import transform_data
from itertools import product

WORKLOAD = ['decreasing']
METRICS = ['cpu']
DEPLOYMENTS = ['frontend']
MODEL_NAMES = ['lstm', 'mlp', 'rf', 'svr', 'xgboost']
WINDOW_SIZES = [10, 20, 30, 40, 50, 60]

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
