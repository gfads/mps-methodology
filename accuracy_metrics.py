def calculate_model_accuracy(y_true, y_pred, metric: str):
    if metric == 'mse':
        return mse(y_true, y_pred)
    elif metric == 'rmse':
        return rmse(y_true, y_pred)
    elif metric == 'nrmse':
        return nrmse(y_true, y_pred)
    elif metric == 'mape':
        return mape(y_true, y_pred)
    elif metric == 'smape':
        return smape(y_true, y_pred)
    elif metric == 'arv':
        return arv(y_true, y_pred)
    elif metric == 'mae':
        return mae(y_true, y_pred)
    else:
        return 'This competence metric is not yet implemented.'


def mape(y_true, y_pred):
    from numpy import mean, abs
    epsilon = 1e-6

    return mean(abs((y_true - y_pred) / (y_true + epsilon))) * 100


def mae(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error

    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error

    return mean_squared_error(y_true, y_pred, squared=False)


def mse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error

    return mean_squared_error(y_true, y_pred)


def nrmse(y_true, y_pred):
    return rmse(y_true, y_pred) / (y_true.max() - y_true.min())


def arv(y_true, y_pred):
    from numpy import repeat

    return mse(y_true, y_pred) / mse(repeat(y_true.mean(), len(y_true)), y_pred)


def smape(y_true, y_pred):
    from numpy import abs, mean
    epsilon = 1e-6

    # return 100 / len(y_true) * sum(2 * abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)))
    return mean(2.0 * abs(y_true - y_pred) / ((abs(y_true) + abs(y_pred)) + epsilon)) * 100
