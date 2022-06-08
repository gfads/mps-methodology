def calculate_model_accuracy(y_true, y_pred, metric: str):
    from accuracy_metrics import rmse, mape

    if metric == 'rmse':
        return rmse(y_true, y_pred)
    elif metric == 'mape':
        return mape(y_true, y_pred)
    elif metric == 'mse':
        return mse(y_true, y_pred)
    elif metric == 'm6':
        return m6(y_true, y_pred)
    else:
        return 'The metric is yet not implemented'


def mape(actual, predicted):
    import numpy as np

    actual = list(actual)
    predicted = list(predicted)
    n = len(actual)
    soma = 0
    for i in range(0, len(actual)):
        if actual[i] > 0.0:
            x = np.abs((predicted[i] - actual[i]) / actual[i])
        else:
            x = 0
        soma = x + soma
    return 100 / n * soma


def rmse(y_true, y_pred):
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    return sqrt(mean_squared_error(y_true, y_pred))


def m6(y_true, y_pred):
    z = [(v[0] - v[1]) ** 2 for v in zip(y_true, y_pred)]

    return sum(z)


def m6_dw(y_true, y_pred, dk):
    x = [(v[0] - v[1]) ** 2 for v in zip(y_true, y_pred)]
    z = [(v[0] - v[1]) ** 2 for v in zip(x, dk)]

    return sum(z)


def mse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)


def smape(actual, predicted):
    import numpy as np
    epsilon = 1e-6
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + epsilon))
