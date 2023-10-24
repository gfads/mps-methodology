def pipeline_sklearn(model, training_sample, competence_measure, lags, validation_sample: list = []):
    from accuracy_metrics import calculate_model_accuracy

    x_train, y_train = training_sample[:, lags[:-1]], training_sample[:, lags[-1]]
    model.fit(x_train, y_train)

    if validation_sample != []:  # Se existe, faça!
        x_val, y_val = validation_sample[:, lags[:-1]], validation_sample[:, lags[-1]]
        predicted = model.predict(x_val)
        accuracy_metric = calculate_model_accuracy(y_val, predicted, competence_measure)
        return model, accuracy_metric
    else:
        return model


def find_better_model(training_models):
    from numpy import Inf

    best_result, best_model = Inf, Inf

    for tm in training_models:
        actual_model = tm[0]
        actual_result = tm[1]

        if actual_result < best_result:
            best_result = actual_result
            best_model = actual_model

    return best_model


def svr_train(training_sample, validation_sample: list = [], lags: list = [], level_grid: str = 'default',
              pool_size: int = 100, competence_measure: str = 'rmse'):
    from sklearn.svm import SVR
    import random

    if level_grid == 'default':
        model = SVR()
        model = pipeline_sklearn(model, training_sample, competence_measure, lags)

        return model
    elif level_grid == 'hard':
        from itertools import product

        kernel: list = ['rbf', 'sigmoid']
        gamma: list = [0.001, 0.01, 0.1, 1]
        epsilon: list = [0.1, 0.001, 0.0001]
        regularization_parameter: list = [0.1, 1, 10, 100, 1000, 10000]

        hyper_param = list(product(kernel, gamma, regularization_parameter, epsilon))

        training_models = []
        for k, g, rp, e in hyper_param:
            training_models.append(
                pipeline_sklearn(SVR(kernel=k, gamma=g, C=rp, epsilon=e, ), training_sample,
                                 competence_measure,
                                 lags, validation_sample=validation_sample))

        return find_better_model(training_models)

    elif level_grid == 'bagging':
        models = bagging(pool_size, training_sample, validation_sample, 'svr', competence_measure, lags)

        return models


def mlp_train(training_sample, validation_sample: list = [], lags: list = [], level_grid: str = 'default',
              pool_size: int = 100, competence_measure: str = 'rmse'):
    from sklearn.neural_network import MLPRegressor
    from itertools import product

    if level_grid == 'default':
        model = MLPRegressor()
        model = pipeline_sklearn(model, training_sample, competence_measure, lags)

        return model
    elif level_grid == 'hard':
        hidden_layer_sizes = [5, 10, 15, 20]
        activation = ['tanh', 'relu', 'logistic']
        solver = ['lbfgs', 'sgd', 'adam']
        max_iter = [100, 500, 1000, 2000, 3000]
        learning_rate = ['constant', 'adaptive']

        hyper_param = list(product(hidden_layer_sizes, activation, solver, max_iter, learning_rate))

        training_models = []

        for hls, a, s, mi, lr in hyper_param:
            training_models.append(
                pipeline_sklearn(
                    MLPRegressor(hidden_layer_sizes=hls, activation=a, solver=s, max_iter=mi, learning_rate=lr),
                    # MLPRegressor(hidden_layer_sizes=hls, activation=a, solver=s),
                    training_sample, competence_measure, lags, validation_sample=validation_sample))

        return find_better_model(training_models)

    elif level_grid == 'bagging':
        models = bagging(pool_size, training_sample, validation_sample, 'mlp', competence_measure, lags)

        return models


def reamostragem(serie, n):
    import numpy as np
    size = len(serie)
    ind_particao = []

    for i in range(n):
        ind_r = np.random.randint(size)
        ind_particao.append(ind_r)

    return ind_particao


def bagging(qtd_modelos, training_sample, validation_sample, name_model, competence_measure, lags):
    models = {'model': [], 'training_sample': [], 'validation_sample': [], 'indices': []}

    for i in range(qtd_modelos):
        print('Training model: ', i)
        indices_training = reamostragem(training_sample, len(training_sample))
        particao = training_sample[indices_training, :]

        if name_model == 'mlp':
            models['model'].append(
                mlp_train(particao, validation_sample=validation_sample, lags=lags, level_grid='hard',
                          competence_measure=competence_measure))
        elif name_model == 'svr':
            models['model'].append(
                svr_train(particao, validation_sample=validation_sample, lags=lags, level_grid='hard',
                          competence_measure=competence_measure))
        elif name_model == 'rf':
            models['model'].append(rf_train(particao, validation_sample=validation_sample, lags=lags, level_grid='hard',
                                            competence_measure=competence_measure))
        elif name_model == 'xgboost':
            models['model'].append(
                xgboost_train(particao, validation_sample=validation_sample, lags=lags, level_grid='hard',
                              competence_measure=competence_measure))
        elif name_model == 'lstm':
            models['model'].append(
                    lstm_train(particao, validation_sample=validation_sample, lags=lags, level_grid='hard',
                               competence_measure=competence_measure))

        elif name_model == 'arima':
            models['model'].append(
                    arima_train(particao, level_grid='hard', window_size=lags, competence_measure=competence_measure))


        models['training_sample'].append(particao)
        models['validation_sample'].append(validation_sample)
        models['indices'].append(indices_training)

    return models


def rf_train(training_sample, validation_sample: list = [], lags: list = [], level_grid='default', pool_size: int = 100,
             competence_measure: str = 'rmse'):
    from sklearn.ensemble import RandomForestRegressor
    from itertools import product

    if level_grid == 'default':
        from sklearn.ensemble import RandomForestRegressor
        model = pipeline_sklearn(RandomForestRegressor(), training_sample, competence_measure, lags)

        return model
    elif level_grid == 'hard':

        min_samples_leaf = [1, 5, 10]
        min_samples_split = [2, 5, 10, 15]
        n_estimators = [100, 500, 1000]


        hyper_param = list(product(min_samples_leaf, min_samples_split, n_estimators))

        training_models = []
        for msl, mss, ne in hyper_param:
            training_models.append(
                pipeline_sklearn(RandomForestRegressor(n_estimators=ne, min_samples_leaf=msl, min_samples_split=mss),
                                 training_sample, competence_measure, lags, validation_sample)
            )

        model = find_better_model(training_models)

        return model

    elif level_grid == 'bagging':
        models = bagging(pool_size, training_sample, validation_sample, 'rf', competence_measure, lags)

        return models


def pipeline_xgboost(parameters, training_sample, competence_measure, lags, validation_sample: list = []):
    from accuracy_metrics import calculate_model_accuracy
    from xgboost import XGBRegressor

    x_train, y_train = training_sample[:, lags[:-1]], training_sample[:, lags[-1]]

    model = XGBRegressor(learning_rate=parameters['learning_rate'], max_depth=parameters['max_depth'],
                         n_estimators=parameters['n_estimators'], reg_alpha=parameters['reg_alpha'],
                         subsample=parameters['subsample'],
                         tree_method="hist")

    model.fit(x_train, y_train)

    if validation_sample != []:
        predicted = model.predict(validation_sample[:, lags[:-1]])
        accuracy_metric = calculate_model_accuracy(validation_sample[:, lags[-1]], predicted, competence_measure)

        return model, accuracy_metric
    else:
        return model


def xgboost_train(training_sample, validation_sample: list = [], lags: list = [], level_grid='default',
                  pool_size: int = 100,
                  competence_measure: str = 'rmse'):
    from itertools import product
    if level_grid == 'default':
        model = pipeline_xgboost(1, {}, training_sample, competence_measure, lags)

        return model
    elif level_grid == 'hard':

        learning_rate = [0.1, 0.05]
        reg_alpha = [1, 5]
        max_depth = [25, 50]
        n_estimators = [100, 150]
        subsample = [0.5, 0.8]

        hyper_param = list(product(learning_rate, reg_alpha, max_depth, n_estimators,
                                   reg_alpha, subsample))

        training_models = []
        for lr, ra, md, ne, re, ssa in hyper_param:
            training_models.append(pipeline_xgboost({'learning_rate': lr, 'reg_alpha': ra, 'max_depth': md,
                                                     'n_estimators': ne, 'subsample': ssa}, training_sample,
                                                    competence_measure,
                                                    lags, validation_sample))

        model = find_better_model(training_models)

        return model

    elif level_grid == 'bagging':
        models = bagging(pool_size, training_sample, validation_sample, 'xgboost', competence_measure, lags)

        return models


def lstm_train(training_sample, validation_sample: list = [], lags: list = [], level_grid='default',
               pool_size: int = 100, competence_measure: str = 'rmse'):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    from numpy import Inf, isnan
    from accuracy_metrics import calculate_model_accuracy
    from itertools import product
    import random

    if level_grid == 'default':
        x_training = training_sample[:, lags[:-1]]
        y_training = training_sample[:, lags[-1]]
        x_training = x_training.reshape((x_training.shape[0], x_training.shape[1], 1))

        lags_size = x_training.shape[1]

        model = Sequential()
        model.add(LSTM(4, input_shape=(lags_size, 1)))
        model.add(Dense(1))
        model.compile(optimizer='Adam', loss='mean_squared_error')
        model.fit(x_training, y_training)  # , epochs=20, verbose=0, batch_size=len(x_training))

        return model
    elif level_grid == 'hard':
        x_training = training_sample[:, lags[:-1]]
        y_training = training_sample[:, lags[-1]]
        x_training = x_training.reshape((x_training.shape[0], x_training.shape[1], 1))

        lags_size = x_training.shape[1]

        epochs = [1, 2, 4, 8, 10]
        learning_rate = [0.05, 0.01, 0.001]
        batches = [64, 128]
        number_of_units = [50, 75, 125]
        number_of_hidden_layers = [2, 3, 4, 5, 6]

        best_accuracy_measure = Inf
        best_model_lstm = Sequential()

        hyper_param = list(product(epochs, learning_rate, batches, number_of_units, number_of_hidden_layers))

        for e, lr, b, nu, nhr in hyper_param:
            model = Sequential()

            for _ in range(0, nhr):
                model.add(LSTM(nu, activation='relu', return_sequences=True, input_shape=(lags_size, 1)))

            model.add(LSTM(nu, activation='relu', input_shape=(lags_size, 1)))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
            model.fit(x_training, y_training, epochs=e, verbose=0, batch_size=b)

            x_validation = validation_sample[:, lags[:-1]]
            x_validation = x_validation.reshape((x_validation.shape[0], x_validation.shape[1], 1))
            forecast = model.predict(x_validation)

            if not isnan(forecast).any():
                accuracy_measure = calculate_model_accuracy(x_validation[:, -1], forecast, competence_measure)
            else:
                accuracy_measure = Inf

            if accuracy_measure < best_accuracy_measure:
                best_accuracy_measure = accuracy_measure
                best_model_lstm = model

        return best_model_lstm

    elif level_grid == 'bagging':
        models = bagging(pool_size, training_sample, validation_sample, 'lstm', competence_measure, lags)

        return models


def d_values(data: list):
    a = 0
    for index in range(len(data) - 1, 0, -1):
        if (data[index] - data[index - 1]) != 1:
            return len(data) - 1 - index
        else:
            a = len(data) - 1

    return a


def find_p_d_q_arima(data, window_size):
    from pmdarima.arima import ADFTest
    from preprocess import select_lag_acf, select_lag_pacf

    adf_test = ADFTest(alpha=0.05)
    dtr = adf_test.should_diff(data)
    d = 0

    if dtr[1]:
        d = 1

    q = d_values(select_lag_acf(data, window_size))
    p = d_values(select_lag_pacf(data, window_size))

    return p, d, q


def arima_train(data: list, level_grid: str, window_size: int = 0, pool_size: int = 150,
                competence_measure: str = 'rmse'):
    from pmdarima.arima import auto_arima

    if level_grid == 'hard':
        p, d, q = find_p_d_q_arima(data, window_size)

        arima_model = auto_arima(data, start_p=0, start_q=0, max_p=p, max_q=q,
                                 seasonal=False, error_action='warn', trace=False, suppress_warnings=True,
                                 stepwise=True)

        return arima_model
    elif level_grid == 'default':
        arima_model = auto_arima(data)

        return arima_model

    elif level_grid == 'bagging':
        models = bagging(pool_size, data, data, 'arima', competence_measure, window_size)

        return models
