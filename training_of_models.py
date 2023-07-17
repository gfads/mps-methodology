def pipeline_sklearn(model, training_sample, competence_measure, validation_sample: list = []):
    from accuracy_metrics import calculate_model_accuracy
    x_train, y_train = training_sample[:, 0:-1], training_sample[:, -1]
    model.fit(x_train, y_train)

    if validation_sample:  # Se existe, faça!
        x_val, y_val = validation_sample[:, 0:-1], validation_sample[:, -1]
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


def svr_train(training_sample, validation_sample: list = [], level_grid: str = 'default', pool_size: int = 100,
              competence_measure: str = 'rmse'):
    from sklearn.svm import SVR

    if level_grid == 'default':
        model = SVR()
        model = pipeline_sklearn(model, training_sample, competence_measure)

        return model
    elif level_grid == 'hard':
        from itertools import product

        kernel = ['rbf', 'sigmoid']
        gamma = [0.1, 0.5, 10, 50, 100, 1000]
        epsilon = [1, 0.1, 0.001, 0.0001, 0.00001]
        regularization_parameter = [0.1, 1, 5, 10, 100, 1000]
        hyper_param = list(product(kernel, gamma, epsilon, regularization_parameter))

        training_models = []
        for k, g, e, rp in hyper_param:
            training_models.append(
                pipeline_sklearn(SVR(C=rp, epsilon=e, kernel=k, gamma=g), training_sample, competence_measure,
                                 validation_sample))

        return find_better_model(training_models)

    elif level_grid == 'bagging':
        models = bagging(pool_size, training_sample, validation_sample, 'svr', competence_measure)

        return models


def mlp_train(training_sample, validation_sample: list = [], level_grid: str = 'default', pool_size: int = 100,
              competence_measure: str = 'rmse'):
    from sklearn.neural_network import MLPRegressor
    from itertools import product

    if level_grid == 'default':
        model = MLPRegressor()
        model = pipeline_sklearn(model, training_sample, competence_measure)

        return model
    elif level_grid == 'hard':
        hidden_layer_sizes = [1, 5, 10, 50, 100]
        activation = ['identity', 'tanh', 'relu', 'logistic']
        solver = ['lbfgs', 'sgd', 'adam']
        max_iter = [1000]
        learning_rate = ['constant', 'invscaling', 'adaptive']
        num_exec = 10
        hyper_param = list(product(hidden_layer_sizes, activation, solver, max_iter, learning_rate, range(0, num_exec)))

        training_models = []
        for hls, a, s, mi, lr, _ in hyper_param:
            training_models.append(
                pipeline_sklearn(
                    MLPRegressor(hidden_layer_sizes=hls, activation=a, solver=s, max_iter=mi, learning_rate=lr),
                    training_sample, competence_measure, validation_sample))

        return find_better_model(training_models)

    # Improving the accuracy of intelligent forecasting models using the Perturbation
    # Theory
    elif level_grid == 'acf':
        from itertools import product

        learning_rate_init = [0.0001, 0.001, 0.00001]
        solver = ['lbfgs', 'sgd', 'adam']
        hidden_layer_sizes = [1, 5, 10, 15, 20]

        hyper_param = list(product(learning_rate_init, solver, hidden_layer_sizes))

        training_models = []
        for lri, s, hls, in hyper_param:
            training_models.append(pipeline_sklearn(MLPRegressor(hidden_layer_sizes=hls, solver=s,
                                                                 learning_rate_init=lri), training_sample,
                                                    competence_measure,
                                                    validation_sample)[0])

        return training_models

    elif level_grid == 'bagging':
        models = bagging(pool_size, training_sample, validation_sample, 'mlp', competence_measure)

        return models


def reamostragem(serie, n):
    import numpy as np
    size = len(serie)
    ind_particao = []

    for i in range(n):
        ind_r = np.random.randint(size)
        ind_particao.append(ind_r)

    return ind_particao


def bagging(qtd_modelos, training_sample, validation_sample, name_model, competence_measure):
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from keras.models import Sequential
    from keras.layers import LSTM, Dense

    models = {'model': [], 'training_sample': [], 'validation_sample': [], 'indices': []}

    for i in range(qtd_modelos):
        print('Training model: ', i)
        indices = reamostragem(training_sample, len(training_sample))
        particao = training_sample[indices, :]

        if name_model == 'mlp':
            model = MLPRegressor()
        elif name_model == 'rf':
            model = RandomForestRegressor()
        elif name_model == 'svr':
            model = SVR()

        if name_model == 'mlp' or name_model == 'rf' or name_model == 'svr':
            models['model'].append(pipeline_sklearn(model, particao, competence_measure, validation_sample))
        elif name_model == 'xgboost':
            models['model'].append(pipeline_xgboost(1, {}, particao, competence_measure, validation_sample)[0])

        elif name_model == 'lstm':
            x_training = particao[:, 0:-1]
            y_training = particao[:, -1]
            x_training = x_training.reshape((x_training.shape[0], x_training.shape[1], 1))
            lags = x_training.shape[1]

            model = Sequential()
            model.add(LSTM(4, input_shape=(lags, 1)))
            model.add(Dense(1))
            model.compile(optimizer='Adam', loss='mean_squared_error')
            model.fit(x_training, y_training)
            models['model'].append(model)

        models['training_sample'].append(particao)
        models['validation_sample'].append(validation_sample)
        models['indices'].append(indices)

    return models


def rf_train(training_sample, validation_sample: list = [], level_grid='default', pool_size: int = 100,
             competence_measure: str = 'rmse'):
    from sklearn.ensemble import RandomForestRegressor
    from itertools import product

    if level_grid == 'default':
        from sklearn.ensemble import RandomForestRegressor
        model = pipeline_sklearn(RandomForestRegressor(), training_sample, competence_measure)

        return model
    elif level_grid == 'hard':
        bootstrap = [True, False]
        max_depth = [10, 60, 100, None]
        max_features = ['auto', 'sqrt']
        min_samples_leaf = [1, 2, 4]
        min_samples_split = [2, 5, 10]
        n_estimators = [100, 200, 300]

        hyper_param = list(
            product(bootstrap, max_depth, max_features, min_samples_leaf, min_samples_split, n_estimators))

        training_models = []
        for b, md, mf, msl, mss, ne in hyper_param:
            training_models.append(
                RandomForestRegressor(bootstrap=b, max_depth=md, max_features=mf, n_estimators=ne, min_samples_leaf=msl,
                                      min_samples_split=mss), training_sample, validation_sample)

        model = find_better_model(training_models)

        return model

    # End-to-End Latency Prediction of Microservices Workflow on Kubernetes:
    # A Comparative Evaluation of Machine Learning Models and Resource Metrics
    elif level_grid == 'acf':
        bootstrap = ['True', 'False']
        max_features = ['auto', 'sqrt', 'log2']
        n_estimators = [10, 100, 250, 500]

        hyper_param = list(
            product(bootstrap, max_features, n_estimators))

        training_models = []

        for b, mf, ne in hyper_param:
            training_models.append(pipeline_sklearn(RandomForestRegressor(bootstrap=b,
                                                                          max_features=mf,
                                                                          n_estimators=ne), training_sample,
                                                    competence_measure,
                                                    validation_sample)[0])

        return training_models

    elif level_grid == 'bagging':
        models = bagging(pool_size, training_sample, validation_sample, 'rf', competence_measure)

        return models


def pipeline_xgboost(num_exec, parameters, training_sample, competence_measure, validation_sample: list = []):
    from accuracy_metrics import calculate_model_accuracy
    from xgboost import DMatrix, train

    training_sample_dmatrix = DMatrix(training_sample[:, 0:-1], training_sample[:, -1])
    model = train(parameters, training_sample_dmatrix, num_exec)

    if validation_sample:
        validation_sample_dmatrix = DMatrix(validation_sample[:, 0:-1], validation_sample[:, -1])
        predicted = model.predict(validation_sample_dmatrix)
        accuracy_metric = calculate_model_accuracy(validation_sample[:, -1], predicted, competence_measure)

        return model, accuracy_metric
    else:
        return model


def xgboost_train(training_sample, validation_sample: list = [], level_grid='default', pool_size: int = 100,
                  competence_measure: str = 'rmse'):
    from itertools import product

    if level_grid == 'default':
        model = pipeline_xgboost(1, {}, training_sample, competence_measure)

        return model
    elif level_grid == 'hard':
        min_child_weight = [1, 5, 10]
        gamma = [0.5, 1, 1.5, 2, 5]
        subsample = [0.6, 0.8, 1.0]
        colsample_bytree = [0.6, 0.8, 1.0]
        max_depth = [3, 4, 5, 6]
        eta = [0.01, 0.3]
        hyper_param = list(product(min_child_weight, gamma, subsample, colsample_bytree, max_depth, eta))

        training_models = []
        for mcw, g, ss, csb, md, e in hyper_param:
            training_models.append(pipeline_xgboost(10, {'min_child_weight': mcw, 'gamma': g, 'subsample': ss,
                                                         'colsample_bytree': csb, 'max_depth': md,
                                                         'eta': e}, training_sample, competence_measure,
                                                    validation_sample))

        model = find_better_model(training_models)

        return model

    elif level_grid == 'bagging':
        models = bagging(pool_size, training_sample, validation_sample, 'xgboost', competence_measure)

        return models


def lstm_train(training_sample, validation_sample: list = [], level_grid='default', pool_size: int = 100,
               competence_measure: str = 'rmse'):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    from numpy import Inf, isnan
    from accuracy_metrics import calculate_model_accuracy
    from itertools import product

    x_training = training_sample[:, 0:-1]
    y_training = training_sample[:, -1]
    x_training = x_training.reshape((x_training.shape[0], x_training.shape[1], 1))
    lags = x_training.shape[1]

    if level_grid == 'default':
        model = Sequential()
        model.add(LSTM(4, input_shape=(lags, 1)))
        model.add(Dense(1))
        model.compile(optimizer='Adam', loss='mean_squared_error')
        model.fit(x_training, y_training)  # , epochs=20, verbose=0, batch_size=len(x_training))

        # model.add(LSTM(4, activation='relu', input_shape=(lags, 1)))
        # model.add(Dense(1))
        # model.compile(optimizer='Adam', loss='mean_squared_error')
        # model.fit(x_training, y_training, epochs=20, verbose=0, batch_size=len(x_training))

        # model.add(LSTM(4, activation='relu', input_shape=(lags, 1)))
        # model.add(Dense(1))
        # model.compile(optimizer='Adam', loss='mean_squared_error')
        # model.fit(x_training, y_training, epochs=100, verbose=0, batch_size=1)

        return model
    elif level_grid == 'hard':

        epochs = [1, 2, 4, 8, 10]
        learning_rate = [0.05, 0.01, 0.001]
        batches = [1, 64, 128]
        number_of_units = [50, 75, 125]
        # number_of_hidden_layers = [2, 3, 4, 5, 6]

        best_accuracy_measure = Inf
        best_model_lstm = Sequential()

        parameters = list(product(range(0, 1), epochs, learning_rate, batches, number_of_units))

        for _, e, lr, b, nu in parameters:
            model = Sequential()
            model.add(LSTM(nu, activation='relu', input_shape=(lags, 1)))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
            model.fit(x_training, y_training, epochs=e, verbose=0, batch_size=b)

            x_validation = validation_sample[:, 0:-1]
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
        models = bagging(pool_size, training_sample, validation_sample, 'lstm', competence_measure)

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


def arima_train(data: list, level_grid: str, window_size: int = 0, pool_size: int = 100,
                competence_measure: str = 'rmse'):
    from pmdarima.arima import auto_arima

    if level_grid == 'hard':
        p, d, q = find_p_d_q_arima(data, window_size)

        if d == 0:
            arima_model = auto_arima(data, start_p=0, start_q=0, max_p=p, max_q=q,
                                     seasonal=False, error_action='warn', trace=False, suppress_warnings=True,
                                     stepwise=True)
        else:
            arima_model = auto_arima(data, start_p=0, d=d, start_q=0, max_p=p, max_d=2, max_q=q,
                                     seasonal=False, error_action='warn', trace=False, suppress_warnings=True,
                                     stepwise=True)

        return arima_model
    elif level_grid == 'default':
        arima_model = auto_arima(data)

        return arima_model
