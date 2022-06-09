def load_model(model_path, model_name):
    from joblib import load

    if model_name[0: 4] == 'lstm':
        from keras.models import load_model
        # model_load = load_model(model_path + '/' + model_name + '.h5')
        model_load = load_model("Pickle Models/" + model_path + '.h5')
    else:
        model_location = model_path + '/' + model_name + '.joblib'
        model_load = load(model_location)

    return model_load


def predict_data(data, model, ml_model: str, *arima_type):
    if ml_model[0: 3] == 'svr' or ml_model[0: 3] == 'mlp' or ml_model[0:2] == 'rf':
        return model.predict(data)
    elif ml_model[0: 7] == 'xgboost':
        from xgboost import DMatrix
        return model.predict(DMatrix(data))
    elif ml_model[0: 4] == 'lstm':
        data = data.reshape((data.shape[0], data.shape[1], 1))
        return model.predict(data).ravel()
    elif ml_model[0: 5] == 'arima':
        if arima_type:
            if arima_type[0] == 'in_sample':
                return model.predict_in_sample(data)[data]
            elif arima_type[0] == 'out_sample':  # Prediz até um elemento N
                return model.predict(data)[-1]
        else:
            return model.predict(len(data))  # Prediz a quantidade de elemento de DAta


def concatenate_results(target_variables, metrics, ml_models, window_sizes, accuracy_metric):
    from pandas import read_csv, DataFrame
    from numpy import array

    for target_variable in target_variables:
        for metric in metrics:
            for window_size in window_sizes:
                path_local = ""
                for ml_metric in accuracy_metric:
                    ac_metric = []
                    for ml_model in ml_models:
                        path_local = 'Results/' + target_variable + '/' + metric + '/' + str(
                            window_size) + '/'
                        ac_metric.append(
                            read_csv(path_local + ml_model + '_' + ml_metric + '.csv')['testing'].tolist()[0])
                    DataFrame(array(ac_metric).reshape(1, len(ml_models)), columns=ml_models).to_csv(path_local + ml_metric + '.csv')


def concatenate_results_with_one_result_dynamic(target_variables, metrics, ml_models, accuracy_metric):
    from pandas import read_csv, DataFrame
    from numpy import array
    from background_functions import try_create_folder

    for target_variable in target_variables:
        for metric in metrics:
            path_local = try_create_folder(
                'Results/' + target_variable + '/' + metric + '/summary/')

            for ml_metric in accuracy_metric:
                ac_metric = []
                for ml_model in ml_models:
                    ac_metric.append(
                        read_csv(path_local + ml_model + '_' + ml_metric + '.csv')['testing'].tolist()[0])
                DataFrame(array(ac_metric).reshape(1, len(ml_models)),
                          columns=[s + '' for s in ml_models], index=[target_variable]).to_csv(path_local + ml_metric + '.csv')


def everyone_folder_by_lag(target_variables, metrics, windows_size, ml_metrics,
                           folder_name: str = 'Results'):
    from pandas import read_csv, DataFrame
    from numpy import array
    from background_functions import try_create_folder

    for target_variable in target_variables:
        for metric in metrics:
            folder_everyone = folder_name + '/' + target_variable + '/' + metric + '/summary_lag'
            try_create_folder(folder_everyone)

            for ml_metric in ml_metrics:
                dados = []
                r = DataFrame
                for window_size in windows_size:
                    r = read_csv(
                        folder_name + '/' + target_variable + '/' + metric + '/' + str(
                            window_size) + '/' + ml_metric + '.csv')
                    for d in r.values[0][1:]:
                        dados.append(d)
                index = r.columns[1:]  # Pegando columns
                DataFrame(array(dados).reshape(len(windows_size), len(index)).transpose(), columns=windows_size,
                          index=index).to_csv(folder_everyone + '/' + ml_metric + '.csv')


def everyone_folder_with_one_result_dynamic(target_variables, metrics, ml_metrics,
                                            folder_name: str = 'Results'):
    from pandas import read_csv, DataFrame, concat
    from background_functions import try_create_folder

    try_create_folder(folder_name + '/summary')
    path_v = try_create_folder(folder_name + '/' + '/summary/better_pool_values/')

    for metric in metrics:
        for ml_metric in ml_metrics:
            df = DataFrame()

            for workload in target_variables:
                path_local = folder_name + '/' + workload + '/' + metric + '/summary/' + ml_metric + '.csv'
                data = read_csv(path_local)
                df = concat([df, data], axis=0)

            # Set index
            new_columns = df.columns.values
            new_columns[0] = 'Workload'
            df.columns = new_columns

            df.to_csv(path_v + metric + '_' + ml_metric + '.csv', index=False)


def calculate_the_oracle(target_variables, metrics, ml_models, accuracy_metrics, window_sizes, a, b,
                         folder_name: str = 'results'):
    from pickle_functions import load_pickle
    from accuracy_metrics import calculate_model_accuracy
    from pandas import DataFrame
    from numpy import array, Inf
    from background_functions import try_create_folder

    path_dir = try_create_folder(folder_name + '/summary/oracle_value_by_serie/')

    prev_dict = {}

    for target_variable in target_variables:
        for metric in metrics:
            for window_size in window_sizes:
                for ml_model in ml_models:
                    if ml_model[0: 4] == a[0: 4]:
                        window_size = b
                    df_pickle = load_pickle(target_variable + metric + ml_model + str(window_size))
                    prev_dict[target_variable + metric + ml_model + str(window_size)] = [
                        df_pickle['y_pred_testing'], df_pickle['y_true_testing']]

    for metric in metrics:
        oracle_by_workload = []
        for target_variable in target_variables:

            pred_test_oracle = []
            true_test_oracle = \
                prev_dict[target_variable + metric  + ml_models[0] + str(window_size)][1]
            for index in range(0, 852):
                better = Inf
                value = Inf

                for window_size in window_sizes:
                    for ml_model in ml_models:
                        if ml_model[0: 4] == a[0: 4]:
                            window_size = b
                        key = target_variable + metric + ml_model + str(window_size)
                        prev = prev_dict[key][0][index]
                        target = prev_dict[key][1][index]
                        if better > abs(target - prev):
                            better = abs(target - prev)
                            value = prev
                pred_test_oracle.append(value)

                for accuracy_metric in accuracy_metrics:
                    oracle_by_workload.append([target_variable,
                                               calculate_model_accuracy(pred_test_oracle, true_test_oracle,
                                                                        accuracy_metric)])

            DataFrame(array(oracle_by_workload).reshape(len(target_variables), 2),
                      columns=['Workload', 'Oracle']).to_csv(
                path_dir + metric + '_' + accuracy_metric + '.csv', index=False)
            print(oracle_by_workload[0][1])


def everyone_folder_with_one_result_dynamic_all(metrics, ml_metrics,
                                                folder_name: str = 'Results'):
    from pandas import read_csv, concat
    from background_functions import try_create_folder

    path_local = try_create_folder(folder_name + '/summary/')
    path_v = try_create_folder(folder_name + '/summary/better_pool_values_aggregate/')

    for metric in metrics:
        for ml_metric in ml_metrics:
            data_by_lag = read_csv(
                path_local + '/better_acurracy/' + metric + '_' + ml_metric + '.csv')
            data_by_pool = read_csv(
                path_local + '/better_pool_values/' + metric + '_' + ml_metric + '.csv')

            df = concat([data_by_lag, data_by_pool], axis=1)
            df.drop(['Workload'], axis=1, inplace=True)

            new_columns = df.columns.values
            new_columns[0] = 'Workload'
            df.columns = new_columns

            df.to_csv(path_v + metric + '_' + ml_metric + '.csv', index=False)


def summary_folder(target_variables, metrics, ml_metrics, ml_models, folder_name: str = 'Results'):
    from pandas import read_csv, DataFrame
    from background_functions import try_create_folder

    try_create_folder(folder_name + '/summary')
    path_l = try_create_folder(folder_name + '/' + '/summary/better_lags/')
    path_v = try_create_folder(folder_name + '/' + '/summary/better_acurracy/')

    for metric in metrics:
        for ml_metric in ml_metrics:
            df = DataFrame()
            df_id = DataFrame()
            for workload in target_variables:
                path_local = folder_name + '/' + workload + '/' + metric + '/summary_lag/' + ml_metric + '.csv'
                data = read_csv(path_local)
                df[workload] = data.min(axis=1)
                df_id[workload] = data.iloc[:, 1:].idxmin(axis=1)

            df = df.transpose()
            df_id = df_id.transpose()
            df.columns = ml_models
            df_id.columns = ml_models

            df.to_csv(path_v + metric + '_' + ml_metric + '.csv')
            df_id.to_csv(path_l + metric + '_' + ml_metric + '.csv')


def resume_folder(target_variables, metrics, ml_metrics, ml_models, folder_name: str = 'results'):
    from pandas import read_csv, DataFrame

    for metric in metrics:
        for ml_metric in ml_metrics:
            df = DataFrame()
            df_id = DataFrame()
            for workload in target_variables:
                path_local = folder_name + '/' + workload + '/' + metric + '/' + '/everyone_by_lag/' + ml_metric + '.csv'
                data = read_csv(path_local)
                df[workload] = data.min(axis=1)
                df_id[workload] = data.iloc[:, 1:].idxmin(axis=1)

            df = df.transpose()
            df_id = df_id.transpose()
            df.columns = ml_models
            df_id.columns = ml_models
            df.to_csv('results/aggregate of results/value/' + metric + '_' + ml_metric + '.csv')
            df_id.to_csv('results/aggregate of results/lag/' + metric + '_' + ml_metric + '.csv')


def create_plot(y_true, y_pred, fig_title, y_label, x_label, path_to_save):
    import matplotlib.pyplot as plt

    plt.rcParams['figure.figsize'] = (15, 7)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(fig_title)
    plt.plot(y_true, label='Real')
    plt.plot(y_pred, label='Predicted')
    plt.legend(loc='best')
    plt.savefig(path_to_save + '.png')
    plt.clf()


def calculate_accuracy_metrics_and_save_pickle(parameters: list, type_calculate: str):
    from pickle_functions import save_pickle, load_pickle
    from accuracy_metrics import calculate_model_accuracy

    for accuracy_metric, metric, ml_model, target_variable, window_size in parameters:
        if ml_model[0:3] == 'mlp':
            x = ml_model[0:3]
        elif ml_model[0:4] == 'lstm':
            x = ml_model[0:4]
        elif ml_model[0:2] == 'rf':
            x = ml_model[0:2]
        else:
            x = ml_model
        file_path = ""
        if type_calculate == '/bagging/':
            file_path = target_variable + "/" + x + type_calculate + metric + "/" + metric + ml_model + str(
                window_size)
        else:
            file_path = target_variable + "/" + x + type_calculate + metric + ml_model + str(
                window_size)

        df_pickle = load_pickle(file_path)
        if ml_model[0:4] == 'lstm':
            model = load_model(file_path, ml_model)
        else:
            model = df_pickle['model']

        # for sample in ['training', 'validation', 'testing']:
        for sample in ['training', 'testing']:
            # for sample in ['testing']:
            y_true = df_pickle[sample + '_sample'][:, -1]
            y_pred = predict_data(df_pickle[sample + '_sample'][:, 0:-1], model, ml_model)
            df_pickle[accuracy_metric + '_' + sample] = calculate_model_accuracy(y_true, y_pred, accuracy_metric)

            df_pickle['y_true_' + sample] = y_true
            df_pickle['y_pred_' + sample] = y_pred

        save_pickle(df_pickle, file_path)


def save_accuracy_metrics(parameters: list, folder_name: str = 'Results'):
    from pandas import DataFrame
    from background_functions import try_create_folder_aggregate
    from pickle_functions import load_pickle
    from numpy import array

    for accuracy_metric, metric, ml_model, target_variable, window_size in parameters:
        file_path = target_variable + "/" + ml_model + "/monolithic/" + metric + ml_model + str(
            window_size)
        df_pickle = load_pickle(file_path)
        path_folder = try_create_folder_aggregate(metric, target_variable, window_size, folder_name)

        if accuracy_metric + '_validation' in df_pickle.keys():
            ac_training = df_pickle[accuracy_metric + '_training']
            ac_validation = df_pickle[accuracy_metric + '_validation']
            ac_testing = df_pickle[accuracy_metric + '_testing']

            accuracy_metric_values = [ac_training, ac_validation, ac_testing]

            DataFrame(array(accuracy_metric_values).reshape(1, 3), columns=['training', 'validation', 'testing'],
                      index=[accuracy_metric]).to_csv(path_folder + '/' + ml_model + '_' + accuracy_metric + '.csv')
        else:  # Only training and testing
            ac_training = df_pickle[accuracy_metric + '_training']
            ac_testing = df_pickle[accuracy_metric + '_testing']

            accuracy_metric_values = [ac_training, ac_testing]
            DataFrame(array(accuracy_metric_values).reshape(1, 2), columns=['training', 'testing'],
                      index=[accuracy_metric]).to_csv(path_folder + '/' + ml_model + '_' + accuracy_metric + '.csv')


def save_accuracy_metrics_dynamic(parameters: list, folder_name: str = 'results'):
    from pandas import DataFrame
    from background_functions import try_create_folder_aggregate
    from pickle_functions import load_pickle
    from numpy import array

    for accuracy_metric, metric, ml_model, target_variable, window_size in parameters:
        df_pickle = load_pickle(target_variable + metric + ml_model + str(window_size))
        path_folder = try_create_folder_aggregate(metric, target_variable, window_size, folder_name)

        ac_testing = df_pickle[accuracy_metric + '_testing']

        accuracy_metric_values = [ac_testing]
        DataFrame(array(accuracy_metric_values), columns=['testing'],
                  index=[accuracy_metric]).to_csv(path_folder + '/' + ml_model + '_' + accuracy_metric + '.csv')


def save_accuracy_metrics_one_model(parameters: list, folder_name: str = 'Results'):
    from pandas import DataFrame
    from background_functions import try_create_folder_aggregate_for_dynamic
    from pickle_functions import load_pickle
    from numpy import array

    for accuracy_metric, metric, ml_model, target_variable in parameters:
        file_path = ""
        if ml_model[-11:] == 'homogeneous':
            file_path = target_variable + "/homogeneous/" + metric + ml_model
        else:
            file_path = target_variable + "/heterogeneous/" + metric + ml_model

        df_pickle = load_pickle(file_path)
        path_folder = try_create_folder_aggregate_for_dynamic(metric, target_variable, 'summary',
                                                              folder_name)

        ac_testing = df_pickle[accuracy_metric + '_testing']

        accuracy_metric_values = [ac_testing]
        DataFrame(array(accuracy_metric_values), columns=['testing'],
                  index=[accuracy_metric]).to_csv(path_folder + '/' + ml_model + '_' + accuracy_metric + '.csv')
        # print(path_folder + '/' + ml_model + '_' + accuracy_metric + '.csv')


def save_figures(parameters: list, folder_name: str = 'Results'):
    from pickle_functions import load_pickle

    for accuracy_metric, metric, ml_model, target_variable, window_size in parameters:
        file_path = target_variable + "/" + ml_model + "/monolithic/" + metric + ml_model + str(
            window_size)
        df_pickle = load_pickle(file_path)

        if accuracy_metric + '_validation' in df_pickle.keys():
            samples = ['training', 'validation', 'testing']
        else:
            samples = ['training', 'testing']

        for sample in samples:
            path_folder = folder_name + '/' + target_variable + '/' + metric + '/' + str(
                window_size) + '/' + ml_model + '_' + sample

            y_true = df_pickle['y_true_' + sample]
            y_pred = df_pickle['y_pred_' + sample]
            create_plot(y_true, y_pred, '' + ml_model.upper() + ' in ' + sample, metric, 'Minutes (m)', path_folder)


def save_figures_dynamic(parameters: list, folder_name: str = 'results'):
    from pickle_functions import load_pickle

    for accuracy_metric, metric, ml_model, target_variable, window_size in parameters:
        df_pickle = load_pickle(target_variable + metric + ml_model + str(window_size))

        for sample in ['testing']:
            path_folder = folder_name + '/' + target_variable + '/' + metric + '/' + '/' + str(
                window_size) + '/' + ml_model + '_' + sample

            y_true = df_pickle['y_true_' + sample]
            y_pred = df_pickle['y_pred_' + sample]
            create_plot(y_true, y_pred, '' + ml_model.upper() + ' in ' + sample, metric, 'Minutes (m)', path_folder)


def save_figures_dynamic_one_model(parameters: list, folder_name: str = 'Results'):
    from pickle_functions import load_pickle

    for accuracy_metric, metric, ml_model, target_variable in parameters:
        file_path = ""
        if ml_model[-11:] == 'homogeneous':
            file_path = target_variable + "/homogeneous/"  + metric + ml_model
        else:
            file_path = target_variable + "/heterogeneous/" + metric + ml_model

        df_pickle = load_pickle(file_path)

        for sample in ['testing']:
            path_folder = folder_name + '/' + target_variable + '/' + metric + '/summary/' + ml_model + '_' + sample
            path_folder = folder_name + '/' + target_variable + '/' + metric + '/summary/' + ml_model + '_' + sample

            y_true = df_pickle['y_true_' + sample]
            y_pred = df_pickle['y_pred_' + sample]
            create_plot(y_true, y_pred, '' + ml_model.upper() + ' in ' + sample, metric, 'Minutes (m)', path_folder)


def load_pickle_static(acurracy_metrics, id_models, path_id):
    from pickle_functions import load_pickle
    from itertools import product

    parameters = list(product(acurracy_metrics, id_models))
    dataset = {}

    for accuracy_metric, id_model in parameters:
        df_pickle = load_pickle(path_id + id_model)

        if id_model[0: 4] == 'lstm':
            print(df_pickle['model'])
            model = load_model(df_pickle['model'], id_model)
        else:
            model = df_pickle['model']

        for sample in ['training', 'validation', 'testing']:
            if df_pickle[sample + '_sample'] != []:
                y_true = df_pickle[sample + '_sample'][:, -1]
                y_pred = predict_data(df_pickle[sample + '_sample'][:, 0:-1], model, id_model)

                dataset[path_id + id_model + sample] = [y_pred, y_true]

    return dataset


def calculate_measures_of_central_tendency(data, central_metric):
    from statistics import fmean, median

    if central_metric == 'mean':
        return fmean(data)

    if central_metric == 'median':
        return median(data)


def forecast_by_a_given_serie(path_id, sample, dataset, ml_models, cm):
    len_sample = len(dataset[path_id + ml_models[0] + sample][0])
    y_true = dataset[path_id + ml_models[0] + sample][1]

    y_pred = []
    for index in range(0, len_sample):
        data = []
        for nm in ml_models:
            data.append(
                dataset[path_id + nm + sample][0][index])

        y_pred.append(calculate_measures_of_central_tendency(data, cm))

    return y_true, y_pred


def static_combination(acurracy_metrics, central_measures, dataset, id_models, name_pickle, path_id):
    from pickle_functions import save_pickle
    from accuracy_metrics import calculate_model_accuracy

    for cm in central_measures:
        df_pickle = {}
        for sample in ['testing']:
            y_true, y_pred = forecast_by_a_given_serie(path_id, sample, dataset, id_models, cm)
            df_pickle['y_true_' + sample] = y_true
            df_pickle['y_pred_' + sample] = y_pred

            for accuracy_metric in acurracy_metrics:
                df_pickle[accuracy_metric + '_' + sample] = calculate_model_accuracy(y_true, y_pred, accuracy_metric)
                print(df_pickle[accuracy_metric + '_' + sample])

        path_split = path_id.split('/', 4)
        save_pickle(df_pickle, path_split[0] + "/" +name_pickle + "/" + path_split[3] + 'static_' + cm + '_' + name_pickle)
        #save_pickle(df_pickle, path_id + 'static_' + cm + '_' + name_pickle)



def prev_dynamic(model, nm, indices_cr, x_cr):
    if nm == 'arima':
        y_pred = predict_data(indices_cr, model, nm, 'in_sample')
    else:
        y_pred = predict_data(x_cr, model, nm)

    return y_pred


def calculate_the_distance_vector_new(competence_region):
    vector_weight = []

    bottom = 0
    for weight_p_competence_region_j in competence_region:
        bottom += (1 / weight_p_competence_region_j)

    for weight_p_competence_region_k in competence_region:
        vector_weight.append((1 / weight_p_competence_region_k) / bottom)

    return vector_weight


def calculate_the_upper_alfa(vector_predict, window_sizes, ml_models):
    for window_size in window_sizes:
        for ml_model in ml_models:
            d = []

            # Dk x sqe(k,i)
            for elemA, elemB in zip(vector_predict['vw_' + str(window_size)],
                                    vector_predict[ml_model + str(window_size)]):
                d.append(elemA * elemB)

            # 1 / Somatório  Dk x sqe(k,i)
            vector_predict[ml_model + str(window_size)] = 1 / sum(d)

    return vector_predict


def calculate_the_down_alfa(vector_predict, window_sizes, ml_models):
    bottom = 0

    for window_size in window_sizes:
        for ml_model in ml_models:
            bottom += vector_predict[ml_model + str(window_size)]

    for window_size in window_sizes:
        for ml_model in ml_models:
            vector_predict[ml_model + str(window_size)] = vector_predict[ml_model + str(window_size)] / bottom
            # print(ml_model + str(window_size), vector_predict[ml_model + str(window_size)] / bottom)

    return vector_predict


def calculate_the_upper_alfa_dws(vector_predict, ws_mls):
    for ws_ml in ws_mls:
        d = []

        # Dk x sqe(k,i)
        for elemA, elemB in zip(vector_predict['vw_' + str(ws_ml[-2:])],
                                vector_predict[ws_ml]):
            d.append(elemA * elemB)

        # 1 / Somatório  Dk x sqe(k,i)
        vector_predict[ws_ml] = 1 / sum(d)

    return vector_predict


def calculate_the_down_alfa_dws(vector_predict, ws_mls):
    bottom = 0

    for ws_ml in ws_mls:
        bottom += vector_predict[ws_ml]

    for ws_ml in ws_mls:
        vector_predict[ws_ml] = vector_predict[ws_ml] / bottom
        # print(ml_model + str(window_size), vector_predict[ml_model + str(window_size)] / bottom)

    return vector_predict

# Criar as chaves

def generate_idmodels(bagging: int, metric: list, model_names: str, type: str, window_sizes: int) -> list:
    """

    :param model_names:
    :param type:
    :type window_sizes: object
    """
    id_models = []

    if type == 'homogeneous':
        for i in range(0, bagging):
            id_models.append(model_names + "bagging" + str(i) + str(window_sizes))
    elif type == 'heteregoneous':
        for j in range(0, len(model_names)):
            id_models.append(model_names[j] + "/monolithic/" + metric[0] + model_names[j] + str(window_sizes[j]))

    return id_models


def generate_path_id(metric: list, workload: list, models: list, type: str):

    if type == 'homogeneous':
        path_id = []
        for wi in range(0, len(workload)):
            path_id.append(workload[wi] + "/" + models[wi] + "/bagging/" + metric[0] + "/" + metric[0])

        return path_id
    else:
        path_id = []
        for wi in range(0, len(workload)):
            path_id.append(workload[wi] + "/")

        #for m in models[wi]:
                #path_id.append(workload[wi] + "/" + str(m) + "/monolithic/" + metric[0])

        return path_id
