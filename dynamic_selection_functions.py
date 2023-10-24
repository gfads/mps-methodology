def load_pickle_dynamic(id_models, path_id):
    from pickle_functions import load_pickle
    from generate_results_fuctions import load_model

    dataset = {}
    for id_model in id_models:
        df_pickle = load_pickle(path_id + id_model)

        print(path_id + id_model)
        if id_model[0:4] == 'lstm':
            model = load_model(df_pickle['model'], id_model)
        else:
            model = df_pickle['model']

        dataset[path_id + id_model] = dict(model=model, training_sample=df_pickle['training_sample'],
                                           validation_sample=df_pickle['validation_sample'],
                                           window_size=df_pickle['window_size'], lag=df_pickle['lag'],
                                           testing_sample=df_pickle['testing_sample'],
                                           total_sample=df_pickle['total_sample'])

    return dataset


def get_maximum_ws(id_models):
    m_ws = 0
    m_id = ''
    for id_model in id_models:
        if m_ws < int(id_model[-2:]):
            m_ws = int(id_model[-2:])
            m_id = id_model

    return m_id, m_id[-2:]


def combine_training_and_validation_data(training_sample, validation_sample):
    from numpy import concatenate

    return concatenate((training_sample, validation_sample))


def pre_process_dynamic_selection_data(dataset, key):
    training_sample = dataset[key]['training_sample']  # Valor real
    validation_sample = dataset[key]['validation_sample']  # Valor real

    if validation_sample != []:
        training_sample = combine_training_and_validation_data(training_sample, validation_sample)

    testing_sample = dataset[key]['testing_sample']  # Valor real

    return training_sample, testing_sample


def process_data_dynamic(dataset, path_id, max_lag_model, max_ws):
    d_rs = {}
    test_size = len(dataset[path_id + max_lag_model]['testing_sample'])  # Valor real
    train, test = pre_process_dynamic_selection_data(dataset, path_id + max_lag_model)
    d_rs[max_ws + 'train'] = train
    d_rs[max_ws + 'test'] = test
    d_rs['test_size'] = test_size

    return d_rs, test_size


def calculate_the_distance_between_the_windows(training_sample, testing_sample):
    from scipy.spatial.distance import euclidean

    competence_region = []

    for i_training in range(0, len(training_sample)):

        d = euclidean(testing_sample, training_sample[i_training, 0:-1])
        competence_region.append(d)

    return competence_region


def collect_the_competence_region(training_sample, competence_region, len_competence_region):
    indices_patterns = range(0, len(training_sample))
    competence_region, indices_patterns = zip(
        *sorted(zip(competence_region, indices_patterns)))

    indices_patterns_l = list(indices_patterns)
    k_patterns_x = training_sample[:, 0:-1][indices_patterns_l[0:len_competence_region]]
    k_patterns_y = training_sample[:, -1][indices_patterns_l[0:len_competence_region]]

    return k_patterns_x, k_patterns_y, indices_patterns_l[0:len_competence_region]


def region_competence_pipeline(crs, d_rs, max_ws, test_size):
    for i_test in range(0, test_size):
        cr = calculate_the_distance_between_the_windows(d_rs[max_ws + 'train'], d_rs[max_ws + 'test'][i_test, 0:-1])
        x_cr, y_cr, indices_cr = collect_the_competence_region(d_rs[max_ws + 'train'], cr, crs)

        d_rs[str(i_test) + max_ws + 'x_cr'] = x_cr
        d_rs[str(i_test) + max_ws + 'y_cr'] = y_cr
        d_rs[str(i_test) + max_ws + 'indices_cr'] = indices_cr
        d_rs[str(i_test) + max_ws + 'dk'] = calculate_the_distance_vector(cr, crs)


def calculate_the_distance_vector(competence_region, len_competence_region):
    vector_weight = []

    bottom = 0
    for weight_p_competence_region_j in competence_region[0:len_competence_region]:
        bottom += (1 / weight_p_competence_region_j)

    for weight_p_competence_region_k in competence_region[0:len_competence_region]:
        vector_weight.append((1 / weight_p_competence_region_k) / bottom)

    return vector_weight


def process_for_dealing_with_heterogeneous_lags(dataset, d_rs, id_models, path_id, max_ws):
    from numpy import array

    for id_model in id_models:
        d_rs[id_model + 'model'] = dataset[path_id + id_model]['model']
        lags = dataset[path_id + id_model]['lag']

        if lags[-1] != int(max_ws):
            lags = ((int(max_ws) - lags[-1]) + array(dataset[path_id + id_model]['lag']))
            lags = lags.tolist()

        d_rs[id_model + 'lag'] = lags


def predict_for_dynamic(model, name_model: str, window_for_arima: int, window_for_others: list, arima_type: str):
    from generate_results_fuctions import predict_data

    if name_model == 'arima':
        y_pred = predict_data(window_for_arima, model, name_model, arima_type)
    else:
        y_pred = predict_data(window_for_others, model, name_model)

    return y_pred


def sqe_new(y_pred, target_y: list):
    vector_aux = []
    for index in range(0, len(y_pred)):
        vector_aux.append(((target_y[index] - y_pred[index]) ** 2))  # SQE

    return vector_aux


def forecast_of_technical_models(accuracy_metric, d_rs, id_models, test_size, max_ws):
    from accuracy_metrics import calculate_model_accuracy

    for id_model in id_models:
        for i_test in range(0, test_size):
            model = d_rs[id_model + 'model']
            lags = d_rs[id_model + 'lag']

            y_pred = predict_for_dynamic(model, id_model, d_rs[str(i_test) + max_ws + 'indices_cr'],
                                         d_rs[str(i_test) + max_ws + 'x_cr'][:, lags[: -1]], 'in_sample')

            d_rs[str(i_test) + id_model + 'y_pred'] = y_pred
            d_rs[str(i_test) + id_model + 'sqe'] = sqe_new(y_pred, d_rs[str(i_test) + max_ws + 'y_cr'])
            d_rs[str(i_test) + id_model + accuracy_metric] = calculate_model_accuracy(y_pred,
                                                                                      d_rs[
                                                                                          str(i_test) + max_ws + 'y_cr'],
                                                                                      accuracy_metric)


def dynamic_selection_algorithms(path_id, id_models, dataset, accuracy_metric, crs):
    # Receber id do modelo que obteve o maior lag
    m_id, m_ws = get_maximum_ws(id_models)
    d_rs, test_size = process_data_dynamic(dataset, path_id, m_id, m_ws)
    region_competence_pipeline(crs, d_rs, m_ws, test_size)
    process_for_dealing_with_heterogeneous_lags(dataset, d_rs, id_models, path_id, m_ws)
    forecast_of_technical_models(accuracy_metric, d_rs, id_models, test_size, m_ws)

    return d_rs, m_id, m_ws, test_size


def dynamic_selection(accuracy_metric: str, d_rs: dict, id_models: list, max_ws: str, path_id: str,
                      test_size: int, metric, deployment, approach):
    from accuracy_metrics import calculate_model_accuracy
    from pickle_functions import save_the_pre_defined_pickle
    from numpy import Inf
    from generate_results_fuctions import predict_data

    predl = []
    targetl = []
    namel = []
    for i_test in range(0, test_size):
        better_result = Inf
        name_model = ''
        select_model = ''
        window = []
        target = []

        for id_model in id_models:
            model = d_rs[id_model + 'model']
            lags = d_rs[id_model + 'lag']
            result = calculate_model_accuracy(d_rs[str(i_test) + max_ws + 'y_cr'],
                                              d_rs[str(i_test) + id_model + 'y_pred'], accuracy_metric)

            if result < better_result:
                better_result = result
                name_model = id_model
                select_model = model

                window = d_rs[max_ws + 'test'][i_test, lags[0:-1]].reshape(1, -1)

                #if name_model[0: 4] == 'lstm':
                #select_model = 'pickle/' + path_id + id_model + '.h5'

                if name_model[0: 5] == 'arima':
                    window = i_test + 1

                target = d_rs[max_ws + 'test'][i_test, -1]

        predl.append(predict_data(window, select_model, name_model, 'out_sample')[0])
        targetl.append(target)
        namel.append([name_model])

    path_split = path_id.split('/', 4)

    save_the_pre_defined_pickle(predl, targetl, namel, accuracy_metric,
                                path_split[0] + "/" + approach + '/' + metric + '/' + deployment + 'dynamic_selection')


def dynamic_weighting(path_id, id_models, dataset, accuracy_metric, d_rs, max_id_model, max_ws, test_size, metric,
                      deployment, approach):
    from pickle_functions import save_the_pre_defined_pickle

    for i_test in range(0, test_size):
        for nm in id_models:
            sqe = d_rs[str(i_test) + nm + 'sqe']
            dk = d_rs[str(i_test) + max_ws + 'dk']

            # Dk x sqe(k,i)
            alpha_upper = []
            for elemA, elemB in zip(sqe, dk):
                alpha_upper.append(elemA * elemB)

            # 1 / Somatório  Dk x sqe(k,i)
            d_rs[str(i_test) + nm + 'alpha_upper'] = 1 / sum(alpha_upper)

    for i_test in range(0, test_size):
        alpha_down = 0

        for nm in id_models:
            alpha_down += d_rs[str(i_test) + nm + 'alpha_upper']

        for nm in id_models:
            d_rs[str(i_test) + nm + 'alpha'] = d_rs[str(i_test) + nm + 'alpha_upper'] / alpha_down

    # print('Conclusão DW')
    predl = []
    for i_test in range(0, test_size):
        pred = 0

        for nm in id_models:
            lags = dataset[path_id + nm]['lag']
            window = d_rs[max_ws + 'test'][i_test, lags[0:-1]].reshape(1, -1)
            y_pred = predict_for_dynamic(d_rs[nm + 'model'], nm, i_test + 1, window, 'out_sample')
            d_rs[str(i_test) + nm + 'y_pred_test'] = y_pred
            pred += d_rs[str(i_test) + nm + 'alpha'] * y_pred

        predl.append(sum(pred))

    testing_sample = dataset[path_id + max_id_model]['testing_sample']

    path_split = path_id.split('/', 4)

    save_the_pre_defined_pickle(testing_sample[0:test_size, -1], predl, "", accuracy_metric,
                                path_split[0] + "/" + approach + '/' + metric + '/' + deployment + 'dynamic_weighting')


def dynamic_weighting_selection(path_id, id_models, dataset, accuracy_metric, d_rs, max_id_model, max_ws,
                                test_size, metric, deployment, approach):
    from pickle_functions import save_the_pre_defined_pickle

    for i_test in range(0, test_size):
        am_error = []
        for id_model in id_models:
            am_error.append(d_rs[str(i_test) + id_model + accuracy_metric])

        d_rs[str(i_test) + 'error_value'] = (max(am_error) - min(am_error)) / 2

        new_id_models = []
        for id_model in id_models:
            if d_rs[str(i_test) + id_model + accuracy_metric] <= d_rs[str(i_test) + 'error_value']:
                new_id_models.append(id_model)

        if not new_id_models:
            for id_model in id_models:
                new_id_models.append(id_model)

        d_rs[str(i_test) + 'new_id_models'] = new_id_models

    for i_test in range(0, test_size):
        for nidmodel in d_rs[str(i_test) + 'new_id_models']:
            sqe = d_rs[str(i_test) + nidmodel + 'sqe']
            dk = d_rs[str(i_test) + max_ws + 'dk']

            # Dk x sqe(k,i)
            alpha_upper = []
            for elemA, elemB in zip(sqe, dk):
                alpha_upper.append(elemA * elemB)

            # 1 / Somatório  Dk x sqe(k,i)
            d_rs[str(i_test) + nidmodel + 'alpha_upper_dws'] = 1 / sum(alpha_upper)

    for i_test in range(0, test_size):
        alpha_down = 0

        for nidmodel in d_rs[str(i_test) + 'new_id_models']:
            alpha_down += d_rs[str(i_test) + nidmodel + 'alpha_upper_dws']

        for nidmodel in d_rs[str(i_test) + 'new_id_models']:
            d_rs[str(i_test) + nidmodel + 'alpha_dws'] = d_rs[str(i_test) + nidmodel + 'alpha_upper_dws'] / alpha_down

        # print('Conclusão DWS')
    predl = []
    for i_test in range(0, test_size):
        pred = 0
        for nidmodel in d_rs[str(i_test) + 'new_id_models']:
            lags = dataset[path_id + nidmodel]['lag']
            window = d_rs[max_ws + 'test'][i_test, lags[0:-1]].reshape(1, -1)
            y_pred = predict_for_dynamic(d_rs[nidmodel + 'model'], nidmodel, i_test + 1, window, 'out_sample')
            d_rs[str(i_test) + nidmodel + 'y_pr ed_test'] = y_pred
            pred += d_rs[str(i_test) + nidmodel + 'alpha_dws'] * y_pred

        predl.append(sum(pred))

    testing_sample = dataset[path_id + max_id_model]['testing_sample']
    path_split = path_id.split('/', 4)

    save_the_pre_defined_pickle(testing_sample[0:test_size, -1], predl, "", accuracy_metric,
                                path_split[
                                    0] + "/" + approach + '/' + metric + '/' + deployment + 'dynamic_weighting_with_selection')
