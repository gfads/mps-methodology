def df_pickle_bagging(model, model_name: str, window_size: int, h_step: int, testing: list, total: list, lags: int,
                      scaler, level_grid: str, file_path: str):
    df_pickle = {'model': model['model'], 'window_size': window_size, 'h_step': h_step,
                 'training_sample': model['training_sample'], 'validation_sample': model['validation_sample'],
                 'testing_sample': testing, 'total_sample': total, 'lag': lags, 'scaler': scaler,
                 'level_grid': level_grid, 'indices': model['indices'], 'number_of_models': len(model['model'])}

    if model_name == 'lstm':
        for i in range(0, df_pickle['number_of_models']):
            path_aux = file_path[:-2] + level_grid + str(i) + str(window_size)
            df_pickle['model'][i].save(filepath='Pickle Models/' + path_aux + '.h5', overwrite=True)
            df_pickle['model'][i] = 'Pickle Models/' + path_aux + '.h5'

    return df_pickle


def df_pickle_only(model, model_name, file_path: str, window_size: int, h_step: int, training: list,
                   validation: list, testing: list, total: list, lags: int, scaler, level_grid: str):
    df_pickle = {'model': model, 'window_size': window_size, 'h_step': h_step, 'training_sample': training,
                 'validation_sample': validation, 'testing_sample': testing, 'total_sample': total, 'lag': lags,
                 'scaler': scaler, 'level_grid': level_grid, 'number_of_models': 1}

    if model_name == 'lstm':
        df_pickle['model'] = 'Pickle Models/' + file_path + '.h5'
        model.save(filepath='Pickle Models/' + file_path + '.h5', overwrite=True)

    return df_pickle


def save_model(model, model_name: str, ws: int, h_step: int, training: list, testing: list,
               total: list, lags: int, scaler, level_grid: str, file_path: str, validation: list = []):
    if level_grid == 'default' or level_grid == 'hard':
        save_pickle(df_pickle_only(model, model_name, file_path, ws, h_step, training, validation, testing, total, lags,
                                   scaler, level_grid), file_path)

    if level_grid == 'bagging':
        df = df_pickle_bagging(model, model_name, ws, h_step, testing, total, lags, scaler, level_grid, file_path)

        for i in range(0, df['number_of_models']):
            df['model'] = model['model'][i]
            df['training_sample'] = model['training_sample'][i]
            df['validation_sample'] = model['validation_sample'][i]
            df['indices'] = model['indices'][i]

            save_pickle(df, file_path[:-2] + level_grid + str(i) + str(ws))


def save_the_pre_defined_pickle(predl, targetl, list_y_models_sequence, accuracy_metric, filename_pickle):
    from accuracy_metrics import calculate_model_accuracy
    from pickle_functions import save_pickle
    from numpy import array

    # Resolver probblema
    df_pickle = {'y_true_testing': targetl, 'y_pred_testing': predl,
                 'y_model_testing_sequence': list_y_models_sequence,
                 accuracy_metric + '_testing': calculate_model_accuracy(targetl, predl, accuracy_metric)}

    save_pickle(df_pickle, filename_pickle)


def ml_models_bagging(model_names, tam):
    model_name_aux = []
    for mn in model_names:
        for b in range(0, tam):
            model_name_aux.append(mn + str(b))

    return model_name_aux


def save_pickle(df_pickle, filename_pickle: str):
    from pickle import dump
    from os import path, makedirs

    if not path.isdir('Pickle Models/' + filename_pickle[0:filename_pickle.rfind('/') + 1]):
        makedirs('Pickle Models/' + filename_pickle[0:filename_pickle.rfind('/') + 1])

    dump(df_pickle, open('Pickle Models/' + filename_pickle + '.sav', 'wb'))


def load_pickle(filename_pickle: str):
    from pickle import load

    return load(open('Pickle Models/' + filename_pickle + '.sav', 'rb'))
