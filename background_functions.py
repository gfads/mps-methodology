def try_create_folder(folder_path: str):
    from os import path, mkdir
    if not path.exists(folder_path):
        mkdir(folder_path)

    return folder_path


def try_create_folder_aggregate(deployment, metric, target_variable, window_size, folder_name) -> str:
    folder_root = folder_name + '/' + target_variable
    folder_metric = folder_name + '/' + target_variable + '/' + metric
    folder_deployment = folder_name + '/' + target_variable + '/' + metric + '/' + deployment
    folder_lag = folder_name + '/' + target_variable + '/' + metric + '/' + deployment + '/' + str(window_size)

    try_create_folder(folder_root)
    try_create_folder(folder_metric)
    try_create_folder(folder_deployment)
    try_create_folder(folder_lag)

    return folder_lag

def try_create_folder_aggregate_for_dynamic(deployment, metric, target_variable, window_size, folder_name) -> str:
    folder_root = folder_name + '/' + target_variable
    folder_metric = folder_name + '/' + target_variable + '/' + metric
    folder_deployment = folder_name + '/' + target_variable + '/' + metric + '/' + deployment
    folder_dynamic_model = folder_name + '/' + target_variable + '/' + metric + '/' + deployment + '/' + str(window_size)

    try_create_folder(folder_root)
    try_create_folder(folder_metric)
    try_create_folder(folder_deployment)
    try_create_folder(folder_dynamic_model)

    return folder_dynamic_model
