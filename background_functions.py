def try_create_folder(folder_path: str):
    from os import path, makedirs

    if not path.exists(folder_path):
        makedirs(folder_path)

    return folder_path


def try_create_folder_aggregate(metric, target_variable, window_size, folder_name) -> str:
    from os import makedirs
    from os import path

    folder = folder_name + '/' + target_variable + '/' + metric + '/' + str(window_size)

    if not path.exists(folder):
        makedirs(folder)

    return folder_name + '/' + target_variable + '/' + metric + '/' + str(window_size)


def try_create_folder_aggregate_for_dynamic(metric, target_variable, window_size, folder_name) -> str:
    from os import makedirs
    from os import path

    folder = folder_name + '/' + target_variable + '/' + metric + '/' + str(window_size)

    if not path.exists(folder):
        makedirs(folder)

    return folder
