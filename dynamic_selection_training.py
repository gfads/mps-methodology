import argparse
from generate_results_fuctions import *
from dynamic_selection_functions import load_pickle_dynamic, dynamic_selection_algorithms, dynamic_selection, \
    dynamic_weighting, dynamic_weighting_selection

CLI = argparse.ArgumentParser()
CLI.add_argument("--approach", type=str)
CLI.add_argument("--workload", type=str)
CLI.add_argument("--metric", type=str)
CLI.add_argument("--lag", type=int)
CLI.add_argument("--lags", nargs="*", type=int)
CLI.add_argument("--pool_size", type=int)
CLI.add_argument("--learning_algorithm", type=str)
CLI.add_argument("--learning_algorithms", nargs="*", type=str)
CLI.add_argument("--deployment", type=str)
CLI.add_argument("--competence_measure", type=str)

args = CLI.parse_args()

approach = args.approach
metrics = args.metric
deployment = args.deployment
workload = args.workload
pool_size = args.pool_size
competence_measure = args.competence_measure


if approach == 'homogeneous':
    lag = args.lag
    learning_algorithm = args.learning_algorithm
else:
    lag = args.lags
    learning_algorithm = args.learning_algorithms

path_id = generate_path_id(metrics, workload, deployment, learning_algorithm, approach)

RoC = 10

id_models = generate_idmodels(metrics, pool_size, deployment, learning_algorithm, approach, lag)
dataset = load_pickle_dynamic(id_models, path_id)

d_rs, max_id_model, max_ws, test_size = dynamic_selection_algorithms(path_id, id_models, dataset, competence_measure,
                                                                     RoC)

print('Starting DS algorithm training')
dynamic_selection(competence_measure, d_rs, id_models, max_ws, path_id, test_size, metrics, deployment, approach)

print('Starting DW algorithm training')
dynamic_weighting(path_id, id_models, dataset, competence_measure, d_rs, max_id_model, max_ws, test_size, metrics, deployment, approach)

print('Starting DWS algorithm training')
dynamic_weighting_selection(path_id, id_models, dataset, competence_measure, d_rs, max_id_model, max_ws, test_size, metrics, deployment, approach)
