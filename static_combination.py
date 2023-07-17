import argparse
from generate_results_fuctions import load_pickle_static, static_combination, generate_path_id, generate_idmodels

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


CENTRAL_MEASURES = ['mean', 'median']

path_id = generate_path_id(metrics, workload, deployment, learning_algorithm, approach)

id_models = generate_idmodels(metrics, pool_size, deployment, learning_algorithm, approach, lag)
dataset = load_pickle_static(competence_measure, id_models, path_id)

print('Starting Mean and Median algorithm training')
static_combination(competence_measure, CENTRAL_MEASURES, dataset, id_models, path_id, approach, deployment, metrics)
