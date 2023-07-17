import argparse
from generate_results_fuctions import *
from itertools import product


CLI = argparse.ArgumentParser()
CLI.add_argument("--workloads", nargs="*", type=str)
CLI.add_argument("--learning_algorithms", nargs="*", type=str)
CLI.add_argument("--deployment", nargs="*", type=str)
CLI.add_argument("--metrics", nargs="*", type=str)
CLI.add_argument("--lags", nargs="*", type=int)
CLI.add_argument("--competence_measure", nargs="*", type=str)
args = CLI.parse_args()

METRICS = args.metrics
MODEL_NAMES = args.learning_algorithms
WINDOW_SIZES = args.lags
DEPLOYMENTS = args.deployment
WORKLOAD = args.workloads
ACCURACY_METRICS = args.competence_measure

parameters = list(product(ACCURACY_METRICS, DEPLOYMENTS, METRICS, MODEL_NAMES, WORKLOAD, WINDOW_SIZES))
calculate_accuracy_metrics_and_save_pickle(parameters, "/monolithic/")
save_accuracy_metrics(parameters)
save_figures(parameters)
concatenate_results(WORKLOAD, METRICS, MODEL_NAMES, WINDOW_SIZES, ACCURACY_METRICS, DEPLOYMENTS)
everyone_folder_by_lag(WORKLOAD, METRICS, WINDOW_SIZES, ACCURACY_METRICS, DEPLOYMENTS)
summary_folder(WORKLOAD, METRICS, ACCURACY_METRICS, MODEL_NAMES, DEPLOYMENTS)