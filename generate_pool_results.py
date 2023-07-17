from generate_results_fuctions import *
from itertools import product
import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument("--workloads", nargs="*", type=str)
CLI.add_argument("--learning_algorithms", nargs="*", type=str)
CLI.add_argument("--deployment", nargs="*", type=str)
CLI.add_argument("--metric", nargs="*", type=str)
CLI.add_argument("--lags", nargs="*", type=int)
CLI.add_argument("--competence_measure", nargs="*", type=str)
args = CLI.parse_args()

METRIC = args.metric
MODEL_NAMES = args.learning_algorithms
WINDOW_SIZES = args.lags
DEPLOYMENTS = args.deployment
WORKLOAD = args.workloads
ACCURACY_METRICS = args.competence_measure

# All models by serie
ML_MODELS = ['static_mean_homogeneous', 'static_median_homogeneous', 'dynamic_selection_homogeneous',
             'dynamic_weighting_homogeneous', 'dynamic_weighting_with_selection_homogeneous',
             'static_mean_heteregoneous', 'static_median_heterogeneous', 'dynamic_selection_heterogeneous',
             'dynamic_weighting_heterogeneous', 'dynamic_weighting_with_selection_heterogeneous']

parameters = list(product(ACCURACY_METRICS, METRIC, ML_MODELS, WORKLOAD, DEPLOYMENTS))

save_accuracy_metrics_one_model(parameters)
save_figures_dynamic_one_model(parameters)
concatenate_results_with_one_result_dynamic(WORKLOAD, METRIC, ML_MODELS, ACCURACY_METRICS, DEPLOYMENTS)
everyone_folder_with_one_result_dynamic(WORKLOAD, METRIC, ACCURACY_METRICS, DEPLOYMENTS)
everyone_folder_with_one_result_dynamic_all(METRIC, ACCURACY_METRICS, DEPLOYMENTS)
