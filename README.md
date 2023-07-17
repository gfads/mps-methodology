# MPS-Methodology

MPS methodology is a project that proposes applying the Multiple Preditors System (MPS) to forecasts time series
extracted from Microservice-Based Applications (MBAs). In literature, works have applied time series forecasting to
predict performance degradation in MBAs. However, all these studies use a single forecast model, which increases the
risk of inaccurate estimates, which can lead the application to undesirable scenarios such as unavailability. MPS
emerges as an alternative to this problem since it uses multiple models in the forecast. MPS's basic idea of the
ensemble
is to combine the strengths of different learning algorithms to build a more accurate and reliable forecasting system.

More reliable and accurate forecasting systems are essential in proactive microservice auto-scaling systems. They
improve the decision-making process of these systems by more reliably estimating microservices trends while mitigating
incorrect adaptations triggered by inaccurate estimates. Consequently, microservices have a reduction in operating
costs, and their customer satisfaction is maintained.

# Installation

## How to install the software?

    $ virtualenv venv
    $ source venv/bin/activate
    $ pip3 install -r requirements.txt

# What do the folders generated by the results mean?

All processor results were stored in the Results folder.

The description of each folder and its respective content is given below:

| Folder                                                                                   | Content description                                                                                   |
|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| [Increasing](Results/increasing)                                                         | It contains the accuracy of monolithic models in Increasing workload.                                 |
| [Decreasing](Results/decreasing)                                                         | It contains the accuracy of monolithic models in Decreasing workload.                                 |
| [Periodic](Results/periodic)                                                             | It contains the accuracy of monolithic models in Periodic workload.                                   |
| [Random](Results/random)                                                                 | It contains the accuracy of monolithic models in Random workload.                                     |
| [Summary](Results/summary)                                                               | It contains the summary with precision of all approaches (monolithic, homogeneous and heterogeneous). |
| [Summary/better_lags](Results/summary/better_lags)                                       | It contains the best lag for each monolithic model.                                                   |
| [Summary/better_acurracy](Results/summary/better_acurracy)                               | It contains the best accuracy values for each monolithic model.                                       |
| [Summary/better_pool_values](Results/summary/better_pool_values)                         | MPS accuracy.                                                                                         |
| [Summary/better_pool_values_aggregate](Results/summary/better_pool_values_aggregate)     | It contains aggregated data of better_pool_values and better_acurracy                                 |
| [Summary/pool_size_homogeneous_analisys](Results/summary/pool_size_homogeneous_analisys) | It contains data from each time series's optimal bagging size analysis.                               |
| [Multiple Metrics](Results/multiple_metrics)                                             | It contains a summary of results on additional metrics beyond RMSE                                    |


# How to regenerate paper results using saved models?

## Download and extract from the models

Download the models
from [OneDrive](https://cinufpe-my.sharepoint.com/:u:/g/personal/wrms_cin_ufpe_br/ESmYfCk1chtAhx-WrHR4gxcBt-lBNubv-DDFOpIBHC_P1Q?e=H0njRw)
and save them inside the MPS Methodology folder. Models were uploaded externally due to their size.

    $ unzip models.zip

## Regenerate all results

Be patient. This process can take a while. Between 15-30 minutes per performance metric.

    $ rm Results/ -r; mkdir Results/
    $ python3 generate_initial_results.py --competence_measure rmse --deployment frontend --lags 10 20 30 40 50 60 --learning_algorithms arima lstm xgboost svr rf mlp --metrics cpu memory responsetime traffic  --workloads decreasing increasing random periodic
    $ python3 generate_pool_results.py --competence_measure rmse --deployment frontend --lags 10 20 30 40 50 60 --learning_algorithms arima lstm xgboost svr rf mlp --metric cpu memory traffic responsetime --workloads decreasing increasing random periodic

If desired, you can generate the values of a specific metric by changing the input command. For example, to generate
results for just the memory metric, do:

    $ python3 generate_initial_results.py --competence_measure rmse --deployment frontend --lags 10 20 30 40 50 60 --learning_algorithms arima lstm xgboost svr rf mlp --metrics memory  --workloads decreasing increasing random periodic
    $ python3 generate_pool_results.py --competence_measure rmse --deployment frontend --lags 10 20 30 40 50 60 --learning_algorithms arima lstm xgboost svr rf mlp --metric memory --workloads decreasing increasing random periodic

# Summary of other information presented throughout the paper.

## Learning algorithm parameters

|                                                       Learning Algorithm                                                        |                                                                                       Library                                                                                       |
|:-------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  [ARIMA](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L408)  |                                      [Pmdarima 1.8.0](http://alkaline-ml.com/pmdarima/1.8.0/modules/generated/pmdarima.arima.auto_arima.html)                                       |
|  [LSTM](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L304)   |                                                    [Keras 2.6.0](https://keras.io/api/layers/recurrent_layers/lstm/#lstm-class)                                                     |
|   [MLP](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L77)    |          [Scikit-learn 0.24.1](https://scikit-learn.org/0.24/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlp#sklearn.neural_network.MLPRegressor)          |
|   [RF](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L189)    | [Scikit-learn 0.24.1](https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random\20forest#sklearn.ensemble.RandomForestRegressor) |
|   [SVR](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L45)    |                              [Scikit-learn 0.24.1](https://scikit-learn.org/0.24/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR)                              |
| [XGBoost](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L260) |                                                   [XGBoost 1.4.0](https://xgboost.readthedocs.io/en/release_1.4.0/parameter.html)                                                   |                                                 |

## Lags and Bagging Size

The results for selecting the homogeneous pool size are
summarised [here](Results/summary/pool_size_homogeneous_analisys). Also, the results of the best monolithic model by
dataset and its lag size are available [here](Results/summary/better_acurracy) and [here](Results/summary/better_lags),
respectively.

The table below summarizes both results.

<table style="border-collapse:collapse;border-spacing:0" class="tg"><thead><tr><th style="border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal" colspan="2">Datasets</th><th style="border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal" colspan="3">Approaches</th></tr></thead><tbody><tr><td style="border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Time Series</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Workload</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Best Monolithic</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Monolithic Lag</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Homogeneous Pool Size</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:middle;word-break:normal" rowspan="4">CPU<br><br></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Decreasing</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">MLP</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">10</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">150</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Increasing</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">RF</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">60</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">110</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Periodic</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">RF</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">10</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">100</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Random</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">RF</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">10</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">130</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:middle;word-break:normal" rowspan="4">Memory<br><br></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Decreasing</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">MLP</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">20</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">80</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Increasing</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">LSTM</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">60</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">40</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Periodic</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">LSTM</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">20</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">20</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Random</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">RF</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">40</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">100</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:middle;word-break:normal" rowspan="4">Response Time<br><br></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Decreasing</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">MLP</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">50</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">150</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Increasing</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">RF</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">30</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">30</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Periodic</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">RF</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">40</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">100</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Random</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">RF</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">40</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">90</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:middle;word-break:normal" rowspan="4">Traffic<br><br></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Decreasing</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">MLP</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">60</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">100</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Increasing</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">MLP</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">20</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">100</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Periodic</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">RF</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">50</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">110</td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Random</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">RF</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">10</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">150</td></tr></tbody></table>

## Time Series

The time series used in the research can be found [here](Time%20Series). Also, we also [plot](Time%20Series/plots)
all-time series and create a [description file](Time%20Series/series-description/serie-description.csv).