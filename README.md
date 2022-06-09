# MPS-Methodology
 
 
 ## Project description
 
This methodology is a project that proposes applying the Multiple Preditors System (MPS) to forecasts time series extracted from Microservice-Based Applications (MBAs). MPS are composed of a set of forecasting models, which, in turn, mitigates the uncertainties inherent in choosing a single forecasting model. Thus, MPS adoption improves the behaviour’s forecast of system objectives, contributing to developing adaptive proactive systems more reliable and robust. Also, the methodology operates with pools homogeneous (all models are trained using a specific learning algorithm) and heterogeneous (trained using different learning algorithms). 

The monolithic models adopted to train are AutoRegressive Integrated Moving Average (ARIMA), Long Short-Term Memory (LSTM), Multilayer Perceptron (MLP), Support Vector Regressor (SVR), Random Forest (RF), and eXtreme Gradient Boosting (XGBoost). For the selection module, three dynamic selection algorithms were chosen: DS, DW, DWS. Also, two static approaches Mean and Median, were adopted. 

## Parameters models

The parameters adopted into models’ training is summarises below:

### Learning algorithm parameters

|                                                        Learning Algorithm                                                       |                                                                                      Library                                                                                      |
|:-------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  [ARIMA](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L408)  |                                      [Pmdarima 1.8.0](http://alkaline-ml.com/pmdarima/1.8.0/modules/generated/pmdarima.arima.auto_arima.html)                                     |
|   [LSTM](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L304)  |                                                    [Keras 2.6.0](https://keras.io/api/layers/recurrent_layers/lstm/#lstm-class)                                                   |
|    [MLP](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L77)   |          [Scikit-learn 0.24.1](https://scikit-learn.org/0.24/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlp#sklearn.neural_network.MLPRegressor)          |
|    [RF](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L189)   | [Scikit-learn 0.24.1](https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random\20forest#sklearn.ensemble.RandomForestRegressor) |
|    [SVR](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L45)   |                              [Scikit-learn 0.24.1](https://scikit-learn.org/0.24/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR)                              |
| [XGBoost](https://github.com/ML-Adapt/mps_methodology/blob/fbc85cb2555e43b3a7af525f6ca1ff20398de993/training_of_models.py#L260) |                                                  [XGBoost 1.4.0](https://xgboost.readthedocs.io/en/release_1.4.0/parameter.html)                                                  |                                                 |


### Lags and Bagging Size

|              |            |                     Approaches                         |||  
|--------------|------------| ----------------| ---------------| ----------------------|
| Time Series              || Best Monolithic | Monolithic Lag | Homogeneous Pool Size |
|              | Decreasing | MLP             | 10             | 150                   |
|   CPU        | Increasing | RF              | 60             | 110                   |
|              | Periodic   | RF              | 10             | 100                   |
|              | Random     | RF              | 10             | 130                   |
|              | Decreasing | MLP             | 20             | 80                    |
| Memory       | Increasing | LSTM            | 60             | 40                    |
|              | Periodic   | LSTM            | 20             | 20                    |
|              | Random     | RF              | 40             | 100                   |
|              | Decreasing | MLP             | 50             | 150                   |
|Response Time | Increasing | RF              | 30             | 30                    |
|              | Periodic   | RF              | 40             | 100                   |
|              | Random     | RF              | 40             | 90                    |
|              | Decreasing | MLP             | 60             | 100                   |
|   Traffic    | Increasing | MLP             | 20             | 100                   |
|              | Periodic   | RF              | 50             | 110                   |
|              | Random     | RF              | 10             | 150                   |

## Models

All models generated during the experimental evaluation saved through the pickle library and are available and organized within the Pickle Models folder.

## Results

The results discussed in the article are available into Results folder.


| Folder                     | Content description                                                              |
|----------------------------|----------------------------------------------------------------------------------|
| Increasing                 | It contains the performance of monolithic models in Increasing workload.         |
| Decreasing                 | It contains the performance of monolithic models in Decreasing workload.         |
| Periodic                   | It contains the performance of monolithic models in Periodic workload.           |
| Random                     | It contains the performance of monolithic models in Random workload.             |
| Summary                    | It contains the performance summary of the approach.                             |
| Summary/better_lags        | It contains the best lag for each monolithic.                                    |
| Summary/better_acurracy    | It contains the best acurray values for each monolithic.                         |
| Summary/better_pool_values | MPS performance.                                                                 |
| Summary/better_pool_values_aggregate | It contains aggregated data of better_pool_values and better_acurracy  |
| Summary/pool_size_homogeneous_analisys   | It contains data from the optimal bagging size analysis for each time series.    |


# Instalattion  
  
## How to install MPS-methodology project?

    $ virtualenv venv
    $ source venv/bin/activate
    $ pip3 install -r requirements.txt
    
    
## How to regenerate results using pickle models?
    $ rm Results/ -r
    $ mkdir Results/
    $ python3 generate_monolith_results.py 
    $ python3 generate_mps_results.py 
    
## How to train new models and generate MPS?

|  File                      | File description                                                         |
|----------------------------|--------------------------------------------------------------------------|
| training_models_main.py    | Generation of monolithic models and homogeneous bagging.                 |
| dynamic_selection_main.py  | Dynamic MPS training using DS, DW and DWS                                |
| static_combination.py      | Training of static MPS using Mean and Median.                           |




