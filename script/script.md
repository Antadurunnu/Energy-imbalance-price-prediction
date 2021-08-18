# script for training and prediction

main file: `rebap.py`

files imported from directory rebap_util: 
- rebap_functions.py defining functions
    - `align` for conveniently combining time series of different periods and time steps
    - `cast` for casting data frames with a datetime column into time series
    - `label_target` for sorting target into classes defined by split_values
    - `remove_outliers` for removing data points with outlier target
    - `load_target`, `load_wind_speed` etc: wrapper functions for loading target and features from certain files in a data directory. names of these wrapper functions and specific feature delays are stored in the dictionary feature_catalogue defined in `rebap.py`under the feature names as keys. 
- rebap.params.py defining
    - `feature_list` containing those keys of `feature_catalogue` to be included in model
    - `prediction_time_shift` setting how much time in advance the prediction is made
    - `backlog` setting how many former data points are used in model
    - `train_from` and `train_until` defining the training period
    - `predict_from` and `predict_until` defining the prediction period
    - `times_std` defining the interval around the (training) target mean outside of which outliers get removed ito multiples of standard deviation; set to None for keeping everything
    - `split_values` defining the list of splitting values for classification, set to None for regression
    - definition of feature scaler
    - definition of dimension reducer (PCA or t-SNE)
    - definition of classifier (a random forest classifier by now; not optimized in no way)
    - definition of regressor (a random forest regressor without optimization as well)

usage: python rebap.py with options -t for training and -p for prediction

classifier/regressor, scaler, and reducer fitted in training mode are written into/loaded from directory `rebap_model`

targets predicted in training and training scores are written into directory `rebap_training`

predicted targets and prediction scores are written into directory `rebap_prediction`

