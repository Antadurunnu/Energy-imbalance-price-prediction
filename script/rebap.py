import pandas as pd
import numpy as np

import pickle
import csv
import sys, getopt



## START

try:
    opts, args = getopt.getopt(sys.argv[1:], 'htpa')
except:
    print('Usage: python rebap.py -t (for training), -p (for prediction), -h (for help)')
    sys.exit()

train, predict = 0, 0
if '-t' in [opt[0] for opt in opts]:
    train = 1
if '-p' in [opt[0] for opt in opts]:
    predict = 1
if '-h' in [opt[0] for opt in opts]:
    print('Usage: python rebap.py -t (for training), -p (for prediction), -h (for help)')
    sys.exit()



## PREPARATION

# import functions
from rebap_util.rebap_functions import *

# import hyperparameters, classifier etc
from rebap_util.rebap_params import *

feature_catalogue = {'wind_speed':{'loader':load_wind_speed, 'delay':8},
                   'imbalance_power':{'loader':load_imbalance_power, 'delay':4},
                   'epex_da':{'loader':load_epex_da, 'delay':0},
                   'epex_da_fc':{'loader':load_epex_da_fc, 'delay':0},
                   'power_ac':{'loader':load_power_ac, 'delay':0},
                   'power_fc':{'loader':load_power_fc, 'delay':0},
                   'wind_onshore_fc':{'loader':load_wind_onshore_fc, 'delay':0},
                   'wind_offshore_fc':{'loader':load_wind_offshore_fc, 'delay':0},
                   'solar_fc':{'loader':load_solar_fc, 'delay':0},
                   'renewables_fc':{'loader':load_renewables_fc, 'delay':0}, 
                   'total_fc':{'loader':load_total_fc, 'delay':0},
                   'imbalance_at':{'loader':load_imbalance_at, 'delay':9},
                   'imbalance_be':{'loader':load_imbalance_be, 'delay':8},
                   'imbalance_dk':{'loader':load_imbalance_dk, 'delay':16},
                   'imbalance_pl':{'loader':load_imbalance_pl, 'delay':9},
                   'consumption_ac':{'loader':load_consumption_ac, 'delay':8},
                   'consumption_fc':{'loader':load_consumption_fc, 'delay':0},
                   'generation_ac':{'loader':load_generation_ac, 'delay':8},
                   #'generation_fc':{'loader':load_generation_fc, 'delay':0},
                   'mrl_energy':{'loader':load_mrl_energy, 'delay':4},
                   'mrl_capacity':{'loader':load_mrl_capacity, 'delay':0},
                   'prl':{'loader':load_prl, 'delay':0},
                   'srl_energy':{'loader':load_srl_energy, 'delay':4},
                   'srl_capacity':{'loader':load_srl_capacity, 'delay':0}
                  }
features_used = {key:val for key, val in feature_catalogue.items() if key in feature_list[:2]}



## DATA IMPORT

# load target
print('loading target...')
df = load_target()
target = df.columns[0]

# generate time variables
print('generating time variables...')
df['month'] = df.index.month
df['weekday'] = df.index.weekday
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df = pd.get_dummies(df, columns=['month', 'weekday', 'hour', 'minute'], drop_first=True)

# load features and integrate with their specific delay into the time series
print('loading features...')
for feature, params in features_used.items():
    dfi = params['loader']()
    dfi = dfi.shift(periods=params['delay']+prediction_time_shift)
    dfi.dropna(axis=0, inplace=True)
    df = align(df, dfi, prefixes=('',''), method='forward')

# split into features and target
X, y = df.drop(target, axis=1), df[[target]]

## DATA PREPARATION

# remove target outliers
if times_std:
    print('removing target outliers...')
    X, y = remove_outliers(X, y, times_std)

# add backlog 
if backlog:
    print('adding backlog of', backlog, 'time steps')
    X = align(*[X.shift(b) for b in range(backlog+1)])

y = y.iloc[-len(X):]

# extract training or prediction period
if train:
    print('extracting training period from', train_from, 'to', train_until, '...')
    X_t = X[X.index >= pd.to_datetime(train_from)]
    X_t = X_t[X_t.index < pd.to_datetime(train_until)]
    y_t = y[y.index >= pd.to_datetime(train_from)]
    y_t = y_t[y_t.index < pd.to_datetime(train_until)]
if predict:
    print('extracting prediction period from', predict_from, 'to', predict_until, '...')
    X_p = X[X.index >= pd.to_datetime(predict_from)]
    X_p = X_p[X_p.index < pd.to_datetime(predict_until)]
    y_p = y[y.index >= pd.to_datetime(predict_from)]
    y_p = y_p[y_p.index < pd.to_datetime(predict_until)]

# scale features
if scaler:
    print('scaling features...')
    if train:
        X_t_scaled = scaler.fit_transform(X_t)
        with open('rebap_model/rebap_scaler', 'wb') as filename:
            pickle.dump(scaler, filename)
    else:
        with open('rebap_model/rebap_scaler', 'rb') as filename:
            scaler = pickle.load(filename)
    if predict:
        X_p_scaled = scaler.transform(X_p)
else:
    X_t_scaled = X_t.copy()
    X_p_scaled = X_p.copy()

# reduce dimension
if reducer:
    print('reducing dimension...')
    if train:
        X_t_scaled_reduced = reducer.fit_transform(X_t_scaled)
        with open('rebap_model/rebap_reducer', 'wb') as filename:
            pickle.dump(reducer, filename)
    else:
        with open('rebap_model/rebap_reducer', 'rb') as filename:
            reducer = pickle.load(filename)
    if predict:
        X_p_scaled_reduced = reducer.transform(X_p_scaled)
else:
    X_t_scaled_reduced = X_t_scaled.copy()
    X_p_scaled_reduced = X_p_scaled.copy()



## CLASSIFICATION

if split_values:

    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

    if train:

        # sort target into classes
        print('sorting training target into classes...')
        y_t = label_target(y_t, split_values)
        
        # artificially balance classes
        from imblearn.over_sampling import SMOTE
        oversample = SMOTE()
        X_t_scaled_reduced, y_t = oversample.fit_resample(X_t_scaled_reduced, np.ravel(y_t))

        # fit classifier
        print('fitting classifier...')
        classifier.fit(X_t_scaled_reduced, np.ravel(y_t))
        
        # save model
        print('saving model...')
        with open('rebap_model/rebap_classifier', 'wb') as filename:
            pickle.dump(classifier, filename)

        # predict
        y_t = pd.DataFrame(y_t)
        y_t.columns = ['true values']
        y_t['predicted values'] = classifier.predict(X_t_scaled_reduced) 
    
        # save prediction
        print('saving training prediction...')
        y_t.to_csv('rebap_training/rebap_classifier_training_prediction.csv')

        # calculate, print, and save training scores
        classifier_training_scores = {'confusion_matrix':confusion_matrix(y_t['true values'], y_t['predicted values']),
                                    'accuracy':accuracy_score(y_t['true values'], y_t['predicted values']),
                                    'recall':recall_score(y_t['true values'], y_t['predicted values'], average=None),
                                    'precision':precision_score(y_t['true values'], y_t['predicted values'], average=None),
                                    'f1':f1_score(y_t['true values'], y_t['predicted values'], average=None)
                                    }
        print('TRAINING SCORES:')
        for key, val in classifier_training_scores.items():
            print(key+':', val)
        with open ('rebap_training/rebap_classifier_training_scores.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            for row in classifier_training_scores.items():
                filewriter.writerow(row)

    else:

        # load model
        print('loading model...')
        with open('rebap_model/rebap_classifier', 'rb') as filename:
            classifier = pickle.load(filename)

    if predict:

        # sort target into classes
        print('sorting prediction target into classes...')
        y_p = label_target(y_p, split_values)   

        # predict
        y_p.columns = ['true values']
        y_p['predicted values'] = classifier.predict(X_p_scaled_reduced) 
    
        # save prediction
        print('saving prediction...')
        y_p.to_csv('rebap_prediction/rebap_classifier_prediction.csv')

        # calculate, print, and save prediction scores
        classifier_prediction_scores = {'confusion_matrix':confusion_matrix(y_p['true values'], y_p['predicted values']),
                                        'accuracy':accuracy_score(y_p['true values'], y_p['predicted values']),
                                        'recall':recall_score(y_p['true values'], y_p['predicted values'], average=None),
                                        'precision':precision_score(y_p['true values'], y_p['predicted values'], average=None),
                                        'f1':f1_score(y_p['true values'], y_p['predicted values'], average=None)
                                        }
        print('PREDICTION SCORES:')
        for key, val in classifier_prediction_scores.items():
            print(key+':', val)
        with open ('rebap_prediction/rebap_classifier_prediction_scores.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            for row in classifier_prediction_scores.items():
                filewriter.writerow(row)

        

## REGRESSION

else:

    from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score

    if train:

        # fit regressor
        print('fitting regressor...')
        regressor.fit(X_t_scaled_reduced, np.ravel(y_t))
        
        # save model and scores
        print('saving model')
        with open('rebap_model/rebap_regressor', 'wb') as filename:
            pickle.dump(regressor, filename)

        # predict
        y_t.columns = ['true values']
        print(y_t.columns)
        y_t['predicted values'] = regressor.predict(X_t_scaled_reduced) 
        print(type(y_t), y_t.columns)
    
        # save prediction
        print('saving training prediction...')
        y_t.to_csv('rebap_training/rebap_regressor_training_prediction.csv')

        # calculate, print, and save training scores 
        regressor_training_scores = {'MAPE':mean_absolute_percentage_error(y_t['true values'], y_t['predicted values']),
                                    'MAE':mean_absolute_error(y_t['true values'], y_t['predicted values']),
                                    'MSE':mean_squared_error(y_t['true values'], y_t['predicted values']),
                                    'RMSE':(mean_squared_error(y_t['true values'], y_t['predicted values']))**.5,
                                    'R2':r2_score(y_t['true values'], y_t['predicted values']),
                                    }
        print('TRAINING SCORES:')
        for key, val in regressor_training_scores.items():
            print(key+':', val)
        with open ('rebap_training/rebap_regressor_training_scores.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            for row in regressor_training_scores.items():
                filewriter.writerow(row)

    else:

        # load model
        print('loading model...')
        with open('rebap_model/rebap_regressor', 'rb') as filename:
            regressor = pickle.load(filename)

    if predict:

        # predict
        y_p.columns = ['true values']
        y_p['predicted values'] = regressor.predict(X_p_scaled_reduced) 
    
        # save prediction
        print('saving prediction...')
        y_p.to_csv('rebap_prediction/rebap_regressor_prediction.csv')

        # calculate, print, and save prediction scores
        y_p_pred = regressor.predict(X_p_scaled_reduced) 
        regressor_prediction_scores = {'MAPE':mean_absolute_percentage_error(y_p['true values'], y_p['predicted values']),
                                        'MAE':mean_absolute_error(y_p['true values'], y_p['predicted values']),
                                        'MSE':mean_squared_error(y_p['true values'], y_p['predicted values']),
                                        'RMSE':(mean_squared_error(y_p['true values'], y_p['predicted values']))**.5,
                                        'R2':r2_score(y_p['true values'], y_p['predicted values']),
                                        }
        print('PREDICTION SCORES:')
        for key, val in regressor_prediction_scores.items():
            print(key+':', val)
        with open ('rebap_prediction/rebap_regressor_prediction_scores.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile)
            for row in regressor_prediction_scores.items():
                filewriter.writerow(row)