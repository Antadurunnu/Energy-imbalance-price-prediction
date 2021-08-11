# Capstone - PREDICTION OF THE GERMAN IMBALANCE ENERGY PRICE

Imbalance energy is the energy fed into or extracted from the power transmission net by the transmission net operator to keep balance. In Germany, costs are captured in time steps of 15 minutes by the imbalance energy price (reBAP). We aim at predicting this highly volatile price for the next several hours from a bunch of features available with a feature-specific delay.

In this Branch the main Notebooks are:

- es_forecast_data_processing.ipynb: The Notebook contains the first steps of the Data Exploration/Mining and Transformation 
and tries to extract useful information from the Rebap price (target value) via Simple Exponential Smoothing, Holt-Winters, ARIMA and Correlation Visualization Tools

- Capstone_PCA_EDA_short.ipynb: The Notebook contains the description of the Input data for the created data frame as well as all
Information for the PCA & Random Forest Classification for our binary approach (<= 0€ and > 0€)
and 5-Bucket approach (0: <= 0€, 1: <= 20€. 2: <= 40€, 3: <= 60€, 4: > 60€)

- Ensemble_Methods.ipynb: The Notebook contains the description of the Input data for the created data frame as well as all
Information for the Ensemble Methods Classification for our binary approach (<= 0€ and > 0€)
and 5-Bucket approach (0: <= 0€, 1: <= 20€. 2: <= 40€, 3: <= 60€, 4: > 60€)

- KATS_time_series.ipynb: KATS Toolkit (from Facebook) with univariate Time Series Analysis of the Rebap Price via Facebook Prophet, 
Theta Method, ARIMA (all via KATS library)

