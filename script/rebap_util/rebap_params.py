# select features from feature_catalogue to be used for training and prediction
feature_list = ['wind_speed', 
                 'imbalance_power', 
                 'epex_da', 
                 'epex_da_fc', 
                 'power_ac', 
                 'power_fc',
                 'wind_onshore_fc', 
                 'wind_offshore_fc', 
                 'solar_fc', 
                 'renewables_fc', 
                 'total_fc',
                 'generation_ac',
                 'consumption_ac',
                 'consumption_fc',
                ]
 
# define prediction time in 15 min time steps
prediction_time_shift = 8

# define backlog in 15 min time steps
backlog = 6

# define training period
train_from = '2019-07-01'
train_until = '2020-07-01'

# define prediction period
predict_from = '2020-07-01'
predict_until = '2020-08-01'

# define interval around target mean outside of which outliers get removed ito multiples of standard deviation; set to None for keeping everything
times_std = None
times_std = 3

# define list of splitting values for classification (via <=) or None for regression
split_values = None
split_values = [0]
split_values = [0,20,40,60]

# define feature scaler
scaler = None
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# define dimensional reduction
reducer = None
from sklearn.decomposition import PCA
reducer = PCA(13)
#from sklearn.manifold import TSNE
#reducer = TSNE(2)

# define classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=-1)

# define regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(max_depth=12, min_samples_leaf=5)
