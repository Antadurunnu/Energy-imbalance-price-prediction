import pandas as pd    

def align(*args, prefixes=None, method='forward', value=None):
    
    '''
    takes arbitrarily many time series as positional arguments and aligns them 
    along the union of all timestamps contained in the intersection of their time intervals
    returns result with time stamp as index
        
    for each argument, first column (including index) in datetime64 format is taken as time stamp
    
    timestamp_name: desired name for time stamp column of result

    prefixes: list of common column name prefixes for each argument
              defaults to the position of that argument in args

    method: how to fill gaps arising from timestamps not present in all arguments
                  'forward' (def; fill gaps with last preceding value available)
                  'backward' (fill gaps with first succeeding value available)

    value: constant value to be filled in instead of applying method
           defaults to 0
    '''
    
    # for each arg, take index or first column in dt format as timestamp
    timestamp_names = []
    for arg in args:
        if not pd.api.types.is_datetime64_any_dtype(arg.index.dtype):
            for col in arg.columns:
                if pd.api.types.is_datetime64_dtype(arg[col].dtype):
                    arg.set_index(col, drop=True, inplace=True)
                    break
            else:
                print('Encountered missing timestamps. No output.')
                return

    # adapt args to common time period
    dt_from = max(arg.index[0] for arg in args)
    dt_to = min(arg.index[-1] for arg in args)
    if dt_from > dt_to: 
        print('Time periods do not overlap. No output.')
        return
    args = tuple(arg[arg.index >= dt_from] for arg in args)
    args = tuple(arg[arg.index <= dt_to] for arg in args)

    # reset column names
    if prefixes == None:
        prefixes = tuple(str(i+1)+'_' for i in range(len(args)))
    else:
        prefixes = tuple(pf+'_' if pf else '' for pf in prefixes)
    for arg, pf in zip(args, prefixes):
        arg.rename({col:pf+col for col in arg.columns}, axis=1, inplace=True)
    
    # concatenate
    df = pd.concat(args, axis=1)
    
    # fill potential gaps ato method
    if value != None: 
        df.fillna(value=value, axis=0, inplace=True)
    else:
        if method == 'forward': 
            df.fillna(method='ffill', axis=0, inplace=True)
        if method == 'backward': 
            df.fillna(method='bfill', axis=0, inplace=True)

    df.dropna(axis=0, inplace=True)
    df.sort_index(inplace=True)
    
    return df



def cast(df, keys, prefixes=None, method='columns'):
    
    '''
    turns data frame with time stamps into time series 
    by casting rows with same time stamp into new columns or a dictionary per time stamp
    
    df: time series
    
    keys: list of column names to be taken as keys for identical time stamps
    
    prefixes: list of prefixes per column listed in keys for the values of that column
              defaults to column names
        
    method: how to cast values for same time stamp
            'columns' (def; entries of rows with same time stamp are casted into new columns 
                labeled by key_colname)
            'dictionary' (entries of rows with same time stamp are stored in one dictionary 
                per non-key column with key column names as keys)
    '''
    
    # take first column in dt format as timestamp
    for col in df.columns:
        if pd.api.types.is_datetime64_dtype(df[col].dtype):
            timestamp_name = col
            break
    else:
        print('No timestamp. No output.')
        return
    
    # check whether keys separate rows with same timestamp
    if df[[timestamp_name]+keys].value_counts().max() > 1:
        print('Keys insufficient. No output')
        return
    
    # prepare prefixes 
    if prefixes == None:
        prefixes = tuple(key+'_' for key in keys)
    else:
        prefixes = tuple(pf+'_' if pf else '' for key, pf in zip(keys, prefixes))
    
    # build iterator for all combinations of key column values
    keylists = [[]]
    for key in keys:
        keylists = [x+[k] for k in df[key].unique() for x in keylists]
    
    # collect columns to be distributed
    cols = [col for col in df.columns if col not in [timestamp_name]+keys]
    
    if method == 'columns':
        dfl = []   
        for keylist in keylists:
            keycode = '_'.join(pf+str(entry) for pf, entry in zip(prefixes, keylist))
            dft = df.copy()
            for key, entry in zip(keys, keylist):
                dft = dft[dft[key] == entry]
            dft.set_index(timestamp_name, drop=True, inplace=True)
            dft = dft[cols]
            dft.rename(columns = {x:keycode+'_'+x for x in cols}, inplace=True)
            dfl.append(dft)
        dfo = pd.concat(dfl, axis=1)
    
    if method == 'dictionary':
        timestamps = df[timestamp_name].unique()
        dic = {ts : {timestamp_name:ts} for ts in timestamps}
        for ts in timestamps:
            for col in cols:
                dic[ts][col] = {'_'.join(pf+str(df.loc[idx, key]) for key, pf in zip(keys, prefixes)) : df.loc[idx, col]
                                for idx in df[df[timestamp_name] == ts].index}
        dfo = pd.DataFrame([val for val in dic.values()])
        dfo.set_index(timestamp_name, drop=True, inplace=True)

    return dfo



def label_target(y, split_values=[0]):
    
    '''
    sorts the entries of y into classes separated via <= by the values given in split_values 
    '''
    
    y_label = pd.DataFrame(y.copy())
    target = y_label.columns[0]
    y_label['target_label'] = [0 for x in y_label[target]]
    for i, split in enumerate(split_values):
        y_label['target_label'] = [i+1 if x > split else c for x, c in zip(y_label[target], y_label['target_label'])]
    y_label.drop(target, axis=1, inplace=True)
    return y_label



def remove_outliers(X, y, times_std=None):
    
    '''
    removes all entries of y outside the interval (mean(y) - times_std * std(y) , mean(y) + times_std * std(y)) and the corresponding entries of X
    '''
    
    if times_std:
        target = y.columns[0]
        target_mean, target_std = y[target].mean(), y[target].std()
        indices_below = y[y[target] < target_mean - times_std * target_std].index 
        indices_above = y[y[target] > target_mean + times_std * target_std].index
        return (X.drop(indices_below, axis=0).drop(indices_above, axis=0), 
                y.drop(indices_below, axis=0).drop(indices_above, axis=0))
    return X.copy(), y.copy()
    
    
    
# wrapper functions for data import

def load_target():
    
    '''
    imports the ane.energy data for target
    '''

    df = pd.DataFrame(pd.read_csv('data/imbalance_de.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col='dt_start_utc')[['rebap_eur_mwh']])
    return df



def load_wind_speed():
    
    '''
    imports the ane.energy data for wind speed per Voronoi area
    '''

    df = pd.read_csv('data/wind_speed_ts.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    return df



def load_imbalance_power():
    
    '''
    imports the ane.energy data for imbalance power
    '''

    df = pd.read_csv('data/imbalance_de.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')[['power_mw']]
    df.columns = ['imbalance_power_mw']
    return df



def load_epex_da():
    
    '''
    imports the ane.energy data for EPEX day ahead price
    '''

    df = pd.read_csv('data/epex_da_de.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')[['epex_da_de_eur_mwh']]
    return df



def load_epex_da_fc():
    
    '''
    imports the ane.energy data for EPEX day ahead price forecast
    '''

    df = pd.read_csv('data/epex_da_prognosis_ts.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')[['newest_fc', 'second_newest_fc']]
    df.columns = ['epex_da_'+col for col in df.columns]
    return df



def load_power_ac():
    
    '''
    imports the ane.energy data for feed by wind farm
    '''

    df = pd.read_csv('data/power_ac.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    df.drop('power_act_total', axis=1, inplace=True)
    return df



def load_power_fc():
    df = pd.read_csv('data/power_fc.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    df.drop('power_fc_total', axis=1, inplace=True)
    return df



def load_wind_onshore_fc():
    
    '''
    imports the ane.energy data for onshore wind feed forecast
    '''

    df = pd.read_csv('data/es_fc_wind_onshore_ts.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    df.drop('total_power_mw', axis=1, inplace=True)
    df.columns = ['wind_onshore_fc_'+col for col in df.columns]
    return df



def load_wind_offshore_fc():
    
    '''
    imports the ane.energy data for offshore wind feed forecast
    '''

    df = pd.read_csv('data/es_fc_wind_offshore_ts.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    df.drop('total_power_mw', axis=1, inplace=True)
    df.columns = ['wind_offshore_fc_'+col for col in df.columns]
    return df



def load_solar_fc():
    
    '''
    imports the ane.energy data for solar feed forecast
    '''

    df = pd.read_csv('data/es_fc_solar_ts.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    df.drop('total_power_mw', axis=1, inplace=True)
    df.columns = ['solar_fc_'+col for col in df.columns]
    return df



def load_renewables_fc():
    
    '''
    imports the ane.energy data for renewables feed forecast
    '''

    df = pd.read_csv('data/es_fc_total_renewables_ts.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    df.drop('total_power_mw', axis=1, inplace=True)
    df.columns = ['renewables_fc_'+col for col in df.columns]
    return df



def load_total_fc():
    
    '''
    imports the ane.energy data for total feed forecast
    '''

    df = pd.read_csv('data/es_fc_total_load_ts.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    df.drop('total_power_mw', axis=1, inplace=True)
    df.columns = ['load_fc_'+col for col in df.columns]
    return df



def load_imbalance_at():
    
    '''
    imports the ENTSOE data for imbalance energy in Austria
    '''

    df = pd.read_csv('data/entsoe_imbalance_at.csv',# OR: 'data/smard_imbalance_at.csv' 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    df.drop('status', axis=1, inplace=True)
    return df



def load_imbalance_be():
    
    '''
    imports the ENTSOE data for imbalance energy in Belgium
    '''

    df = pd.read_csv('data/entsoe_imbalance_be.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    df.drop('status', axis=1, inplace=True)
    return df



def load_imbalance_dk():
    
    '''
    imports the ENTSOE data for imbalance energy in Denmark
    '''

    df = pd.read_csv('data/entsoe_imbalance_dk.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    df.drop('status', axis=1, inplace=True)
    return df



def load_imbalance_pl():
    
    '''
    imports the ENTSOE data for imbalance energy in Poland
    '''

    df = pd.read_csv('data/entsoe_imbalance_pl.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    df.drop('status', axis=1, inplace=True)
    return df



def load_consumption_ac():
    
    '''
    imports and reformats the SMARD data for power consumption
    '''

    df = pd.read_csv('data/smard_consumption_ac.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    # replace occasional entries - by 0:
    for col in df.columns:
        df[col] = [0 if x == '-' else x for x in df[col]]
    df = df.astype('float64')    
    return df



def load_consumption_fc():
    
    '''
    imports and reformats the SMARD data for power consumption forecast
    '''

    df = pd.read_csv('data/smard_consumption_fc.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    # replace occasional entries - by 0:
    for col in df.columns:
        df[col] = [0 if x == '-' else x for x in df[col]]
    df = df.astype('float64')    
    return df



def load_generation_ac():
    
    '''
    imports and reformats the SMARD data for power generation
    '''

    df = pd.read_csv('data/smard_generation_ac.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc').astype('str')
    # replace occasional entries - by 0
    for col in df.columns:
        df[col] = ['0' if x == '-' else x.replace('.','').replace(',','.') for x in df[col]]
    # remove dots and replace comma by dot in German format columns
    german_format_cols = ['gen_biomass_ac_mwh', 'gen_waterpower_ac_mwh', 'gen_wind_offshore_ac_mwh', 'gen_wind_onshore_ac_mwh',
       'gen_solar_ac_mwh', 'gen_other_renewables_ac_mwh', 'gen_nuclear_ac_mwh', 'gen_lignite_ac_mwh']
    for col in german_format_cols:
        df[col] = [x.replace('.','').replace(',','.') for x in df[col]]
    df = df.astype('float64')    
    return df



def load_generation_fc():
    
    '''
    placeholder for import and reformatting of the SMARD data for power generation forecast
    '''

    pass
    #df = pd.read_csv('data/smard_generation_fc.csv', 
    #                 parse_dates=['dt_start_utc'], 
    #                 index_col = 'dt_start_utc')
    # replace occasional entries - by 0:
    #for col in df.columns:
    #    df[col] = [0 if x == '-' else x for x in df[col]]
    #df = df.astype('float64')    
    #return df

    
    
def load_mrl_energy():
    
    '''
    imports and reformats the SMARD data for the Minutenreserve demand
    '''

    df = pd.read_csv('data/smard_mrl.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')[['mrl_energy_pos_mwh', 
                                                  'mrl_energy_neg_mwh',
                                                  'mrl_energy_price_pos_eur_mwh', 
                                                  'mrl_energy_price_neg_eur_mwh']]
    # replace occasional entries - by 0:
    for col in df.columns:
        df[col] = [0 if x == '-' else x for x in df[col]]
    # remove dots and replace comma by dot in German format columns
    german_format_cols = ['mrl_energy_price_pos_eur_mwh', 'mrl_energy_price_neg_eur_mwh']
    for col in german_format_cols:
        df[col] = [x.replace('.','').replace(',','.') for x in df[col]]
    df = df.astype('float64')    
    return df



def load_mrl_capacity():
    
    '''
    imports and reformats the SMARD data for the Minutenreserve capacity
    '''

    df = pd.read_csv('data/smard_mrl.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')[['mrl_capacity_pos_mwh', 
                                                  'mrl_capacity_neg_mwh',
                                                  'mrl_capacity_price_pos_eur_mwh', 
                                                  'mrl_capacity_price_neg_eur_mwh']]
    # replace occasional entries - by 0:
    for col in df.columns:
        df[col] = [0 if x == '-' else x for x in df[col]]
    # remove dots and replace comma by dot in German format columns
    german_format_cols = ['mrl_capacity_price_pos_eur_mwh', 'mrl_capacity_price_neg_eur_mwh']
    for col in german_format_cols:
        df[col] = [x.replace('.','').replace(',','.') for x in df[col]]
    df = df.astype('float64')    
    return df



def load_prl():
    
    '''
    imports and reformats the SMARD data for Primärregelreserve
    '''

    df = pd.read_csv('data/smard_prl.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')
    # replace occasional entries - by 0:
    for col in df.columns:
        df[col] = [0 if x == '-' else x for x in df[col]]
    # remove dots and replace comma by dot in German format columns
    german_format_cols = ['prl_capacity_price_eur_mwh']
    for col in german_format_cols:
        df[col] = [x.replace('.','').replace(',','.') for x in df[col]]
    df = df.astype('float64')    
    return df



def load_srl_energy():
    
    '''
    imports and reformats the SMARD data for Sekundärregelreserve demand
    '''

    df = pd.read_csv('data/smard_srl.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')[['srl_energy_pos_mwh', 
                                                  'srl_energy_neg_mwh',
                                                  'srl_energy_price_pos_eur_mwh', 
                                                  'srl_energy_price_neg_eur_mwh']]
    # replace occasional entries - by 0:
    for col in df.columns:
        df[col] = [0 if x == '-' else x for x in df[col]]
    # remove dots and replace comma by dot in German format columns
    german_format_cols = ['srl_energy_pos_mwh', 'srl_energy_neg_mwh']
    for col in german_format_cols:
        df[col] = [x.replace('.','').replace(',','.') for x in df[col]]
    df = df.astype('float64')    
    return df



def load_srl_capacity():
    
    '''
    imports and reformats the SMARD data for Sekundärregelreserve capacity
    '''

    df = pd.read_csv('data/smard_srl.csv', 
                     parse_dates=['dt_start_utc'], 
                     index_col = 'dt_start_utc')[['srl_capacity_pos_mwh', 
                                                  'srl_capacity_neg_mwh',
                                                  'srl_capacity_price_pos_eur_mwh', 
                                                  'srl_capacity_price_neg_eur_mwh']]
    # replace occasional entries - by 0:
    for col in df.columns:
        df[col] = [0 if x == '-' else x for x in df[col]]
    # remove dots and replace comma by dot in German format columns
    german_format_cols = ['srl_capacity_pos_mwh', 'srl_capacity_neg_mwh']
    for col in german_format_cols:
        df[col] = [x.replace('.','').replace(',','.') for x in df[col]]
    df = df.astype('float64')    
    return df