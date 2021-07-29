import pandas as pd
import numpy as np

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