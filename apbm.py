import pandas as pd
import numpy as np

def align(*args, timestamp_name='timestamp', prefixes=None,
          method='forward', value=None):
    
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
        if pd.api.types.is_datetime64_any_dtype(arg.index.dtype):
            arg.reset_index(drop=False, inplace=True)
            timestamp_names.append(arg.columns[0])            
        else:
            for col in arg.columns:
                if pd.api.types.is_datetime64_dtype(arg[col].dtype):
                    timestamp_names.append(col)
                    break
            else:
                print('Encountered missing timestamps. No output.')
                return

    # adapt args to common time period
    dt_from = max(arg.loc[arg.index[0], ts] for arg, ts in zip(args,timestamp_names))
    dt_to = min(arg.loc[arg.index[-1], ts] for arg, ts in zip(args,timestamp_names))
    if dt_from > dt_to: 
        print('Time periods do not intersect. No output.')
        return
    args = tuple(arg[arg[ts] >= dt_from][arg[ts] <= dt_to] for arg, ts in zip(args,timestamp_names))

    # reset column names
    if prefixes == None:
        prefixes = tuple(str(i+1)+'_' for i in range(len(args)))
    else:
        prefixes = tuple(pf+'_' if pf else '' for pf in prefixes)
    for arg, ts, pf in zip(args, timestamp_names, prefixes):
        arg.rename({col:pf+col for col in arg.columns if col != ts}, axis=1, inplace=True)
        arg.rename({ts:timestamp_name}, axis=1, inplace=True)
    
    # combine into dict
    arg = args[0]
    dic = {arg.loc[idx,timestamp_name] : {col:arg.loc[idx,col] for col in arg.columns} for idx in arg.index}
    for arg in args[1:]:
        for idx in arg.index:
            try:
                dic[arg.loc[idx,timestamp_name]].update({col:arg.loc[idx,col] for col in arg.columns})
            except:
                dic.update({arg.loc[idx,timestamp_name] : {col:arg.loc[idx,col] for col in arg.columns}})

    # turn into time ser
    df = pd.DataFrame([val for val in dic.values()])
    
    # fill potential gaps ato method
    if value != None: 
        df.fillna(value=value, axis=0, inplace=True)
    else:
        if method == 'forward': 
            df.fillna(method='ffill', axis=0, inplace=True)
        if method == 'backward': 
            df.fillna(method='bfill', axis=0, inplace=True)

    df.set_index(timestamp_name, drop=True, inplace=True)
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