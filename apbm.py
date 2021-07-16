import pandas as pd
import numpy as np

def align(*args, timestamp_name='timestamp', prefixes=None,
          method='forward'):
    
    '''
    takes arbitrarily many time series as positional arguments 
    and aligns them along the union of all timestamps contained in the intersection of their time intervals
        
    first column of each argument which is in datetime64 format is recognized as that argument's time stamp
    
    timestamp_name: desired name for time stamp column of result

    prefixes: list of common column name prefix for each arg
              defaults to the position of arg in args

    match_method: how to fill gaps arising from timestamps not present in all arguments
                  'forward' (def; fill gaps with last preceding value available)
                  'backward' (... with first succeeding value available)
    '''
    
    # for each arg, take first column in dt format as timestamp
    timestamp_names = []
    for arg in args:
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
    ts = pd.DataFrame([val for key, val in dic.items()])
    
    # fill potential gaps ato method
    if method == 'forward': 
        ts.fillna(method='ffill', axis=0, inplace=True)
    if method == 'backward': 
        ts.fillna(method='bfill', axis=0, inplace=True)
    
    return ts