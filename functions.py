import pandas as pd
import numpy as np

def df_melt(df, id_cols, var_name='time'):
    """
    Takes a dataframe in wide format and melt into long format
    df - dataframe to be melted
    id_cols - columns with descriptive information
    var_name - variable to set to long format, in this case 'time'
    """
    
    melt_df = pd.melt(df, id_vars=id_cols, var_name=var_name)
    melt_df[var_name] = pd.to_datetime(melt_df[var_name], infer_datetime_format=True)
    
    return melt_df

######
def retrieve_zip_info(df, zipcode, zip_col_name, info=['City', 'State', 'Metro']):
    """
    From dataframe with zip codes and corresponding, city, state and metro area names
    Returns city, state and metro names by specifying zip code and column index of city/state/metro
    """
    zip_info = {}
    zip_info['City'] = list(df.loc[df[zip_col_name]==zipcode][info[0]])[0]
    zip_info['State'] = list(df.loc[df[zip_col_name]==zipcode][info[1]])[0]
    zip_info['Metro'] = list(df.loc[df[zip_col_name]==zipcode][info[2]])[0]
    
    return zip_info