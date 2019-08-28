import pandas as pd
import numpy as np
from fbprophet import Prophet as proph

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

###### Functions for annualised returns
def annualised_returns(df):
    """Function to calculate the annualised return for each zipcode from 2012-01-01 to 2018-01-01
    Input: dataframe with zipcodes, and the associated annual return for each zipcode in a given year.   
    -----------------------------------------------------------------------------------------------------------
    Output: dataframe with two columns, one for the zipcodes and another for the annualised returns associated
    with each zipcode during the above mentioned period (ie. Jan-2012 to Jan-2018).
    """
    annualised_return = {} #Given that the result will be one figure, it is best to store it in a dictionary
    #where the key will be the zipcode and the value will be the annualised return.
    for zipcode in df['RegionName']:
        returns = list(df.loc[(df['RegionName'] == zipcode)]['returns']) 
        tot_return = 1
        for r in returns:
            tot_return = tot_return * r
        annualised = (tot_return ** (1/len(returns))) - 1
        annualised_return[zipcode] = annualised
    #Turning the pandas dictionary with the annualised returns into a df. 
    zipcode_ann_returns_df = pd.DataFrame(list(annualised_return.items()),
                                          columns=['RegionName', 'Ann_returns'])
    #Sorting the dataframe to show the zipcodes with the highest annualised returns in order to have a peak into
    #which zipcodes have performed the best for the timeframe selected.
    zipcode_ann_returns_df = zipcode_ann_returns_df.sort_values('Ann_returns', ascending=False)
    return zipcode_ann_returns_df

###### Functions for prophet
def prophet_forecast(df, zipcodes_list):
    """ Function that when inputed a dataframe and a list of zipcodes, retrieves a dictionary containing each
    zipcode as a key and the forecasted values from the Prophet model associated with that zipcode as values.
    """
    forecasts = {}    
    for zipcode in zipcodes_list:
        returns = df.loc[(df['RegionName'] == zipcode)][['time', 'value']]
        returns = returns.rename(columns={'time': 'ds','value': 'y'})

        Model = proph(interval_width=0.95)
        Model.fit(returns)

        future_dates = Model.make_future_dataframe(periods=36, freq='MS')
        forecast = Model.predict(future_dates)

        forecasts[zipcode] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    return forecasts

def dict_to_df(dictionary):
    """ Function that strips the dictionary into individual dataframes and appends one after the other 
    to create a merged dataframe.
    """
    merged = pd.DataFrame(data=None)
    for i in dictionary.keys():
        df = dictionary[i]
        df['RegionName'] = i
        merged = pd.concat([merged, df], axis=0)
    return merged
