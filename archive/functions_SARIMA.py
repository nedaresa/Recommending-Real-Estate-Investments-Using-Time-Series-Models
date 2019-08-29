# functions used for SARIMA time series modeling 
import pandas as pd
import numpy as np
import statsmodels.api as sm

def retrieving_zipcode_info(df, intersection):
    '''
    This function takes a list of zipcodes and outputs a dictionary with 
    zipcodes as keys and dataframe with time index as values per zipcode
    '''
    top20zipcode = {}
    
    for zipcode in intersection:
        returns = df.loc[(df['RegionName'] == zipcode)][['time', 'value']]
        returns = returns.set_index('time')

        top20zipcode[zipcode] = returns

    return top20zipcode


def model_SARIMA_zipcode(df, pdq,pdqs):
    '''
    This function first runs a grid with pdq and seasonal pdq parameters calculated
    in the SARIMA notebook for the example zipcode 32905 and gets the best AIC value. 
    Then forecasts for all zipcodes up to 60 steps in future and gets confidence intervals of forecasts.
    '''
    ans = []
    for comb in pdq:
        for combs in pdqs:
            try:
                mod = sm.tsa.statespace.SARIMAX(df,
                                                order=comb,
                                                seasonal_order=combs,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                output = mod.fit()
                ans.append([comb, combs, output.aic])

            except:
                continue
                
    ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
    
    SARIMA_MODEL = sm.tsa.statespace.SARIMAX(df,
                                order=ans_df.loc[ans_df['aic'].idxmin()]['pdq'],
                                seasonal_order=ans_df.loc[ans_df['aic'].idxmin()]['pdqs'],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    output = SARIMA_MODEL.fit()
    # Get forecast 60 steps ahead in future
    prediction = output.get_forecast(steps=60)

    # Get confidence intervals of forecasts
    pred_conf = prediction.conf_int()
    return prediction


def predicted_annualised_returns(df):
    '''
    This function gets a dataframe and calculates predicted annualized returns for all zipcodes (20).  
    '''
    pred_annualised_return = {}
    for zipcode in df['zipcode']:
        returns = list(df.loc[(df['zipcode'] == zipcode)]['pred_returns'])
        
        tot_return = 1
        for r in returns:
            tot_return = tot_return * r 
        
        pred_annualised = (tot_return ** (1/len(returns))) - 1
        pred_annualised_return[zipcode] = pred_annualised 
        
    return pred_annualised_return
