
#### A description of the SARIMA time series model on 29 selected zipcodes.

- Based on the every zipcode's annualized return using prophet price forcasts in a timeframe of 3 years (2018 until 2021), 29 zipcodes with a predicted annualized return of above 15% were selected for SARIMA forcast.

- Find pdq and seasonal pdq parameters within a selected range that are associated with the best AIC value.

- Perform SARIMA model on an example zipcode in the train dataset to predict for up to 5 years in future using pdq and seasonal pdq parameters associated with the best AIC value.

- Perform SARIMA on all 29 zipcodes to predict median prices per zipcode for up to 5 years in future using pdq and seasonal pdq parameters associated with the best AIC value. 

- Calculate predicted annualized return for each of the 29 zipcodes using SARIMA predictions from 2018 to 2023. 

- Sort zipcodes by their predicted annualized return values in descending order.

- To select top 5 zipcodes that are most promising for real estate investment, go wodn the list of sorted zipcodes and compare their predicted prices to real prices as in the test dataset. Then, pick the 5 zipcodes with predicted values closest to real values (analysis not included in SARIMA_model.ipynb).