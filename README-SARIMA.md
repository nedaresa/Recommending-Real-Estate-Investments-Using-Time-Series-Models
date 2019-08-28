Objective
Using the dataset from zillow, select the top five zipcodes that see to be most promising for real estate investment. Here, I describe a workflow that performs a SARIMA time series model on 20 selected zipcodes.

Workflow
Based on the prophet model evaluation of price forcast and calculated predicted annualized returns for every zipcode in a timeframe of 3 years (2018-06 until 2021?), 20 zipcodes with a predicted annualized return of above 15% were selected for SARIMA forcast performed in this workflow.

Modify the train dataset to only include the top 20 zipcodes.

Find pdq and seasonal pdq parameters within a selected range that are associated with the best AIC value.

Perform SARIMA model on one example zipcode in the train dataset to predict for up to 5 years in future using pdq and seasonal pdq parameters associated with the best AIC value.

Perform SARIMA on all the 20 zipcodes to predict for up to 5 years in future using pdq and seasonal pdq parameters associated with the best AIC value.

Calculate predicted annualized return values using test dataset and sort them based on a descending order.

Top zipcodes sorted by their predicted annualized return values are further explored based on a comparison of predicted vs real prices to narrow down to a selection of top 5.