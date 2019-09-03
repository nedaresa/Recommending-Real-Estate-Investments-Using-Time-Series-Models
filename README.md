
Juan Julian Herranz, Neda Jabbari, Boping Liu

Aug 30th, 2019


### Objective 

Using the dataset from zillow, select the top five zipcodes that are most promising for real estate investment. 

Data source: 

https://www.zillow.com/research/data/

Upon arrival at the page, navigate to the “Home Values” Section. From the “Data Type” dropdown select “ZHVI Single-Family Home Time Series” and from the “Geography” dropdown select “Zip Code”. At time of writing the lastest month in the data is 2019-07.

Workflow:

- Subset the dataset to include 2012 and after.
- Subset the train set to include values starting from 2012-01 until 2018-06. Define the rest as test set.
- Calculate annualized returns for all zipcodes in the train set. 
- Subset zipcodes based on those with higher than 15% annualized returns. 
- performs a prophet time series model on selected zipcodes.
- Calculate annualized returns for every zipcode in the train set using prophet predictions. 
- Subset zipcodes to only include those with >15% predicted annualized return. 
- Perform a SARIMA time series model on the selected zipcodes.
- Calculate annualized returns for every zipcode in the train set using SARIMA predictions. 
- Sort zipcodes by their predicted annualized return values in descending order. 
- Select top 5 zipcode candidates for real estate investment, start from the top zipcodes in the sorted list and compare their predicted prices to real prices as in the test dataset. 
- Pick the 5 zipcodes with predicted values closest to real values.  


#### A description of the prophet workflow.


- Select zipcodes with a predicted annualized return of above 15% (29 zipcodes) for downstream SARIMA forcast.


#### A description of the SARIMA time series model on 29 selected zipcodes.

- Based on the every zipcode's annualized return using prophet price forcasts in a timeframe of 3 years (2018 until 2021), 29 zipcodes with a predicted annualized return of above 15% were selected for SARIMA forcast.


- Perform SARIMA model on an example zipcode in the train dataset to predict for up to 5 years in future using pdq and seasonal pdq parameters associated with the best AIC value.

- Perform SARIMA on all 29 zipcodes to predict median prices per zipcode for up to 5 years in future using pdq and seasonal pdq parameters associated with the best AIC value. 


- Sort zipcodes by their predicted annualized return values in descending order. 

#### Selection of top 5 zipcodes

- To select top 5 zipcodes that are most promising for real estate investment, go down the list of sorted zipcodes and compare their predicted prices to real prices as in the test dataset. Then, pick the 5 zipcodes with predicted values closest to real values.