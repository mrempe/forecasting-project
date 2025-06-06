# main.py in the 
# ForcastingForInvestors project
#
# This code was originally translated from an .ipynb file 
# written by Eli Brunette [GitHub link here]




import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
#from scikit-learn.metrics import mean_squared_error

# plotting stuff
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn for enhanced visualization
import plotly.express as px # Plotly for interactive plots

from datetime import datetime, timedelta # Date operations
#import pmdarima as pm       # for SARIMAX model      

# time series analysis and modeling
from statsmodels.tsa.stattools import adfuller # Augmented Dickey-Fuller Test for stationarity check
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

import pdb  # for breakpoints

from utils import perform_adf_test
from utils import plot_differencing_acf_pacf



def main():
    
    # Load the dataset
    df = pd.read_csv('./data/redfin-data2.csv', parse_dates=['Month of Period End'], encoding='utf-16-le', sep='\t')
    print(df.head())

    # Only going to keep data that we want to keep for forecasting and features that might be # interesting to investigate in the future.

    df = df[['Region', 'Month of Period End', 'Median Sale Price', 'Homes Sold', 'Inventory', 'Days on Market', 'New Listings']]

    # # Only looking at the Spokane, WA region just for this example. Along with other cleanup methods to prepare for modeling.
    #df = df[df['Region'] == 'Spokane, WA']
    df = df[df['Region'].isin(['Spokane, WA metro area'])]    # Spokane metro area, not just Spokane
    #df = df[df['Region'] == 'Seattle, WA']
    df['Price'] = df['Median Sale Price'].str.replace('$', '', regex=False)\
                                 .str.replace('K', '', regex=False)\
                                 .astype(float) * 1000
    df['Price'] = df['Price'].astype(int)
    
    df['Homes Sold'] = df['Homes Sold'].str.replace(',','',regex=False).astype(int)
    df['Inventory']  = df['Inventory'].str.replace(',', '', regex=False).astype(int)
    df['New Listings'] = df['New Listings'].str.replace(',','',regex=False).astype(int)
    df = df.drop(columns=['Region', 'Median Sale Price'])

    df.head()

    # Sort by date
    df = df.sort_values('Month of Period End')
    df.set_index('Month of Period End', inplace=True)

    df.head()

    for lag in range(1, 13):  # use last 12 months
        df[f'price_lag_{lag}'] = df['Price'].shift(lag)

    #MJR: introduce seasonality to use with a SARIMAX model
    # df['Price_diff'] = df['Price'].diff(periods=12)
    # df.info()


    df.dropna(inplace=True)     # this removes all rows with NaN
    df[:-12]                    # show all rows except last 12 rows

    #breakpoint()

    # Set up training data and test data
    train   = df[:-12]   # all but last year
    test    = df[-12:]   # last 12 months

    x_train = train.drop('Price', axis=1)   # the predictors in the training data, so don't include price since that's the outcome var
    y_train = train['Price']                # the outcome var in the training data

    x_test  = test.drop('Price', axis=1)    # the predictors in the test data, don't include price
    y_test  = test['Price']                 # the outcome in the test data

    print(x_train)

    model = XGBRegressor()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    

    
    # Plot the prediction with the actual for the last 12 months (test data only)
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.legend()
    plt.title('Housing Price Forecast')
    plt.show(block=False)



    # now that we see how well it trained, let's see how it predicts future prices
    predictions = []
    last_known = x_test.iloc[-1].copy()
    print(last_known)
    
    for _ in range(12):  # forecast 12 months
        input_data = last_known.values.reshape(1, -1)
        next_price = model.predict(input_data)[0]
        predictions.append(next_price)

        # shift lags and insert new prediction
        for i in range(12, 1, -1):
            last_known[f'price_lag_{i}'] = last_known[f'price_lag_{i-1}']
        last_known['price_lag_1'] = next_price

    # --- Create forecast time index ---
    forecast_index  = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    forecast_series = pd.Series(predictions, index=forecast_index)

    print(forecast_index)
    print(forecast_series)

    # --- Plot actual vs forecast ---
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[-24:], df['Price'].iloc[-24:], label='Actual')
    plt.plot(forecast_series.index, forecast_series, label='Forecast', linestyle='--')
    plt.title("XGBoost Housing Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    print("End of forcasting-project!")


    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    # New approach: try using a SARIMAX approach
    perform_adf_test(df['Price'])  # this tells me that data are not stationary. since not, we do differencing
    plot_differencing_acf_pacf(df['Price'])

    p, d, q = 2, 1, 2

    S_model = SARIMAX(df['Price'],
                    order=(p, d, q),
                    seasonal_order=(p, d, q, 12))
    S_model = S_model.fit()
    print(S_model.summary())
    forecast = S_model.predict(len(df["Price"]), len(df['Price'])+30)
    print(forecast)

    #plot forecast
    plt.figure(figsize=(10, 5))
    plt.plot(df['Price'], label='Actual')
    plt.plot(forecast, label='Forecast')
    plt.title("Seattle Housing Price Forecast using SARIMAX")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()


    plt.show()  # keep all figures open even when script ends

if __name__ == "__main__":
    main()
