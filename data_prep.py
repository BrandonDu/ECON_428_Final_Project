# !pip install pandas_datareader
# !pip install tensorflow
# !pip install yfinance
# !pip install openai
# !pip install httpx
# !pip install requests beautifulsoup4
# !pip install selenium
# !pip install webdriver-manager
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
import pandas_datareader
import yfinance as yf
import httpx
import os

# Download past year's basic price data for all S&P 500 companies
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# Function to get S&P 500 list
def get_sp500_symbols():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500 = table[0]
    return sp500['Symbol'].tolist()


# Example subset of S&P 500 stocks
# sp500_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
sp500_stocks = get_sp500_symbols()
# sp500_stocks = ['SPOT']
print(sp500_stocks)

# Calculate dates for the past year
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

output_directory = "./S&P 500 Past Year Data/"

for symbol in sp500_stocks:
    try:
        # Yahoo Finance sometimes requires a replacement for symbols like "BRK.B" to "BRK-B"
        symbol_yf = symbol.replace('.', '-')

        # Fetch historical data
        stock_data = yf.download(symbol_yf, start=start_date, end=end_date)

        # Generate filename and save to CSV
        filename = f"{output_directory}{symbol}_past_year.csv"
        stock_data.to_csv(filename)

        print(f"Data for {symbol} saved to {filename}")
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")

print("Data download completed.")