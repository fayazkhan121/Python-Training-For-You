import yfinance as yf
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

# Function to convert currency using a public API
def convert_currency(amount, from_currency, to_currency):
    url = f"https://api.exchangerate.host/convert?from={from_currency}&to={to_currency}&amount={amount}"
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json()
        return result['result']
    else:
        raise Exception("Error fetching currency conversion rates")

# Fetch stock data for the past 30 days
def main():
    ticker = "AAPL"  # Apple Inc.
    base_currency = "USD"
    target_currency = "GBP"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    print(f"Fetching stock data for {ticker} from {start_date.date()} to {end_date.date()}...")
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    print("Converting stock prices to GBP...")
    stock_data['Close_GBP'] = stock_data['Close'].apply(
        lambda x: convert_currency(x, base_currency, target_currency)
    )

    print("Processing data and plotting...")
    # Calculate daily percentage change
    stock_data['Daily Change (%)'] = stock_data['Close_GBP'].pct_change() * 100

    # Plot the closing price in GBP
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Date'], stock_data['Close_GBP'], label=f"{ticker} Closing Price in GBP")
    plt.xlabel("Date")
    plt.ylabel("Price (GBP)")
    plt.title(f"{ticker} Closing Price in GBP (Last 30 Days)")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the daily percentage change
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Date'], stock_data['Daily Change (%)'], label="Daily Change (%)", color="orange")
    plt.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Daily Change (%)")
    plt.title(f"{ticker} Daily Percentage Change (Last 30 Days)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
