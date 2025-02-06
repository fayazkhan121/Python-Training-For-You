import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class TradingStrategy:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = None
        
    def fetch_data(self):
        """Fetch historical data from Yahoo Finance"""
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        
    def calculate_indicators(self):
        """Calculate technical indicators"""
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        self.data['SMA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA50'] = self.data['Close'].rolling(window=50).mean()
        
        # MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
    def prepare_features(self):
        """Prepare features for machine learning"""
        self.data['Target'] = np.where(self.data['Close'].shift(-1) > self.data['Close'], 1, 0)
        
        features = ['RSI', 'SMA20', 'SMA50', 'MACD', 'Signal_Line']
        self.data = self.data.dropna()
        
        X = self.data[features]
        y = self.data['Target']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    def train_model(self, X_train, y_train):
        """Train Random Forest model"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
    def backtest(self, X_test, y_test):
        """Backtest the strategy"""
        predictions = self.model.predict(X_test)
        
        # Calculate returns
        test_data = self.data.iloc[-len(X_test):]
        test_data['Predicted'] = predictions
        test_data['Returns'] = test_data['Close'].pct_change()
        test_data['Strategy_Returns'] = test_data['Returns'] * test_data['Predicted'].shift(1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        cumulative_returns = (1 + test_data['Strategy_Returns'].fillna(0)).cumprod()
        sharpe_ratio = np.sqrt(252) * (test_data['Strategy_Returns'].mean() / test_data['Strategy_Returns'].std())
        
        return {
            'accuracy': accuracy,
            'cumulative_returns': cumulative_returns.iloc[-1],
            'sharpe_ratio': sharpe_ratio
        }

def main():
    # Initialize strategy
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    strategy = TradingStrategy(symbol, start_date, end_date)
    
    # Execute strategy
    strategy.fetch_data()
    strategy.calculate_indicators()
    X_train, X_test, y_train, y_test = strategy.prepare_features()
    strategy.train_model(X_train, y_train)
    
    # Get results
    results = strategy.backtest(X_test, y_test)
    
    print(f"Strategy Results for {symbol}:")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Cumulative Returns: {results['cumulative_returns']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    main()
