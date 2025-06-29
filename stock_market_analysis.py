#!/usr/bin/env python3
"""
Stock Market Analysis Project
Analyzing Apple, Microsoft, Netflix, and Google stock performance
using various data science techniques including ML predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Additional Libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)

class StockMarketAnalyzer:
    def __init__(self, tickers=['AAPL', 'MSFT', 'NFLX', 'GOOG'], period='3mo'):
        """
        Initialize the Stock Market Analyzer
        
        Parameters:
        tickers (list): List of stock tickers to analyze
        period (str): Time period for data collection (e.g., '3mo', '1y')
        """
        self.tickers = tickers
        self.period = period
        self.data = {}
        self.technical_indicators = {}
        self.models = {}
        self.scalers = {}
        
    def collect_data(self, use_api=True):
        """
        Collect stock data either from API or local file
        
        Parameters:
        use_api (bool): If True, use yfinance API; if False, load from local file
        """
        if use_api:
            print("Collecting data from Yahoo Finance API...")
            for ticker in self.tickers:
                try:
                    stock = yf.Ticker(ticker)
                    self.data[ticker] = stock.history(period=self.period)
                    print(f"✓ Collected data for {ticker}: {len(self.data[ticker])} records")
                except Exception as e:
                    print(f"✗ Error collecting data for {ticker}: {e}")
        else:
            print("Loading data from local file...")
            try:
                # Load from the provided dataset
                df = pd.read_csv('stocks.csv')
                for ticker in self.tickers:
                    self.data[ticker] = df[df['Ticker'] == ticker].copy()
                    self.data[ticker]['Date'] = pd.to_datetime(self.data[ticker]['Date'])
                    self.data[ticker].set_index('Date', inplace=True)
                    print(f"✓ Loaded data for {ticker}: {len(self.data[ticker])} records")
            except Exception as e:
                print(f"✗ Error loading local data: {e}")
                print("Falling back to API data collection...")
                self.collect_data(use_api=True)
    
    def calculate_technical_indicators(self):
        """Calculate various technical indicators for each stock"""
        print("\nCalculating technical indicators...")
        
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            
            # Moving Averages
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volatility
            df['Volatility'] = df['Close'].rolling(window=20).std()
            
            # Price Changes
            df['Daily_Return'] = df['Close'].pct_change()
            df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
            
            # Volume Indicators
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            self.technical_indicators[ticker] = df
            print(f"✓ Calculated indicators for {ticker}")
    
    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        print("\nPerforming Exploratory Data Analysis...")
        
        # Create a comprehensive EDA report
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=('Stock Price Comparison', 'Daily Returns Distribution',
                          'Volume Analysis', 'Correlation Heatmap',
                          'Moving Averages', 'RSI Analysis',
                          'Volatility Comparison', 'Cumulative Returns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Stock Price Comparison
        for ticker in self.tickers:
            df = self.technical_indicators[ticker]
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Close'], name=ticker, mode='lines'),
                row=1, col=1
            )
        
        # 2. Daily Returns Distribution
        for ticker in self.tickers:
            df = self.technical_indicators[ticker]
            returns = df['Daily_Return'].dropna()
            fig.add_trace(
                go.Histogram(x=returns, name=ticker, opacity=0.7, nbinsx=30),
                row=1, col=2
            )
        
        # 3. Volume Analysis
        for ticker in self.tickers:
            df = self.technical_indicators[ticker]
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name=ticker, opacity=0.7),
                row=2, col=1
            )
        
        # 4. Correlation Heatmap
        # Combine all closing prices
        combined_data = pd.DataFrame()
        for ticker in self.tickers:
            combined_data[ticker] = self.technical_indicators[ticker]['Close']
        
        corr_matrix = combined_data.corr()
        
        # Create correlation heatmap
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                      colorscale='RdBu', zmid=0),
            row=2, col=2
        )
        
        # 5. Moving Averages
        for ticker in self.tickers:
            df = self.technical_indicators[ticker]
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MA_20'], name=f'{ticker} MA20', mode='lines'),
                row=3, col=1
            )
        
        # 6. RSI Analysis
        for ticker in self.tickers:
            df = self.technical_indicators[ticker]
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name=f'{ticker} RSI', mode='lines'),
                row=3, col=2
            )
        
        # Add RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=2)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=2)
        
        # 7. Volatility Comparison
        for ticker in self.tickers:
            df = self.technical_indicators[ticker]
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Volatility'], name=f'{ticker} Vol', mode='lines'),
                row=4, col=1
            )
        
        # 8. Cumulative Returns
        for ticker in self.tickers:
            df = self.technical_indicators[ticker]
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Cumulative_Return'], name=f'{ticker} CumRet', mode='lines'),
                row=4, col=2
            )
        
        fig.update_layout(height=1200, width=1200, title_text="Comprehensive Stock Market Analysis")
        fig.show()
        
        # Print summary statistics
        self.print_summary_statistics()
    
    def print_summary_statistics(self):
        """Print comprehensive summary statistics"""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        for ticker in self.tickers:
            df = self.technical_indicators[ticker]
            print(f"\n{ticker} Analysis:")
            print("-" * 30)
            print(f"Total Records: {len(df)}")
            print(f"Date Range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"Current Price: ${df['Close'].iloc[-1]:.2f}")
            print(f"Price Range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            print(f"Average Daily Return: {df['Daily_Return'].mean():.4f}")
            print(f"Volatility (Std Dev): {df['Daily_Return'].std():.4f}")
            print(f"Total Return: {(df['Cumulative_Return'].iloc[-1] - 1) * 100:.2f}%")
            print(f"Average Volume: {df['Volume'].mean():,.0f}")
    
    def feature_engineering(self):
        """Create features for machine learning models"""
        print("\nPerforming Feature Engineering...")
        
        self.features = {}
        self.targets = {}
        
        for ticker in self.tickers:
            df = self.technical_indicators[ticker].copy()
            
            # Create lag features
            for lag in [1, 2, 3, 5]:
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
                df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
                df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)
            
            # Create rolling statistics
            for window in [5, 10, 20]:
                df[f'Close_MA_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
                df[f'Return_Std_{window}'] = df['Daily_Return'].rolling(window=window).std()
            
            # Price momentum features
            df['Price_Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
            df['Price_Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
            df['Price_Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
            
            # Technical indicator features
            df['MA_Ratio'] = df['MA_5'] / df['MA_20']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            df['RSI_Signal'] = np.where(df['RSI'] > 70, -1, np.where(df['RSI'] < 30, 1, 0))
            
            # Volume features
            df['Volume_Price_Trend'] = df['Volume'] * df['Daily_Return']
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Target variable (next day's return)
            df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
            
            # Remove NaN values
            df = df.dropna()
            
            # Select features for ML
            feature_columns = [col for col in df.columns if col not in 
                             ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            
            self.features[ticker] = df[feature_columns]
            self.targets[ticker] = df['Target']
            
            print(f"✓ Created {len(feature_columns)} features for {ticker}")
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\nTraining Machine Learning Models...")
        
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }
        
        for ticker in self.tickers:
            print(f"\nTraining models for {ticker}...")
            
            X = self.features[ticker]
            y = self.targets[ticker]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[ticker] = scaler
            self.models[ticker] = {}
            
            for model_name, model in models_to_train.items():
                # Train model
                if model_name == 'SVR':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.models[ticker][model_name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                print(f"  {model_name}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
    
    def evaluate_models(self):
        """Evaluate and compare model performance"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        for i, ticker in enumerate(self.tickers):
            row = i // 2
            col = i % 2
            
            # Get model metrics
            model_names = list(self.models[ticker].keys())
            mse_scores = [self.models[ticker][name]['mse'] for name in model_names]
            r2_scores = [self.models[ticker][name]['r2'] for name in model_names]
            
            # Plot MSE comparison
            axes[row, col].bar(model_names, mse_scores, color=['blue', 'green', 'red', 'orange'])
            axes[row, col].set_title(f'{ticker} - Mean Squared Error')
            axes[row, col].set_ylabel('MSE')
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add R² scores as text
            for j, (name, r2) in enumerate(zip(model_names, r2_scores)):
                axes[row, col].text(j, mse_scores[j] + max(mse_scores) * 0.01, 
                                  f'R²={r2:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        for ticker in self.tickers:
            print(f"\n{ticker} Model Performance:")
            print("-" * 40)
            for model_name, metrics in self.models[ticker].items():
                print(f"{model_name:20} | MSE: {metrics['mse']:.6f} | "
                      f"MAE: {metrics['mae']:.6f} | R²: {metrics['r2']:.4f}")
    
    def predict_next_day(self, ticker, model_name='Random Forest'):
        """Predict next day's return for a specific stock"""
        if ticker not in self.models:
            print(f"No models trained for {ticker}")
            return None
        
        if model_name not in self.models[ticker]:
            print(f"Model {model_name} not found for {ticker}")
            return None
        
        # Get latest features
        latest_features = self.features[ticker].iloc[-1:].copy()
        
        # Make prediction
        model = self.models[ticker][model_name]['model']
        scaler = self.scalers[ticker]
        
        if model_name == 'SVR':
            latest_features_scaled = scaler.transform(latest_features)
            prediction = model.predict(latest_features_scaled)[0]
        else:
            prediction = model.predict(latest_features)[0]
        
        current_price = self.technical_indicators[ticker]['Close'].iloc[-1]
        predicted_price = current_price * (1 + prediction)
        
        print(f"\n{ticker} Next Day Prediction ({model_name}):")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Return: {prediction:.4f} ({prediction*100:.2f}%)")
        print(f"Predicted Price: ${predicted_price:.2f}")
        
        return prediction
    
    def generate_trading_signals(self):
        """Generate trading signals based on technical analysis"""
        print("\nGenerating Trading Signals...")
        
        signals = {}
        
        for ticker in self.tickers:
            df = self.technical_indicators[ticker]
            
            # Initialize signal columns
            df['Signal'] = 0
            df['Signal_Strength'] = 0
            
            # RSI signals
            df.loc[df['RSI'] < 30, 'Signal'] += 1  # Oversold
            df.loc[df['RSI'] > 70, 'Signal'] -= 1  # Overbought
            
            # Moving average signals
            df.loc[df['Close'] > df['MA_20'], 'Signal'] += 1  # Above MA
            df.loc[df['Close'] < df['MA_20'], 'Signal'] -= 1  # Below MA
            
            # MACD signals
            df.loc[df['MACD'] > df['MACD_Signal'], 'Signal'] += 1  # Bullish
            df.loc[df['MACD'] < df['MACD_Signal'], 'Signal'] -= 1  # Bearish
            
            # Bollinger Bands signals
            df.loc[df['Close'] < df['BB_Lower'], 'Signal'] += 1  # Oversold
            df.loc[df['Close'] > df['BB_Upper'], 'Signal'] -= 1  # Overbought
            
            # Volume signals
            df.loc[df['Volume'] > df['Volume_MA'] * 1.5, 'Signal_Strength'] += 1  # High volume
            
            signals[ticker] = df[['Close', 'Signal', 'Signal_Strength', 'RSI', 'MA_20']].tail(10)
        
        # Display signals
        for ticker, signal_df in signals.items():
            print(f"\n{ticker} Recent Trading Signals:")
            print("-" * 50)
            print(signal_df.to_string())
            
            # Current recommendation
            latest_signal = signal_df['Signal'].iloc[-1]
            latest_strength = signal_df['Signal_Strength'].iloc[-1]
            
            if latest_signal > 0:
                recommendation = "BUY"
            elif latest_signal < 0:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            print(f"\nCurrent Recommendation: {recommendation}")
            print(f"Signal Strength: {latest_strength}")
    
    def run_complete_analysis(self):
        """Run the complete stock market analysis pipeline"""
        print("🚀 Starting Stock Market Analysis")
        print("=" * 50)
        
        # Step 1: Data Collection
        self.collect_data(use_api=True)
        
        # Step 2: Technical Indicators
        self.calculate_technical_indicators()
        
        # Step 3: Exploratory Data Analysis
        self.exploratory_data_analysis()
        
        # Step 4: Feature Engineering
        self.feature_engineering()
        
        # Step 5: Model Training
        self.train_models()
        
        # Step 6: Model Evaluation
        self.evaluate_models()
        
        # Step 7: Predictions
        print("\n" + "="*60)
        print("NEXT DAY PREDICTIONS")
        print("="*60)
        for ticker in self.tickers:
            self.predict_next_day(ticker)
        
        # Step 8: Trading Signals
        self.generate_trading_signals()
        
        print("\n✅ Analysis Complete!")
        print("=" * 50)

def main():
    """Main function to run the stock market analysis"""
    
    # Initialize analyzer
    analyzer = StockMarketAnalyzer(tickers=['AAPL', 'MSFT', 'NFLX', 'GOOG'], period='3mo')
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    return analyzer

if __name__ == "__main__":
    # Run the analysis
    analyzer = main()
    
    # Additional custom analysis can be added here
    print("\n" + "="*60)
    print("ADDITIONAL INSIGHTS")
    print("="*60)
    
    # Compare performance across stocks
    performance_comparison = {}
    for ticker in analyzer.tickers:
        df = analyzer.technical_indicators[ticker]
        performance_comparison[ticker] = {
            'Total_Return': (df['Cumulative_Return'].iloc[-1] - 1) * 100,
            'Volatility': df['Daily_Return'].std() * np.sqrt(252) * 100,  # Annualized
            'Sharpe_Ratio': (df['Daily_Return'].mean() / df['Daily_Return'].std()) * np.sqrt(252),
            'Max_Drawdown': ((df['Close'] / df['Close'].expanding().max()) - 1).min() * 100
        }
    
    performance_df = pd.DataFrame(performance_comparison).T
    print("\nPerformance Comparison:")
    print(performance_df.round(2))
    
    # Find best and worst performers
    best_return = performance_df['Total_Return'].idxmax()
    worst_return = performance_df['Total_Return'].idxmin()
    best_sharpe = performance_df['Sharpe_Ratio'].idxmax()
    
    print(f"\nBest Performer (Total Return): {best_return} ({performance_df.loc[best_return, 'Total_Return']:.2f}%)")
    print(f"Worst Performer (Total Return): {worst_return} ({performance_df.loc[worst_return, 'Total_Return']:.2f}%)")
    print(f"Best Risk-Adjusted Return (Sharpe): {best_sharpe} ({performance_df.loc[best_sharpe, 'Sharpe_Ratio']:.2f})") 
