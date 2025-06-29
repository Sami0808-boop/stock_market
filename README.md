# Stock Market Analysis Project 📈

A comprehensive Python-based stock market analysis tool that analyzes and compares the performance of Apple (AAPL), Microsoft (MSFT), Netflix (NFLX), and Google (GOOG) using various data science techniques including machine learning.

## 🎯 Project Overview

This project provides a complete analysis of stock market data including:
- **Data Collection**: Automated data gathering from Yahoo Finance API
- **Technical Analysis**: Moving averages, RSI, MACD, Bollinger Bands, and more
- **Exploratory Data Analysis**: Comprehensive visualizations and statistical analysis
- **Machine Learning**: Multiple ML models for price prediction
- **Trading Signals**: Automated signal generation based on technical indicators

## 🚀 Features

### 📊 Data Analysis
- Real-time data collection from Yahoo Finance API
- Historical price data analysis (3 months by default)
- Volume and price trend analysis
- Correlation analysis between different stocks

### 📈 Technical Indicators
- **Moving Averages**: 5, 20, 50-day moving averages
- **RSI (Relative Strength Index)**: Momentum oscillator
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility indicators
- **Volume Analysis**: Volume moving averages and ratios
- **Volatility Metrics**: Rolling standard deviation

### 🤖 Machine Learning Models
- **Linear Regression**: Baseline model for price prediction
- **Random Forest**: Ensemble method for robust predictions
- **Support Vector Regression (SVR)**: Advanced regression technique
- **Decision Tree**: Interpretable tree-based model

### 📊 Visualization
- Interactive Plotly charts
- Comprehensive EDA dashboards
- Performance comparison plots
- Technical indicator visualizations

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd stock-market-analysis
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the analysis**:
```bash
python stock_market_analysis.py
```

## 📋 Usage

### Basic Usage
```python
from stock_market_analysis import StockMarketAnalyzer

# Initialize analyzer
analyzer = StockMarketAnalyzer(tickers=['AAPL', 'MSFT', 'NFLX', 'GOOG'])

# Run complete analysis
analyzer.run_complete_analysis()
```

### Custom Analysis
```python
# Custom tickers and time period
analyzer = StockMarketAnalyzer(
    tickers=['TSLA', 'AMZN', 'NVDA'], 
    period='6mo'
)

# Individual steps
analyzer.collect_data(use_api=True)
analyzer.calculate_technical_indicators()
analyzer.exploratory_data_analysis()
analyzer.feature_engineering()
analyzer.train_models()
analyzer.evaluate_models()

# Get predictions
prediction = analyzer.predict_next_day('AAPL', 'Random Forest')
```

## 📊 Output Examples

### Performance Metrics
```
AAPL Model Performance:
----------------------------------------
Linear Regression    | MSE: 0.000123 | MAE: 0.008234 | R²: 0.2345
Random Forest        | MSE: 0.000098 | MAE: 0.006789 | R²: 0.3456
SVR                  | MSE: 0.000112 | MAE: 0.007123 | R²: 0.2987
Decision Tree        | MSE: 0.000134 | MAE: 0.008567 | R²: 0.2123
```

### Trading Signals
```
AAPL Recent Trading Signals:
--------------------------------------------------
Date       | Close   | Signal | Signal_Strength | RSI   | MA_20
2024-01-15 | 185.92  | 1      | 0               | 45.2  | 182.34
2024-01-16 | 186.01  | 2      | 1               | 48.7  | 182.89
...
```

## 📈 Key Insights

The analysis provides insights into:
- **Trend Analysis**: Price movements and patterns
- **Volatility Assessment**: Risk measurement and comparison
- **Correlation Analysis**: Relationships between different stocks
- **Performance Comparison**: Total returns, Sharpe ratios, and drawdowns
- **Trading Opportunities**: Buy/sell signals based on technical indicators

## 🔧 Customization

### Adding New Technical Indicators
```python
def custom_indicator(self, df):
    # Add your custom indicator here
    df['Custom_Indicator'] = df['Close'].rolling(window=14).mean()
    return df
```

### Adding New ML Models
```python
from sklearn.ensemble import GradientBoostingRegressor

models_to_train = {
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    # ... existing models
}
```

## 📁 Project Structure

```
stock-market-analysis/
├── stock_market_analysis.py    # Main analysis script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── stocks.csv                  # Sample dataset (if using local data)
└── notebooks/                  # Jupyter notebooks (optional)
    ├── eda_notebook.ipynb
    └── ml_notebook.ipynb
```

## 🎓 Learning Objectives

This project demonstrates:
- **Data Science Workflow**: Complete pipeline from data collection to model deployment
- **Financial Analysis**: Technical indicators and market analysis techniques
- **Machine Learning**: Feature engineering, model selection, and evaluation
- **Data Visualization**: Interactive charts and comprehensive dashboards
- **Python Programming**: Object-oriented design and modular code structure

## 📚 Technical Concepts Covered

- **Time Series Analysis**: Handling temporal data and trends
- **Feature Engineering**: Creating predictive features from raw data
- **Model Evaluation**: MSE, MAE, R² metrics and cross-validation
- **Technical Analysis**: Financial indicators and trading signals
- **Data Visualization**: Matplotlib, Seaborn, and Plotly



## 🙏 Acknowledgments

- Yahoo Finance API for providing stock data
- Scikit-learn for machine learning algorithms
- Plotly for interactive visualizations
- Pandas and NumPy for data manipulation
