# Stock Price Predictor using Time Series Analysis
# This project shows how to grab stock data, clean it up, and build ML models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, symbol, period='2y'):
        """
        Set up the Stock Predictor
        
        Args:
            symbol (str): Stock symbol (like 'AAPL', 'GOOGL')
            period (str): How much data to grab ('1y', '2y', '5y', 'max')
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.scaler = MinMaxScaler()
        self.model = None
        
    def fetch_data(self):
        """Grab stock data from Yahoo Finance (it's free!)"""
        try:
            stock = yf.Ticker(self.symbol)
            data = stock.history(period=self.period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            self.data = data
            print(f"Successfully fetched {len(data)} days of data for {self.symbol}")
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def create_features(self):
        """Create technical indicators and features that might help predict prices"""
        if self.data is None:
            print("No data available. Please fetch data first.")
            return
        
        df = self.data.copy()
        
        # Basic price features that traders care about
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Volume'] = df['Close'] * df['Volume']
        
        # Price momentum (how much it moved in the last 5 days)
        df['Price_Momentum'] = df['Close'].pct_change(periods=5) * 100
        
        # Volume trend (is trading picking up or slowing down?)
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Trend'] = ((df['Volume'] - df['Volume_MA_20']) / df['Volume_MA_20']) * 100
        
        # Moving averages (smooth out the noise)
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI - tells us if the stock is overbought or oversold
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands - show if price is outside normal range
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = rolling_mean + (rolling_std * 2)
        df['BB_Lower'] = rolling_mean - (rolling_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators (is there unusual trading activity?)
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Look at previous days' prices (maybe there's a pattern)
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        
        # What we're trying to predict (tomorrow's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        self.data = df
        return df
    
    def prepare_data(self, test_size=0.2):
        """Get the data ready for machine learning"""
        if self.data is None:
            print("No data available. Please create features first.")
            return None, None, None, None
        
        # Pick the features we think will be useful
        feature_columns = [
            'Open', 'High', 'Low', 'Volume', 'Price_Change', 'High_Low_Pct',
            'MA_5', 'MA_10', 'MA_20', 'RSI', 'BB_Position', 'Volume_Ratio',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
            'Price_Momentum', 'Volume_Trend'
        ]
        
        # Clean up any missing data
        df_clean = self.data.dropna()
        
        X = df_clean[feature_columns]
        y = df_clean['Target']
        
        # Split chronologically (newest data for testing, like real life)
        split_index = int(len(df_clean) * (1 - test_size))
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Scale the features so they all play nice together
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train a couple different models and see which works better"""
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        trained_models = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"Trained {name}")
        
        return trained_models
    
    def evaluate_models(self, models, X_test, y_test):
        """See how well our models actually perform"""
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'RMSE': np.sqrt(mse),
                'predictions': y_pred
            }
            
            print(f"\n{name} Performance:")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {np.sqrt(mse):.4f}")
        
        return results
    
    def plot_predictions(self, y_test, predictions, model_name):
        """Make a nice chart showing actual vs predicted prices"""
        plt.figure(figsize=(12, 6))
        
        # Convert to pandas Series for easier plotting
        y_test_series = pd.Series(y_test.values, index=range(len(y_test)))
        pred_series = pd.Series(predictions, index=range(len(predictions)))
        
        plt.plot(y_test_series.index, y_test_series.values, label='Actual', alpha=0.7)
        plt.plot(pred_series.index, pred_series.values, label='Predicted', alpha=0.7)
        
        plt.title(f'{self.symbol} Stock Price Prediction - {model_name}')
        plt.xlabel('Time (Days)')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model, feature_names):
        """Show which features the model thinks are most important"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance')
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
    
    def predict_next_day(self, model):
        """Predict next day's stock price"""
        if self.data is None:
            print("No data available.")
            return None
        
        # Get the latest data point
        latest_data = self.data.iloc[-1:].copy()
        
        feature_columns = [
            'Open', 'High', 'Low', 'Volume', 'Price_Change', 'High_Low_Pct',
            'MA_5', 'MA_10', 'MA_20', 'RSI', 'BB_Position', 'Volume_Ratio',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5'
        ]
        
        X_latest = latest_data[feature_columns].dropna()
        
        if len(X_latest) == 0:
            print("Not enough data for prediction.")
            return None
        
        X_latest_scaled = self.scaler.transform(X_latest)
        prediction = model.predict(X_latest_scaled)[0]
        
        current_price = self.data['Close'].iloc[-1]
        change_pct = ((prediction - current_price) / current_price) * 100
        
        print(f"\nNext Day Prediction for {self.symbol}:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${prediction:.2f}")
        print(f"Expected Change: {change_pct:+.2f}%")
        
        return prediction

# Example usage and demonstration
def main():
    # Initialize predictor for Apple stock
    predictor = StockPredictor('AAPL', period='2y')
    
    print("=== Stock Price Predictor Demo ===\n")
    
    # Step 1: Fetch data
    print("1. Fetching stock data...")
    data = predictor.fetch_data()
    
    if data is not None:
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Step 2: Create features
    print("\n2. Creating technical indicators and features...")
    predictor.create_features()
    
    # Step 3: Prepare data
    print("\n3. Preparing data for training...")
    X_train, X_test, y_train, y_test = predictor.prepare_data()
    
    if X_train is not None:
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
    
    # Step 4: Train models
    print("\n4. Training models...")
    models = predictor.train_models(X_train, y_train)
    
    # Step 5: Evaluate models
    print("\n5. Evaluating models...")
    results = predictor.evaluate_models(models, X_test, y_test)
    
    # Step 6: Visualize results
    print("\n6. Creating visualizations...")
    
    # Plot predictions for the best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['R2'])
    best_model = models[best_model_name]
    
    predictor.plot_predictions(y_test, results[best_model_name]['predictions'], best_model_name)
    
    # Plot feature importance
    if best_model_name == 'Random Forest':
        feature_names = [
            'Open', 'High', 'Low', 'Volume', 'Price_Change', 'High_Low_Pct',
            'MA_5', 'MA_10', 'MA_20', 'RSI', 'BB_Position', 'Volume_Ratio',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
            'Price_Momentum', 'Volume_Trend'
        ]
        predictor.plot_feature_importance(best_model, feature_names)
    
    # Step 7: Make next day prediction
    print("\n7. Making next day prediction...")
    predicted_price = predictor.predict_next_day(best_model)
    
    # Prepare response data
    response_data = {
        'current_price': data['Close'].iloc[-1],
        'predicted_price': predicted_price,
        'expected_change': ((predicted_price - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100,
        'prediction_confidence': max(results[best_model_name]['R2'], 0.5),
        'technical_analysis': {
            'rsi': predictor.data['RSI'].iloc[-1],
            'macd': 0,  # Assuming MACD is not available in the data
            'support_level': 0,  # Assuming support level is not available in the data
            'resistance_level': 0,  # Assuming resistance level is not available in the data
        },
        'risk_metrics': {
            'volatility': predictor.data['Price_Change'].std() * 100,
            'sharpe_ratio': 0,  # Assuming sharpe ratio is not available in the data
            'max_drawdown': 0,  # Assuming max drawdown is not available in the data
            'price_momentum': float(predictor.data['Price_Momentum'].iloc[-1]),
            'volume_trend': float(predictor.data['Volume_Trend'].iloc[-1])
        },
        'historical_data': data.to_dict('records'),
        'model_performance': results[best_model_name]
    }
    
    # Debug print
    print("\nRisk Metrics being sent:")
    print(f"Price Momentum: {response_data['risk_metrics']['price_momentum']}")
    print(f"Volume Trend: {response_data['risk_metrics']['volume_trend']}")
    
    return predictor, results, response_data

if __name__ == "__main__":
    # Run the demo
    predictor, results, response_data = main()
    
    print("\n=== Project Complete! ===")
    print("This stock predictor demonstrates:")
    print("• Data collection from APIs")
    print("• Feature engineering with technical indicators")
    print("• Time series data handling")
    print("• Model training and evaluation")
    print("• Data visualization")
    print("• Making future predictions")