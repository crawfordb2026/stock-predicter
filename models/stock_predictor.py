import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import ta
from textblob import TextBlob
import yfinance as yf
import logging

class EnhancedStockPredictor:
    def __init__(self, symbol, period='1y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.models = {}
        self.scalers = {}
        self.logger = logging.getLogger(__name__)

    def fetch_data(self):
        """Fetch and prepare stock data with technical indicators"""
        try:
            # Fetch stock data
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            
            # Add technical indicators
            self._add_technical_indicators()
            
            # Add market indicators
            self._add_market_indicators()
            
            return True
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            return False

    def _add_technical_indicators(self):
        """Add technical indicators to the dataset"""
        # Trend indicators
        self.data['SMA_20'] = ta.trend.sma_indicator(self.data['Close'], window=20)
        self.data['SMA_50'] = ta.trend.sma_indicator(self.data['Close'], window=50)
        self.data['EMA_20'] = ta.trend.ema_indicator(self.data['Close'], window=20)
        
        # Momentum indicators
        self.data['RSI'] = ta.momentum.rsi(self.data['Close'])
        self.data['MACD'] = ta.trend.macd_diff(self.data['Close'])
        self.data['Stoch'] = ta.momentum.stoch(self.data['High'], self.data['Low'], self.data['Close'])
        
        # Volatility indicators
        self.data['BB_upper'], self.data['BB_middle'], self.data['BB_lower'] = ta.volatility.bollinger_bands(self.data['Close'])
        self.data['ATR'] = ta.volatility.average_true_range(self.data['High'], self.data['Low'], self.data['Close'])
        
        # Volume indicators
        self.data['OBV'] = ta.volume.on_balance_volume(self.data['Close'], self.data['Volume'])
        self.data['MFI'] = ta.volume.money_flow_index(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'])

    def _add_market_indicators(self):
        """Add market indicators and correlations"""
        # Fetch S&P 500 data
        sp500 = yf.Ticker('^GSPC')
        sp500_data = sp500.history(period=self.period)
        
        # Calculate correlation
        self.data['SP500_Correlation'] = self.data['Close'].rolling(window=20).corr(sp500_data['Close'])
        
        # Add VIX data
        vix = yf.Ticker('^VIX')
        vix_data = vix.history(period=self.period)
        self.data['VIX'] = vix_data['Close']

    def prepare_data(self, sequence_length=10):
        """Prepare data for model training"""
        # Select features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 
                   'EMA_20', 'RSI', 'MACD', 'Stoch', 'BB_upper', 'BB_middle', 
                   'BB_lower', 'ATR', 'OBV', 'MFI', 'SP500_Correlation', 'VIX']
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(self.data) - sequence_length):
            X.append(self.data[features].iloc[i:(i + sequence_length)].values)
            y.append(self.data['Close'].iloc[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test

    def build_lstm_model(self, input_shape):
        """Build and compile LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_models(self, X_train, y_train):
        """Train multiple models"""
        # LSTM Model
        lstm_model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        self.models['lstm'] = lstm_model
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        self.models['random_forest'] = rf_model
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        self.models['gradient_boosting'] = gb_model
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        self.models['linear_regression'] = lr_model

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and return performance metrics"""
        results = {}
        
        for name, model in self.models.items():
            if name == 'lstm':
                predictions = model.predict(X_test)
            else:
                predictions = model.predict(X_test.reshape(X_test.shape[0], -1))
            
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            results[name] = {
                'MSE': mse,
                'R2': r2,
                'predictions': predictions
            }
        
        return results

    def predict_next_day(self, confidence_threshold=0.7):
        """Make ensemble prediction with confidence score"""
        if not self.models:
            raise ValueError("Models not trained. Call train_models first.")
        
        # Get the last sequence of data
        last_sequence = self.data.iloc[-10:][self.features].values
        last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        
        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            if name == 'lstm':
                pred = model.predict(last_sequence)[0][0]
            else:
                pred = model.predict(last_sequence.reshape(1, -1))[0]
            predictions.append(pred)
        
        # Calculate ensemble prediction
        ensemble_prediction = np.mean(predictions)
        
        # Calculate confidence score based on model agreement
        confidence = 1 - (np.std(predictions) / np.mean(predictions))
        
        if confidence < confidence_threshold:
            self.logger.warning(f"Low confidence prediction: {confidence}")
        
        return {
            'prediction': ensemble_prediction,
            'confidence': confidence,
            'model_predictions': dict(zip(self.models.keys(), predictions))
        }

    def get_support_resistance(self):
        """Calculate support and resistance levels"""
        # Simple implementation using recent highs and lows
        recent_highs = self.data['High'].rolling(window=20).max()
        recent_lows = self.data['Low'].rolling(window=20).min()
        
        resistance = recent_highs.iloc[-1]
        support = recent_lows.iloc[-1]
        
        return {
            'support': support,
            'resistance': resistance
        }

    def get_risk_metrics(self):
        """Calculate risk metrics"""
        returns = self.data['Close'].pct_change()
        
        metrics = {
            'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': (self.data['Close'] / self.data['Close'].cummax() - 1).min()
        }
        
        return metrics 