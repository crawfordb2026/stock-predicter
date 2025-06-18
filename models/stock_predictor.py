import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# import ta  # COMMENTED OUT - Could potentially make network calls
# from textblob import TextBlob  # COMMENTED OUT - Could make network calls for language processing
# import yfinance as yf  # COMMENTED OUT - Using simulated data instead
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
        """Grab stock data and add all the fancy technical indicators"""
        try:
            # COMMENTED OUT - All external API calls removed, using simulated data instead
            # Get the raw stock data
            # stock = yf.Ticker(self.symbol)
            # self.data = stock.history(period=self.period)
            
            # Calculate all the technical indicators traders love
            # self._add_technical_indicators()
            
            # Add broader market context
            # self._add_market_indicators()
            
            print(f"Note: Using simulated data for {self.symbol} instead of external APIs")
            return False  # Indicate that no real data was fetched
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            return False

    def _add_technical_indicators(self):
        """Calculate all the technical indicators that might help predict prices"""
        # COMMENTED OUT - ta library could potentially make network calls
        # Trend indicators (which way is it going?)
        # self.data['SMA_20'] = ta.trend.sma_indicator(self.data['Close'], window=20)
        # self.data['SMA_50'] = ta.trend.sma_indicator(self.data['Close'], window=50)
        # self.data['EMA_20'] = ta.trend.ema_indicator(self.data['Close'], window=20)
        
        # Momentum indicators (how fast is it moving?)
        # self.data['RSI'] = ta.momentum.rsi(self.data['Close'])
        # self.data['MACD'] = ta.trend.macd_diff(self.data['Close'])
        # self.data['Stoch'] = ta.momentum.stoch(self.data['High'], self.data['Low'], self.data['Close'])
        
        # Volatility indicators (how wild is it getting?)
        # self.data['BB_upper'], self.data['BB_middle'], self.data['BB_lower'] = ta.volatility.bollinger_bands(self.data['Close'])
        # self.data['ATR'] = ta.volatility.average_true_range(self.data['High'], self.data['Low'], self.data['Close'])
        
        # Volume indicators (how much interest is there?)
        # self.data['OBV'] = ta.volume.on_balance_volume(self.data['Close'], self.data['Volume'])
        # self.data['MFI'] = ta.volume.money_flow_index(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'])
        
        print("Note: Technical indicators disabled to avoid external dependencies")

    def _add_market_indicators(self):
        """Add context from the broader market"""
        # Get S&P 500 data for market context
        # sp500 = yf.Ticker('^GSPC')
        # sp500_data = sp500.history(period=self.period)
        
        # See how correlated this stock is with the overall market
        # self.data['SP500_Correlation'] = self.data['Close'].rolling(window=20).corr(sp500_data['Close'])
        
        # Add fear index (VIX) - when this spikes, everything goes crazy
        # vix = yf.Ticker('^VIX')
        # vix_data = vix.history(period=self.period)
        # self.data['VIX'] = vix_data['Close']

    def prepare_data(self, sequence_length=10):
        """Get the data ready for our machine learning models"""
        # Pick all the features we think matter
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 
                   'EMA_20', 'RSI', 'MACD', 'Stoch', 'BB_upper', 'BB_middle', 
                   'BB_lower', 'ATR', 'OBV', 'MFI', 'SP500_Correlation', 'VIX']
        
        # Create sequences for the LSTM (it needs to see patterns over time)
        X, y = [], []
        for i in range(len(self.data) - sequence_length):
            X.append(self.data[features].iloc[i:(i + sequence_length)].values)
            y.append(self.data['Close'].iloc[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into training and testing (chronologically, like real life)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test

    def build_lstm_model(self, input_shape):
        """Build our fancy neural network for time series prediction"""
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
        """Train our ensemble of different models"""
        # LSTM - the neural network that's good with time series
        lstm_model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        self.models['lstm'] = lstm_model
        
        # Random Forest - good at finding complex patterns
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        self.models['random_forest'] = rf_model
        
        # Gradient Boosting - builds on mistakes of previous models
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        self.models['gradient_boosting'] = gb_model
        
        # Linear Regression - simple but often effective baseline
        lr_model = LinearRegression()
        lr_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        self.models['linear_regression'] = lr_model

    def evaluate_models(self, X_test, y_test):
        """See how well each of our models actually performs"""
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
        """Make our best guess at tomorrow's price using all models"""
        if not self.models:
            raise ValueError("Models not trained. Call train_models first.")
        
        # Get the most recent data sequence
        last_sequence = self.data.iloc[-10:][self.features].values
        last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        
        # Ask each model for their opinion
        predictions = []
        for name, model in self.models.items():
            if name == 'lstm':
                pred = model.predict(last_sequence)[0][0]
            else:
                pred = model.predict(last_sequence.reshape(1, -1))[0]
            predictions.append(pred)
        
        # Average all the predictions (wisdom of crowds)
        ensemble_prediction = np.mean(predictions)
        
        # See how much the models agree (higher agreement = higher confidence)
        confidence = 1 - (np.std(predictions) / np.mean(predictions))
        
        if confidence < confidence_threshold:
            self.logger.warning(f"Low confidence prediction: {confidence}")
        
        return {
            'prediction': ensemble_prediction,
            'confidence': confidence,
            'model_predictions': dict(zip(self.models.keys(), predictions))
        }

    def get_support_resistance(self):
        """Figure out the support and resistance levels traders watch"""
        # Look at recent highs and lows to find key levels
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