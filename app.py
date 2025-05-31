from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import json
import logging
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.sentiment_analyzer import SentimentAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://127.0.0.1:5000", 
            "http://localhost:5000",
            "http://127.0.0.1:5500", 
            "http://localhost:5500",
            "http://127.0.0.1:8000", 
            "http://localhost:8000",
            "https://stock-predicter.onrender.com",
            "https://stock-predicter-*.onrender.com"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "supports_credentials": True
    }
})

# Initialize sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()

# Route to serve the main web interface
@app.route('/')
def home():
    try:
        return send_file('web/index.html')
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        return f"Error: {str(e)}", 500

# Route to serve static files (CSS, JS, images)
@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory('web/static', filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {str(e)}")
        return f"Error serving {filename}: {str(e)}", 404

# Explicit routes for CSS and JS files (backup solution)
@app.route('/static/css/style.css')
def serve_css():
    try:
        return send_file('web/static/css/style.css', mimetype='text/css')
    except Exception as e:
        logger.error(f"Error serving CSS: {str(e)}")
        return f"Error serving CSS: {str(e)}", 404

@app.route('/static/js/main.js')
def serve_js():
    try:
        return send_file('web/static/js/main.js', mimetype='application/javascript')
    except Exception as e:
        logger.error(f"Error serving JS: {str(e)}")
        return f"Error serving JS: {str(e)}", 404

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, period=20, num_std=2):
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    return upper_band, middle_band, lower_band

def calculate_atr(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_obv(data):
    close = data['Close']
    volume = data['Volume']
    obv = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    return obv

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': 'Server is running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        period = data.get('period', '2y')
        
        logging.info(f"Received prediction request for {symbol} over {period}")
        
        # Validate inputs
        if not symbol or not period:
            return jsonify({'error': 'Missing symbol or period parameter'}), 400
        
        # Get historical data with timeout protection
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return jsonify({'error': f'Could not fetch data for {symbol}. Please try a different stock.'}), 400
        
        if len(hist) < 30:
            return jsonify({'error': 'Not enough historical data available for prediction'}), 400
            
        # Calculate features with shorter windows for shorter periods
        window_20 = min(20, len(hist) // 2)
        window_50 = min(50, len(hist) // 2)
        
        hist['SMA_20'] = hist['Close'].rolling(window=window_20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=window_50).mean()
        hist['RSI'] = calculate_rsi(hist['Close'], period=min(14, len(hist) // 3))
        hist['MACD'], hist['Signal'] = calculate_macd(hist['Close'], 
                                                     fast=min(12, len(hist) // 4),
                                                     slow=min(26, len(hist) // 3),
                                                     signal=min(9, len(hist) // 5))
        hist['BB_upper'], hist['BB_middle'], hist['BB_lower'] = calculate_bollinger_bands(
            hist['Close'], period=min(20, len(hist) // 2))
        hist['ATR'] = calculate_atr(hist, period=min(14, len(hist) // 3))
        hist['OBV'] = calculate_obv(hist)
        
        # Forward fill NaN values instead of dropping them
        hist = hist.fillna(method='ffill')
        # Backward fill any remaining NaN values
        hist = hist.fillna(method='bfill')
        
        if len(hist) < 20:
            return jsonify({'error': 'Not enough valid data points after processing'}), 400
        
        # Prepare features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 
                   'RSI', 'MACD', 'Signal', 'BB_upper', 'BB_middle', 'BB_lower', 
                   'ATR', 'OBV']
        
        X = hist[features].values
        y = hist['Close'].values
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Add price normalization to prevent bias
        price_scaler = MinMaxScaler()
        y_scaled = price_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Prepare data for LSTM
        sequence_length = 10
        X_lstm = []
        y_lstm = []
        for i in range(len(X_scaled) - sequence_length):
            X_lstm.append(X_scaled[i:(i + sequence_length)])
            y_lstm.append(y_scaled[i + sequence_length])
        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        
        # Split data
        train_size = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
        
        # Split LSTM data
        lstm_train_size = int(len(X_lstm) * 0.8)
        X_lstm_train, X_lstm_test = X_lstm[:lstm_train_size], X_lstm[lstm_train_size:]
        y_lstm_train, y_lstm_test = y_lstm[:lstm_train_size], y_lstm[lstm_train_size:]
        
        # Train models (optimized for speed)
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor(
            n_estimators=50,  # Reduced from 100 for speed
            random_state=42,
            max_depth=3,      # Reduced depth for speed
            min_samples_split=20,
            n_jobs=-1         # Use all CPU cores
        )
        gb_model = GradientBoostingRegressor(
            n_estimators=50,  # Reduced from 200 for speed
            random_state=42,
            max_depth=3,      # Reduced depth for speed
            learning_rate=0.1, # Increased learning rate for faster convergence
            min_samples_split=20,
            subsample=0.8,
            max_features='sqrt'
        )
        
        # Build and train LSTM model (optimized for speed)
        lstm_model = Sequential([
            LSTM(units=16, return_sequences=True, input_shape=(sequence_length, X_scaled.shape[1])),  # Reduced units
            Dropout(0.1),     # Reduced dropout
            LSTM(units=8, return_sequences=False),  # Reduced units
            Dropout(0.1),     # Reduced dropout
            Dense(units=1, activation='linear')
        ])
        lstm_model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )
        lstm_model.fit(
            X_lstm_train, 
            y_lstm_train, 
            epochs=5,         # Reduced from 20 for speed
            batch_size=64,    # Increased batch size for speed
            verbose=0,
            validation_split=0.1
        )
        
        # Train other models
        lr_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Make predictions
        lr_pred = lr_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        lstm_pred = lstm_model.predict(X_lstm_test).flatten()
        
        # Inverse transform predictions back to original scale
        lr_pred = price_scaler.inverse_transform(lr_pred.reshape(-1, 1)).flatten()
        rf_pred = price_scaler.inverse_transform(rf_pred.reshape(-1, 1)).flatten()
        gb_pred = price_scaler.inverse_transform(gb_pred.reshape(-1, 1)).flatten()
        lstm_pred = price_scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
        
        # Ensure all predictions have the same length
        min_length = min(len(lr_pred), len(rf_pred), len(gb_pred), len(lstm_pred))
        lr_pred = lr_pred[-min_length:]
        rf_pred = rf_pred[-min_length:]
        gb_pred = gb_pred[-min_length:]
        lstm_pred = lstm_pred[-min_length:]
        y_test = y[-min_length:]  # Use original y values for metrics
        
        # Calculate metrics
        lr_mse = mean_squared_error(y_test, lr_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        gb_mse = mean_squared_error(y_test, gb_pred)
        gb_r2 = r2_score(y_test, gb_pred)
        gb_mae = mean_absolute_error(y_test, gb_pred)  # Added MAE for Gradient Boosting
        gb_rmse = np.sqrt(gb_mse)  # Added RMSE for Gradient Boosting
        lstm_mse = mean_squared_error(y_test, lstm_pred)
        lstm_r2 = r2_score(y_test, lstm_pred)
        
        # Get current price and predicted price (ensemble of all models)
        current_price = hist['Close'].iloc[-1]
        
        # Prepare latest data for prediction
        latest_data = X_scaled[-sequence_length:].reshape(1, sequence_length, X_scaled.shape[1])
        latest_data_flat = X_scaled[-1:].reshape(1, -1)
        
        # Get predictions from all models
        lr_next = price_scaler.inverse_transform(lr_model.predict(latest_data_flat).reshape(-1, 1))[0][0]
        rf_next = price_scaler.inverse_transform(rf_model.predict(latest_data_flat).reshape(-1, 1))[0][0]
        gb_next = price_scaler.inverse_transform(gb_model.predict(latest_data_flat).reshape(-1, 1))[0][0]
        lstm_next = price_scaler.inverse_transform(lstm_model.predict(latest_data).reshape(-1, 1))[0][0]
        
        # Calculate weighted ensemble prediction with bias correction
        total_r2 = abs(lr_r2) + abs(rf_r2) + abs(gb_r2) + abs(lstm_r2)
        min_weight = 0.1
        lr_weight = max(min_weight, abs(lr_r2) / total_r2)
        rf_weight = max(min_weight, abs(rf_r2) / total_r2)
        gb_weight = max(min_weight, abs(gb_r2) / total_r2)
        lstm_weight = max(min_weight, abs(lstm_r2) / total_r2)
        
        # Normalize weights
        total_weight = lr_weight + rf_weight + gb_weight + lstm_weight
        lr_weight /= total_weight
        rf_weight /= total_weight
        gb_weight /= total_weight
        lstm_weight /= total_weight
        
        # Calculate ensemble prediction
        predicted_price = (
            lr_next * lr_weight +
            rf_next * rf_weight +
            gb_next * gb_weight +
            lstm_next * lstm_weight
        )
        
        # Calculate prediction confidence based on model agreement
        all_predictions = [lr_next, rf_next, gb_next, lstm_next]
        prediction_std = np.std(all_predictions)
        prediction_mean = np.mean(all_predictions)
        
        # Debug logging
        print(f"Individual predictions: LR={lr_next:.2f}, RF={rf_next:.2f}, GB={gb_next:.2f}, LSTM={lstm_next:.2f}")
        print(f"Prediction std: {prediction_std:.4f}, mean: {prediction_mean:.2f}")
        
        # Calculate confidence based on model agreement (improved approach)
        # Higher standard deviation = lower confidence
        max_reasonable_std = current_price * 0.15  # 15% of current price as max std (more realistic)
        confidence_raw = max(0, 1 - (prediction_std / max_reasonable_std))
        
        # Scale confidence more realistically: 25-95% range
        confidence = 0.25 + (confidence_raw * 0.70)  # Maps 0-1 to 25-95%
        confidence = max(0.25, min(0.95, confidence))  # Bound between 25-95%
        
        print(f"Raw confidence: {confidence_raw:.4f}, Final confidence: {confidence:.4f}")
        
        # Add constraints to prevent extreme predictions
        max_change = 0.15
        min_price = current_price * (1 - max_change)
        max_price = current_price * (1 + max_change)
        predicted_price = max(min_price, min(max_price, predicted_price))
        
        expected_change = ((predicted_price - current_price) / current_price) * 100
        
        # Calculate support and resistance levels
        recent_highs = hist['High'].rolling(window=20).max()
        recent_lows = hist['Low'].rolling(window=20).min()
        support_level = recent_lows.iloc[-1]
        resistance_level = recent_highs.iloc[-1]
        
        # Get latest technical indicators
        latest_rsi = hist['RSI'].iloc[-1]
        latest_macd = hist['MACD'].iloc[-1]
        
        # Calculate risk metrics
        returns = hist['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))  # Annualized Sharpe ratio
        max_drawdown = (hist['Close'] / hist['Close'].cummax() - 1).min()  # Maximum drawdown
        
        # Calculate sentiment metrics based on price and volume data
        # Recent price momentum (last 5 days)
        recent_returns = returns.tail(5)
        price_momentum = recent_returns.mean() * 100  # Convert to percentage
        
        # Volume trend (comparing recent volume to 20-day average)
        recent_volume = hist['Volume'].tail(5).mean()
        avg_volume = hist['Volume'].tail(20).mean()
        volume_trend = ((recent_volume / avg_volume) - 1) * 100  # Convert to percentage
        
        # Price volatility trend (comparing recent volatility to historical)
        recent_volatility = returns.tail(5).std() * np.sqrt(252) * 100
        vol_trend = ((recent_volatility / volatility) - 1) * 100
        
        # Calculate overall sentiment score
        # Weighted combination of price momentum, volume trend, and volatility
        overall_sentiment = (
            price_momentum * 0.5 +  # Price momentum has highest weight
            volume_trend * 0.3 +    # Volume trend has medium weight
            (-vol_trend * 0.2)      # Lower volatility is positive, hence negative sign
        )
        
        # Prepare historical data for chart
        dates = hist.index[-min_length:].strftime('%Y-%m-%d').tolist()
        actual_prices = y_test.tolist()
        predicted_prices = ((lr_pred + rf_pred + gb_pred + lstm_pred) / 4).tolist()
        
        return jsonify({
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'expected_change': round(expected_change, 2),
            'prediction_confidence': round(confidence * 100, 2),  # Convert to percentage
            'technical_analysis': {
                'rsi': round(latest_rsi, 2),
                'macd': round(latest_macd, 2),
                'support_level': round(support_level, 2),
                'resistance_level': round(resistance_level, 2)
            },
            'risk_metrics': {
                'volatility': round(volatility * 100, 2),  # Convert to percentage
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown * 100, 2),  # Convert to percentage
                'price_momentum': round(price_momentum, 2),
                'volume_trend': round(volume_trend, 2)
            },
            'sentiment_analysis': {
                'overall_sentiment': round(overall_sentiment, 2),
                'news_sentiment': {
                    'overall_sentiment': {
                        'polarity': round(price_momentum, 2)  # Use price momentum as news sentiment
                    }
                },
                'social_sentiment': {
                    'overall_social_sentiment': round(volume_trend, 2)  # Use volume trend as social sentiment
                }
            },
            'model_performance': {
                'lstm': {
                    'mse': round(lstm_mse, 2),
                    'r2': round(lstm_r2, 2)
                },
                'linear_regression': {
                    'mse': round(lr_mse, 2),
                    'r2': round(lr_r2, 2)
                },
                'random_forest': {
                    'mse': round(rf_mse, 2),
                    'r2': round(rf_r2, 2)
                },
                'gradient_boosting': {
                    'mse': round(gb_mse, 2),
                    'r2': round(gb_r2, 2),
                    'mae': round(gb_mae, 2),
                    'rmse': round(gb_rmse, 2)
                }
            },
            'chart_data': {
                'dates': dates,
                'actual': actual_prices,
                'predicted': predicted_prices
            }
        })
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Get host from environment variable or default to localhost
    host = os.environ.get('HOST', '127.0.0.1')
    
    # Check if running in production
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info(f"Starting Flask server on {host}:{port} (debug={debug})")
    app.run(debug=debug, port=port, host=host) 