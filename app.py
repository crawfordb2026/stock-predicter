from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Finnhub configuration
FINNHUB_API_KEY = "d0tlobhr01qlvahcuok0d0tlobhr01qlvahcuokg"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

@app.route('/')
def home():
    """Serve the main HTML page"""
    try:
        return send_file('web/index.html')
    except Exception as e:
        return f"Error loading page: {str(e)}", 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('web/static', filename)

@app.route('/static/css/style.css')
def serve_css():
    """Serve CSS file specifically"""
    return send_from_directory('web/static/css', 'style.css')

@app.route('/static/js/main.js')
def serve_js():
    """Serve JavaScript file specifically"""
    return send_from_directory('web/static/js', 'main.js')

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(window=period).mean()

def calculate_obv(data):
    """Calculate On Balance Volume"""
    obv = []
    obv_value = 0
    for i in range(len(data)):
        if i == 0:
            obv_value = data['Volume'].iloc[i]
        else:
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv_value += data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv_value -= data['Volume'].iloc[i]
        obv.append(obv_value)
    return pd.Series(obv, index=data.index)

def fetch_finnhub_data(symbol, period):
    """Fetch historical stock data from Finnhub API"""
    try:
        # Calculate date range based on period
        end_date = datetime.now()
        period_days = {
            '3mo': 90, '6mo': 180, '1y': 365, 
            '2y': 730, '5y': 1825, 'max': 2000
        }
        days = period_days.get(period, 730)
        start_date = end_date - timedelta(days=days)
        
        # Convert to Unix timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # Finnhub stock candles endpoint
        url = f"{FINNHUB_BASE_URL}/stock/candle"
        params = {
            'symbol': symbol,
            'resolution': 'D',  # Daily resolution
            'from': start_timestamp,
            'to': end_timestamp,
            'token': FINNHUB_API_KEY
        }
        
        logger.info(f"Fetching Finnhub data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if data is valid
        if data.get('s') != 'ok' or not data.get('c'):
            logger.error(f"Finnhub API error for {symbol}: {data}")
            return None
            
        # Convert to DataFrame
        df_data = {
            'Open': data['o'],
            'High': data['h'], 
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data['v']
        }
        
        # Convert timestamps to dates
        dates = [datetime.fromtimestamp(ts) for ts in data['t']]
        
        df = pd.DataFrame(df_data, index=dates)
        df.index.name = 'Date'
        df = df.sort_index()
        
        logger.info(f"Successfully fetched {len(df)} data points for {symbol} from Finnhub")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching Finnhub data for {symbol}: {str(e)}")
        return None
    except KeyError as e:
        logger.error(f"Data format error from Finnhub for {symbol}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching Finnhub data for {symbol}: {str(e)}")
        return None

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'Server is running', 'message': 'Stock Predictor API is operational'})

@app.route('/debug/<symbol>', methods=['GET'])
def debug_data(symbol):
    """Debug endpoint to test data fetching"""
    try:
        data = fetch_finnhub_data(symbol, '1y')
        if data is not None:
            return jsonify({
                'symbol': symbol,
                'data_points': len(data),
                'date_range': f"{data.index[0].date()} to {data.index[-1].date()}",
                'latest_price': float(data['Close'].iloc[-1]),
                'sample_data': {
                    'first_row': {
                        'date': str(data.index[0].date()),
                        'open': float(data['Open'].iloc[0]),
                        'high': float(data['High'].iloc[0]),
                        'low': float(data['Low'].iloc[0]),
                        'close': float(data['Close'].iloc[0]),
                        'volume': int(data['Volume'].iloc[0])
                    },
                    'last_row': {
                        'date': str(data.index[-1].date()),
                        'open': float(data['Open'].iloc[-1]),
                        'high': float(data['High'].iloc[-1]),
                        'low': float(data['Low'].iloc[-1]),
                        'close': float(data['Close'].iloc[-1]),
                        'volume': int(data['Volume'].iloc[-1])
                    }
                }
            })
        else:
            return jsonify({'error': f'No data available for {symbol}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        period = data.get('period', '2y')
        
        logger.info(f"Received prediction request for {symbol} over {period}")
        
        # Validate inputs
        if not symbol or not period:
            return jsonify({'error': 'Missing symbol or period parameter'}), 400
        
        # Get historical data from Finnhub
        hist = fetch_finnhub_data(symbol, period)
        
        if hist is None or len(hist) == 0:
            return jsonify({'error': f'Unable to fetch data for {symbol}. Please check the symbol and try again.'}), 400
        
        if len(hist) < 30:
            return jsonify({'error': f'Not enough historical data available for {symbol}. Got {len(hist)} data points, need at least 30.'}), 400
            
        # Calculate technical indicators
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
        
        # Fill NaN values
        hist = hist.fillna(method='ffill').fillna(method='bfill')
        
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
        
        price_scaler = MinMaxScaler()
        y_scaled = price_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Prepare LSTM data
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
        
        lstm_train_size = int(len(X_lstm) * 0.8)
        X_lstm_train, X_lstm_test = X_lstm[:lstm_train_size], X_lstm[lstm_train_size:]
        y_lstm_train, y_lstm_test = y_lstm[:lstm_train_size], y_lstm[lstm_train_size:]
        
        # Train models (optimized for speed)
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor(
            n_estimators=50, random_state=42, max_depth=3,
            min_samples_split=20, n_jobs=-1
        )
        gb_model = GradientBoostingRegressor(
            n_estimators=50, random_state=42, max_depth=3,
            learning_rate=0.1, min_samples_split=20,
            subsample=0.8, max_features='sqrt'
        )
        
        # Build LSTM model
        lstm_model = Sequential([
            LSTM(units=16, return_sequences=True, input_shape=(sequence_length, X_scaled.shape[1])),
            Dropout(0.1),
            LSTM(units=8, return_sequences=False),
            Dropout(0.1),
            Dense(units=1, activation='linear')
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=64, verbose=0, validation_split=0.1)
        
        # Train models
        lr_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Make predictions
        lr_pred = lr_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        lstm_pred = lstm_model.predict(X_lstm_test).flatten()
        
        # Inverse transform predictions
        lr_pred = price_scaler.inverse_transform(lr_pred.reshape(-1, 1)).flatten()
        rf_pred = price_scaler.inverse_transform(rf_pred.reshape(-1, 1)).flatten()
        gb_pred = price_scaler.inverse_transform(gb_pred.reshape(-1, 1)).flatten()
        lstm_pred = price_scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
        
        # Ensure consistent lengths
        min_length = min(len(lr_pred), len(rf_pred), len(gb_pred), len(lstm_pred))
        lr_pred = lr_pred[-min_length:]
        rf_pred = rf_pred[-min_length:]
        gb_pred = gb_pred[-min_length:]
        lstm_pred = lstm_pred[-min_length:]
        y_test = y[-min_length:]
        
        # Calculate metrics
        lr_mse = mean_squared_error(y_test, lr_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        gb_mse = mean_squared_error(y_test, gb_pred)
        gb_r2 = r2_score(y_test, gb_pred)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        gb_rmse = np.sqrt(gb_mse)
        lstm_mse = mean_squared_error(y_test, lstm_pred)
        lstm_r2 = r2_score(y_test, lstm_pred)
        
        # Choose best model
        models = {
            'Linear Regression': {'mse': lr_mse, 'r2': lr_r2, 'predictions': lr_pred},
            'Random Forest': {'mse': rf_mse, 'r2': rf_r2, 'predictions': rf_pred},
            'Gradient Boosting': {'mse': gb_mse, 'r2': gb_r2, 'predictions': gb_pred},
            'LSTM': {'mse': lstm_mse, 'r2': lstm_r2, 'predictions': lstm_pred}
        }
        
        best_model_name = min(models.keys(), key=lambda k: models[k]['mse'])
        best_predictions = models[best_model_name]['predictions']
        
        # Calculate current and predicted prices
        current_price = float(hist['Close'].iloc[-1])
        predicted_price = float(best_predictions[-1])
        expected_change = ((predicted_price - current_price) / current_price) * 100
        
        # Calculate technical analysis
        latest_rsi = float(hist['RSI'].iloc[-1]) if not pd.isna(hist['RSI'].iloc[-1]) else 50.0
        latest_macd = float(hist['MACD'].iloc[-1]) if not pd.isna(hist['MACD'].iloc[-1]) else 0.0
        support_level = float(hist['Close'].rolling(window=20).min().iloc[-1])
        resistance_level = float(hist['Close'].rolling(window=20).max().iloc[-1])
        
        # Calculate risk metrics
        returns = hist['Close'].pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252) * 100)  # Annualized volatility
        sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min() * 100)
        
        # Price momentum and volume trend
        price_momentum = float(((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100)
        volume_trend = float(((hist['Volume'].iloc[-5:].mean() - hist['Volume'].iloc[-10:-5].mean()) / hist['Volume'].iloc[-10:-5].mean()) * 100)
        
        # Prepare chart data
        chart_dates = [date.strftime('%Y-%m-%d') for date in hist.index[-min_length:]]
        chart_actual = [float(price) for price in y_test]
        chart_predicted = [float(price) for price in best_predictions]
        
        response_data = {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'expected_change': expected_change,
            'prediction_confidence': float(models[best_model_name]['r2'] * 100),
            'best_model': best_model_name,
            'technical_analysis': {
                'rsi': latest_rsi,
                'macd': latest_macd,
                'support_level': support_level,
                'resistance_level': resistance_level
            },
            'risk_metrics': {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'price_momentum': price_momentum,
                'volume_trend': volume_trend
            },
            'model_performance': {
                'linear_regression': {'mse': float(lr_mse), 'r2': float(lr_r2)},
                'random_forest': {'mse': float(rf_mse), 'r2': float(rf_r2)},
                'gradient_boosting': {'mse': float(gb_mse), 'r2': float(gb_r2), 'mae': float(gb_mae), 'rmse': float(gb_rmse)},
                'lstm': {'mse': float(lstm_mse), 'r2': float(lstm_r2)}
            },
            'chart_data': {
                'dates': chart_dates,
                'actual': chart_actual,
                'predicted': chart_predicted
            },
            'data_source': 'Finnhub',
            'data_points': len(hist)
        }
        
        logger.info(f"Successfully generated prediction for {symbol} using {best_model_name}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Flask server on 0.0.0.0:{port} (debug={debug_mode})")
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 