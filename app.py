import os
# Set TensorFlow environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for compatibility

from flask import Flask, request, jsonify, send_from_directory, send_file
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
import yfinance as yf
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=['*'])  # Allow all origins for Render deployment

# Finnhub configuration
FINNHUB_API_KEY = "d0tlt41r01qlvahcvqv0d0tlt41r01qlvahcvqvg"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

@app.route('/')
def home():
    """Serve the main HTML page"""
    try:
        return send_from_directory('web', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return f"Error: {str(e)}", 500

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

def fetch_stock_data(symbol, period):
    """Fetch historical stock data from Yahoo Finance using yfinance"""
    try:
        logger.info(f"Fetching Yahoo Finance data for {symbol} over {period}")
        
        # Fetch data using yfinance with interval='1d' to ensure daily data
        stock = yf.Ticker(symbol)
        
        # Get today's date
        today = pd.Timestamp.now().date()
        
        # Fetch data with end date as today and retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = stock.history(period=period, interval='1d', timeout=30)
                if not df.empty:
                    break
                logger.warning(f"Empty data on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                
        if df.empty:
            logger.error(f"No data found for symbol {symbol} after {max_retries} attempts")
            return None
            
        # If the last date in our data is not today, try to fetch one more day
        if df.index[-1].date() < today:
            try:
                extra_data = stock.history(period='1d', interval='1d', timeout=30)
                if not extra_data.empty:
                    df = pd.concat([df, extra_data])
                    df = df[~df.index.duplicated(keep='last')]  # Remove any duplicates
            except Exception as e:
                logger.warning(f"Could not fetch latest data: {str(e)}")
            
        logger.info(f"Successfully fetched {len(df)} days of data for {symbol} from Yahoo Finance")
        logger.info(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'Server is running', 'message': 'Stock Predictor API is operational'})

@app.route('/debug/<symbol>', methods=['GET'])
def debug_data(symbol):
    """Debug endpoint to test data fetching"""
    try:
        data = fetch_stock_data(symbol, '1y')
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
        # Add request validation
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        symbol = data.get('symbol', 'AAPL').upper().strip()
        period = data.get('period', '2y')
        
        logger.info(f"Received prediction request for {symbol} over {period}")
        
        # Validate inputs
        if not symbol or len(symbol) > 10:
            return jsonify({'error': 'Invalid symbol parameter'}), 400
            
        if period not in ['1y', '2y', '5y', 'max']:
            return jsonify({'error': 'Invalid period parameter'}), 400
        
        # Get historical data from Yahoo Finance
        hist = fetch_stock_data(symbol, period)
        
        if hist is None or len(hist) == 0:
            return jsonify({'error': f'Unable to fetch data for {symbol}. Please check the symbol and try again.'}), 400
        
        # Calculate technical indicators
        hist['SMA_5'] = hist['Close'].rolling(window=5).mean()
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = ema12 - ema26
        hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        hist['BB_middle'] = hist['Close'].rolling(window=20).mean()
        bb_std = hist['Close'].rolling(window=20).std()
        hist['BB_upper'] = hist['BB_middle'] + (bb_std * 2)
        hist['BB_lower'] = hist['BB_middle'] - (bb_std * 2)
        hist['ATR'] = calculate_atr(hist)
        hist['OBV'] = calculate_obv(hist)
        hist['Price_Momentum'] = hist['Close'].pct_change(periods=5) * 100
        hist['Volume_MA_20'] = hist['Volume'].rolling(window=20).mean()
        hist['Volume_Trend'] = ((hist['Volume'] - hist['Volume_MA_20']) / hist['Volume_MA_20']) * 100
        returns = hist['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = float(drawdown.min() * 100)
        current_price = float(hist['Close'].iloc[-1])
        current_rsi = float(hist['RSI'].iloc[-1])
        current_macd = float(hist['MACD'].iloc[-1])
        current_momentum = float(hist['Price_Momentum'].iloc[-1])
        current_volume_trend = float(hist['Volume_Trend'].iloc[-1])
        # Prepare features for ML models
        hist = hist.dropna()
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'Signal', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'OBV', 'Price_Momentum', 'Volume_Trend']
        if len(hist) < 30:
            return jsonify({'error': 'Not enough data to make a prediction.'}), 400
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        X = hist[feature_cols].values
        y = hist['Close'].values
        
        # Scale both features and target
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_scaled = X_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))
        
        # Use last row for prediction
        X_pred = X_scaled[-1].reshape(1, -1)
        
        # 1. Linear Regression
        lr = LinearRegression()
        lr.fit(X_scaled[:-1], y_scaled[:-1])
        lr_pred = y_scaler.inverse_transform(lr.predict(X_pred).reshape(-1, 1))[0][0]
        
        # 2. Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_scaled[:-1], y_scaled[:-1])
        rf_pred = y_scaler.inverse_transform(rf.predict(X_pred).reshape(-1, 1))[0][0]
        
        # 3. Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
        gb.fit(X_scaled[:-1], y_scaled[:-1])
        gb_pred = y_scaler.inverse_transform(gb.predict(X_pred).reshape(-1, 1))[0][0]
        
        # 4. LSTM
        seq_len = 10
        X_lstm = []
        y_lstm = []
        for i in range(seq_len, len(X_scaled)):
            X_lstm.append(X_scaled[i-seq_len:i])
            y_lstm.append(y_scaled[i])
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        
        lstm_model = Sequential([
            LSTM(32, input_shape=(seq_len, X_lstm.shape[2]), return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X_lstm[:-1], y_lstm[:-1], epochs=5, batch_size=8, verbose=0)
        
        X_lstm_pred = X_scaled[-seq_len:].reshape(1, seq_len, X_lstm.shape[2])
        lstm_pred = y_scaler.inverse_transform(lstm_model.predict(X_lstm_pred))[0][0]
        
        # Calculate predictions and validate them
        predictions = [lr_pred, rf_pred, gb_pred, lstm_pred]
        
        # Remove any extreme outliers (more than 20% from current price)
        valid_predictions = [p for p in predictions if abs(p - current_price) / current_price < 0.20]
        
        if not valid_predictions:
            # If all predictions are invalid, use a more conservative approach
            recent_returns = hist['Close'].pct_change().tail(20).mean()
            predicted_price = current_price * (1 + recent_returns)
            confidence = 30.0  # Low confidence when using fallback
        else:
            # Use the average of valid predictions
            predicted_price = np.mean(valid_predictions)
            
            # Calculate confidence based on:
            # 1. Model agreement (how close predictions are to each other)
            # 2. Recent prediction accuracy
            # 3. Market volatility
            
            # 1. Model agreement (0-40 points)
            std_dev = np.std(valid_predictions)
            mean_price = np.mean(valid_predictions)
            agreement_score = 40 * (1 - min(1, std_dev / mean_price))
            
            # 2. Recent prediction accuracy (0-30 points)
            # Use last 5 days to evaluate model accuracy
            recent_actual = hist['Close'].tail(5).values
            recent_pred = []
            for i in range(5):
                if i < len(X_scaled) - 1:
                    recent_pred.append(np.mean([
                        y_scaler.inverse_transform(lr.predict(X_scaled[i:i+1]).reshape(-1, 1))[0][0],
                        y_scaler.inverse_transform(rf.predict(X_scaled[i:i+1]).reshape(-1, 1))[0][0],
                        y_scaler.inverse_transform(gb.predict(X_scaled[i:i+1]).reshape(-1, 1))[0][0]
                    ]))
            
            if len(recent_pred) > 0:
                accuracy = 1 - np.mean(np.abs(np.array(recent_pred) - recent_actual) / recent_actual)
                accuracy_score = 30 * max(0, accuracy)
            else:
                accuracy_score = 15  # Default middle score if not enough data
            
            # 3. Market volatility (0-30 points)
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)
            volatility_score = 30 * (1 - min(1, volatility))
            
            # Total confidence score (30-95 range)
            confidence = min(95, max(30, agreement_score + accuracy_score + volatility_score))
            
            # Add debug logging
            logger.info(f"Confidence components - Agreement: {agreement_score:.1f}, Accuracy: {accuracy_score:.1f}, Volatility: {volatility_score:.1f}")
            logger.info(f"Final confidence score: {confidence:.1f}")
            
            # Ensure prediction is within reasonable bounds
            max_change = 0.05  # 5% maximum change
            predicted_price = np.clip(
                predicted_price,
                current_price * (1 - max_change),
                current_price * (1 + max_change)
            )
            
            # Calculate the expected change percentage
            expected_change = ((predicted_price - current_price) / current_price) * 100
            
            # Prepare chart data - get the most recent 30 days
            recent_data = hist.tail(30)
            chart_dates = recent_data.index.strftime('%Y-%m-%d').tolist()
            actual_prices = recent_data['Close'].tolist()
            
            # Add the prediction as a single point
            # Find the next trading day after the last historical date
            last_date = recent_data.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            
            # Keep adding days until we find a weekday (0-4 are Monday-Friday)
            while next_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
                next_date += pd.Timedelta(days=1)
            
            # Format the next trading day
            next_date_str = next_date.strftime('%Y-%m-%d')
            
            # Debug log to verify dates
            logger.info(f"Last historical date: {last_date.strftime('%Y-%m-%d')}")
            logger.info(f"Next trading day: {next_date_str}")
            
            # Add the prediction date and price
            chart_dates.append(next_date_str)
            actual_prices.append(None)  # No actual price for the prediction date
            
            # Create prediction line with only the last point
            historical_predictions = [None] * (len(actual_prices) - 1)  # None for all historical points
            historical_predictions.append(predicted_price)  # Add the prediction as the last point
            
            # Debug log the final chart dates
            logger.info(f"Chart dates: {chart_dates}")
            
            response_data = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'expected_change': expected_change,
                'prediction_confidence': float(confidence),
                'best_model': 'Ensemble (LR, RF, GB, LSTM)',
                'technical_analysis': {
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'support_level': float(hist['BB_lower'].iloc[-1]),
                    'resistance_level': float(hist['BB_upper'].iloc[-1])
                },
                'risk_metrics': {
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': max_drawdown,
                    'price_momentum': current_momentum,
                    'volume_trend': current_volume_trend
                },
                'model_performance': {
                    'model_agreement': float(agreement_score),
                    'recent_accuracy': float(accuracy_score),
                    'volatility_impact': float(volatility_score)
                },
                'chart_data': {
                    'dates': chart_dates,
                    'actual': [float(p) if p is not None else None for p in actual_prices],
                    'predicted': [float(p) if p is not None else None for p in historical_predictions]
                },
                'data_source': 'Yahoo Finance',
                'data_points': len(hist),
                'note': f'Prediction based on {period} of historical data using ensemble of 4 models.'
            }
            logger.info(f"Successfully generated prediction for {symbol}")
            return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test that we can import all required modules
        import yfinance
        import pandas
        import numpy
        import sklearn
        import tensorflow
        
        return jsonify({
            'status': 'healthy',
            'message': 'Stock Predictor API is operational',
            'dependencies': 'all modules loaded successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting Flask server on 0.0.0.0:{port} (debug=False)")
    app.run(host='0.0.0.0', port=port, debug=False) 