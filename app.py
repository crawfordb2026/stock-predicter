import os
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Gotta set these TensorFlow settings before importing it, or it'll spam the console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Keep TensorFlow quiet 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off some optimizations that can break things

# Additional cloud platform optimizations
if os.environ.get('RENDER') or os.environ.get('HEROKU') or os.environ.get('RAILWAY'):
    # Running on cloud platform - be extra careful with TensorFlow
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Even quieter on cloud
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only on cloud

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

#set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=['*'])  #let everyone access this for now (needed for Render)

@app.route('/')
def home():
    """Just serve up our main page"""
    try:
        return send_from_directory('web', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Handle all the CSS, JS, and other static files"""
    return send_from_directory('web/static', filename)

@app.route('/static/css/style.css')
def serve_css():
    """Make sure our CSS loads properly"""
    return send_from_directory('web/static/css', 'style.css')

@app.route('/static/js/main.js')
def serve_js():
    """Make sure our JavaScript loads properly"""
    return send_from_directory('web/static/js', 'main.js')

def calculate_rsi(prices, period=14):
    """Calculate RSI - basically tells us if a stock is overbought or oversold"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD helps us spot trend changes - super useful for predictions"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Bollinger Bands show us price ranges - helps identify breakouts"""
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def calculate_atr(data, period=14):
    """Average True Range - shows us how volatile a stock has been"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(window=period).mean()

def calculate_obv(data):
    """On Balance Volume - helps us see if money is flowing in or out"""
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
    """Grab historical stock data - using simulated data for the demo"""
    try:
        logger.info(f"Fetching data for {symbol} over {period}")
        
        # Figure out how many days we need
        if period == '1y':
            days = 365
        elif period == '2y':
            days = 730
        elif period == '5y':
            days = 1825
        else:
            days = 365
            
        # Create our date range
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.Timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Only keep business days (no weekends)
        business_days = dates[dates.dayofweek < 5]
        
        # Base prices for different stocks (roughly realistic)
        symbol_base_prices = {
            'AAPL': 180.0, 'GOOGL': 2800.0, 'MSFT': 340.0, 'AMZN': 3200.0,
            'TSLA': 800.0, 'META': 320.0, 'NFLX': 450.0, 'NVDA': 900.0,
            'SPY': 420.0, 'QQQ': 380.0, 'IWM': 200.0, 'VTI': 240.0
        }
        
        base_price = symbol_base_prices.get(symbol, 150.0)
        
        # Create realistic-looking price data
        np.random.seed(hash(symbol) % 2**32)
        
        n_days = len(business_days)
        prices = [base_price]
        
        # Generate price movements that actually look like real stocks
        for i in range(1, n_days):
            # Add some seasonal trends
            trend = 0.0002 * np.sin(2 * np.pi * i / 252)  # Annual cycle
            
            # Random daily movement
            daily_return = np.random.normal(trend, 0.02)
            
            # Make volatility cluster (like real markets)
            if i > 5:
                recent_vol = np.std([np.log(prices[j]/prices[j-1]) for j in range(max(1, i-5), i)])
                daily_return *= (1 + recent_vol * 2)
            
            new_price = prices[-1] * (1 + daily_return)
            new_price = max(new_price, base_price * 0.3)  # don't let it crash too hard
            new_price = min(new_price, base_price * 3.0)   # Don't let it moon too hard
            prices.append(new_price)
        
        # Build the full OHLC dataset
        data = []
        for i, date in enumerate(business_days):
            close = prices[i]
            
            # Create realistic intraday movement
            daily_range = abs(np.random.normal(0, 0.015))
            high = close * (1 + daily_range)
            low = close * (1 - daily_range)
            
            # Make sure the open makes sense
            if i == 0:
                open_price = close * np.random.uniform(0.995, 1.005)
            else:
                open_price = prices[i-1] * np.random.uniform(0.998, 1.002)
            
            # Fix any OHLC inconsistencies
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume (more volume when price moves a lot)
            price_change = abs(close - (prices[i-1] if i > 0 else close))
            base_volume = 1000000 + hash(f"{symbol}{i}") % 5000000
            volume_multiplier = 1 + (price_change / close) * 10
            volume = int(base_volume * volume_multiplier)
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        # Put it all in a nice DataFrame
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        logger.info(f"Successfully retrieved {len(df)} days of data for {symbol}")
        logger.info(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
        return df
        
    except Exception as e:
        logger.error(f"Error retrieving data for {symbol}: {str(e)}")
        return None

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'Server is running', 'message': 'Stock Predictor API is operational'})

@app.route('/debug/<symbol>', methods=['GET'])
def debug_data(symbol):
    """Quick debug check to see if our data fetching works"""
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
        logger.info("=== PREDICTION REQUEST STARTED ===")
        
        # Make sure they sent us proper JSON
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        symbol = data.get('symbol', 'AAPL').upper().strip()
        period = data.get('period', '2y')
        
        logger.info(f"Received prediction request for {symbol} over {period}")
        logger.info(f"Running on environment: {'cloud' if os.environ.get('RENDER') else 'local'}")
        
        # Basic input validation
        if not symbol or len(symbol) > 10:
            return jsonify({'error': 'Invalid symbol parameter'}), 400
            
        if period not in ['1y', '2y', '5y', 'max']:
            return jsonify({'error': 'Invalid period parameter'}), 400
        
        # Clean the symbol to prevent any injection issues
        symbol = ''.join(c for c in symbol if c.isalnum()).upper()
        if not symbol:
            return jsonify({'error': 'Symbol contains no valid characters'}), 400
        
        logger.info("Input validation passed")
        
        # Grab the historical data
        logger.info("Fetching historical data...")
        hist = fetch_stock_data(symbol, period)
        
        if hist is None or len(hist) == 0:
            logger.error(f"Failed to fetch data for {symbol}")
            return jsonify({'error': f'Unable to fetch data for {symbol}. Please check the symbol and try again.'}), 400
        
        logger.info(f"Successfully fetched {len(hist)} data points")
        
        # Calculate all our technical indicators 
        logger.info("Calculating technical indicators...")
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
        
        logger.info("Technical indicators calculated successfully")
        
        # Get our data ready for the ML models
        logger.info("Preparing data for ML models...")
        hist = hist.dropna()
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'Signal', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'OBV', 'Price_Momentum', 'Volume_Trend']
        if len(hist) < 30:
            return jsonify({'error': 'Not enough data to make a prediction.'}), 400
        
        logger.info(f"Data prepared with {len(hist)} rows and {len(feature_cols)} features")
        
        # Keep things consistent with random seeds
        np.random.seed(42)
        tf.random.set_seed(42)
        logger.info("Random seeds set for reproducibility")
        
        X = hist[feature_cols].values
        y = hist['Close'].values
        
        # Scale everything so the models can work with it properly
        logger.info("Scaling data for ML models...")
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_scaled = X_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))
        
        # Use the latest data for our prediction
        X_pred = X_scaled[-1].reshape(1, -1)
        logger.info("Data scaling completed successfully")
        
        # Time to run our ensemble of models!
        logger.info("Starting ML model ensemble training...")
        
        # Model 1: Linear Regression (simple but effective baseline)
        try:
            lr = LinearRegression()
            lr.fit(X_scaled[:-1], y_scaled[:-1])
            lr_pred = y_scaler.inverse_transform(lr.predict(X_pred).reshape(-1, 1))[0][0]
            logger.info("Linear Regression model successfully trained")
        except Exception as lr_error:
            logger.warning(f"Linear Regression failed, using simple trend: {str(lr_error)}")
            recent_trend = hist['Close'].pct_change().tail(5).mean()
            lr_pred = current_price * (1 + recent_trend)
        
        # Model 2: Random Forest (good at finding complex patterns)
        try:
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_scaled[:-1], y_scaled[:-1])
            rf_pred = y_scaler.inverse_transform(rf.predict(X_pred).reshape(-1, 1))[0][0]
            logger.info("Random Forest model successfully trained")
        except Exception as rf_error:
            logger.warning(f"Random Forest failed, using moving average: {str(rf_error)}")
            rf_pred = hist['Close'].rolling(window=20).mean().iloc[-1]
        
        # Model 3: Gradient Boosting (great at correcting mistakes)
        try:
            gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
            gb.fit(X_scaled[:-1], y_scaled[:-1])
            gb_pred = y_scaler.inverse_transform(gb.predict(X_pred).reshape(-1, 1))[0][0]
            logger.info("Gradient Boosting model successfully trained")
        except Exception as gb_error:
            logger.warning(f"Gradient Boosting failed, using median price: {str(gb_error)}")
            gb_pred = hist['Close'].tail(10).median()
        
        # Model 4: LSTM (the fancy neural network for time series)
        seq_len = 10
        X_lstm = []
        y_lstm = []
        for i in range(seq_len, len(X_scaled)):
            X_lstm.append(X_scaled[i-seq_len:i])
            y_lstm.append(y_scaled[i])
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        
        # Try to use LSTM, but fall back gracefully if it fails (common on cloud platforms)
        try:
            logger.info("Attempting to create LSTM model...")
            lstm_model = Sequential([
                LSTM(32, input_shape=(seq_len, X_lstm.shape[2]), return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            logger.info("LSTM model architecture created successfully")
            
            logger.info("Compiling LSTM model...")
            lstm_model.compile(optimizer='adam', loss='mse')
            logger.info("LSTM model compiled successfully")
            
            logger.info("Training LSTM model...")
            lstm_model.fit(X_lstm[:-1], y_lstm[:-1], epochs=5, batch_size=8, verbose=0)
            logger.info("LSTM model training completed")
            
            logger.info("Making LSTM prediction...")
            X_lstm_pred = X_scaled[-seq_len:].reshape(1, seq_len, X_lstm.shape[2])
            lstm_pred = y_scaler.inverse_transform(lstm_model.predict(X_lstm_pred))[0][0]
            logger.info("LSTM model successfully trained and used for prediction")
        except Exception as lstm_error:
            logger.warning(f"LSTM model failed on cloud platform (using fallback): {str(lstm_error)}")
            logger.warning(f"LSTM error type: {type(lstm_error).__name__}")
            # Use a simple trend-based prediction as fallback for LSTM
            recent_trend = hist['Close'].pct_change().tail(10).mean()
            lstm_pred = current_price * (1 + recent_trend)
        
        # Combine all our predictions and check for sanity
        predictions = [lr_pred, rf_pred, gb_pred, lstm_pred]
        
        # Toss out any crazy predictions (more than 20% change is suspicious)
        valid_predictions = [p for p in predictions if abs(p - current_price) / current_price < 0.20]
        
        if not valid_predictions:
            # If all models went crazy, fall back to simple trend analysis
            recent_returns = hist['Close'].pct_change().tail(20).mean()
            predicted_price = current_price * (1 + recent_returns)
            confidence = 30.0  # Low confidence when using fallback
        else:
            # Average the sane predictions
            predicted_price = np.mean(valid_predictions)
            
            # Figure out how confident we should be in this prediction
            # Based on three things:
            # 1. Do our models agree with each other?
            # 2. How well have they been doing lately?
            # 3. How volatile is this stock?
            
            # 1. Model agreement - if they all say similar things, that's good
            std_dev = np.std(valid_predictions)
            mean_price = np.mean(valid_predictions)
            agreement_score = 40 * (1 - min(1, std_dev / mean_price))
            
            # 2. Recent accuracy - have we been right lately?
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
                accuracy_score = 15  # Just give it an average score if we can't calculate
            
            # 3. Market volatility - wild markets are harder to predict
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)
            volatility_score = 30 * (1 - min(1, volatility))
            
            # Combine all confidence factors (keep it between 30-95%)
            confidence = min(95, max(30, agreement_score + accuracy_score + volatility_score))
            
            # Some debug info for nerds like us
            logger.info(f"Confidence components - Agreement: {agreement_score:.1f}, Accuracy: {accuracy_score:.1f}, Volatility: {volatility_score:.1f}")
            logger.info(f"Final confidence score: {confidence:.1f}")
            
            # Don't let the prediction be too crazy (max 5% change)
            max_change = 0.05
            predicted_price = np.clip(
                predicted_price,
                current_price * (1 - max_change),
                current_price * (1 + max_change)
            )
            
            # Calculate how much change we're predicting
            expected_change = ((predicted_price - current_price) / current_price) * 100
            
            # Prepare the chart data - last 30 days plus our prediction
            recent_data = hist.tail(30)
            chart_dates = recent_data.index.strftime('%Y-%m-%d').tolist()
            actual_prices = recent_data['Close'].tolist()
            
            # Figure out the next trading day for our prediction
            last_date = recent_data.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            
            # Skip weekends (market's closed!)
            while next_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
                next_date += pd.Timedelta(days=1)
            
            next_date_str = next_date.strftime('%Y-%m-%d')
            
            # Debug info
            logger.info(f"Last historical date: {last_date.strftime('%Y-%m-%d')}")
            logger.info(f"Next trading day: {next_date_str}")
            
            # Add our prediction to the chart
            chart_dates.append(next_date_str)
            actual_prices.append(None)  # No actual price yet (it's the future!)
            
            # Show prediction as just the final point
            historical_predictions = [None] * (len(actual_prices) - 1)
            historical_predictions.append(predicted_price)
            
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
                'data_source': 'Simulated Market Data (Demo)',
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
    """Simple health check to make sure everything's working"""
    try:
        # Make sure all our important modules are still there
        import pandas
        import numpy
        import sklearn
        import tensorflow
        
        # Test basic functionality
        test_data = fetch_stock_data('AAPL', '1y')
        can_predict = test_data is not None and len(test_data) > 30
        
        return jsonify({
            'status': 'healthy',
            'message': 'Stock Predictor API is operational',
            'dependencies': 'all modules loaded successfully',
            'data_generation': 'working' if can_predict else 'limited',
            'environment': 'cloud' if os.environ.get('RENDER') or os.environ.get('HEROKU') else 'local'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'message': str(e),
            'environment': 'cloud' if os.environ.get('RENDER') or os.environ.get('HEROKU') else 'local'
        }), 500

if __name__ == '__main__':
    # Render gives us the port via environment variable
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting Flask server on 0.0.0.0:{port} (debug=False)")
    app.run(host='0.0.0.0', port=port, debug=False) 