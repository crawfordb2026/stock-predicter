# Advanced Stock Price Predictor

A sophisticated web application that uses multiple machine learning models to predict stock prices for the next trading day. The system combines LSTM neural networks, Random Forest, Gradient Boosting, and Linear Regression models to provide accurate ensemble predictions with confidence scores.

## 🚀 Features

- **Multi-Model Ensemble**: Combines 4 different ML models (LSTM, Random Forest, Gradient Boosting, Linear Regression)
- **Real-Time Predictions**: Next-day stock price predictions with confidence scores
- **Technical Analysis**: RSI, MACD, Support/Resistance levels
- **Risk Metrics**: Volatility, Sharpe Ratio, Max Drawdown, Price Momentum, Volume Trend
- **Interactive Charts**: Real-time price charts with predicted vs actual prices
- **Model Performance Tracking**: Individual model performance metrics and weightings
- **Responsive Web Interface**: Clean, modern UI with detailed explanations

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Flask**: Web framework
- **TensorFlow/Keras**: LSTM neural networks
- **scikit-learn**: Machine learning models
- **yfinance**: Stock data API
- **pandas & numpy**: Data processing

### Frontend
- **HTML5/CSS3/JavaScript**
- **Chart.js**: Interactive charts
- **Responsive design**: Mobile-friendly interface

### Machine Learning Models
1. **LSTM (35% weight)**: Best for capturing long-term patterns and temporal dependencies
2. **Random Forest (25% weight)**: Robust ensemble method, handles multiple features well
3. **Gradient Boosting (25% weight)**: Excellent for complex pattern recognition
4. **Linear Regression (15% weight)**: Baseline model for trend analysis

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚦 Usage

### Running the Application

1. **Start the Flask server**
```bash
python app.py
```
or use the provided script:
```bash
./start_server.sh
```

2. **Open your browser**
Navigate to `http://127.0.0.1:5000`

3. **Use the application**
- Select a stock symbol from the dropdown
- Choose a time period (3 months to 5 years)
- Click "Predict" to get next-day price prediction

### API Endpoints

- `GET /test`: Health check endpoint
- `POST /predict`: Main prediction endpoint

**Request format:**
```json
{
    "symbol": "AAPL",
    "period": "2y"
}
```

## 📊 Model Performance

The system uses an ensemble approach where each model contributes based on its historical performance:

- **LSTM (35%)**: Specializes in temporal dependencies and long-term patterns
- **Random Forest (25%)**: Robust against overfitting, handles feature interactions
- **Gradient Boosting (25%)**: Excellent accuracy for complex non-linear relationships
- **Linear Regression (15%)**: Provides baseline trend analysis

### Confidence Scoring

Prediction confidence is calculated based on model agreement:
- **High (80-95%)**: Models agree closely
- **Medium (50-80%)**: Moderate disagreement
- **Low (25-50%)**: Significant disagreement

## 🔧 Configuration

### Supported Stocks
The application includes 20 major stocks:
- AAPL (Apple), MSFT (Microsoft), GOOGL (Google)
- AMZN (Amazon), TSLA (Tesla), META (Meta)
- NVDA (NVIDIA), JPM (JPMorgan), V (Visa)
- And more...

### Time Periods
- **3 months**: Focus on recent market behavior
- **6 months**: Short-term trends
- **1 year**: Medium-term analysis
- **2 years**: Balanced analysis (recommended)
- **5 years**: Long-term trend identification

## 📁 Project Structure

```
stock-predictor/
├── app.py                 # Main Flask application
├── main.py               # Alternative main script
├── requirements.txt      # Python dependencies
├── start_server.sh      # Server startup script
├── web/
│   ├── index.html       # Main web interface
│   └── static/
│       ├── css/
│       │   └── style.css # Application styles
│       └── js/
│           └── main.js  # Frontend JavaScript
├── utils/              # Utility modules
├── models/            # Model definitions
└── data/             # Data storage
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals and do your own research before making investment choices.

## 🙏 Acknowledgments

- **yfinance**: For providing easy access to Yahoo Finance data
- **TensorFlow**: For the powerful machine learning framework
- **scikit-learn**: For the comprehensive ML algorithms
- **Chart.js**: For beautiful interactive charts 