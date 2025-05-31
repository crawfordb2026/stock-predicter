#!/bin/bash

# Stock Predictor Server Startup Script

echo "🚀 Starting Advanced Stock Predictor Server..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/update requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Allow users to override port and host
PORT=${PORT:-5000}
HOST=${HOST:-127.0.0.1}

echo "🌐 Server will run on http://${HOST}:${PORT}"
echo "📁 Open web/index.html in your browser to use the application"
echo "⚡ Press Ctrl+C to stop the server"
echo ""

# Start the Flask server
python app.py 