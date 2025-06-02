#!/bin/bash

# Stock Predictor Server Startup Script

echo "🚀 Starting Advanced Stock Predictor Server..."

# Check if running on Render (cloud) or locally
if [ "$RENDER" = "true" ]; then
    echo "☁️ Detected Render cloud environment"
    # On Render, skip local setup and use cloud settings
    export HOST=0.0.0.0
    export PORT=${PORT:-10000}  # Render assigns PORT automatically
    echo "🌐 Server starting on $HOST:$PORT"
    python app.py
else
    echo "💻 Detected local development environment"
    
    # Kill any existing Flask servers on port 5000
    echo "🔪 Stopping any existing servers..."
    pkill -f "python app.py" || true
    pkill -f "flask" || true
    # Kill everything on port 5000 more aggressively
    sudo lsof -ti:5000 | xargs sudo kill -9 2>/dev/null || true
    lsof -ti:5000 | xargs kill -9 2>/dev/null || true
    # Also kill common processes that use port 5000
    pkill -f "AirPlay" 2>/dev/null || true
    pkill -f "ControlCenter" 2>/dev/null || true
    sleep 3

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

    # Set environment variables for local development
    export FLASK_APP=app.py
    export FLASK_ENV=development
    export HOST=127.0.0.1
    export PORT=8080

    echo "🌐 Server will run on http://$HOST:$PORT"
    echo "📁 Open web/index.html in your browser to use the application"
    echo "⚡ Press Ctrl+C to stop the server"
    echo ""

    # Start the Flask server
    python app.py
fi 