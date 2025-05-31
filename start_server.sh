#!/bin/bash

# Kill any processes on ports 5000, 5001, and 5500
echo "Killing existing processes..."
lsof -ti:5000 | xargs kill -9 2>/dev/null
lsof -ti:5001 | xargs kill -9 2>/dev/null
lsof -ti:5500 | xargs kill -9 2>/dev/null

# Start Flask server
echo "Starting Flask server..."
python3 app.py 