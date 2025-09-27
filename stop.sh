#!/bin/bash
# Stop the application

echo "Stopping PNG to SVG Converter..."

# Find and kill Python process running app.py
PID=$(ps aux | grep 'python app.py' | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
    kill $PID
    echo "Server stopped (PID: $PID)"
else
    echo "Server not running"
fi

# Also check for any Flask processes
FLASK_PID=$(ps aux | grep 'flask' | grep -v grep | awk '{print $2}')

if [ -n "$FLASK_PID" ]; then
    kill $FLASK_PID
    echo "Flask process stopped (PID: $FLASK_PID)"
fi