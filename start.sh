#!/bin/bash
# Start the application

echo "Starting PNG to SVG Converter..."

# Check Python version
python3 --version

# Activate virtual environment
if [ -d "venv39" ]; then
    source venv39/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: No virtual environment found"
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r backend/requirements.txt

# Create uploads directory
mkdir -p backend/uploads

# Start backend server
echo "Starting backend server on port 8000..."
cd backend && python app.py