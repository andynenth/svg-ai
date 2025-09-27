#!/bin/bash
# Deploy the application

echo "Deploying PNG to SVG Converter..."

# Pull latest code
echo "Pulling latest code..."
git pull origin main

# Install backend dependencies
echo "Installing backend dependencies..."
pip install -r backend/requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p backend/uploads
mkdir -p backups

# Set permissions
chmod 755 backend/uploads

# Restart server
echo "Restarting server..."
./stop.sh
./start.sh