#!/bin/bash
set -e

# Oracle Cloud Deployment Script for SVG-AI
echo "=== SVG-AI Oracle Cloud Deployment ==="

# Create application directory
sudo mkdir -p /opt/svg-ai
sudo chown ubuntu:ubuntu /opt/svg-ai
cd /opt/svg-ai

# Clone repository (if not already done)
if [ ! -d ".git" ]; then
    echo "Setting up application..."
    # Create minimal structure
    mkdir -p backend frontend

    # Create basic backend app
    cat > backend/app.py << 'EOF'
from flask import Flask, jsonify, request, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        "message": "SVG-AI Oracle Cloud Server",
        "status": "running",
        "endpoints": ["/api/", "/api/health"]
    })

@app.route('/api/')
def api_root():
    return jsonify({"message": "SVG-AI API", "version": "1.0"})

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "service": "svg-ai"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
EOF

    # Create requirements.txt
    cat > requirements.txt << 'EOF'
flask>=2.0.0
pillow>=10.0.0
requests>=2.31.0
EOF
fi

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test basic setup
echo "Testing application..."
python -c "from backend.app import app; print('Flask app imported successfully')"

echo "=== Deployment completed successfully! ==="
echo "Next steps:"
echo "1. Start application: cd /opt/svg-ai && source venv/bin/activate && python backend/app.py"
echo "2. Configure Nginx"
echo "3. Setup SSL"