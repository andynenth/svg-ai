#!/usr/bin/env python3
"""
Local SVG-AI Server Launcher
Simple launcher to run SVG-AI locally without import issues
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up environment
os.chdir(project_root)

try:
    # Import and run the Flask app
    from backend.app import app

    print("🚀 Starting SVG-AI Local Server...")
    print("📡 Frontend: http://localhost:5000/")
    print("🔗 API: http://localhost:5000/api/")
    print("⭐ Ready for PNG to SVG conversion!")
    print("Press Ctrl+C to stop")

    # Run the Flask development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Avoid double imports
    )

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Try running: python -m backend.app")
except Exception as e:
    print(f"❌ Error starting server: {e}")