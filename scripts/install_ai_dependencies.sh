#!/bin/bash
set -e

echo "🚀 Installing AI dependencies for SVG-AI Phase 1..."

# Check environment
echo "📋 Verifying environment..."
python3 --version || { echo "❌ Python 3 not found"; exit 1; }
pip3 --version || { echo "❌ pip3 not found"; exit 1; }

# Verify we're in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️ Warning: Not in virtual environment"
    echo "Please run: source venv39/bin/activate"
    exit 1
fi

echo "✅ Using virtual environment: $VIRTUAL_ENV"

# Verify we're in correct project directory
if [[ ! -f "requirements_ai_phase1.txt" ]]; then
    echo "❌ requirements_ai_phase1.txt not found"
    echo "Please run from project root directory"
    exit 1
fi

echo "✅ Found requirements_ai_phase1.txt"

# Install PyTorch CPU
echo "📦 Installing PyTorch CPU..."
pip3 install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
if python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')" 2>/dev/null; then
    echo "✅ PyTorch CPU installation successful"
else
    echo "❌ PyTorch installation failed"
    exit 1
fi

# Install scikit-learn (downgrade if needed)
echo "📦 Installing scikit-learn 1.3.2..."
pip3 install scikit-learn==1.3.2
if python3 -c "import sklearn; print(f'✅ scikit-learn {sklearn.__version__} installed')" 2>/dev/null; then
    echo "✅ scikit-learn installation successful"
else
    echo "❌ scikit-learn installation failed"
    exit 1
fi

# Install Reinforcement Learning packages
echo "📦 Installing Reinforcement Learning packages..."
pip3 install stable-baselines3==2.0.0
if python3 -c "from stable_baselines3 import PPO; print('✅ Stable-Baselines3 installed')" 2>/dev/null; then
    echo "✅ Stable-Baselines3 installation successful"
else
    echo "❌ Stable-Baselines3 installation failed"
    exit 1
fi

pip3 install gymnasium==0.29.1
if python3 -c "import gymnasium as gym; print('✅ Gymnasium installed')" 2>/dev/null; then
    echo "✅ Gymnasium installation successful"
else
    echo "❌ Gymnasium installation failed"
    exit 1
fi

# Install Genetic Algorithm library
echo "📦 Installing DEAP (Genetic Algorithms)..."
pip3 install deap==1.4.1
if python3 -c "import deap; print('✅ DEAP installed')" 2>/dev/null; then
    echo "✅ DEAP installation successful"
else
    echo "❌ DEAP installation failed"
    exit 1
fi

# Install Transformers (minimal)
echo "📦 Installing Transformers (minimal)..."
pip3 install transformers==4.36.0 --no-deps
pip3 install tokenizers==0.15.0
if python3 -c "import transformers; print('✅ Transformers installed')" 2>/dev/null; then
    echo "✅ Transformers installation successful"
else
    echo "❌ Transformers installation failed"
    exit 1
fi

echo ""
echo "✅ All AI dependencies installed successfully!"
echo ""
echo "🔍 Running verification script..."
python3 scripts/verify_ai_setup.py

echo ""
echo "🎉 Phase 1 AI dependency installation complete!"
echo "Ready to proceed with Phase 1 development."