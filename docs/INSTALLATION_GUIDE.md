# Installation Guide for New Developers

## Overview

This guide helps new developers set up the SVG-AI Enhanced Conversion Pipeline with all AI capabilities. The project combines VTracer (Rust-based vectorization) with AI-enhanced parameter optimization.

## Prerequisites

### System Requirements
- **Python**: 3.9+ (recommended: Python 3.9.22)
- **Operating System**: macOS, Linux, or Windows
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Disk Space**: 2GB free space
- **Git**: Latest version

### Development Tools (Recommended)
- **IDE**: VS Code, PyCharm, or similar
- **Terminal**: Bash/Zsh (macOS/Linux) or Git Bash (Windows)
- **Package Manager**: pip (included with Python)

## Quick Start (5 minutes)

```bash
# 1. Clone the repository
git clone <repository-url>
cd svg-ai

# 2. Create Python virtual environment
python3 -m venv venv39
source venv39/bin/activate  # On Windows: venv39\Scripts\activate

# 3. Install basic dependencies
pip install -r requirements.txt

# 4. Install VTracer (requires special handling on macOS)
export TMPDIR=/tmp  # macOS only
pip install vtracer

# 5. Verify basic installation
python3 convert.py --help
```

## Complete Installation (15 minutes)

### Step 1: Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv venv39
source venv39/bin/activate

# Upgrade pip to latest version
pip install --upgrade pip
```

### Step 2: Core Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# Install VTracer with platform-specific handling
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    export TMPDIR=/tmp
    pip install vtracer
else
    # Linux/Windows
    pip install vtracer
fi
```

### Step 3: AI Dependencies (Optional but Recommended)

```bash
# Option A: Use automated script
./scripts/install_ai_dependencies.sh

# Option B: Manual installation
pip install -r requirements_ai_phase1.txt

# Option C: Individual packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn==1.3.2
pip install stable-baselines3==2.0.0
pip install gymnasium==0.28.1
pip install deap==1.4
```

### Step 4: Development Dependencies

```bash
# Install development tools
pip install -r requirements/dev.txt

# Install testing tools
pip install pytest coverage pytest-cov
```

### Step 5: Verification

```bash
# Verify AI setup
python3 scripts/verify_ai_setup.py

# Run basic tests
python3 -m pytest tests/ai_modules/ -v

# Test conversion pipeline
python3 convert.py data/logos/simple_geometric/circle_00.png
```

## Detailed Setup Instructions

### Python Environment Setup

#### macOS
```bash
# Install Python 3.9 using Homebrew
brew install python@3.9

# Create virtual environment
python3.9 -m venv venv39
source venv39/bin/activate
```

#### Linux (Ubuntu/Debian)
```bash
# Install Python 3.9
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev

# Create virtual environment
python3.9 -m venv venv39
source venv39/bin/activate
```

#### Windows
```powershell
# Download Python 3.9 from python.org
# During installation, check "Add Python to PATH"

# Create virtual environment
python -m venv venv39
venv39\Scripts\activate
```

### VTracer Installation Troubleshooting

VTracer is a Rust-based library that can sometimes have installation issues:

#### macOS Issues
```bash
# If installation fails with permission errors:
export TMPDIR=/tmp
pip install vtracer

# If you get "Microsoft Visual C++ 14.0 is required" error:
# This shouldn't happen on macOS, but if it does:
xcode-select --install
```

#### Linux Issues
```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install build essentials
sudo apt install build-essential

# Try installing VTracer again
pip install vtracer
```

#### Windows Issues
```powershell
# Install Microsoft Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Install Rust
# Download from: https://rustup.rs/

# Try installing VTracer again
pip install vtracer
```

### AI Dependencies Detailed

#### PyTorch Installation
```bash
# CPU-only version (recommended for development)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU version (if you have CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

#### Scikit-learn
```bash
# Install with specific version for compatibility
pip install scikit-learn==1.3.2

# Verify installation
python3 -c "from sklearn import __version__; print(f'Scikit-learn {__version__}')"
```

#### Stable-Baselines3 (Reinforcement Learning)
```bash
# Install stable-baselines3 with dependencies
pip install stable-baselines3[extra]==2.0.0

# Verify installation
python3 -c "import stable_baselines3; print('Stable-Baselines3 OK')"
```

## Project Structure

After installation, your project should look like this:

```
svg-ai/
├── backend/
│   ├── ai_modules/           # AI module implementations
│   ├── converters/           # Conversion logic
│   └── utils/               # Utilities
├── tests/
│   ├── ai_modules/          # AI module tests
│   ├── data/               # Test data
│   └── utils/              # Test utilities
├── scripts/                # Automation scripts
├── docs/                   # Documentation
├── data/                   # Sample data
├── requirements.txt        # Core dependencies
├── requirements_ai_phase1.txt  # AI dependencies
└── CLAUDE.md              # Project instructions
```

## Configuration

### Environment Variables

Create a `.env` file (optional):
```bash
# Development settings
PYTHONPATH=.
AI_LOG_LEVEL=DEBUG
ENABLE_AI_OPTIMIZATION=true
MAX_CONCURRENT_CONVERSIONS=4

# Performance settings
AI_CACHE_SIZE=100
AI_MEMORY_LIMIT=200  # MB
```

### IDE Setup

#### VS Code
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv39/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true
}
```

#### PyCharm
1. Open project in PyCharm
2. Go to File → Settings → Project → Python Interpreter
3. Add interpreter → Existing environment
4. Select `venv39/bin/python`

## Testing Your Installation

### Basic Functionality Test
```bash
# Test basic conversion
python3 convert.py data/logos/simple_geometric/circle_00.png

# Test with optimization
python3 optimize_iterative.py data/logos/simple_geometric/circle_00.png --target-ssim 0.9
```

### AI Modules Test
```bash
# Test AI imports
python3 scripts/test_ai_imports.py

# Test AI pipeline
python3 -c "
from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
print('AI modules working!')
"
```

### Comprehensive Test Suite
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run with coverage
coverage run -m pytest tests/ai_modules/
coverage report

# Run integration tests
python3 scripts/complete_integration_test.sh
```

## Common Issues and Solutions

### Issue 1: Python Version Mismatch
**Problem**: AI modules require Python 3.9+
**Solution**:
```bash
python3 --version  # Should show 3.9.x
# If not, install Python 3.9 and recreate virtual environment
```

### Issue 2: VTracer Import Error
**Problem**: `ImportError: No module named 'vtracer'`
**Solution**:
```bash
# Reinstall with platform-specific handling
pip uninstall vtracer
export TMPDIR=/tmp  # macOS only
pip install vtracer
```

### Issue 3: PyTorch Installation Issues
**Problem**: PyTorch fails to install or import
**Solution**:
```bash
# Use CPU-only version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue 4: Permission Errors (macOS)
**Problem**: Permission denied during installation
**Solution**:
```bash
# Use TMPDIR workaround
export TMPDIR=/tmp
pip install --user <package-name>
```

### Issue 5: Memory Issues
**Problem**: Out of memory during AI processing
**Solution**:
```bash
# Reduce concurrent processing
export MAX_CONCURRENT_CONVERSIONS=2

# Or process smaller images
python3 convert.py input.png --max-size 512
```

## Development Workflow

### Daily Development
```bash
# Activate environment
source venv39/bin/activate

# Pull latest changes
git pull

# Install any new dependencies
pip install -r requirements.txt

# Run tests before coding
python3 -m pytest tests/ai_modules/ -x

# Start development...
```

### Before Committing
```bash
# Run full test suite
python3 -m pytest tests/ -v

# Check code coverage
coverage run -m pytest tests/ai_modules/
coverage report --fail-under=60

# Run integration tests
python3 scripts/complete_integration_test.sh

# Format code (if using formatter)
black backend/ tests/
```

## Performance Optimization

### For Development
- Use CPU-only PyTorch (faster installation)
- Limit concurrent processing to 2-4 threads
- Use smaller test images (256x256 or 512x512)

### For Production
- Install GPU-enabled PyTorch if available
- Increase concurrent processing based on system resources
- Use larger cache sizes for better performance

## Getting Help

### Documentation
- [AI Modules README](docs/ai_modules/README.md)
- [API Documentation](docs/api/README.md)
- [Usage Examples](docs/examples/README.md)
- [Troubleshooting Guide](docs/ai_modules/troubleshooting.md)

### Testing
```bash
# Quick test
python3 scripts/continuous_testing.py --quick

# Watch mode (runs tests when files change)
python3 scripts/continuous_testing.py --watch
```

### Support
1. Check existing documentation
2. Run diagnostic tools:
   ```bash
   python3 scripts/verify_ai_setup.py
   python3 scripts/test_ai_imports.py
   ```
3. Review test output for specific error messages
4. Check GitHub issues for known problems

## Next Steps

After successful installation:

1. **Explore Examples**: Try the examples in `docs/examples/README.md`
2. **Run Benchmarks**: Execute `python3 scripts/performance_validation.py`
3. **Study Documentation**: Read through the AI modules documentation
4. **Experiment**: Try different logo types and optimization settings
5. **Contribute**: Add tests or documentation improvements

## Appendix

### Complete Dependency List

#### Core Dependencies
- click>=8.1.0
- numpy>=1.26.0
- opencv-python>=4.10.0
- Pillow>=10.0.0
- vtracer>=0.6.11

#### AI Dependencies
- torch>=2.1.0+cpu
- torchvision>=0.16.0+cpu
- scikit-learn==1.3.2
- stable-baselines3==2.0.0
- gymnasium==0.28.1
- deap==1.4

#### Development Dependencies
- pytest>=7.0.0
- pytest-cov>=4.0.0
- coverage>=7.0.0
- black>=23.0.0
- pylint>=2.17.0

### Version Compatibility Matrix

| Python | PyTorch | Scikit-learn | Status |
|--------|---------|--------------|--------|
| 3.9.x  | 2.1.0+  | 1.3.2       | ✅ Recommended |
| 3.10.x | 2.1.0+  | 1.3.2       | ✅ Supported |
| 3.11.x | 2.1.0+  | 1.3.2       | ⚠️ Experimental |
| 3.8.x  | 2.1.0+  | 1.3.2       | ❌ Not supported |

---
**Installation Guide Version**: 1.0
**Last Updated**: September 28, 2024
**Compatible with**: Phase 1 (Foundation & Dependencies)

🤖 Generated with [Claude Code](https://claude.ai/code)