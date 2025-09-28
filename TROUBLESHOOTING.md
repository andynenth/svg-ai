# Troubleshooting Guide - AI Dependencies

## Phase 1 AI Installation Issues

### Common Installation Problems

#### 1. PyTorch CPU Installation Fails

**Error**: `Could not find a version that satisfies the requirement torch==2.1.0+cpu`

**Solution**:
```bash
# Ensure you're using the correct find-links URL
pip3 install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# If still failing, try without version pinning
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Verification**:
```bash
python3 -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

#### 2. scikit-learn Version Conflict

**Error**: `pip check` shows conflicts with scikit-learn versions

**Solution**:
```bash
# Force downgrade to compatible version
pip3 install scikit-learn==1.3.2 --force-reinstall

# Verify compatibility
pip3 check
```

#### 3. Virtual Environment Not Activated

**Error**: `Warning: Not in virtual environment`

**Solution**:
```bash
# Activate venv39 from project root
source venv39/bin/activate

# Verify activation
echo $VIRTUAL_ENV
# Should show: /Users/nrw/python/svg-ai/venv39
```

#### 4. Stable-Baselines3 Import Error

**Error**: `No module named 'stable_baselines3'`

**Solution**:
```bash
# Install with specific dependencies
pip3 install stable-baselines3[extra]==2.0.0

# If fails, install gymnasium first
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.0.0
```

#### 5. Memory Issues During Installation

**Error**: Installation killed or crashes

**Solution**:
```bash
# Install packages one by one with limited memory
pip3 install torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
pip3 install torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir

# Clear pip cache
pip3 cache purge
```

### Environment-Specific Issues

#### macOS Intel (x86_64)

**Issue**: Architecture mismatch errors

**Solution**:
```bash
# Verify architecture
uname -m
# Should show: x86_64

# Use CPU-specific PyTorch
pip3 install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

#### Python 3.9.22 Compatibility

**Issue**: Package version incompatibility

**Solution**:
```bash
# Verify Python version
python3 --version
# Should show: Python 3.9.22

# Use exact versions from requirements_ai_phase1.txt
pip3 install -r requirements_ai_phase1.txt
```

### Verification and Testing

#### Check AI Setup Status

```bash
# Run comprehensive verification
python3 scripts/verify_ai_setup.py

# Expected output for successful installation:
# Package imports: 10/10 successful
# PyTorch performance: ✅ Good
# RL components: ✅ Working
# Genetic algorithms: ✅ Working
# Transformers: ✅ Working
# Memory: ✅ Sufficient
```

#### Individual Package Testing

```bash
# Test PyTorch
python3 -c "import torch; x=torch.randn(100,100); y=torch.mm(x,x.t()); print('PyTorch working')"

# Test RL components
python3 -c "import gymnasium as gym; from stable_baselines3 import PPO; print('RL working')"

# Test genetic algorithms
python3 -c "from deap import base, creator, tools; print('DEAP working')"

# Test transformers
python3 -c "import transformers; print('Transformers working')"
```

### Performance Issues

#### Slow PyTorch Performance

**Issue**: Matrix operations taking >1 second

**Diagnosis**:
```bash
python3 -c "
import torch, time
start = time.time()
x = torch.randn(1000, 1000)
y = torch.mm(x, x.t())
print(f'Time: {time.time() - start:.3f}s')
"
```

**Solution**:
- Ensure CPU optimized version: `torch==2.1.0+cpu`
- Check system load and close unnecessary applications
- Verify sufficient RAM (8GB recommended)

#### Memory Usage Too High

**Issue**: System running out of memory

**Diagnosis**:
```bash
python3 -c "
import psutil
mem = psutil.virtual_memory()
print(f'Total: {mem.total/(1024**3):.1f}GB')
print(f'Available: {mem.available/(1024**3):.1f}GB')
print(f'Used: {mem.percent:.1f}%')
"
```

**Solution**:
- Close other applications
- Use `--no-cache-dir` during installation
- Install packages individually

### Recovery Procedures

#### Complete AI Environment Reset

```bash
# Remove AI packages
pip3 uninstall torch torchvision stable-baselines3 gymnasium deap transformers tokenizers scikit-learn -y

# Reinstall core packages
pip3 install -r requirements.txt

# Clean reinstall AI packages
./scripts/install_ai_dependencies.sh
```

#### Restore to Pre-AI State

```bash
# Check git history
git log --oneline

# Reset to before AI installation
git checkout master
git branch -D phase1-foundation

# Start fresh
git checkout -b phase1-foundation
```

### Getting Help

#### Information to Provide

When reporting issues, include:

1. **System Information**:
   ```bash
   python3 --version
   pip3 --version
   uname -a
   echo $VIRTUAL_ENV
   ```

2. **Error Output**:
   ```bash
   python3 scripts/verify_ai_setup.py 2>&1 | tee ai_setup_error.log
   ```

3. **Package Status**:
   ```bash
   pip3 list | grep -E "(torch|sklearn|stable|gymnasium|deap|transformers)"
   ```

4. **Memory/Disk Status**:
   ```bash
   df -h
   vm_stat | head -5
   ```

#### Contact Information

- Project repository: Include error logs and system information
- AI setup issues: Reference this troubleshooting guide
- Performance problems: Include benchmark results from verify_ai_setup.py