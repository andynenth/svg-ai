# Version Update Process

This document describes the process for updating dependency versions in the SVG-AI project.

## Overview

The project uses pinned dependency versions in `requirements.txt` to ensure reproducible builds and deployment consistency. This document outlines how to safely update these versions.

## Current Dependency Strategy

- **All versions are pinned** to specific releases (e.g., `numpy==2.0.2`)
- **Dependencies are organized** by function (core, image processing, web server, etc.)
- **Compatibility is validated** before version updates are committed

## Update Process

### 1. Generate Current Versions

```bash
# Activate the virtual environment
source venv39/bin/activate

# Generate complete list of current versions
pip freeze > requirements_pinned.txt
```

### 2. Review Compatibility

Before updating `requirements.txt`, review the generated versions for:

- **Major version changes** that could break compatibility
- **Security updates** that should be prioritized
- **Dependencies with conflicting requirements**

### 3. Test Dependency Compatibility

```bash
# Check for dependency conflicts
python -m pip check

# Test core functionality
python -c "
import backend.converters.smart_auto_converter
import backend.utils.color_detector
import numpy as np
import PIL
import vtracer
print('All core imports successful!')
"
```

### 4. Update Requirements.txt

Update the pinned versions in `requirements.txt` while maintaining the organized structure:

```txt
# Core dependencies
pillow==11.3.0       # Updated from 10.1.0
numpy==2.0.2         # Updated from 1.24.3 - MAJOR VERSION
click==8.1.8         # Updated from 8.1.7
requests==2.32.5     # Updated from 2.31.0
```

### 5. Handle Major Version Updates

For major version updates (like numpy 1.x → 2.x), additional steps are required:

1. **Check dependent packages** for compatibility
2. **Update conflicting dependencies** (e.g., scikit-learn for numpy 2.0 support)
3. **Run comprehensive tests** to ensure functionality
4. **Update documentation** if APIs have changed

### 6. Validate the Update

```bash
# Reinstall from updated requirements
pip install -r requirements.txt

# Run dependency check
python -m pip check

# Run test suite
pytest tests/

# Test core functionality
python test_e2e.py
```

## Troubleshooting Common Issues

### Dependency Conflicts

If `pip check` reports conflicts:

1. **Identify the conflicting packages**
2. **Find compatible versions** using dependency graphs
3. **Update the newer package** to support dependencies
4. **Consider downgrading** if newer versions break compatibility

Example: numpy 2.0.2 conflicted with scikit-learn 1.3.2
- **Solution**: Updated scikit-learn to 1.6.1 which supports numpy 2.x

### Import Errors

If core functionality tests fail:

1. **Check for API changes** in major version updates
2. **Review deprecation warnings** in previous versions
3. **Update code** to use new APIs if necessary
4. **Consider pinning to older version** if changes are breaking

## Version Update Schedule

- **Security updates**: Apply immediately when available
- **Minor updates**: Monthly review and update cycle
- **Major updates**: Quarterly review with thorough testing
- **Pre-release updates**: Only for development/testing environments

## Rolling Back Updates

If issues are discovered after updating:

1. **Revert requirements.txt** to previous working versions
2. **Reinstall dependencies**: `pip install -r requirements.txt`
3. **Test functionality** to ensure rollback succeeded
4. **Document the issue** for future reference

## Dependencies by Category

### Core Dependencies
- **pillow**: Image processing library
- **numpy**: Numerical computing (affects many other packages)
- **click**: CLI interface
- **requests**: HTTP client

### Critical Path Dependencies
- **vtracer**: Core conversion functionality (pin to tested versions)
- **fastapi**: Web API framework
- **opencv-python**: Image processing and analysis

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting
- **mypy**: Static type checking
- **flake8**: Code linting

## Notes

- Always test in development environment before production deployment
- Document any API changes or migration steps required
- Keep `requirements_pinned.txt` as backup of working environment
- Consider using `pip-tools` for more sophisticated dependency management in future