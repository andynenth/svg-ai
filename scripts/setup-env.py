#!/usr/bin/env python3
"""
Environment setup and validation script for SVG-AI deployment
"""

import os
import sys
import secrets
from pathlib import Path

def generate_secret_key():
    """Generate a secure secret key"""
    return secrets.token_urlsafe(32)

def validate_environment(env_name):
    """Validate environment configuration"""
    required_vars = {
        'development': ['FLASK_ENV'],
        'production': ['SECRET_KEY', 'FLASK_ENV', 'REDIS_URL'],
        'testing': ['FLASK_ENV']
    }

    missing_vars = []
    for var in required_vars.get(env_name, []):
        if not os.environ.get(var):
            missing_vars.append(var)

    return missing_vars

def setup_environment(env_name):
    """Setup environment variables for specified environment"""
    env_file = f".env.{env_name}"

    print(f"Setting up environment: {env_name}")

    # Base configuration
    config = {
        'FLASK_ENV': env_name,
        'PYTHONPATH': '/home/app',
        'REDIS_URL': 'redis://localhost:6379',
        'UPLOAD_FOLDER': 'uploads',
        'MAX_CONTENT_LENGTH': '16777216'  # 16MB
    }

    # Environment-specific configuration
    if env_name == 'production':
        config['SECRET_KEY'] = generate_secret_key()
        config['FLASK_DEBUG'] = '0'
        config['WORKERS'] = '4'
    elif env_name == 'development':
        config['SECRET_KEY'] = 'dev-secret-key-not-for-production'
        config['FLASK_DEBUG'] = '1'
    elif env_name == 'testing':
        config['SECRET_KEY'] = 'test-secret-key'
        config['FLASK_DEBUG'] = '1'
        config['TESTING'] = '1'

    # Write environment file
    with open(env_file, 'w') as f:
        f.write(f"# Environment configuration for {env_name}\n")
        f.write(f"# Generated automatically by setup-env.py\n\n")

        for key, value in config.items():
            f.write(f"{key}={value}\n")

    print(f"✓ Environment file created: {env_file}")

    # Set permissions for production
    if env_name == 'production':
        os.chmod(env_file, 0o600)
        print(f"✓ Secure permissions set for {env_file}")

    return config

def check_dependencies():
    """Check if required dependencies are available"""
    checks = []

    # Check Python version
    if sys.version_info >= (3, 9):
        checks.append(("Python 3.9+", True))
    else:
        checks.append(("Python 3.9+", False))

    # Check if required packages can be imported
    required_packages = [
        'flask', 'redis', 'docker', 'safety', 'bandit'
    ]

    for package in required_packages:
        try:
            __import__(package)
            checks.append((f"Package: {package}", True))
        except ImportError:
            checks.append((f"Package: {package}", False))

    return checks

def main():
    """Main setup function"""
    if len(sys.argv) != 2:
        print("Usage: python setup-env.py <environment>")
        print("Environments: development, production, testing")
        sys.exit(1)

    env_name = sys.argv[1].lower()

    if env_name not in ['development', 'production', 'testing']:
        print(f"Error: Unknown environment '{env_name}'")
        print("Valid environments: development, production, testing")
        sys.exit(1)

    print("SVG-AI Environment Setup")
    print("=" * 40)

    # Check dependencies
    print("\nChecking dependencies...")
    checks = check_dependencies()
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")

    if not all(passed for _, passed in checks):
        print("\n⚠️  Some dependency checks failed. Please install missing dependencies.")

    # Setup environment
    print(f"\nSetting up {env_name} environment...")
    config = setup_environment(env_name)

    # Validate environment
    missing_vars = validate_environment(env_name)
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before deployment.")
    else:
        print(f"\n✓ Environment validation passed for {env_name}")

    # Show next steps
    print("\nNext steps:")
    print(f"1. Source the environment file: source .env.{env_name}")
    print(f"2. Deploy using: ./scripts/deploy.sh {env_name}")

    if env_name == 'production':
        print("\n🔒 Production environment notes:")
        print("- Secret key has been generated automatically")
        print("- Environment file has secure permissions (600)")
        print("- Review all settings before deployment")

if __name__ == "__main__":
    main()