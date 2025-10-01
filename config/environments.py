# config/environments.py
import os
from typing import Dict, Any

class BaseConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    FLASK_ENV = 'development'
    TESTING = False

class ProductionConfig(BaseConfig):
    DEBUG = False
    FLASK_ENV = 'production'
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')  # Must be set in production

class TestingConfig(BaseConfig):
    TESTING = True
    DEBUG = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}