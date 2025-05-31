"""
Configuration file for Stock Predictor application
"""
import os

class Config:
    """Base configuration"""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    
    # Server settings
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = int(os.environ.get('PORT', 5000))
    
    # CORS settings
    CORS_ORIGINS = [
        f"http://{HOST}:{PORT}",
        f"http://localhost:{PORT}",
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "http://127.0.0.1:5500", 
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000"
    ]

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'
    
    # In production, listen on all interfaces
    HOST = '0.0.0.0'
    
    # Production CORS - more restrictive
    CORS_ORIGINS = [
        "https://yourdomain.com",
        "http://localhost:5000"
    ]

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 