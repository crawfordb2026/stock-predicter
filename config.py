"""
My configuration settings for the Stock Predictor app
"""
import os

class Config:
    """The basic settings that everything inherits from"""
    # Flask stuff
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    
    # Where to run the server
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = int(os.environ.get('PORT', 5000))
    
    # CORS settings - who's allowed to talk to our API
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
    """Settings for when I'm developing locally"""
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    """Settings for when this is running live on the internet"""
    DEBUG = False
    FLASK_ENV = 'production'
    
    # In production, listen on all interfaces (not just localhost)
    HOST = '0.0.0.0'
    
    # Production CORS - way more restrictive for security
    CORS_ORIGINS = [
        "https://yourdomain.com",
        "http://localhost:5000"
    ]

# Easy way to switch between configs
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 