from flask import Flask
from .config import Config
from .utils.logging import get_logger

logger = get_logger(__name__)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    logger.info("Initializing app")

    app.pipe = None

    from .routes.generate import generate_bp
    from .routes.health import health_bp

    app.register_blueprint(generate_bp)
    app.register_blueprint(health_bp)

    @app.before_request
    def load_model():
        if app.pipe is None:
            from .models.controlnet import init_models
            logger.info("Loading model on first request")
            app.pipe = init_models(app)

    logger.info("Application initialized successfully")
    return app