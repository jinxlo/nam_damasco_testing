# /namwoo_app/__init__.py
import os
import logging
from logging.config import dictConfig
from flask import Flask
import redis
from .config.config import Config
from .utils.logging_utils import JsonFormatter

# --- Logging Configuration (Unchanged) ---
log_level_env = os.environ.get('LOG_LEVEL', 'INFO').upper()
_default_basedir_for_logs = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir_path = os.path.join(getattr(Config, 'basedir', _default_basedir_for_logs) , 'logs')
os.makedirs(log_dir_path, exist_ok=True)

logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            '()': JsonFormatter,
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'level': log_level_env,
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
        'app_file': {
            'level': log_level_env,
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'filename': os.path.join(log_dir_path, 'app.log'),
            'maxBytes': 10485760,
            'backupCount': 5,
            'encoding': 'utf8',
        },
        'sync_file': {
            'level': log_level_env,
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'filename': os.path.join(log_dir_path, 'sync.log'),
            'maxBytes': 5242880,
            'backupCount': 3,
            'encoding': 'utf8',
        },
        'json_file': {
            'level': log_level_env,
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'json',
            'filename': getattr(Config, 'LOG_JSON_FILE', os.path.join(log_dir_path, 'app.json')),
            'maxBytes': 10485760,
            'backupCount': 5,
            'encoding': 'utf8',
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'app_file', 'json_file'],
            'level': log_level_env,
            'propagate': True
        },
        'werkzeug': {'handlers': ['console', 'app_file', 'json_file'], 'level': 'INFO', 'propagate': False,},
        'sqlalchemy.engine': {'handlers': ['console', 'app_file', 'json_file'], 'level': 'WARNING', 'propagate': False,},
        # APScheduler logger removed as the package is no longer a direct dependency
        'sync': {'handlers': ['console', 'sync_file', 'json_file'], 'level': log_level_env, 'propagate': False,},
        'celery': {'handlers': ['console', 'app_file', 'json_file'], 'level': log_level_env, 'propagate': False,},
        'namwoo_app': {'handlers': ['console', 'app_file', 'json_file'], 'level': log_level_env, 'propagate': False}
    }
}
dictConfig(logging_config)
logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    logger.info("--- Creating Flask Application Instance ---")
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize shared Redis client (Unchanged)
    try:
        app.redis_client = redis.Redis.from_url(app.config["REDIS_URL"])
        logger.info(f"Redis client initialized using URL: {app.config['REDIS_URL']}")
    except Exception as e:
        logger.exception(f"Failed to initialize Redis client: {e}")
        app.redis_client = None

    logger.info(f"Flask Environment: {app.config.get('FLASK_ENV', 'not_set')}")
    logger.info(f"Debug Mode: {app.config.get('DEBUG', False)}")

    # Initialize database (Unchanged)
    from .utils import db_utils
    db_utils.init_db(app)

    logger.info("Dependent services will be initialized as needed within their respective service files.")

    # --- BLUEPRINT REGISTRATION --- (Unchanged)
    from .api import api_bp as api_module_blueprint
    app.register_blueprint(api_module_blueprint)
    logger.info(f"Main API Blueprint '{api_module_blueprint.name}' registered under url_prefix: {api_module_blueprint.url_prefix}")
    
    # --- CELERY CONFIGURATION LINKING --- (Unchanged)
    try:
        from .celery_app import celery_app as celery_application_instance
        celery_config_keys_to_pass = [
            'CELERY_BROKER_URL', 
            'CELERY_RESULT_BACKEND', 
            'CELERY_TASK_SERIALIZER',
        ]
        celery_flask_config = {key: app.config[key] for key in celery_config_keys_to_pass if key in app.config}
        
        if celery_flask_config:
            celery_application_instance.conf.update(celery_flask_config)
            logger.info(f"Celery instance config updated from Flask app config for keys: {list(celery_flask_config.keys())}")
        else:
            logger.info("No specific Celery configurations found in Flask app.config to update Celery instance.")
    except ImportError:
        logger.warning("celery_app not found. Celery configurations skipped.")
    except Exception as e_celery_conf:
        logger.error(f"Error during Celery configuration linking: {e_celery_conf}", exc_info=True)


    # ==============================================================================
    # === SCHEDULER BLOCK REMOVED ==================================================
    # The entire `if not app.debug...` block for APScheduler has been deleted.
    # ==============================================================================


    register_cli_commands(app)

    logger.info("--- Namwoo Application Initialization Complete ---")
    return app


def register_cli_commands(app):
    
    # ==============================================================================
    # === `run-sync` COMMAND REMOVED ===============================================
    # The @app.cli.command("run-sync") block has been deleted.
    # ==============================================================================

    @app.cli.command("create-db")
    def create_db_command():
        logger.info("Database table creation triggered via CLI.")
        print("--- Creating Database Tables ---")
        try:
            with app.app_context(): 
                from .utils import db_utils
                from .models import Base 
                if db_utils.engine:
                    print("Creating tables from SQLAlchemy models...")
                    Base.metadata.create_all(bind=db_utils.engine)
                    print("Database tables (from models) created successfully.")
                    logger.info("Database tables (from models) created successfully via CLI.")

                    from sqlalchemy import text
                    with db_utils.engine.connect() as connection:
                        with connection.begin(): 
                            logger.info("Ensuring pgvector extension exists in the database (CLI)...")
                            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                            logger.info("pgvector extension check complete (CLI).")
                else:
                    print("Error: Database engine not initialized.")
                    logger.error("Database engine not initialized in create-db command.")
        except Exception as e:
            logger.exception("Error during database table creation via CLI.")
            print(f"An error occurred during table creation: {e}")

    logger.info("Custom CLI commands registered.")

# Ensure Celery Tasks Are Imported so the worker can find them
import namwoo_app.celery_tasks