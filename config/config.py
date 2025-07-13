# namwoo_app/config/config.py
# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

# ==============================================================================
# === START OF ADDED BLOCK FOR TESTING =========================================
# This is the ONLY addition.
# This block checks if we are running tests and loads the special .env.test file.
# It has NO effect on your production server.
# ==============================================================================

# Determine the absolute path to the project's root directory (e.g., 'ec2-user/').
# This goes up two levels from this file's directory (namwoo_app/config/).
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Check the FLASK_ENV environment variable, which our test setup (conftest.py) sets.
if os.environ.get('FLASK_ENV') == 'testing':
    # If we are testing, define the path to our special test environment file.
    test_dotenv_path = os.path.join(project_root_dir, '.env.test')
    
    # If the .env.test file exists, load it.
    if os.path.exists(test_dotenv_path):
        # The `override=True` flag ensures that variables in this file
        # (like the test DATABASE_URL) will replace any others from the main .env.
        load_dotenv(dotenv_path=test_dotenv_path, override=True)
        print(f"DEBUG [config.py]: LOADED TEST CONFIG from: {test_dotenv_path}")

# ==============================================================================
# === END OF ADDED BLOCK FOR TESTING ===========================================
# ==============================================================================


# --- Your Original Code Starts Here (Unchanged) ---
# This part of the code still runs in all environments. On your EC2 server,
# the block above will be skipped, and only this part will run.
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(basedir, '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"DEBUG [config.py]: Loaded .env from: {dotenv_path}")
else:
    print(f"WARNING: .env file not found at {dotenv_path}")

class Config:
    # --- Flask App ---
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-insecure-secret-key')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'production')
    DEBUG = FLASK_ENV == 'development'

    # --- Logging ---
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
    LOG_DIR = os.path.join(basedir, 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, 'app.log')
    SYNC_LOG_FILE = os.path.join(LOG_DIR, 'sync.log')
    LOG_JSON_FILE = os.path.join(LOG_DIR, 'app.json')

    # --- AI Configuration (Standardized) ---
    AI_PROVIDER = os.environ.get('AI_PROVIDER', 'openai_chat').lower()
    
    # OpenAI Specific
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_ASSISTANT_ID = os.environ.get('OPENAI_ASSISTANT_ID')
    OPENAI_EMBEDDING_MODEL = os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    OPENAI_CHAT_MODEL = os.environ.get('OPENAI_CHAT_MODEL', 'gpt-4o-mini')
    OPENAI_MAX_TOKENS = int(os.environ.get('OPENAI_MAX_TOKENS', 1024))
    EMBEDDING_DIMENSION = int(os.environ.get('EMBEDDING_DIMENSION', 1536))

    # Google Specific
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    GOOGLE_GEMINI_MODEL = os.environ.get('GOOGLE_GEMINI_MODEL', 'gemini-1.5-flash-latest')
    GOOGLE_MAX_TOKENS = int(os.environ.get('GOOGLE_MAX_TOKENS', 2048))

    # --- PostgreSQL Database ---
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)
    
    # Add this line to ensure conftest.py can find the variable by this name.
    DATABASE_URL = SQLALCHEMY_DATABASE_URI

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = DEBUG

    # --- Celery 5+ Configuration ---
    broker_url = os.environ.get('broker_url', 'redis://localhost:6379/0')
    result_backend = os.environ.get('result_backend', 'redis://localhost:6379/0')
    task_serializer = os.environ.get('task_serializer', 'json')
    result_serializer = os.environ.get('result_serializer', 'json')
    accept_content = [os.environ.get('accept_content', 'json')]
    timezone = os.environ.get('timezone', 'America/Caracas')
    enable_utc = os.environ.get('enable_utc', 'true').lower() == 'true'
    REDIS_URL = os.environ.get('REDIS_URL', broker_url)

    # --- Support Board Configuration ---
    SUPPORT_BOARD_API_URL = os.environ.get('SUPPORT_BOARD_API_URL')
    SUPPORT_BOARD_API_TOKEN = os.environ.get('SUPPORT_BOARD_API_TOKEN')
    SUPPORT_BOARD_WEBHOOK_SECRET = os.environ.get('SUPPORT_BOARD_WEBHOOK_SECRET')

    # --- Bot and Agent User IDs in Support Board ---
    SUPPORT_BOARD_DM_BOT_USER_ID = os.environ.get('SUPPORT_BOARD_DM_BOT_USER_ID')
    COMMENT_BOT_PROXY_USER_ID = os.environ.get('COMMENT_BOT_PROXY_USER_ID')
    COMMENT_BOT_INITIATION_TAG = os.environ.get('COMMENT_BOT_INITIATION_TAG')
    _agent_ids_str = os.environ.get('SUPPORT_BOARD_AGENT_IDS', '')
    SUPPORT_BOARD_AGENT_IDS = {id.strip() for id in _agent_ids_str.split(',') if id.strip()} if _agent_ids_str else set()
    HUMAN_TAKEOVER_PAUSE_MINUTES = int(os.environ.get('HUMAN_TAKEOVER_PAUSE_MINUTES', 43200))
    
    # --- Department IDs ---
    SUPPORT_BOARD_SUPPORT_DEPARTMENT_ID = int(os.getenv("SUPPORT_BOARD_SUPPORT_DEPARTMENT_ID", "1"))
    SUPPORT_BOARD_SALES_DEPARTMENT_ID = int(os.getenv("SUPPORT_BOARD_SALES_DEPARTMENT_ID", 2))
    SUPPORT_BOARD_ATENCION_AL_CLIENTE_ID = int(os.getenv("SUPPORT_BOARD_ATENCION_AL_CLIENTE_ID", 3))
    SUPPORT_DEPARTMENT_ID = int(os.getenv("SUPPORT_DEPARTMENT_ID", "1"))
    SALES_DEPARTMENT_ID = int(os.getenv("SALES_DEPARTMENT_ID", "2"))
    ATENCION_AL_CLIENT_ID = int(os.getenv("ATENCION_AL_CLIENT_ID", "3"))

    # --- WhatsApp Cloud API ---
    WHATSAPP_CLOUD_API_TOKEN = os.environ.get('WHATSAPP_CLOUD_API_TOKEN')
    WHATSAPP_PHONE_NUMBER_ID = os.environ.get('WHATSAPP_PHONE_NUMBER_ID')
    WHATSAPP_DEFAULT_COUNTRY_CODE = os.environ.get('WHATSAPP_DEFAULT_COUNTRY_CODE', '')
    WHATSAPP_API_VERSION = os.environ.get('WHATSAPP_API_VERSION', 'v19.0')
    LOG_WA_MESSAGES_INTERNALLY = os.environ.get('LOG_WA_MESSAGES_INTERNALLY', 'false').lower() == 'true'

    # --- Application Specific ---
    MAX_HISTORY_MESSAGES = int(os.environ.get('MAX_HISTORY_MESSAGES', 16))
    PRODUCT_SEARCH_LIMIT = max(5, int(os.environ.get('PRODUCT_SEARCH_LIMIT', 10)))
    IDEMPOTENCY_TTL = int(os.getenv("IDEMPOTENCY_TTL", "300"))

    # --- Damasco Specific ---
    DAMASCO_RECEIVER_API_URL = os.environ.get('DAMASCO_RECEIVER_API_URL')
    DAMASCO_API_SECRET = os.environ.get('DAMASCO_API_SECRET')

    if not DAMASCO_API_SECRET:
        print("WARNING [Config]: DAMASCO_API_SECRET is not set. Receiver endpoint will reject requests.")

    # --- System Prompt for AI Assistant (from file) ---
    SYSTEM_PROMPT_FILE = os.path.join(basedir, 'data', 'system_prompt.txt')
    try:
        with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
            SYSTEM_PROMPT = f.read().strip()
        print(f"INFO [Config]: Loaded system prompt from {SYSTEM_PROMPT_FILE}")
    except FileNotFoundError:
        print(f"ERROR [Config]: system_prompt.txt not found at {SYSTEM_PROMPT_FILE}. Using fallback.")
        SYSTEM_PROMPT = "Default fallback system prompt content here."


# --- Config Sanity Check ---
if __name__ != "__main__":
    print(f"--- Config Initialized ---")
    print(f"ENV: {Config.FLASK_ENV}, DEBUG={Config.DEBUG}, AI Provider: {Config.AI_PROVIDER}")
    print(f"DB URI: {'SET' if Config.SQLALCHEMY_DATABASE_URI else 'MISSING'}")
    print(f"Celery broker_url: {Config.broker_url}")
    print(f"Celery result_backend: {Config.result_backend}")
    print(f"Support Board URL: {Config.SUPPORT_BOARD_API_URL}")
    print(f"DM Bot User ID: {Config.SUPPORT_BOARD_DM_BOT_USER_ID}")
    print(f"Comment Bot Proxy User ID: {Config.COMMENT_BOT_PROXY_USER_ID}")
    print(f"Human Agent IDs: {Config.SUPPORT_BOARD_AGENT_IDS}")
    print(f"Comment Bot Initiation Tag: '{Config.COMMENT_BOT_INITIATION_TAG}' (None or empty means not used)")
    print(f"WhatsApp Config Loaded: {'Yes' if Config.WHATSAPP_CLOUD_API_TOKEN else 'No'}")
    print(f"Damasco Receiver API URL: {Config.DAMASCO_RECEIVER_API_URL}")
    print(f"Damasco API Secret Loaded: {'Yes' if Config.DAMASCO_API_SECRET else 'No'}")
    print(f"Support Dept ID: {Config.SUPPORT_BOARD_SUPPORT_DEPARTMENT_ID}")
    print(f"Sales Dept ID: {Config.SUPPORT_BOARD_SALES_DEPARTMENT_ID}")
    print(f"Support Dept (generic): {Config.SUPPORT_DEPARTMENT_ID}")
    print(f"Sales Dept (generic): {Config.SALES_DEPARTMENT_ID}")
    print(f"--------------------")