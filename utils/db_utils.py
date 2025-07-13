# namwoo_app/utils/db_utils.py

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session, Session as SQLAlchemySession
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from typing import Optional, Generator, List, Dict

import datetime
from datetime import timezone

from ..models import Base
from ..models.conversation_pause import ConversationPause
from ..config import Config

logger = logging.getLogger(__name__)

db_session: scoped_session = None
_engine = None
_SessionFactory = None


def init_db(app) -> bool:
    """
    Initialize the SQLAlchemy engine and session factories using app config.
    This function remains unchanged and is essential for the application.
    """
    global _engine, _SessionFactory, db_session
    if _engine is not None:
        return True

    db_uri = app.config.get('SQLALCHEMY_DATABASE_URI')
    if not db_uri:
        logger.error("SQLALCHEMY_DATABASE_URI not configured. Database features will fail.")
        return False

    try:
        loggable_db_uri = '@'.join(db_uri.split('@')[1:]) if '@' in db_uri else db_uri
        logger.info(f"Initializing database engine for: {loggable_db_uri}")
        
        _engine = create_engine(
            db_uri,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=app.config.get('SQLALCHEMY_ECHO', False)
        )
        _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        db_session = scoped_session(_SessionFactory)
        logger.info("SQLAlchemy engine and session factory have been configured successfully.")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        return False

@contextmanager
def get_db_session() -> Generator[Optional[SQLAlchemySession], None, None]:
    """
    Context manager for safe database sessions. This is a core utility and remains unchanged.
    """
    if not db_session:
        logger.error("db_session not initialized. Cannot create DB session.")
        yield None 
        return

    session: SQLAlchemySession = db_session()
    try:
        yield session
    except SQLAlchemyError as e:
        logger.error(f"DB Session SQLAlchemy error: {e}", exc_info=True)
        session.rollback()
        raise
    except Exception as e:
        logger.error(f"DB Session unexpected error: {e}", exc_info=True)
        session.rollback()
        raise
    finally:
        db_session.remove()

def create_all_tables(app) -> bool:
    """
    Creates all tables. This is a core setup utility and remains unchanged.
    """
    if not _engine:
        logger.error("Database engine not initialized. Cannot create tables.")
        return False
    try:
        logger.info("Ensuring all tables from SQLAlchemy models exist...")
        Base.metadata.create_all(bind=_engine)
        with _engine.connect() as connection:
            with connection.begin():
                logger.info("Ensuring pgvector extension exists...")
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        logger.info("Table and extension check complete.")
        return True
    except Exception as e:
        logger.error(f"Error during create_all_tables: {e}", exc_info=True)
        return False

# --- DEPRECATED HISTORY FUNCTIONS ---
# The following two functions represent a legacy history management system.
# The current implementation uses support_board_service.get_sb_conversation_data
# as the single source of truth for conversation history. These functions should
# be considered for removal to avoid confusion and code duplication.

def fetch_history(session_id: str) -> Optional[List[Dict]]:
    """
    DEPRECATED: Fetches conversation history from a custom table.
    This is not used by the current AI providers, which rely on Support Board's history.
    """
    logger.warning("DEPRECATED: fetch_history was called. This function is obsolete.")
    with get_db_session() as session:
        if not session: return None
        from ..models.history import ConversationHistory
        try:
            record = session.query(ConversationHistory).filter_by(session_id=session_id).first()
            return record.history_data if record else []
        except Exception as e:
            logger.exception(f"Error in deprecated fetch_history for session {session_id}: {e}")
            return None

def save_history(session_id: str, history_list: List[Dict]) -> bool:
    """
    DEPRECATED: Saves conversation history to a custom table.
    This is not used by the current AI providers.
    """
    logger.warning("DEPRECATED: save_history was called. This function is obsolete.")
    with get_db_session() as session:
        if not session: return False
        from ..models.history import ConversationHistory
        try:
            record = session.query(ConversationHistory).filter_by(session_id=session_id).first()
            if record:
                record.history_data = history_list
            else:
                record = ConversationHistory(session_id=session_id, history_data=history_list)
                session.add(record)
            session.commit()
            return True
        except Exception as e:
            logger.exception(f"Error in deprecated save_history for session {session_id}: {e}")
            return False

# --- CONVERSATION PAUSE MANAGEMENT ---
# This logic is essential for business rules and remains unchanged.

def is_conversation_paused(conversation_id: str) -> bool:
    """Checks if a conversation is currently paused."""
    with get_db_session() as session:
        if not session: return False
        try:
            now_utc = datetime.datetime.now(timezone.utc)
            pause_record = session.query(ConversationPause)\
                .filter(ConversationPause.conversation_id == conversation_id)\
                .filter(ConversationPause.paused_until > now_utc)\
                .first()
            return pause_record is not None
        except Exception as e:
            logger.exception(f"Error checking pause status for conversation {conversation_id}: {e}")
            return False

def get_pause_record(conversation_id: str) -> Optional[ConversationPause]:
    """Retrieves the current active pause record for a conversation, if any."""
    with get_db_session() as session:
        if not session: return None
        try:
            now_utc = datetime.datetime.now(timezone.utc)
            return session.query(ConversationPause)\
                .filter(ConversationPause.conversation_id == conversation_id)\
                .filter(ConversationPause.paused_until > now_utc)\
                .first()
        except Exception as e:
            logger.exception(f"Error getting pause record for conversation {conversation_id}: {e}")
            return None

def pause_conversation_for_duration(conversation_id: str, duration_seconds: int):
    """Sets or updates a pause for a conversation."""
    with get_db_session() as session:
        if not session: return
        try:
            pause_until_time = datetime.datetime.now(timezone.utc) + datetime.timedelta(seconds=duration_seconds)
            pause_record = session.query(ConversationPause).filter_by(conversation_id=conversation_id).first()
            if pause_record:
                pause_record.paused_until = pause_until_time
            else:
                pause_record = ConversationPause(conversation_id=conversation_id, paused_until=pause_until_time)
                session.add(pause_record)
            session.commit()
            logger.info(f"Set/updated pause for conversation {conversation_id} until {pause_until_time.isoformat()}.")
        except Exception as e:
            logger.exception(f"Error pausing conversation {conversation_id}: {e}")

def unpause_conversation(conversation_id: str):
    """Removes any active pause for a conversation."""
    with get_db_session() as session:
        if not session: return
        try:
            # This logic correctly deletes the pause record regardless of expiry
            num_deleted = session.query(ConversationPause).filter_by(conversation_id=conversation_id).delete()
            if num_deleted > 0:
                session.commit()
                logger.info(f"Deleted pause record for conversation {conversation_id}, effectively unpausing it.")
            else:
                logger.info(f"No pause record found to delete for conversation {conversation_id}.")
        except Exception as e:
            logger.exception(f"Error unpausing conversation {conversation_id}: {e}")