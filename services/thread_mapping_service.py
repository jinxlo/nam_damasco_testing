# namwoo/services/thread_mapping_service.py
import logging
from typing import Optional

# Correctly import the utility for getting a DB session and the SQLAlchemy model
from ..utils import db_utils
from ..models.thread_map import ConversationThreadMap

logger = logging.getLogger(__name__)

def get_thread_id(sb_conversation_id: str) -> Optional[str]:
    """
    Retrieves the OpenAI thread_id for a given Support Board conversation ID from the database.
    """
    if not sb_conversation_id:
        return None

    # Use the established context manager for safe database sessions
    with db_utils.get_db_session() as session:
        if not session:
            logger.error(f"DB session not available for get_thread_id of conv {sb_conversation_id}")
            return None
        
        try:
            # Query the mapping using the session object
            mapping = session.query(ConversationThreadMap).filter(
                ConversationThreadMap.sb_conversation_id == sb_conversation_id
            ).first()
            
            if mapping:
                logger.info(f"Found existing thread_id '{mapping.openai_thread_id}' for Conv '{sb_conversation_id}'.")
                return mapping.openai_thread_id
            
        except Exception as e:
            logger.exception(f"Database error while fetching thread_id for conv {sb_conversation_id}: {e}")

    return None

def store_thread_id(sb_conversation_id: str, thread_id: str) -> bool:
    """
    Stores a new mapping between a Support Board conversation ID and an OpenAI thread_id.
    Returns True on success, False on failure.
    """
    if not sb_conversation_id or not thread_id:
        logger.warning("store_thread_id called with empty sb_conversation_id or thread_id.")
        return False

    with db_utils.get_db_session() as session:
        if not session:
            logger.error(f"DB session not available for store_thread_id of conv {sb_conversation_id}")
            return False

        # Check if a mapping already exists to prevent duplicates.
        existing_mapping = session.query(ConversationThreadMap).filter(
            ConversationThreadMap.sb_conversation_id == sb_conversation_id
        ).first()

        if existing_mapping:
            logger.warning(f"Attempted to store a thread mapping for Conv '{sb_conversation_id}', but one already exists.")
            return True # Consider it a success if the mapping is already there.

        try:
            new_mapping = ConversationThreadMap(
                sb_conversation_id=sb_conversation_id,
                openai_thread_id=thread_id
            )
            session.add(new_mapping)
            session.commit()
            logger.info(f"Successfully stored new mapping for Conv '{sb_conversation_id}': '{thread_id}'.")
            return True
        except Exception as e:
            logger.exception(f"Database error while storing new thread mapping for conv {sb_conversation_id}: {e}")
            session.rollback()
            return False