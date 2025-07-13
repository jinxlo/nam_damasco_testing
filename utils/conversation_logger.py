import logging
from typing import Optional

conversation_logger = logging.getLogger('conversation')


def log_conversation_event(
    event_type: str,
    conversation_id: str,
    user_id: str,
    source: str,
    message: Optional[str] = None,
) -> None:
    """Log a conversation event with standardized metadata."""
    conversation_logger.info(
        message or "",
        extra={
            "event_type": event_type,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "source": source,
        },
    )
