# NAMWOO/models/thread_map.py
# -*- coding: utf-8 -*-

from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ConversationThreadMap(Base):
    """
    SQLAlchemy model for mapping Support Board conversation IDs to OpenAI thread IDs.
    This is essential for the Assistants API to maintain conversation state.
    """
    __tablename__ = 'conversation_thread_map'

    # The Support Board conversation ID is the primary key as it's the unique identifier in our system.
    sb_conversation_id = Column(String, primary_key=True, nullable=False)
    
    # The corresponding thread_id from the OpenAI Assistants API.
    openai_thread_id = Column(String, nullable=False, index=True)

    # Timestamp for when the mapping was created.
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ConversationThreadMap(sb_conversation_id='{self.sb_conversation_id}', openai_thread_id='{self.openai_thread_id}')>"