# namwoo_app/services/openai_service.py
import logging
from typing import Optional, List

from openai import OpenAI, APIError, APITimeoutError
from ..config import Config

logger = logging.getLogger(__name__)

# Initialize the client at the module level.
# It's thread-safe and can be reused.
try:
    # Ensure that OPENAI_API_KEY is loaded via your Config class
    if not Config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in configuration. Embedding service will fail.")
        client = None
    else:
        client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            timeout=getattr(Config, 'OPENAI_EMBEDDING_TIMEOUT', 20.0) # Use a specific timeout
        )
        logger.info("OpenAI client for embedding service initialized.")
except Exception as e:
    logger.exception("Failed to initialize OpenAI client for embedding service.")
    client = None

def generate_product_embedding(text: str) -> Optional[List[float]]:
    """
    Generates a vector embedding for a given text using the configured OpenAI model.

    Args:
        text (str): The text content to embed.

    Returns:
        Optional[List[float]]: A list of floats representing the embedding, or None on error.
    """
    if not client:
        logger.error("OpenAI client is not initialized; cannot generate embedding.")
        return None

    if not text or not isinstance(text, str):
        logger.warning("generate_product_embedding called with empty or invalid text.")
        return None

    # Get the embedding model name from your configuration
    model = getattr(Config, 'OPENAI_EMBEDDING_MODEL', "text-embedding-3-small")
    
    try:
        # Replace newlines, as recommended by OpenAI for embedding quality
        text = text.replace("\n", " ")
        
        logger.debug(f"Requesting embedding for text (first 100 chars): '{text[:100]}...' using model: {model}")
        
        response = client.embeddings.create(input=[text], model=model)
        
        embedding_vector = response.data[0].embedding
        
        logger.info(f"Successfully generated embedding for text (first 100 chars): '{text[:100]}...'")
        return embedding_vector

    except (APIError, APITimeoutError) as e:
        logger.error(f"OpenAI API error during embedding generation: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during embedding generation: {e}")
        return None