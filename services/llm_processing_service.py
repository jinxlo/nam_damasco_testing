# namwoo_app/services/llm_processing_service.py
import logging
from typing import Optional

from openai import OpenAI, APIError, APITimeoutError
from ..config import Config
from ..utils.text_utils import strip_html_to_text

logger = logging.getLogger(__name__)

# --- START OF MODIFICATION: Create a dedicated, self-contained client ---
# This client is ONLY for the purpose of summarization within this service.
try:
    # It correctly reads the API key from your Config, which reads from .env
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in configuration. Summarization service will fail.")
    
    # Initialize the client specifically for this service's needs.
    llm_client = OpenAI(
        api_key=Config.OPENAI_API_KEY,
        timeout=getattr(Config, 'OPENAI_REQUEST_TIMEOUT', 30.0)
    )
    logger.info("OpenAI client for LLM Processing Service (Summarization) initialized.")
except Exception as e:
    logger.exception("Failed to initialize OpenAI client for LLM Processing Service.")
    llm_client = None
# --- END OF MODIFICATION ---


def generate_llm_product_summary(
    html_description: Optional[str],
    item_name: Optional[str] = None
) -> Optional[str]:
    """
    Generates a product summary using the OpenAI Chat Completions API directly.
    It first strips HTML from the description before sending to the LLM.
    """
    # --- START OF MODIFICATION: Simplified logic ---
    if not llm_client:
        logger.error("OpenAI client not available for summarization. Check API key configuration.")
        return None

    if not html_description:
        logger.debug("No HTML description provided for summarization.")
        return None

    plain_text_description = strip_html_to_text(html_description)
    if not plain_text_description or len(plain_text_description) < 40: # Don't summarize very short text
        logger.debug(f"Description for '{item_name or 'Unknown'}' is too short after stripping HTML; skipping summarization.")
        return None

    # This service will now always use OpenAI for summarization.
    # The provider switch is removed, fixing the AttributeError.
    model = getattr(Config, "OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
    logger.info(f"Generating product summary for '{item_name or 'Unknown'}' using direct call to OpenAI model: {model}")

    system_prompt = (
        "Eres un experto en marketing de productos de tecnología. Tu tarea es tomar la siguiente descripción de un "
        "producto y reescribirla como un resumen atractivo, conciso y comercial de 2 a 4 oraciones. "
        "Enfócate en los beneficios clave y las características más importantes. Evita la jerga técnica excesiva. "
        "No uses frases como 'Este producto es' o 'En resumen'. Simplemente escribe el resumen."
    )
    
    user_prompt = f"Producto: {item_name or 'No especificado'}\n\nDescripción a resumir:\n\"\"\"{plain_text_description}\"\"\""

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=256
        )
        summary = response.choices[0].message.content

        if summary:
            logger.info(f"Successfully generated summary via OpenAI for '{item_name or 'Unknown'}'.")
            return summary.strip()
        else:
            logger.warning(f"Summarization via OpenAI for '{item_name or 'Unknown'}' returned no summary text.")
            return None

    except (APIError, APITimeoutError) as e:
        logger.error(f"OpenAI API Error during summarization for '{item_name}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during summarization for '{item_name}': {e}")
        return None
    # --- END OF MODIFICATION ---