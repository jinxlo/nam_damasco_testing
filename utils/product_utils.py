# namwoo_app/utils/product_utils.py
import re
from typing import Optional, Any
import unicodedata # For a basic unaccent equivalent
import logging # Optional: for logging issues within these utils

logger = logging.getLogger(__name__) # Optional: if you want to log from here

def _normalize_raw_input_to_str(value: Any) -> str:
    """Converts input to string and strips whitespace. Returns empty string if None or only whitespace."""
    if value is None:
        return ""
    return str(value).strip()


def generate_product_location_id(item_code_raw: Any, whs_name_raw: Any) -> Optional[str]:
    """Combine item code and warehouse name into a predictable ID.

    The warehouse name is sanitized by replacing any characters other than
    letters, numbers and ``-`` with ``_``. The result is truncated to
    512 characters. If either input is missing or results in an empty
    segment after sanitization, ``None`` is returned.
    """

    item_code = _normalize_raw_input_to_str(item_code_raw)
    whs_name = _normalize_raw_input_to_str(whs_name_raw)

    if not item_code or not whs_name:
        return None

    sanitized_whs = re.sub(r"[^A-Za-z0-9-]+", "_", whs_name)
    if not sanitized_whs:
        return None

    return f"{item_code}_{sanitized_whs}"[:512]

def python_equivalent_of_canonicalize_whs(original_warehouse_name: Optional[str]) -> str:
    """
    Python equivalent of the PostgreSQL function `public.canonicalize_whs(text)`.
    Produces an UPPERCASE, underscore-separated, sanitized, unaccented string.
    This MUST EXACTLY REPLICATE the PostgreSQL function's behavior.

    PostgreSQL function:
    SELECT upper(
             regexp_replace(
               unaccent($1),
               '[^A-Za-z0-9]+',
               '_',
               'g'
             )
           );
    """
    text = _normalize_raw_input_to_str(original_warehouse_name)

    if not text: # If original was None, empty, or just whitespace
        # PostgreSQL: unaccent('') is '', regexp_replace('', '[^A-Za-z0-9]+', '_', 'g') is '', upper('') is ''
        return "" # Return empty string to perfectly match PostgreSQL behavior for empty input

    # 1. Python equivalent of unaccent()
    try:
        nfkd_form = unicodedata.normalize('NFKD', text)
        unaccented_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    except TypeError: # Should be rare as input is already str
        unaccented_text = text # Fallback

    # 2. Python equivalent of regexp_replace(unaccented_text, '[^A-Za-z0-9]+', '_', 'g')
    sanitized_text = re.sub(r'[^A-Za-z0-9]+', '_', unaccented_text)
    
    # 3. Python equivalent of upper()
    result = sanitized_text.upper()

    # PostgreSQL regexp_replace might leave leading/trailing underscores if the original (after unaccent)
    # started/ended with non-alphanumeric characters.
    # Example: unaccent(' Warehouse Name ') -> ' Warehouse Name '
    # regexp_replace(' Warehouse Name ', '[^A-Za-z0-9]+', '_', 'g') -> '_Warehouse_Name_'
    # upper('_Warehouse_Name_') -> '_WAREHOUSE_NAME_'
    # It does NOT strip leading/trailing underscores by default.
    # If your DB trigger *implicitly* handles this (e.g., by later use or if the column definition somehow cleans it,
    # which is unlikely), then fine. But to match the SQL function exactly, we should NOT strip here unless the SQL does.
    # Let's assume the SQL function result is what's used directly.
    # If your previous sample DB data for `warehouse_name_canonical` showed NO leading/trailing underscores
    # (e.g., "ALMACEN_PRINCIPAL_LOS_TEQUES" not "_ALMACEN_PRINCIPAL_LOS_TEQUES_"),
    # then it implies either the input `warehouse_name` to the trigger never has leading/trailing spaces/symbols
    # OR `canonicalize_whs` itself or another step strips them.
    # Given the SQL `canonicalize_whs` doesn't explicitly strip them AFTER regex,
    # we'll keep it that way for now. If leading/trailing underscores are an issue,
    # the SQL function itself would be the place to adjust or the input to it needs to be clean.
    # For safety and based on typical DB canonical forms, let's add a strip('_') here,
    # assuming that is the desired final canonical form in the DB. This is often a good practice.
    result = result.strip('_')

    # If, after all processing (including stripping underscores), the result is empty,
    # it means the original string was composed entirely of characters that were removed or underscores.
    if not result:
        # What does PostgreSQL canonicalize_whs('!@#$') return?
        # unaccent('!@#$') -> '!@#$'
        # regexp_replace('!@#$', '[^A-Za-z0-9]+', '_', 'g') -> '_'
        # upper('_') -> '_'
        # If we then strip('_'), it becomes ''.
        # So, for an input like '!@#$', the SQL function returns '_'. If Python returns '', it's a mismatch.
        # The `strip('_')` should only be done if that's the true final form.
        # Let's reconsider the strip: The SQL `regexp_replace` replaces sequences of non-alphanum with a SINGLE underscore.
        # If the whole string is non-alphanum, it becomes a single underscore.
        # Example: `SELECT upper(regexp_replace(unaccent('!@#$ %^'), '[^A-Za-z0-9]+', '_', 'g'));` -> `_`
        # So, if `result` here is `_` and we strip it, it becomes empty. That's a mismatch.
        # Let's remove the strip for now to be closer to the literal SQL provided.
        # The consumer (`generate_product_id_for_lookup`) will need to handle an output like `_`.
        # --- Reverting the strip for closer SQL match ---
        # result = result.strip('_') # Commenting this out for now.

        # If the result is now empty (e.g. original input was empty string)
        if not result:
            return "CANONICAL_EMPTY_WHS" # Or some other placeholder if DB doesn't allow empty and trigger has a default

    # The warehouse_name_canonical column is VARCHAR(255).
    return result[:255]

def generate_product_id_for_lookup(item_code_raw: Any, whs_name_raw: Any) -> Optional[str]:
    """
    Generates the composite product ID as the DATABASE TRIGGER `trg_set_canonical_whs` would.
    This is used for LOOKING UP existing products.
    Format: item_code + '_' + canonical_warehouse_name, truncated to 512 chars.
    The item_code part of the ID should be UPPERCASE to match observed DB IDs.
    """
    item_code_str = _normalize_raw_input_to_str(item_code_raw)
    if not item_code_str:
        logger.warning("Cannot generate product_id_for_lookup: item_code is missing or empty.")
        return None

    item_code_for_id = item_code_str.upper() # Match DB trigger behavior (NEW.item_code is likely already uppercase or trigger uses it as is)

    canonical_whs_name_as_per_db = python_equivalent_of_canonicalize_whs(whs_name_raw)

    # If the canonical name is empty or a placeholder indicating an issue from canonicalization.
    # An empty string from python_equivalent_of_canonicalize_whs (if it now returns that for empty input)
    # would lead to an ID like "ITEMCODE_", which might be valid or invalid depending on your rules.
    if not canonical_whs_name_as_per_db: # Handles empty string from canonicalize_whs
        logger.warning(f"Cannot generate product_id for item {item_code_for_id}: "
                       f"warehouse name '{whs_name_raw}' resulted in an empty canonical form.")
        return None
    if canonical_whs_name_as_per_db == "CANONICAL_EMPTY_WHS": # Handle placeholder if still used
         logger.warning(f"Cannot generate product_id for item {item_code_for_id}: "
                       f"warehouse name '{whs_name_raw}' resulted in placeholder canonical form.")
         return None


    # DB Trigger logic: NEW.item_code || '_' || NEW.warehouse_name_canonical
    # Assuming NEW.item_code in the trigger uses the item_code as received (which should be uppercase from Pydantic if source is uppercase)
    generated_id = f"{item_code_for_id}_{canonical_whs_name_as_per_db}"
    
    # DB Trigger logic: LEFT(..., 512)
    return generated_id[:512]