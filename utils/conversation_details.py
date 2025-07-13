# namwoo_app/utils/conversation_details.py
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# --- In-Memory Cache for Conversation Reservation Details ---
# The key is the conversation_id (string).
# The value is a dictionary holding all reservation data for that conversation.
_conversation_reservations: Dict[str, Dict[str, Any]] = {}


# --- Core Data Keys ---
# Using constants for keys prevents typos and makes the code more maintainable.
KEY_ITEM_CODE = "itemCode_seleccionado"
KEY_ITEM_NAME = "nombre_producto_seleccionado"
KEY_FULL_NAME = "full_name"
KEY_CEDULA = "cedula"
KEY_PHONE = "telefono"
KEY_EMAIL = "correo"
KEY_DELIVERY_METHOD = "delivery_method"
KEY_BRANCH_NAME = "branch_name_seleccionado"
KEY_BRANCH_ADDRESS = "branch_address_seleccionada"
KEY_DELIVERY_ADDRESS = "delivery_address"
KEY_PAYMENT_METHOD = "payment_method"
KEY_ACCESSORIES = "accessories" # To store selected accessories
KEY_PRICE_USD = "price_usd" # ADDED
KEY_PRICE_VES = "price_ves" # ADDED
KEY_TOTAL_USD = "total_usd" # ADDED - For final total including accessories
KEY_TOTAL_VES = "total_ves" # ADDED - For final total including accessories


def _get_or_create_reservation(conversation_id: str) -> Dict[str, Any]:
    """
    Internal helper to safely get the reservation dictionary for a conversation,
    creating it if it doesn't exist.
    """
    if conversation_id not in _conversation_reservations:
        logger.debug(f"Creating new reservation data store for conversation_id: {conversation_id}")
        _conversation_reservations[conversation_id] = {}
    return _conversation_reservations[conversation_id]


def store_reservation_detail(conversation_id: str, key: str, value: Any) -> None:
    """
    Stores a single piece of reservation data for a given conversation.

    Args:
        conversation_id: The unique ID of the conversation.
        key: The data key to store (e.g., 'full_name', 'itemCode_seleccionado').
        value: The data to store.
    """
    if not conversation_id or not key:
        logger.warning("store_reservation_detail called with empty conversation_id or key.")
        return

    reservation = _get_or_create_reservation(conversation_id)
    reservation[key] = value
    logger.info(f"Stored detail '{key}' for conversation {conversation_id}.")
    logger.debug(f"Current reservation state for {conversation_id}: {reservation}")


def get_reservation_details(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the complete dictionary of reservation details for a conversation.
    Returns None if no data exists for that conversation.
    """
    return _conversation_reservations.get(conversation_id)


def get_specific_detail(conversation_id: str, key: str) -> Optional[Any]:
    """
    Retrieves a single specific detail from a conversation's reservation data.
    """
    reservation = get_reservation_details(conversation_id)
    if reservation:
        return reservation.get(key)
    return None


def get_missing_details(conversation_id: str) -> List[str]:
    """
    Checks the reservation data for a conversation and returns a list of
    keys that are still missing for a complete reservation.
    """
    reservation = get_reservation_details(conversation_id)
    if not reservation:
        # If no reservation has started, these are the first things to ask for.
        return [KEY_FULL_NAME, KEY_CEDULA, KEY_PHONE, KEY_EMAIL, KEY_DELIVERY_METHOD]

    # These details are always required for the base reservation.
    required_details = [
        KEY_FULL_NAME,
        KEY_CEDULA,
        KEY_PHONE,
        KEY_EMAIL,
        KEY_DELIVERY_METHOD,
    ]
    
    missing = [key for key in required_details if key not in reservation]

    # Conditional logic: if delivery method is pickup, we need a branch.
    if reservation.get(KEY_DELIVERY_METHOD) == "retiro_en_tienda" and KEY_BRANCH_NAME not in reservation:
        missing.append(KEY_BRANCH_NAME)
    
    # Conditional logic: if delivery method is home delivery, we need an address.
    if reservation.get(KEY_DELIVERY_METHOD) == "entrega_a_domicilio" and KEY_DELIVERY_ADDRESS not in reservation:
        missing.append(KEY_DELIVERY_ADDRESS)
        
    # Payment method is always the last thing asked after all other details are complete.
    if not missing: # Only check for payment method if nothing else is missing
        if KEY_PAYMENT_METHOD not in reservation:
            missing.append(KEY_PAYMENT_METHOD)

    logger.debug(f"Missing details for conversation {conversation_id}: {missing}")
    return missing


def clear_reservation_details(conversation_id: str) -> None:
    """
    Removes all stored reservation data for a specific conversation,
    for example, after a successful reservation or timeout.
    """
    if conversation_id in _conversation_reservations:
        del _conversation_reservations[conversation_id]
        logger.info(f"Cleared all reservation details for conversation {conversation_id}.")