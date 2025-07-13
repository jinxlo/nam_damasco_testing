# -*- coding: utf-8 -*-
import requests
import logging
import json
import re # Import regex module for cleaning phone numbers
from flask import current_app
from typing import Optional, List, Dict, Any
from ..utils.conversation_logger import log_conversation_event

# Assuming Config is correctly imported and loads .env variables
from ..config import Config

logger = logging.getLogger(__name__)

# --- PRIVATE HELPER: Make Support Board API Call ---
def _call_sb_api(payload: Dict) -> Optional[Any]:
    """Internal helper to make POST requests to the Support Board API."""
    api_url = current_app.config.get('SUPPORT_BOARD_API_URL')
    api_token = current_app.config.get('SUPPORT_BOARD_API_TOKEN')

    if not api_url or not api_token:
        logger.error("Support Board API URL or Token is not configured.")
        return None

    payload['token'] = api_token
    function_name = payload.get('function', 'N/A')
    logger.debug(f"Calling SB API URL: {api_url} with function: {function_name}")
    try:
        log_payload = payload.copy()
        if 'token' in log_payload:
            log_payload['token'] = '***' + log_payload['token'][-4:] if len(log_payload.get('token','')) > 4 else '***'
        log_payload_str = json.dumps(log_payload)
    except Exception:
        log_payload_str = str(payload)
    logger.debug(f"Payload for {function_name} (requests data param): {log_payload_str}")

    try:
        response = requests.post(api_url, data=payload, timeout=20)
        response.raise_for_status()
        response_json = response.json()
        try:
            log_response_str = json.dumps(response_json)
        except Exception:
            log_response_str = str(response_json)
        logger.debug(f"Raw SB API response for {function_name}: {log_response_str}")

        if response_json.get("success") is True:
            # --- START OF FIX: Handle `response: false` from API ---
            # The API can return `false` for the 'response' key on a successful call
            # if the requested entity (e.g., user, conversation) is not found.
            response_data = response_json.get("response")
            if response_data is False:
                logger.warning(f"SB API for {function_name} reported success but returned 'false' in the response payload. Treating as not-found.")
                return None
            return response_data
            # --- END OF FIX ---
        else:
            error_detail = response_json.get("response", f"API call failed for function '{function_name}' with success=false or missing")
            logger.error(f"Support Board API reported failure for {function_name}: {error_detail}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error calling Support Board API ({function_name}): {e}", exc_info=True)
        if e.response is not None:
            logger.error(f"Response body from failed request: {e.response.text[:500]}")
        return None
    except requests.exceptions.JSONDecodeError as e:
        raw_text = getattr(response, 'text', 'N/A')
        logger.error(f"Failed to decode JSON response from Support Board API ({function_name}): {e}. Response text: {raw_text[:500]}", exc_info=True)
        return None
    except Exception as e:
        logger.exception(f"Unexpected error calling SB API ({function_name}): {e}")
        return None

# --- Public Function: Get Conversation Data ---
def get_sb_conversation_data(conversation_id: str) -> Optional[Dict]:
    """Fetches the full conversation details from Support Board."""
    payload = {
        'function': 'get-conversation',
        'conversation_id': conversation_id
    }
    logger.info(f"Attempting to fetch full conversation data from Support Board API for ID: {conversation_id}")
    response_data = _call_sb_api(payload)
    if isinstance(response_data, dict):
        if "messages" in response_data and "details" in response_data:
            logger.info(f"Successfully fetched conversation data for SB conversation {conversation_id}")
            return response_data
        else:
            logger.warning(f"SB API get-conversation success reported, but response for {conversation_id} might be incomplete or malformed. Response: {response_data}")
            return response_data
    else:
        # This will now correctly catch both API failures and the "not found" (`response: false`) case from the robust _call_sb_api
        logger.error(f"Failed to fetch or parse valid conversation data dictionary for SB conversation {conversation_id}. Raw response from _call_sb_api call was not a valid dictionary: {response_data}")
        return None

# --- NEW PUBLIC FUNCTION: Add Message to SB Conversation ---
def add_message_to_sb_conversation(conversation_id: str, message_data: dict) -> bool:
    """
    Adds a message to a Support Board conversation.
    Used for logging virtual tool calls.
    The message_data can contain 'message' for text or 'payload' for structured data.
    """
    bot_user_id = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID)
    if not bot_user_id:
        logger.error("Cannot add internal message: SUPPORT_BOARD_DM_BOT_USER_ID not configured.")
        return False
        
    payload = {
        'function': 'send-message',
        'user_id': bot_user_id,
        'conversation_id': conversation_id,
        'message': message_data.get('message', ''), # Text part, can be empty
        'payload': message_data.get('payload', '')   # Payload part, can be empty
    }
    
    logger.info(f"Adding internal/payload message to SB conv {conversation_id} as User ID {bot_user_id}")
    response = _call_sb_api(payload)

    if response:
        logger.info(f"Successfully added message to conv {conversation_id}. Response: {response}")
        return True
    else:
        logger.error(f"Failed to add message to conv {conversation_id}.")
        return False


def extract_waid_from_conversation(conversation_id: str, message_id: str) -> Optional[str]:
    """Return WA message ID from the conversation details if present."""
    conv_data = get_sb_conversation_data(conversation_id)
    if not conv_data:
        return None
    messages = conv_data.get("messages", [])
    for msg in messages:
        if str(msg.get("id")) == str(message_id):
            payload_str = msg.get("payload")
            if isinstance(payload_str, str):
                try:
                    inner = json.loads(payload_str)
                    waid = inner.get("waid")
                    if waid:
                        logger.info(
                            f"Extracted WAID {waid} from conversation {conversation_id} message {message_id}"
                        )
                        return waid
                except ValueError:
                    logger.warning(
                        f"Invalid JSON payload in conversation {conversation_id} message {message_id}: {payload_str}"
                    )
            break
    logger.debug(
        f"Could not find WAID in conversation {conversation_id} message {message_id}"
    )
    return None

# --- PUBLIC HELPER: Assign Conversation to Department ---
def assign_conversation_to_department(conversation_id: str, department_id: int):
    """Assign a conversation to a specific department using the SB API."""
    payload = {
        "function": "update-conversation-department",
        "conversation_id": str(conversation_id),
        "department": department_id,
        "token": current_app.config["SUPPORT_BOARD_API_TOKEN"],
    }
    return _call_sb_api(payload)

# --- PRIVATE HELPER: Get User PSID (for FB/IG) ---
def _get_user_psid(user_id: str) -> Optional[str]:
    """Fetches user details and extracts the PSID (Facebook/Instagram ID)."""
    logger.info(f"Attempting to fetch user details for User ID: {user_id} to get PSID.")
    payload = {'function': 'get-user', 'user_id': user_id, 'extra': 'true'}
    user_data = _call_sb_api(payload)
    if user_data and isinstance(user_data, dict):
        details_list = user_data.get('details', [])
        expected_slug = 'facebook-id' # As per common SB setup, adjust if different
        if isinstance(details_list, list):
            for detail in details_list:
                if isinstance(detail, dict) and detail.get('slug') == expected_slug:
                    psid = detail.get('value')
                    if psid and isinstance(psid, str) and psid.strip():
                        logger.info(f"Found PSID (using slug '{expected_slug}') for User ID {user_id}")
                        return psid.strip()
                    else:
                        logger.warning(f"Found '{expected_slug}' detail slug for user {user_id} but its value is empty or invalid: '{psid}'")
            logger.warning(f"Could not find PSID using slug '{expected_slug}' in the details list for User ID: {user_id}. Details received: {details_list}")
            return None
        else:
             logger.warning(f"User details for {user_id} received, but 'details' key is not a list: {details_list}")
             return None
    else:
        logger.error(f"Failed to fetch or parse valid user details dictionary for User ID: {user_id} needed for PSID lookup.")
        return None

# --- CORRECTED PRIVATE HELPER: Get User WAID (for WhatsApp) ---
def _get_user_waid(user_id: str) -> Optional[str]:
    """
    Fetches user details from SB API using 'get-user' + 'extra=true'
    and extracts/formats the WAID. Ignores any pre-fetched data.
    Prioritizes 'phone' detail slug, falls back to 'first_name' if it looks like a number.
    Requires WHATSAPP_DEFAULT_COUNTRY_CODE in config for numbers missing the prefix.
    """
    logger.info(f"Attempting to get WAID for User ID: {user_id}. ALWAYS fetching user details via 'get-user' + 'extra=true'.")
    phone_number = None
    user_first_name = None
    user_details_data = None # Ensure we fetch fresh data

    # --- ALWAYS Fetch user details using get-user + extra=true ---
    logger.debug(f"Fetching user details for {user_id} using 'get-user' + 'extra=true'.")
    payload = {'function': 'get-user', 'user_id': str(user_id), 'extra': 'true'}
    user_details_data = _call_sb_api(payload)
    # --- End Fetch ---

    if user_details_data and isinstance(user_details_data, dict):
        # Extract first name for potential fallback
        user_first_name = user_details_data.get('first_name')
        details_list = user_details_data.get('details', []) # Expecting a list here from get-user

        if isinstance(details_list, list):
            # Try to find the phone number in the details list
            for detail in details_list:
                if isinstance(detail, dict) and detail.get('slug') == 'phone':
                    phone_value = detail.get('value')
                    if phone_value and isinstance(phone_value, str) and phone_value.strip():
                        phone_number = phone_value.strip()
                        logger.info(f"Found phone number for User ID {user_id} in 'phone' detail from get-user response.")
                        break # Found it, stop looking
                    else:
                        logger.warning(f"Found 'phone' detail slug for user {user_id} but its value is empty or invalid: '{phone_value}'")

            if not phone_number:
                logger.warning(f"Could not find valid 'phone' detail in the details list returned by get-user for User ID {user_id}. Details received: {details_list}. Will check first_name as fallback.")
        else:
            logger.warning(f"User details fetched via get-user for {user_id} received, but 'details' key is not a list or is missing. Response: {user_details_data}")
    else:
        logger.error(f"Failed to fetch or parse valid user details dictionary via get-user for User ID: {user_id} needed for WAID lookup.")
        return None # Cannot proceed without user details

    if not phone_number:
        if user_first_name and isinstance(user_first_name, str):
            cleaned_first_name = user_first_name.strip()
            if re.fullmatch(r'[\d\s\-\(\)\+]+', cleaned_first_name) and len(re.sub(r'\D', '', cleaned_first_name)) >= 7:
                logger.info(f"Using 'first_name' field '{cleaned_first_name}' as fallback phone number for User ID {user_id}.")
                phone_number = cleaned_first_name
            else:
                 logger.warning(f"'first_name' for user {user_id} ('{cleaned_first_name}') does not appear to be a valid phone number. Cannot use as fallback.")
        else:
            logger.warning(f"Cannot fallback to first_name for user {user_id}: field is missing, not a string, or empty.")


    if not phone_number:
        logger.error(f"Could not determine phone number for User ID {user_id} from get-user details or first_name fallback.")
        return None

    waid = re.sub(r'\D', '', phone_number) # Remove non-digits

    if not phone_number.lstrip().startswith('+') and not waid.startswith(Config.WHATSAPP_DEFAULT_COUNTRY_CODE or ''):
        default_cc = Config.WHATSAPP_DEFAULT_COUNTRY_CODE
        if default_cc:
            default_cc_digits = re.sub(r'\D', '', default_cc)
            if default_cc_digits:
                logger.warning(f"Phone number '{phone_number}' for user {user_id} appears to be missing country code prefix. Prepending default: '{default_cc_digits}'.")
                waid = default_cc_digits + waid
            else:
                logger.error(f"Configured WHATSAPP_DEFAULT_COUNTRY_CODE ('{default_cc}') contains no digits. Cannot prepend.")
                return None
        else:
            logger.error(f"Phone number '{phone_number}' for user {user_id} is missing country code prefix, and WHATSAPP_DEFAULT_COUNTRY_CODE is not set or is invalid. Cannot form valid WAID.")
            return None
    elif not phone_number.lstrip().startswith('+') and waid.startswith(Config.WHATSAPP_DEFAULT_COUNTRY_CODE or ''):
        logger.debug(f"Phone number '{phone_number}' for user {user_id} seemed to be missing '+' but already started with the default country code. Assuming it's correct.")


    logger.info(f"Successfully derived WAID '{waid}' for User ID {user_id}.")
    return waid


# --- PRIVATE HELPER: Send Messenger/Instagram Message (External Delivery via SB API) ---
def _send_messenger_message(
    psid: str,
    page_id: str,
    message_text: str,
    conversation_id: str,
    triggering_message_id: Optional[str]
) -> bool:
    """Sends a message via the SB messenger-send-message API."""
    logger.debug(f"[_send_messenger_message CALLED] Conv ID: {conversation_id}")
    logger.debug(f"[_send_messenger_message] Received triggering_message_id: {repr(triggering_message_id)} (Type: {type(triggering_message_id)})")

    bot_user_id = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID) if Config.SUPPORT_BOARD_DM_BOT_USER_ID else None
    if not bot_user_id:
        logger.warning(f"SUPPORT_BOARD_DM_BOT_USER_ID not configured. This might affect how messages are attributed or if internal logging for the DM bot itself works as expected for this specific _send_messenger_message call, though external send will proceed if other params are fine.")

    logger.info(f"Attempting to send Messenger/IG message via specific SB API for Conv ID {conversation_id} to PSID: ...{psid[-6:]} on Page ID: {page_id}")

    payload = {
        'function': 'messenger-send-message',
        'psid': psid,
        'facebook_page_id': page_id,
        'message': message_text
    }

    if triggering_message_id is not None and str(triggering_message_id).strip() != '':
        logger.info(f"Including metadata (triggering message ID: {triggering_message_id}) in messenger-send-message call for conv {conversation_id}.")
        payload['metadata'] = str(triggering_message_id)
    else:
        logger.warning(f"No triggering message ID available/valid for conv {conversation_id}. Sending messenger message without metadata. Dashboard linking might fail.")

    try:
        log_payload_msg = json.dumps(payload)
    except Exception:
        log_payload_msg = str(payload)
    logger.debug(f"[_send_messenger_message] Final payload before API call: {log_payload_msg}")

    response_data = _call_sb_api(payload)
    logger.debug(f"[_send_messenger_message] response_data from _call_sb_api: {response_data} (Type: {type(response_data)})") 

    is_list = isinstance(response_data, list)
    logger.debug(f"[_send_messenger_message] Check: isinstance(response_data, list) = {is_list}") 

    if is_list:
        list_len = len(response_data)
        logger.debug(f"[_send_messenger_message] Check: len(response_data) = {list_len}") 
        if list_len > 0:
            first_element_is_dict = isinstance(response_data[0], dict)
            logger.debug(f"[_send_messenger_message] Check: isinstance(response_data[0], dict) = {first_element_is_dict}")
            if first_element_is_dict:
                has_recipient_id = 'recipient_id' in response_data[0]
                has_message_id = 'message_id' in response_data[0]
                logger.debug(f"[_send_messenger_message] Check: 'recipient_id' in response_data[0] = {has_recipient_id}")
                logger.debug(f"[_send_messenger_message] Check: 'message_id' in response_data[0] = {has_message_id}")

                if has_recipient_id and has_message_id:
                    fb_message_id = response_data[0].get('message_id', 'N/A')
                    logger.info(f"Messenger/IG message acknowledged as successful by SB API (FB Msg ID: {fb_message_id}) for Conv ID {conversation_id} to PSID ...{psid[-6:]} (Parsed from list response)")
                    log_conversation_event(
                        event_type="bot_reply",
                        conversation_id=conversation_id,
                        user_id=psid,
                        source="facebook",
                        message=message_text,
                    )
                    return True

    is_dict = isinstance(response_data, dict)
    logger.debug(f"[_send_messenger_message] Fallback Check: isinstance(response_data, dict) = {is_dict}")
    if is_dict:
        has_recipient_id_dict = 'recipient_id' in response_data
        has_message_id_dict = 'message_id' in response_data
        logger.debug(f"[_send_messenger_message] Fallback Check: 'recipient_id' in response_data = {has_recipient_id_dict}")
        logger.debug(f"[_send_messenger_message] Fallback Check: 'message_id' in response_data = {has_message_id_dict}")
        if has_recipient_id_dict and has_message_id_dict:
            fb_message_id = response_data.get('message_id', 'N/A')
            logger.info(f"Messenger/IG message acknowledged as successful by SB API (FB Msg ID: {fb_message_id}) for Conv ID {conversation_id} to PSID ...{psid[-6:]} (Parsed from dict response)")
            log_conversation_event(
                event_type="bot_reply",
                conversation_id=conversation_id,
                user_id=psid,
                source="facebook",
                message=message_text,
            )
            return True

    is_true_bool = response_data is True
    logger.debug(f"[_send_messenger_message] Fallback Check: response_data is True = {is_true_bool}")
    if is_true_bool:
        logger.warning(f"Messenger/IG message API call for Conv ID {conversation_id} returned 'True', which differs from documented structure, but treating as success.")
        log_conversation_event(
            event_type="bot_reply",
            conversation_id=conversation_id,
            user_id=psid,
            source="facebook",
            message=message_text,
        )
        return True

    logger.error(f"Failed to send Messenger/IG message via SB API for Conv ID {conversation_id} to PSID ...{psid[-6:]}. Unexpected response structure after all checks: {response_data}")
    return False


# --- PRIVATE HELPER: Add Message Internally to SB (Dashboard Visibility) ---
def _add_internal_sb_message(conversation_id: str, message_text: str, bot_user_id: str) -> bool:
    """Adds a message internally to the SB conversation using send-message."""
    if not bot_user_id:
        logger.error("Cannot add internal SB message: Bot User ID not provided or configured.")
        return False

    logger.info(f"Adding bot reply internally to SB conversation ID: {conversation_id} as User ID: {bot_user_id}")
    payload = {
        'function': 'send-message',
        'user_id': bot_user_id,
        'conversation_id': conversation_id,
        'message': message_text,
        'attachments': json.dumps([]) # Send as empty JSON array string
    }
    response_data = _call_sb_api(payload) 

    if isinstance(response_data, dict) and ('id' in response_data or 'message-id' in response_data):
        internal_msg_id = response_data.get('id', response_data.get('message-id', 'N/A'))
        logger.info(f"Internal SB message added successfully (Internal Msg ID: {internal_msg_id}) to conversation {conversation_id}")
        return True
    elif response_data is True:
         logger.info(f"Internal SB message add attempt reported 'response': True for conversation {conversation_id}, treating as success.")
         return True
    else:
        logger.error(f"Failed to add internal SB message to conversation {conversation_id}. API response: {response_data}")
        return False


# --- PUBLIC HELPER: Log Bot Message to SB Dashboard (No External Send) ---
def log_bot_message_to_support_board(conversation_id: str, message: str) -> bool:
    """Logs a bot/system message into Support Board UI without sending it to the user."""
    from flask import current_app
    import requests

    api_url = current_app.config.get("SUPPORT_BOARD_API_URL")
    api_token = current_app.config.get("SUPPORT_BOARD_API_TOKEN")

    payload = {
        "token": api_token,
        "function": "send-message",
        "conversation_id": conversation_id,
        "user_id": 0,
        "message": message,
        "attachments": json.dumps([]),
    }

    try:
        response = requests.post(api_url, data=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception:
        return False


# --- NEW PRIVATE HELPER: Send WhatsApp Message DIRECTLY via Meta Cloud API ---
def _send_whatsapp_cloud_api(recipient_waid: str, message_text: str) -> bool:
    token = Config.WHATSAPP_CLOUD_API_TOKEN
    phone_number_id = Config.WHATSAPP_PHONE_NUMBER_ID
    api_version = Config.WHATSAPP_API_VERSION

    if not token or not phone_number_id:
        logger.error("WhatsApp Cloud API Token or Phone Number ID not configured. Cannot send direct message.")
        return False
    api_url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload_dict = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_waid,
        "type": "text",
        "text": { "preview_url": False, "body": message_text }
    }
    logger.info(f"Attempting to send direct WhatsApp message via Meta Cloud API to WAID: ...{recipient_waid[-6:]}")
    logger.debug(f"Direct WhatsApp API URL: {api_url}")
    try: log_payload = payload_dict.copy(); logger.debug(f"Direct WhatsApp API Payload: {json.dumps(log_payload)}")
    except Exception: logger.debug(f"Direct WhatsApp API Payload (fallback log): {str(payload_dict)}")

    try:
        response = requests.post(api_url, headers=headers, json=payload_dict, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        try: log_response_str = json.dumps(response_json)
        except Exception: log_response_str = str(response_json)
        logger.debug(f"Direct WhatsApp API Raw Response: {log_response_str}")
        if isinstance(response_json, dict) and \
           response_json.get("messaging_product") == "whatsapp" and \
           isinstance(response_json.get("messages"), list) and \
           len(response_json["messages"]) > 0 and \
           isinstance(response_json["messages"][0], dict) and \
           "id" in response_json["messages"][0]:
            message_wamid = response_json["messages"][0]["id"]
            logger.info(f"Direct WhatsApp API call successful. Message WAMID: {message_wamid}")
            return True
        else:
            logger.error(f"Direct WhatsApp API call returned unexpected success structure: {response_json}")
            return False
    except requests.exceptions.HTTPError as http_err:
        response_text = http_err.response.text if http_err.response else "N/A"
        logger.error(f"Direct WhatsApp API HTTP error: {http_err.response.status_code} - {response_text}", exc_info=False)
        return False
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Direct WhatsApp API request error: {req_err}", exc_info=True)
        return False
    except Exception as e:
        logger.exception(f"Unexpected error during direct WhatsApp API call: {e}")
        return False

# --- NEW PRIVATE HELPER: Send Telegram Message (External Delivery via SB API) ---
def _send_telegram_message(
    chat_id: str,
    message_text: str,
    conversation_id: Optional[str]
) -> bool:
    logger.info(f"Attempting to send Telegram message via SB API to Chat ID: {chat_id} for SB Conv ID: {conversation_id or 'N/A'}")
    payload = {
        'function': 'telegram-send-message',
        'chat_id': chat_id,
        'message': message_text,
        'attachments': json.dumps([])
    }
    if conversation_id: payload['conversation_id'] = conversation_id
    response_data = _call_sb_api(payload)
    if isinstance(response_data, dict) and \
       response_data.get("ok") is True and \
       isinstance(response_data.get("result"), dict) and \
       "message_id" in response_data["result"]:
        tg_message_id = response_data["result"].get("message_id", "N/A")
        logger.info(f"Telegram message acknowledged as successful by SB API (TG Msg ID: {tg_message_id}) for Chat ID {chat_id}, SB Conv ID {conversation_id or 'N/A'}")
        return True
    elif response_data is True:
         logger.warning(f"Telegram message API call for Chat ID {chat_id} returned 'True', which differs from documented structure, but treating as success.")
         return True
    else:
        logger.error(f"Failed to send Telegram message via SB API for Chat ID {chat_id}, SB Conv ID {conversation_id or 'N/A'}. Unexpected response: {response_data}")
        return False


# --- REVISED PUBLIC FUNCTION: send_reply_to_channel ---
def send_reply_to_channel(
    conversation_id: str,
    message_text: str,
    source: Optional[str],
    target_user_id: str,
    conversation_details: Optional[Dict],
    triggering_message_id: Optional[str]
) -> bool:
    if not message_text or not message_text.strip():
        logger.warning(f"Attempted to send empty reply to conversation {conversation_id}. Skipping.")
        return False

    effective_source = source.strip().lower() if isinstance(source, str) and source.strip() else 'web'
    logger.info(f"Routing reply for conversation {conversation_id} to target customer User ID {target_user_id} via effective source channel '{effective_source}'")

    external_success = False
    dm_bot_user_id = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID) if Config.SUPPORT_BOARD_DM_BOT_USER_ID else None

    if effective_source == 'wa':
        logger.info(f"Processing WA reply for conversation {conversation_id} using Direct Cloud API.")
        recipient_waid = _get_user_waid(target_user_id)
        if not recipient_waid:
             logger.error(f"Cannot send WA reply to conv {conversation_id}: Failed to get recipient WAID for user {target_user_id}.")
             return False
        logger.info(f"Step 1 (WA - Direct): Sending externally via Meta Cloud API for conv {conversation_id}")
        external_success = _send_whatsapp_cloud_api(recipient_waid, message_text)
        if external_success:
            logger.info(
                f"Step 2 (WA - Direct): External send successful for conv {conversation_id}."
            )
            if Config.LOG_WA_MESSAGES_INTERNALLY:
                logger.info(
                    "Adding message internally via SB send-message because LOG_WA_MESSAGES_INTERNALLY=True"
                )
                if dm_bot_user_id:
                    internal_add_success = _add_internal_sb_message(
                        conversation_id=conversation_id,
                        message_text=message_text,
                        bot_user_id=dm_bot_user_id,
                    )
                    if not internal_add_success:
                        logger.error(
                            f"Failed to add WA message internally to SB dashboard for conv {conversation_id} after successful direct external send."
                        )
                else:
                    logger.error(
                        "Cannot add WA message internally to SB dashboard: SUPPORT_BOARD_DM_BOT_USER_ID not configured."
                    )
            else:
                logger.info(
                    "Skipping internal SB message add to avoid duplication because LOG_WA_MESSAGES_INTERNALLY=False"
                )
            log_bot_message_to_support_board(conversation_id, message_text)
            log_conversation_event(
                event_type="bot_reply",
                conversation_id=conversation_id,
                user_id=target_user_id,
                source="whatsapp",
                message=message_text,
            )
        else:
             logger.error(f"Direct external WA send via Meta Cloud API failed for conv {conversation_id}.")
        return external_success

    elif effective_source in ['fb', 'ig']:
        logger.info(f"Processing FB/IG reply for conversation {conversation_id} using SB API.")
        conv_details = conversation_details
        if not conv_details:
            logger.info(f"Conversation details not provided for FB/IG {conversation_id}, fetching...")
            conv_details = get_sb_conversation_data(conversation_id)
            if not conv_details:
                logger.error(f"Cannot send FB/IG reply to conv {conversation_id}: Failed to fetch conversation details.")
                return False
        psid = _get_user_psid(target_user_id)
        page_id = conv_details.get('details', {}).get('extra')
        page_id_str = str(page_id).strip() if page_id else None
        if psid and page_id_str:
            logger.info(f"Step 1 (FB/IG - SB): Sending externally via messenger-send-message for conv {conversation_id}")
            external_success = _send_messenger_message(
                psid=psid,
                page_id=page_id_str,
                message_text=message_text,
                conversation_id=conversation_id,
                triggering_message_id=triggering_message_id
            )
            if external_success:
                logger.info(f"Step 2 (FB/IG - SB): External send successful for conv {conversation_id}. Adding message internally via SB send-message.")
                if dm_bot_user_id: # Check the specifically fetched DM Bot ID
                    internal_add_success = _add_internal_sb_message(
                        conversation_id=conversation_id,
                        message_text=message_text,
                        bot_user_id=dm_bot_user_id # Use the DM Bot ID for internal logging
                    )
                    if not internal_add_success:
                        logger.error(f"Failed to add FB/IG message internally to SB dashboard for conv {conversation_id} after successful external send.")
                else:
                    logger.error("Cannot add FB/IG message internally to SB dashboard: SUPPORT_BOARD_DM_BOT_USER_ID not configured.")
                log_conversation_event(
                    event_type="bot_reply",
                    conversation_id=conversation_id,
                    user_id=target_user_id,
                    source="facebook" if effective_source == 'fb' else 'instagram',
                    message=message_text,
                )
            else:
                logger.error(f"External FB/IG send via messenger-send-message failed for conv {conversation_id}.")
            return external_success
        else:
            error_details_list = []
            if not psid: error_details_list.append(f"PSID not found for user {target_user_id}")
            if not page_id_str: error_details_list.append("Page ID not found in conversation details 'extra' field")
            reason = ", ".join(error_details_list)
            logger.error(f"Cannot send FB/IG reply to conv {conversation_id}: Required IDs missing ({reason}).")
            return False

    elif effective_source == 'tg':
        logger.info(f"Processing Telegram reply for conversation {conversation_id} using SB API.")
        conv_details = conversation_details
        if not conv_details:
            logger.info(f"Conversation details not provided for Telegram conv {conversation_id}, fetching...")
            conv_details = get_sb_conversation_data(conversation_id)
            if not conv_details:
                logger.error(f"Cannot send Telegram reply to conv {conversation_id}: Failed to fetch conversation details.")
                return False
        chat_id_from_extra = conv_details.get('details', {}).get('extra')
        if not chat_id_from_extra:
            logger.error(f"Cannot send Telegram reply to conv {conversation_id}: chat_id (from details.extra) not found in conversation details. Details: {conv_details.get('details')}")
            return False
        chat_id = str(chat_id_from_extra).strip()
        if not chat_id:
            logger.error(f"Cannot send Telegram reply to conv {conversation_id}: chat_id (from details.extra) is empty after stripping. Original: '{chat_id_from_extra}'")
            return False

        logger.info(f"Step 1 (TG - SB): Sending externally via telegram-send-message for conv {conversation_id} to Chat ID {chat_id}")
        external_success = _send_telegram_message(
            chat_id=chat_id,
            message_text=message_text,
            conversation_id=conversation_id
        )
        if external_success:
            logger.info(f"Step 2 (TG - SB): External send successful for conv {conversation_id}. Adding message internally via SB send-message.")
            if dm_bot_user_id: # Check the specifically fetched DM Bot ID
                internal_add_success = _add_internal_sb_message(
                    conversation_id=conversation_id,
                    message_text=message_text,
                    bot_user_id=dm_bot_user_id # Use the DM Bot ID for internal logging
                )
                if not internal_add_success:
                    logger.error(f"Failed to add Telegram message internally to SB dashboard for conv {conversation_id} after successful external send.")
            else:
                logger.error("Cannot add Telegram message internally to SB dashboard: SUPPORT_BOARD_DM_BOT_USER_ID not configured.")
            log_conversation_event(
                event_type="bot_reply",
                conversation_id=conversation_id,
                user_id=chat_id,
                source="telegram",
                message=message_text,
            )
        else:
            logger.error(f"External Telegram send via SB API failed for conv {conversation_id}.")
        return external_success

    else:
        logger.warning(f"Unhandled conversation source '{effective_source}' for conv {conversation_id}. Message not sent.")
        return False


# --- NEW PUBLIC FUNCTIONS: Order Confirmation Template & Sales Routing ---
def send_order_confirmation_template(user_id: str, conversation_id: str, variables: list):
    """Send WhatsApp order confirmation template with provided variables."""
    payload = {
        "function": "messaging-platforms-send-template",
        "user": {"id": user_id},
        "conversation_id": conversation_id,
        "source": {"source": "wa"},
        "template": {
            "name": "confirmacion_datos_cliente",
            "language": "es",
            "parameters": variables,
        },
    }
    return _call_sb_api(payload)


def send_template_by_phone_number(phone_number: str, template_params: list):
    """Send WhatsApp template using a raw phone number in international format."""
    payload = {
        "function": "messaging-platforms-send-template",
        "source": {"source": "wa"},
        "phone": phone_number,
        "template": {
            "name": "confirmacion_datos_cliente",
            "language": "es",
            "parameters": template_params,
        },
    }
    return _call_sb_api(payload)


def delete_message(message_id: str) -> bool:
    """Delete a message from Support Board by ID."""
    payload = {
        "function": "delete-message",
        "message_id": str(message_id),
    }
    result = _call_sb_api(payload)
    if result:
        logger.info(f"Deleted Support Board message {message_id}")
        return True
    else:
        logger.error(f"Failed to delete Support Board message {message_id}. Response: {result}")
        return False


def route_conversation_to_sales(conversation_id: str) -> None:
    """Assign conversation to the Sales department and disable the bot."""
    sales_department_id = Config.SUPPORT_BOARD_SALES_DEPARTMENT_ID
    if not sales_department_id:
        logger.error(f"Cannot route conv {conversation_id} to Sales: SUPPORT_BOARD_SALES_DEPARTMENT_ID is not configured.")
        return

    logger.info(f"Routing conversation {conversation_id} to Sales department (ID: {sales_department_id}) and enabling human takeover.")
    try:
        assign_conversation_to_department(conversation_id, int(sales_department_id))
    except (ValueError, TypeError):
        logger.error(f"Cannot route to Sales: SUPPORT_BOARD_SALES_DEPARTMENT_ID ('{sales_department_id}') is not a valid integer.")
        return

    _call_sb_api(
        {
            "function": "sb-human-takeover",
            "conversation_id": conversation_id,
        }
    )

def route_conversation_to_support(conversation_id: str) -> None:
    """Assign conversation to the general Support department and disable the bot."""
    support_department_id = Config.SUPPORT_BOARD_SUPPORT_DEPARTMENT_ID
    if not support_department_id:
        logger.error(f"Cannot route conv {conversation_id} to Support: SUPPORT_BOARD_SUPPORT_DEPARTMENT_ID is not configured.")
        return
        
    logger.info(f"Routing conversation {conversation_id} to Support department (ID: {support_department_id}) and enabling human takeover.")
    try:
        assign_conversation_to_department(conversation_id, int(support_department_id))
    except (ValueError, TypeError):
        logger.error(f"Cannot route to Support: SUPPORT_BOARD_SUPPORT_DEPARTMENT_ID ('{support_department_id}') is not a valid integer.")
        return

    _call_sb_api(
        {
            "function": "sb-human-takeover",
            "conversation_id": conversation_id,
        }
    )