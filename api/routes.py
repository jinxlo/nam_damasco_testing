# NAMWOO/routes.py
# -*- coding: utf-8 -*-

import logging
import datetime
import hmac
import hashlib
import json
from datetime import timedelta, timezone

from flask import request, jsonify, current_app, abort
from sqlalchemy import text

# --- Local Imports ---
from ..utils import db_utils
from ..config import Config
from ..extensions import get_redis_client
from ..models.conversation_pause import ConversationPause
from ..utils.text_utils import split_full_name

# --- Service Imports ---
# CORRECTED: Only import the main AI service dispatcher.
from ..services import ai_service
from ..services import support_board_service

# This utility function is used for a special feature.
from ..services.ai_service import extract_customer_info_via_llm

from . import api_bp

logger = logging.getLogger(__name__)


def _validate_sb_webhook_secret(request):
    """(This helper function remains unchanged)"""
    secret = current_app.config.get('SUPPORT_BOARD_WEBHOOK_SECRET')
    if not secret:
        return True
    signature_header = request.headers.get('X-Sb-Signature')
    if not signature_header:
        return False
    try:
        method, signature_hash = signature_header.split('=', 1)
        if method != 'sha1': return False
        request_data_bytes = request.get_data()
        mac = hmac.new(secret.encode('utf-8'), msg=request_data_bytes, digestmod=hashlib.sha1)
        if hmac.compare_digest(mac.hexdigest(), signature_hash):
            return True
        return False
    except Exception:
        return False


@api_bp.route('/sb-webhook', methods=['POST'])
def handle_support_board_webhook():
    """
    Receives 'message-sent' webhooks, handles deduplication, determines sender type,
    and delegates customer messages to the unified AI service dispatcher.
    """
    # 1. Webhook parsing logic
    try:
        body = request.get_json(force=True)
        if not body: abort(400, description="Invalid payload: Empty body.")
    except Exception:
        abort(400, description="Invalid JSON payload received.")

    if body.get('function') != 'message-sent':
        return jsonify({"status": "ok", "message": "Webhook type ignored"}), 200

    data = body.get('data', {})
    sb_conversation_id = data.get('conversation_id')
    sender_user_id_str = str(data.get('user_id'))
    customer_user_id_str = str(data.get('conversation_user_id'))
    triggering_message_id = data.get('message_id')
    new_user_message_text = data.get('message')
    conversation_source = data.get('conversation_source')
    order_vars = data.get('order_confirmation_variables')

    if not all([sb_conversation_id, sender_user_id_str, customer_user_id_str]):
        return jsonify({"status": "error", "message": "Webhook payload missing required ID fields"}), 200

    # --- RESTORED: Original two-part deduplication logic ---
    payload_str = data.get('payload', "{}")
    wa_message_id = None
    if isinstance(payload_str, str):
        try:
            wa_message_id = json.loads(payload_str).get('waid')
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Could not parse waid from payload: {payload_str}")

    if not wa_message_id and sb_conversation_id and triggering_message_id is not None:
        wa_message_id = support_board_service.extract_waid_from_conversation(
            str(sb_conversation_id), str(triggering_message_id)
        )

    # Part 1: Atomic WAID check
    if wa_message_id:
        dedupe_key = f"waid:{sb_conversation_id}:{wa_message_id}"
        logger.debug(f"Checking WAID de-dup key: {dedupe_key}")
        try:
            first = current_app.redis_client.set(dedupe_key, "1", nx=True, ex=30)
        except Exception as e:
            logger.error(f"Redis error during WAID deduplication: {e}")
            first = True
        if not first:
            logger.info(f"DEDUP_SKIP {dedupe_key}")
            if triggering_message_id is not None:
                if support_board_service.delete_message(str(triggering_message_id)):
                    logger.info(f"Deleted duplicate message {triggering_message_id}")
                else:
                    logger.error(f"Failed to delete duplicate message {triggering_message_id}")
            return "", 200
        logger.info(f"DEDUP_NEW {dedupe_key}")

    # Part 2: Generic idempotency check
    redis_client = get_redis_client()
    if redis_client:
        unique_msg_id = wa_message_id or str(triggering_message_id)
        cache_key = f"sb_webhook_processed:{sb_conversation_id}:{unique_msg_id}"
        try:
            if redis_client.exists(cache_key):
                logger.warning(f"Duplicate webhook skipped by generic check: {cache_key}")
                return jsonify({"status": "duplicate"}), 200
            redis_client.setex(cache_key, current_app.config["IDEMPOTENCY_TTL"], "processed")
        except Exception as e:
            logger.error(f"Redis failure in idempotency check: {str(e)}")
    # --- END OF RESTORED LOGIC ---

    if order_vars and isinstance(order_vars, list) and len(order_vars) == 8:
        support_board_service.send_order_confirmation_template(
            user_id=customer_user_id_str, conversation_id=str(sb_conversation_id), variables=order_vars)
        support_board_service.route_conversation_to_sales(str(sb_conversation_id))
        return jsonify({"status": "ok", "message": "Order confirmation sent"}), 200

    # --- RESTORED: Full Sender Identification Logic ---
    DM_BOT_ID_STR = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID)
    COMMENT_BOT_PROXY_USER_ID_STR = str(Config.COMMENT_BOT_PROXY_USER_ID) if Config.COMMENT_BOT_PROXY_USER_ID else None
    HUMAN_AGENT_IDS_SET = Config.SUPPORT_BOARD_AGENT_IDS
    COMMENT_BOT_INITIATION_TAG = Config.COMMENT_BOT_INITIATION_TAG
    pause_minutes = Config.HUMAN_TAKEOVER_PAUSE_MINUTES

    if not DM_BOT_ID_STR:
         logger.critical("FATAL: SUPPORT_BOARD_DM_BOT_USER_ID not configured.")
         return jsonify({"status": "error", "message": "Internal configuration error: DM Bot User ID missing."}), 200

    logger.info(f"Processing webhook for SB Conv ID: {sb_conversation_id} from Sender: {sender_user_id_str}, Customer: {customer_user_id_str}")

    # Rule 1: Message from DM Bot (Namwoo) itself (echo)
    if sender_user_id_str == DM_BOT_ID_STR:
        logger.info(f"Ignoring own message echo from DM bot (ID: {sender_user_id_str}) in conv {sb_conversation_id}.")
        return jsonify({"status": "ok", "message": "Bot message echo ignored"}), 200

    # Rule 2: Message from the Comment Bot's Proxy User ID (e.g., user "1")
    if COMMENT_BOT_PROXY_USER_ID_STR and sender_user_id_str == COMMENT_BOT_PROXY_USER_ID_STR:
        is_comment_bot_message = False
        if COMMENT_BOT_INITIATION_TAG:
            if COMMENT_BOT_INITIATION_TAG in (new_user_message_text or ""):
                is_comment_bot_message = True
                logger.info(f"Message from Comment Bot (proxy ID: {sender_user_id_str}, with tag) in conv {sb_conversation_id}. DM Bot will not reply.")
            else:
                logger.info(f"Message from human admin using proxy ID {sender_user_id_str} (tag mismatch) in conv {sb_conversation_id}. Pausing DM Bot.")
                db_utils.pause_conversation_for_duration(str(sb_conversation_id), duration_seconds=pause_minutes * 60)
                return jsonify({"status": "ok", "message": "Human admin (proxy ID), bot paused"}), 200
        else:
            is_comment_bot_message = True
            logger.info(f"Message from Comment Bot's proxy (ID: {sender_user_id_str}, no tag) in conv {sb_conversation_id}. DM Bot will not reply.")
        
        if is_comment_bot_message:
            return jsonify({"status": "ok", "message": "Comment bot proxy message processed"}), 200

    # Rule 3: Message from a configured DEDICATED HUMAN AGENT
    if sender_user_id_str in HUMAN_AGENT_IDS_SET:
        logger.info(f"Human agent {sender_user_id_str} message in conv {sb_conversation_id}. Pausing bot.")
        db_utils.pause_conversation_for_duration(str(sb_conversation_id), duration_seconds=pause_minutes * 60)
        return jsonify({"status": "ok", "message": "Human agent message received, bot paused"}), 200

    # Rule 4: Message from the CUSTOMER
    if sender_user_id_str == customer_user_id_str:
        if db_utils.is_conversation_paused(str(sb_conversation_id)):
            logger.info(f"Conv {sb_conversation_id} is paused in DB. Bot will not reply.")
            return jsonify({"status": "ok", "message": "Conversation explicitly paused"}), 200

        conversation_data = support_board_service.get_sb_conversation_data(str(sb_conversation_id))
        is_implicitly_human_handled = False
        if conversation_data and conversation_data.get('messages'):
            for msg in reversed(conversation_data['messages']):
                msg_sender_id = str(msg.get('user_id'))
                if msg_sender_id == customer_user_id_str: continue
                if msg_sender_id == DM_BOT_ID_STR: break
                
                is_hist_comment_bot = False
                if COMMENT_BOT_PROXY_USER_ID_STR and msg_sender_id == COMMENT_BOT_PROXY_USER_ID_STR:
                    msg_text_history = msg.get('message', '')
                    if COMMENT_BOT_INITIATION_TAG and COMMENT_BOT_INITIATION_TAG in msg_text_history:
                        is_hist_comment_bot = True
                    elif not COMMENT_BOT_INITIATION_TAG:
                        is_hist_comment_bot = True
                
                if is_hist_comment_bot: break
                
                is_implicitly_human_handled = True
                logger.info(f"Implicit human takeover detected in conv {sb_conversation_id}. Last non-bot/customer message from: {msg_sender_id}.")
                break
        
        if is_implicitly_human_handled:
            return jsonify({"status": "ok", "message": "Implicit human takeover, bot will not reply"}), 200

        if new_user_message_text:
            try:
                customer_data = extract_customer_info_via_llm(new_user_message_text)
                required_keys = ["full_name", "cedula", "telefono", "correo", "direccion", "productos", "total"]
                if customer_data and all(customer_data.get(k) for k in required_keys):
                    nombre, apellido = split_full_name(str(customer_data["full_name"]))
                    params = [str(nombre), str(apellido), str(customer_data["cedula"]).strip(), str(customer_data["telefono"]).strip(), str(customer_data["correo"]).strip(), str(customer_data["direccion"]).strip(), str(customer_data["productos"]).strip(), str(customer_data["total"]).strip()]
                    phone = str(customer_data.get("telefono", "")).strip()
                    if phone:
                        support_board_service.send_template_by_phone_number(phone_number=phone, template_params=params)
                        support_board_service.route_conversation_to_sales(str(sb_conversation_id))
                        return jsonify({"status": "ok", "message": "Template sent via phone from extracted data"}), 200
            except Exception as e:
                logger.exception(f"LLM info extraction failed: {e}")
        
        # --- UNIFIED AI SERVICE CALL (THE ONLY CHANGE FROM THE OLD FILE) ---
        logger.info(f"Conv {sb_conversation_id} is active. Delegating to unified AI service.")
        try:
            ai_service.process_new_message(
                sb_conversation_id=str(sb_conversation_id),
                new_user_message=new_user_message_text,
                conversation_source=conversation_source,
                sender_user_id=sender_user_id_str,
                customer_user_id=customer_user_id_str,
                triggering_message_id=str(triggering_message_id) if triggering_message_id else None
            )
            return jsonify({"status": "ok", "message": "AI processing initiated"}), 200
        except Exception as e:
            logger.exception(f"CRITICAL: The AI service dispatcher failed for conv {sb_conversation_id}: {e}")
            return jsonify({"status": "error", "message": "Critical error in AI service dispatcher"}), 500

    # Rule 5: Final fallback for any other sender type
    logger.warning(f"Message in conv {sb_conversation_id} from unhandled sender {sender_user_id_str}. Pausing bot.")
    db_utils.pause_conversation_for_duration(str(sb_conversation_id), duration_seconds=pause_minutes * 60)
    return jsonify({"status": "ok", "message": "Unhandled sender, bot paused"}), 200


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Performs a health check on the application and its database connection."""
    logger.debug("Health check endpoint hit.")
    db_ok = False
    try:
        with db_utils.get_db_session() as session:
            session.execute(text("SELECT 1"))
            db_ok = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}", exc_info=True)
        db_ok = False
    return jsonify({"status": "ok", "database_connected": db_ok}), 200


@api_bp.route('/supportboard/test', methods=['GET'])
def handle_support_board_test():
    """A simple test endpoint to confirm the API blueprint is active."""
    endpoint_name = "/api/supportboard/test"
    logger.info(f"--- TEST HIT --- Endpoint {endpoint_name} was successfully reached.")
    return jsonify({
        "status": "success", 
        "message": f"Namwoo endpoint {endpoint_name} reached successfully!",
        "timestamp": datetime.datetime.now(timezone.utc).isoformat()
    }), 200