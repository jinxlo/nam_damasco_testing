# NAMWOO/services/providers/openai_assistant_provider.py
# -*- coding: utf-8 -*-
import logging
import json
import time
from urllib.parse import urlparse
from typing import List, Dict, Optional, Any, Tuple

import redis
from openai import OpenAI
from openai.types.beta.threads import Run

# Import local services and utils
from .. import product_service
from .. import support_board_service
from .. import geolocation_service
from .. import thread_mapping_service
from ...config import Config
from ...utils import conversation_location
from ...utils import conversation_details
from ...utils import message_parser

logger = logging.getLogger(__name__)

class OpenAIAssistantProvider:
    def __init__(self, api_key: str, assistant_id: str):
        if not api_key or not assistant_id:
            raise ValueError("API key and Assistant ID are required for OpenAIAssistantProvider.")
        
        self.client = OpenAI(api_key=api_key)
        self.assistant_id = assistant_id

        # Polling configuration
        self.polling_interval_seconds = 1
        self.run_timeout_seconds = 120

        # Application-level lock (requires Config.REDIS_URL)
        self.redis = redis.Redis.from_url(getattr(Config, "REDIS_URL", "redis://localhost:6379/0"))

        logger.info(f"OpenAIAssistantProvider initialized for Assistant ID '{self.assistant_id}'.")

    def _get_or_create_thread_id(self, sb_conversation_id: str) -> str:
        thread_id = thread_mapping_service.get_thread_id(sb_conversation_id)
        if not thread_id:
            logger.info(f"No existing thread for Conv {sb_conversation_id}. Creating a new one.")
            thread = self.client.beta.threads.create()
            thread_id = thread.id
            thread_mapping_service.store_thread_id(sb_conversation_id, thread_id)
        return thread_id

    def _wait_for_thread_free(self, thread_id: str):
        """
        Polls until there are no active runs on the thread.
        Cancels the run and raises if timeout is exceeded.
        """
        start = time.time()
        resp = self.client.beta.threads.runs.list(thread_id=thread_id, limit=1)
        if not resp.data:
            return  # no runs yet

        run = resp.data[0]
        while run.status in ("queued", "in_progress", "requires_action"):
            if time.time() - start > self.run_timeout_seconds:
                logger.warning(f"Run {run.id} hung; cancelling after {self.run_timeout_seconds}s")
                self.client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                raise RuntimeError(f"Run {run.id} did not finish in time.")
            time.sleep(self.polling_interval_seconds)
            run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    def _handle_geolocation_injection(self, thread_id: str, new_user_message: str) -> bool:
        location_data = message_parser.extract_location_from_text(new_user_message)
        if not location_data:
            return False

        logger.info(f"Location URL detected. Injecting geo-context into thread {thread_id}.")
        geo_details = geolocation_service.get_location_details(
            latitude=location_data['latitude'],
            longitude=location_data['longitude']
        )
        if geo_details and not geo_details.get("error"):
            # original URL
            self.client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=new_user_message
            )
            # formatted context block
            nearby = "\n".join(
                f"- {s['branch_name']} (a {s['distance_km']} km)"
                for s in geo_details.get("nearby_stores", [])
            )
            context_message = (
                "[CONTEXTO DE UBICACIÓN PROPORCIONADO POR EL USUARIO]\n"
                f"Dirección Detectada: {geo_details.get('formatted_address', 'No disponible')}\n"
                f"Tiendas Cercanas:\n{nearby}\n"
                "[FIN DEL CONTEXTO DE UBICACIÓN]"
            )
            self.client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=context_message
            )
            logger.info(f"Injected location context into thread {thread_id}.")
            return True
        return False

    # --- START OF REVISED FIX: PRIORITIZE TRANSCRIPT OVER AUDIO ATTACHMENTS ---
    def _prepare_message_content(self, conversation_data: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Parses the recent conversation history to bundle contiguous user messages
        (text and images) into a single logical input for the AI. This prevents
        race conditions where separate webhooks for text and images cause duplicate bot replies.
        
        It now intelligently ignores audio attachments when a text transcript is present.
        
        Returns a tuple: (content_parts, last_message_id)
        """
        if not conversation_data or not conversation_data.get("messages"):
            return None, None

        user_messages_block = []
        # Iterate backwards from the most recent message
        for message in reversed(conversation_data["messages"]):
            # We only care about messages from the customer
            if message.get("user_type") in ["lead", "user"]:
                user_messages_block.insert(0, message)
            else:
                # Stop when we hit a message from the bot or an agent
                break
        
        if not user_messages_block:
            return None, None

        content_parts = []
        image_urls = set() # Use a set to avoid duplicate images
        text_parts = []
        
        for message in user_messages_block:
            # Aggregate text from the block. This includes transcripts of voice notes.
            if message.get('message'):
                text_parts.append(message['message'])
            
            # Aggregate attachments, but only if they are actual images.
            attachments_str = message.get("attachments")
            if attachments_str and isinstance(attachments_str, str):
                try:
                    attachments = json.loads(attachments_str)
                    supported_image_extensions = ('.png', '.jpeg', '.jpg', '.gif', '.webp')
                    
                    for attachment in attachments:
                        # Standard check for valid URL structure
                        if not (isinstance(attachment, list) and len(attachment) == 2 and attachment[1].startswith('http')):
                            continue

                        url = attachment[1]
                        try:
                            # Parse the URL and check if it's a supported *image* format
                            parsed_url = urlparse(url)
                            if parsed_url.path.lower().endswith(supported_image_extensions):
                                image_urls.add(url)
                            else:
                                # This will now correctly ignore .mp3 files and other non-image attachments
                                logger.info(f"Ignoring non-image attachment: {url}. Transcript should be in 'message' field if available.")
                        except Exception as e:
                            logger.warning(f"Could not parse or validate attachment URL '{url}': {e}")

                except json.JSONDecodeError:
                    logger.warning(f"Could not parse attachments JSON from message: {attachments_str}")

        # Build the final content list for the API
        for url in sorted(list(image_urls)): # Sort to maintain a consistent order
            content_parts.append({"type": "image_url", "image_url": {"url": url}})
        
        full_text = " ".join(text_parts).strip()
        if full_text:
            content_parts.append({"type": "text", "text": full_text})
        
        last_message_id = user_messages_block[-1].get("id")

        return content_parts if content_parts else None, last_message_id
    # --- END OF REVISED FIX ---

    def process_message(
        self,
        sb_conversation_id: str,
        new_user_message: Optional[str],
        conversation_data: Dict[str, Any],
        reservation_context: Dict[str, Any]
    ) -> Optional[str]:
        lock_key = f"lock:conv:{sb_conversation_id}"
        with self.redis.lock(lock_key, timeout=self.run_timeout_seconds + 10, blocking_timeout=60):
            try:
                thread_id = self._get_or_create_thread_id(sb_conversation_id)

                self._wait_for_thread_free(thread_id)

                # Check the last message processed in the OpenAI thread
                thread_messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1, order="desc")
                last_processed_id = None
                if thread_messages.data and thread_messages.data[0].metadata:
                    last_processed_id = thread_messages.data[0].metadata.get('sb_message_id')

                # Prepare the content by bundling recent user messages
                final_content, latest_user_message_id = self._prepare_message_content(conversation_data)

                # If the latest message from the user has already been processed, abort.
                if latest_user_message_id and (latest_user_message_id == last_processed_id):
                    logger.warning(f"Skipping processing for Conv {sb_conversation_id}. Message block ending in {latest_user_message_id} already processed.")
                    return None

                if final_content:
                    # Special handling for geolocation links is still separate
                    is_location_injection = False
                    if new_user_message:
                         is_location_injection = self._handle_geolocation_injection(thread_id, new_user_message)
                    
                    if not is_location_injection:
                        logger.info(f"Adding new bundled message to thread {thread_id}, tagged with last message ID {latest_user_message_id}.")
                        self.client.beta.threads.messages.create(
                            thread_id=thread_id,
                            role="user",
                            content=final_content,
                            metadata={'sb_message_id': latest_user_message_id} # Tag the message
                        )
                else:
                    logger.warning(f"No new user message content to process for Conv {sb_conversation_id}. Skipping message creation.")
                    return None # Abort if there's nothing to process
                
                instructions = Config.SYSTEM_PROMPT
                if reservation_context:
                    header = "\n\n--- CONTEXTO DE RESERVA ---\n"
                    footer = "\n---------------------------\n"
                    ctx = "\n".join(f"- {k}: {v}" for k,v in reservation_context.items())
                    instructions += f"{header}Estado actual de la reserva:\n{ctx}{footer}"

                run = self.client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=self.assistant_id,
                    instructions=instructions
                )
                logger.info(f"Created Run {run.id} for Thread {thread_id}.")

                start = time.time()
                while time.time() - start < self.run_timeout_seconds:
                    run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

                    if run.status == 'completed':
                        logger.info(f"Run {run.id} completed.")
                        msgs = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
                        last_message = msgs.data[0]
                        if last_message.role == "assistant" and last_message.content and last_message.content[0].type == "text":
                            return last_message.content[0].text.value
                        else:
                            logger.info(f"Run {run.id} completed after a tool call with no subsequent text response. No message to send.")
                            return None

                    if run.status == 'requires_action':
                        logger.info(f"Run {run.id} requires action.")
                        tool_outputs = self._execute_tool_calls(run.required_action.submit_tool_outputs.tool_calls, sb_conversation_id)
                        self.client.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread_id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )

                    if run.status in ('failed', 'cancelled', 'expired'):
                        logger.error(f"Run {run.id} ended with {run.status}: {run.last_error}")
                        return f"Lo siento, la operación falló con estado: {run.status}."

                    time.sleep(self.polling_interval_seconds)

                logger.error(f"Run {run.id} timed out after {self.run_timeout_seconds}s")
                return "Lo siento, la operación tardó demasiado en completarse."

            except Exception as e:
                logger.exception(f"Error in OpenAIAssistantProvider for Conv {sb_conversation_id}: {e}")
                return "Ocurrió un error inesperado con nuestro asistente. Por favor, intenta de nuevo."

    def _execute_tool_calls(self, tool_calls: List[Any], sb_conversation_id: str) -> List[Dict[str, str]]:
        tool_outputs = []
        for tc in tool_calls:
            fn = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON for {fn}: {tc.function.arguments}")
                args = {}

            logger.info(f"Tool requested: {fn} args={args} for Conv {sb_conversation_id}")
            output: Any = {}

            try:
                if fn == "find_products":
                    query, city_arg = args.get("query"), args.get("city")
                    if city_arg:
                        conversation_location.set_conversation_city(sb_conversation_id, city_arg)
                        warehouses = conversation_location.get_city_warehouses(sb_conversation_id)
                        if not warehouses:
                            output = {"status": "city_not_served", "city": city_arg}
                        else:
                            res = product_service.find_products(query=query, warehouse_names=warehouses)
                            if not res or not (res.get("products_grouped") or res.get("product_details")):
                                output = {"status": "not_found_in_city", "city": city_arg}
                            else:
                                output = res
                    else:
                        output = product_service.find_products(query=query, warehouse_names=None)

                elif fn == "get_available_brands":
                    brands = product_service.get_available_brands_by_category(category=args.get("category", "CELULAR"))
                    output = {"status": "success", "brands": brands} if brands else {"status": "not_found"}

                elif fn == "get_branch_address":
                    output = product_service.get_branch_address(
                        branch_name=args.get("branchName"),
                        city=args.get("city")
                    )

                elif fn == "query_accessories":
                    warehouses = conversation_location.get_city_warehouses(sb_conversation_id)
                    res = product_service.query_accessories(main_product_item_code=args.get("itemCode"), city_warehouses=warehouses)
                    output = {"status": "success", "accessories_list": res} if res else {"status": "not_found"}

                elif fn == "get_location_details_from_address":
                    output = geolocation_service.get_location_details_from_address(address=args.get("address"))

                elif fn == "save_customer_reservation_details":
                    saved = [k for k,v in args.items() if conversation_details.store_reservation_detail(sb_conversation_id, k, v)]
                    if 'city' in args:
                        conversation_location.set_conversation_city(sb_conversation_id, args['city'])
                    output = {"status": "success", "message": f"OK. Detalles guardados: {', '.join(saved)}."} if saved else {"status": "no_action"}

                elif fn == "send_whatsapp_order_summary_template":
                    output = {"status": "success", "message": "OK_TEMPLATE_SENT"}

                elif fn == "route_to_sales_department":
                    support_board_service.route_conversation_to_sales(sb_conversation_id)
                    output = {"status": "success", "message": "Conversation has been routed to the Sales department."}

                elif fn == "route_to_human_support":
                    support_board_service.route_conversation_to_support(sb_conversation_id)
                    output = {"status": "success", "message": "Conversation has been routed to the Support department."}

                else:
                    output = {"status": "error", "message": f"Herramienta desconocida '{fn}'."}

            except Exception as ex:
                logger.exception(f"Error executing {fn}: {ex}")
                output = {"status": "error", "message": f"Error interno en {fn}: {ex}"}

            tool_outputs.append({
                "tool_call_id": tc.id,
                "output": json.dumps(output, ensure_ascii=False)
            })

        return tool_outputs