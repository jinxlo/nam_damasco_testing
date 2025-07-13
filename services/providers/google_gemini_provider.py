# NAMWOO/services/providers/google_gemini_provider.py
# -*- coding: utf-8 -*-
import logging
import json
from typing import List, Dict, Optional, Any, Union
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, BadRequestError

# --- CORRECTED IMPORTS ---
# Go up one level '..' from 'providers' to the 'services' directory.
from .. import product_service
from .. import support_board_service
# Go up two levels '...' from 'providers' to the 'namwoo_app' root for config/utils.
from ...config import Config
from ...utils import conversation_location

logger = logging.getLogger(__name__)

# This tool schema is specific to the original google_service.py implementation.
# It is kept here for this provider. If it were identical to the OpenAI one,
# we would import it from tools_schema.py.
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "search_local_products",
            "description": "Busca en el catálogo de productos de la tienda utilizando una consulta en lenguaje natural.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "La consulta del usuario que describe el producto."},
                    "filter_stock": {"type": "boolean", "description": "Si es True, solo devuelve productos con stock.", "default": True},
                    "warehouse_names": {"type": "array", "items": {"type": "string"}, "description": "Opcional. Lista de almacenes para limitar la búsqueda."},
                },
                "required": ["query_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_product_details",
            "description": "Obtiene información detallada y actualizada de un producto específico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_identifier": {"type": "string", "description": "El SKU o ID compuesto del producto."},
                    "identifier_type": {"type": "string", "enum": ["sku", "composite_id"], "description": "Especifica el tipo de identificador."},
                },
                "required": ["product_identifier", "identifier_type"],
            },
        },
    },
]


class GoogleGeminiProvider:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Google API key is required for GoogleGeminiProvider.")
        
        google_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        timeout_seconds = getattr(Config, 'GOOGLE_REQUEST_TIMEOUT', 60.0)
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=google_base_url,
            timeout=timeout_seconds,
        )
        self.model = getattr(Config, "GOOGLE_GEMINI_MODEL", "gemini-1.5-flash-latest")
        self.max_tokens = getattr(Config, "GOOGLE_MAX_TOKENS", 1024)
        self.temperature = getattr(Config, "GOOGLE_TEMPERATURE", 0.7)
        self.tool_call_retry_limit = 2
        logger.info(f"GoogleGeminiProvider initialized for model '{self.model}'.")

    def process_message(
        self,
        sb_conversation_id: str,
        new_user_message: Optional[str],
        conversation_data: Dict[str, Any],
        reservation_context: Dict[str, Any] # Note: reservation_context is not used in the original Gemini logic, but is kept for interface consistency
    ) -> Optional[str]:
        logger.info(f"[Gemini Provider] Handling SB Conv {sb_conversation_id}")
        
        if new_user_message:
            detected_city = conversation_location.detect_city_from_text(new_user_message)
            if detected_city:
                conversation_location.set_conversation_city(sb_conversation_id, detected_city)

        sb_history_list = (conversation_data.get("messages", []) if conversation_data else [])
        api_history = self._format_sb_history(sb_history_list)

        if not api_history:
            if new_user_message:
                api_history = [{"role": "user", "content": new_user_message}]
            else:
                logger.error(f"[Gemini Provider] No history and no new message for Conv {sb_conversation_id}.")
                return "Lo siento, no pude procesar tu solicitud."

        messages_for_api = [{"role": "system", "content": Config.SYSTEM_PROMPT}] + api_history
        
        final_assistant_response: Optional[str] = None
        try:
            tool_call_count = 0
            while tool_call_count <= self.tool_call_retry_limit:
                call_params: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages_for_api,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "tools": tools_schema,
                    "tool_choice": "auto"
                }
                
                response = self.client.chat.completions.create(**call_params)
                response_msg_obj = response.choices[0].message
                messages_for_api.append(response_msg_obj.model_dump(exclude_none=True))
                
                if not response_msg_obj.tool_calls:
                    final_assistant_response = response_msg_obj.content
                    break

                tool_outputs = self._execute_tool_calls(response_msg_obj.tool_calls, sb_conversation_id)
                messages_for_api.extend(tool_outputs)
                tool_call_count += 1

            if not final_assistant_response:
                logger.warning(f"[Gemini Provider] Tool call limit reached for Conv {sb_conversation_id}. No final response generated.")
                final_assistant_response = "Parece que estoy teniendo dificultades para completar tu solicitud. ¿Podrías intentar de otra manera?"

        except (RateLimitError, APITimeoutError) as e:
            logger.warning(f"[Gemini Provider] API Error for conv {sb_conversation_id}: {e}")
            final_assistant_response = "El servicio de IA (Google) está experimentando un problema. Por favor, intenta más tarde."
        except Exception as e:
            logger.exception(f"[Gemini Provider] Unexpected error for conv {sb_conversation_id}: {e}")
            final_assistant_response = "Ocurrió un error inesperado al procesar tu solicitud con Google AI."
            
        return final_assistant_response

    def _execute_tool_calls(self, tool_calls: List[Any], sb_conversation_id: str) -> List[Dict[str, str]]:
        tool_outputs_for_api: List[Dict[str, str]] = []
        for tc in tool_calls:
            fname = tc.function.name
            tool_call_id = tc.id
            logger.info(f"[Gemini Provider] Requested tool: {fname} with args: {tc.function.arguments}")
            
            tool_content_str: str
            try:
                args = json.loads(tc.function.arguments)
                if fname == "search_local_products":
                    query = args.get("query_text")
                    warehouses = args.get("warehouse_names") or conversation_location.get_city_warehouses(sb_conversation_id)
                    results = product_service.search_local_products(query_text=query, warehouse_names=warehouses)
                    tool_content_str = self._format_search_results(results)
                elif fname == "get_live_product_details":
                    identifier, id_type = args.get("product_identifier"), args.get("identifier_type")
                    details = product_service.get_live_product_details_by_sku(identifier) if id_type == "sku" else product_service.get_live_product_details_by_id(identifier)
                    tool_content_str = self._format_live_details(details, id_type)
                else:
                    tool_content_str = json.dumps({"status": "error", "message": f"Herramienta desconocida '{fname}'."})
            except Exception as e_tool:
                logger.exception(f"[Gemini Provider] Error executing tool {fname}: {e_tool}")
                tool_content_str = json.dumps({"status": "error", "message": f"Error interno ejecutando la herramienta {fname}."})
            
            tool_outputs_for_api.append({"tool_call_id": tool_call_id, "role": "tool", "name": fname, "content": tool_content_str})
        return tool_outputs_for_api

    def _format_sb_history(self, sb_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # This is a simplified version of the history formatter. A full implementation would handle payloads.
        openai_messages: List[Dict[str, Any]] = []
        bot_user_id_str = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID)
        for msg in sb_messages:
            role = "assistant" if str(msg.get("user_id")) == bot_user_id_str else "user"
            text_content = msg.get("message", "").strip()
            if text_content:
                openai_messages.append({"role": role, "content": text_content})
        return openai_messages

    def _format_search_results(self, results: Optional[List[Dict[str, Any]]]) -> str:
        if results is None: return json.dumps({"status": "error", "message": "Error interno al buscar productos."})
        if not results: return json.dumps({"status": "not_found", "message": "No se encontraron productos."})
        return json.dumps({"status": "success", "products": results}, indent=2, ensure_ascii=False)

    def _format_live_details(self, details: Union[Optional[Dict], Optional[List]], id_type: str) -> str:
        if not details: return json.dumps({"status": "not_found", "message": f"No se encontró el producto con el {id_type} especificado."})
        return json.dumps({"status": "success", "product_details": details}, indent=2, ensure_ascii=False)