# NAMWOO/services/tools_schema.py
# -*- coding: utf-8 -*-
from typing import List, Dict, Any

# Import the shared keys for reservation details to ensure consistency
from ..utils.conversation_details import (
    KEY_ITEM_CODE, KEY_ITEM_NAME, KEY_FULL_NAME, KEY_CEDULA, KEY_PHONE,
    KEY_EMAIL, KEY_DELIVERY_METHOD, KEY_BRANCH_NAME, KEY_DELIVERY_ADDRESS,
    KEY_PAYMENT_METHOD, KEY_ACCESSORIES, KEY_PRICE_USD, KEY_PRICE_VES
)

# This list defines all functions the AI can call. It is the single source of truth for the project.
tools_schema: List[Dict[str, Any]] = [
    {
        "type": "function", "function": {
            "name": "find_products",
            "description": "Searches the product catalog by keyword, SKU, or brand. If 'city' is provided, it filters for stock in that city's stores. Otherwise, it performs a nationwide search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User's search term (e.g., 'iPhone 15', 'Honor', 'celular 16gb ram')."},
                    "city": {"type": "string", "description": "Optional. City to check for in-store stock."}
                }, "required": ["query"]
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "get_available_brands",
            "description": "Gets a list of available brands for a given product category, like 'CELULAR'. Use this when the user asks for 'marcas'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category to get brands for. Defaults to 'CELULAR'.",
                        "default": "CELULAR"
                    }
                }, "required": []
            }
        }
    },
    { 
        "type": "function", "function": {
            "name": "get_branch_address",
            "description": "Obtiene la dirección completa de una sucursal específica de Damasco basada en su nombre.",
            "parameters": {"type": "object", "properties": {
                    "branchName": {"type": "string", "description": "El nombre exacto de la sucursal (ej. 'La California', 'CCCT')."},
                    "city": {"type": "string", "description": "Opcional. La ciudad donde se encuentra la sucursal, si se conoce."}
                },"required": ["branchName"] 
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "query_accessories",
            "description": "Busca accesorios relevantes (ej. forros, cargadores) para un producto principal, basado en su código de item (SKU).",
            "parameters": {"type": "object", "properties": {
                    "itemCode": {"type": "string", "description": "El itemCode (SKU) del producto principal."}
                },"required": ["itemCode"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_location_details_from_address",
            "description": "Busca tiendas cercanas basado en una dirección de texto del usuario (ej. calle, sector, ciudad). Usar cuando el usuario describe su ubicación con palabras en lugar de compartir un pin GPS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "La dirección textual del usuario (ej. 'Petare, Caracas', 'Avenida Libertador')."}
                },
                "required": ["address"]
            }
        }
    },
    {
        "type": "function", "function": {
            "name": "save_customer_reservation_details",
            "description": "Guarda o actualiza la información del cliente para su reserva actual. Debe llamarse cada vez que el usuario proporciona un dato personal.",
            "parameters": {"type": "object", "properties": {
                    KEY_FULL_NAME: {"type": "string"}, KEY_CEDULA: {"type": "string"},
                    KEY_PHONE: {"type": "string"}, KEY_EMAIL: {"type": "string"},
                    "city": {"type": "string", "description": "La ciudad del cliente para la entrega o retiro."},
                    KEY_DELIVERY_METHOD: {"type": "string", "enum": ["retiro_en_tienda", "envio_a_domicilio_caracas", "envio_nacional"]},
                    KEY_BRANCH_NAME: {"type": "string"}, KEY_DELIVERY_ADDRESS: {"type": "string"},
                    KEY_PAYMENT_METHOD: {"type": "string"}, KEY_ITEM_CODE: {"type": "string"},
                    KEY_ITEM_NAME: {"type": "string"}, KEY_PRICE_USD: {"type": "number"},
                    KEY_PRICE_VES: {"type": "number"}, KEY_ACCESSORIES: {"type": "array", "items": {"type": "string"}}
                }, "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_order_summary_template",
            "description": "Envía la plantilla de resumen de pedido por WhatsApp al cliente.",
            "parameters": { "type": "object", "properties": {
                    "customer_platform_user_id": {"type": "string"}, "conversation_id": {"type": "string"},
                    "template_variables": {"type": "array", "items": {"type": "string"}, "minItems": 8, "maxItems": 8}
                }, "required": ["customer_platform_user_id", "conversation_id", "template_variables"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_location_details_from_user",
            "description": "Information automatically extracted from a GPS location shared by the user. Contains the formatted address and a sorted list of nearby stores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "formatted_address": {
                        "type": "string",
                        "description": "The full, human-readable address of the user's location."
                    },
                    "nearby_stores": {
                        "type": "array",
                        "description": "A list of nearby stores, sorted from closest to farthest.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "branch_name": {"type": "string"},
                                "distance_km": {"type": "number"}
                            }
                        }
                    }
                },
                "required": ["nearby_stores"]
            }
        }
    },
    # --- START OF MODIFICATION ---
    {
        "type": "function",
        "function": {
            "name": "route_to_sales_department",
            "description": "Use this function ONLY after the reservation is fully confirmed and you have said the final closing message. This transfers the conversation to a human sales agent to handle payment and finalizes your involvement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string", "description": "The ID of the current conversation to be transferred."}
                },
                "required": ["conversation_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_human_support",
            "description": "Use this function when you cannot help the user or when they explicitly ask to speak to a human agent. You must ask for their confirmation before calling this function. This transfers the conversation to the support department and finalizes your involvement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string", "description": "The ID of the current conversation to be transferred."}
                },
                "required": ["conversation_id"]
            }
        }
    }
    # --- END OF MODIFICATION ---
]