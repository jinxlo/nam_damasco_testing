# NAMWOO/api/receiver_routes.py

import logging
from flask import request, jsonify, current_app

# Import the NEW, EFFICIENT batch processing Celery task
from ..celery_tasks import process_products_batch_task

from . import api_bp

logger = logging.getLogger(__name__)

# This helper function is no longer suitable for the new nested data structure.
# The flattening and key conversion logic will now be handled directly in the route.
# def _convert_api_input_to_snake_case_for_task(data_camel: dict) -> dict:
#    ...


@api_bp.route('/receive-products', methods=['POST'])
def receive_data():
    """
    Receives a JSON list of NEW-FORMAT product entries (with nested availability).
    Flattens the data, validates the API token, and enqueues ONE Celery task
    for the entire flattened batch. Returns HTTP 202 Accepted.
    """
    # --- Authentication and initial request validation (remains the same) ---
    auth_token = request.headers.get('X-API-KEY')
    expected_token = current_app.config.get('DAMASCO_API_SECRET')

    if not expected_token:
        logger.critical("DAMASCO_API_SECRET not configured. Cannot authenticate request.")
        return jsonify({"status": "error", "message": "Server misconfiguration"}), 500

    if not auth_token or auth_token != expected_token:
        logger.warning("Unauthorized /receive-products request. Invalid or missing API token.")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if not request.is_json:
        logger.error("Invalid request: Content-Type not application/json.")
        return jsonify({"status": "error", "message": "Content-Type must be application/json."}), 415

    try:
        # This payload is expected to be in the NEW nested format.
        new_format_payload = request.get_json()
    except Exception as e_json:
        logger.error(f"Invalid JSON received: {e_json}", exc_info=True)
        return jsonify({"status": "error", "message": f"Invalid JSON format: {e_json}"}), 400

    if not isinstance(new_format_payload, list):
        logger.error("Invalid data received. Expected a JSON list of product entries.")
        return jsonify({"status": "error", "message": "Invalid format: Expected a list."}), 400

    if not new_format_payload:
        logger.info("Received an empty list of products. No action taken.")
        return jsonify({
            "status": "accepted",
            "message": "Received empty product list. No tasks enqueued.",
            "tasks_enqueued": 0
        }), 202

    logger.info(f"Received {len(new_format_payload)} new-format product entries. Preparing and flattening batch for Celery.")

    # --- MODIFIED LOGIC: Flattening the nested API data ---
    # This single list will contain all the "product-location" records with snake_case keys.
    products_batch_snake = []
    
    for nested_product in new_format_payload:
        if not isinstance(nested_product, dict) or not nested_product.get("itemCode"):
            logger.warning(f"Skipping an entry in the received batch because it's not a valid product object or is missing an itemCode: {str(nested_product)[:200]}")
            continue

        # Extract and snake_case the core product details once per product
        core_details = {
            "item_code": nested_product.get("itemCode"),
            "item_name": nested_product.get("itemName"),
            "price": nested_product.get("price"),
            "price_bolivar": nested_product.get("priceBolivar"),
            "category": nested_product.get("category"),
            "sub_category": nested_product.get("subCategory"),
            "line": nested_product.get("line"),
            "brand": nested_product.get("brand"),
            "specifitacion": nested_product.get("specifitacion"),
            "item_group_name": nested_product.get("itemGroupName"),
            "description": nested_product.get("description"),
        }

        availability_list = nested_product.get("availability")
        if not isinstance(availability_list, list):
            logger.warning(f"Product {core_details.get('item_code')} has no 'availability' list. Skipping.")
            continue
            
        # Create a separate flat record for each location in the availability list
        for location_info in availability_list:
            if not isinstance(location_info, dict) or not location_info.get("whsName"):
                logger.warning(f"Skipping a location entry for item {core_details.get('item_code')} due to missing 'whsName' or invalid format.")
                continue

            # Create a new dictionary for each flattened record by combining core and location details
            flat_record = core_details.copy()
            flat_record.update({
                "warehouse_name": location_info.get("whsName"),
                "branch_name": location_info.get("branchName"),
                "stock": location_info.get("stock"),
                "store_address": location_info.get("storeAddress")
            })
            
            products_batch_snake.append(flat_record)

    # 2. Enqueue ONE task for the entire flattened batch.
    if not products_batch_snake:
        logger.warning("No valid product-location records found in payload after flattening.")
        return jsonify({"status": "accepted", "message": "No valid items to process.", "tasks_enqueued": 0}), 202
    
    try:
        # The 'process_products_batch_task' receives the flat list it has always expected.
        process_products_batch_task.delay(products_batch_snake)
        
        response_summary = {
            "status": "accepted",
            "message": "Product data batch received, flattened, and one task enqueued for processing.",
            "original_products_received": len(new_format_payload),
            "total_product_locations_enqueued": len(products_batch_snake)
        }
        logger.info(f"Successfully enqueued one batch task for {len(products_batch_snake)} product-location records.")
        return jsonify(response_summary), 202

    except Exception as e_celery:
        logger.critical(f"CRITICAL: Failed to enqueue batch task for Celery: {e_celery}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Failed to enqueue task to the message broker. Check broker connectivity."
        }), 503