# /home/ec2-user/namwoo_app/namwoo_app/celery_tasks.py

import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, ValidationError, validator
from decimal import Decimal, InvalidOperation
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError
from celery.exceptions import Ignore, MaxRetriesExceededError, OperationalError as CeleryOperationalError

from .celery_app import celery_app, FlaskTask
# --- START OF MODIFICATION: Corrected Imports ---
# Import the specific services that this file actually uses.
from .services import product_service, openai_service, llm_processing_service
# --- END OF MODIFICATION ---
from .utils import db_utils, product_utils
from .models.product import Product
from .config import Config

logger = logging.getLogger(__name__)

# --- Pydantic Model for Validating Incoming Snake_Case Product Data ---
class DamascoProductDataSnake(BaseModel):
    item_code: str
    item_name: str
    description: Optional[str] = None
    specifitacion: Optional[str] = None
    stock: int
    price: Optional[Decimal] = None
    price_bolivar: Optional[Decimal] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    brand: Optional[str] = None
    line: Optional[str] = None
    item_group_name: Optional[str] = None
    warehouse_name: str
    branch_name: Optional[str] = None
    store_address: Optional[str] = None

    @validator('price', 'price_bolivar', pre=True, allow_reuse=True)
    def validate_prices_to_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None: return None
        if isinstance(v, (int, float, str)):
            try: return Decimal(str(v))
            except InvalidOperation: return None
        if isinstance(v, Decimal): return v
        return None

    class Config:
        extra = 'allow'
        validate_assignment = True

# --- NEW, EFFICIENT, AND ROBUST BATCH PROCESSING TASK ---
@celery_app.task(
    bind=True,
    base=FlaskTask,
    name='namwoo_app.celery_tasks.process_products_batch_task',
    max_retries=Config.CELERY_TASK_MAX_RETRIES if hasattr(Config, 'CELERY_TASK_MAX_RETRIES') else 3,
    default_retry_delay=Config.CELERY_TASK_RETRY_DELAY if hasattr(Config, 'CELERY_TASK_RETRY_DELAY') else 300,
    acks_late=True
)
def process_products_batch_task(self, products_batch_snake_case: List[Dict[str, Any]]):
    task_id = self.request.id
    batch_size = len(products_batch_snake_case)
    logger.info(f"Task {task_id}: Starting batch processing for {batch_size} products.")

    if not products_batch_snake_case:
        logger.info(f"Task {task_id}: Received an empty batch. Nothing to do.")
        return {"status": "success_empty_batch", "processed_count": 0}

    validated_items_for_processing = []
    product_ids_for_db_lookup = []

    # --- STEP 1: VALIDATE AND PREPARE IDS FOR DB LOOKUP ---
    for raw_item_data_snake in products_batch_snake_case:
        try:
            validated_product_pydantic = DamascoProductDataSnake(**raw_item_data_snake)
            
            id_for_lookup = product_utils.generate_product_id_for_lookup(
                validated_product_pydantic.item_code,
                validated_product_pydantic.warehouse_name
            )

            if id_for_lookup:
                validated_items_for_processing.append(
                    (id_for_lookup, validated_product_pydantic, raw_item_data_snake)
                )
                product_ids_for_db_lookup.append(id_for_lookup)
            else:
                logger.error(f"Task {task_id}: Failed to generate lookup ID for item {raw_item_data_snake.get('item_code')} "
                             f"at warehouse {raw_item_data_snake.get('warehouse_name')}. Skipping.")
        except ValidationError as e:
            logger.error(f"Task {task_id}: Pydantic validation failed for item {raw_item_data_snake.get('item_code')}. "
                         f"Skipping. Error: {e.errors()}")
            
    if not validated_items_for_processing:
        logger.warning(f"Task {task_id}: No items survived validation/ID generation. Exiting.")
        return {"status": "failed_all_invalid", "processed_count": 0}

    # --- STEP 2: FETCH EXISTING DATA ---
    existing_products_map = {}
    try:
        with db_utils.get_db_session() as session:
            if product_ids_for_db_lookup:
                existing_db_entries = session.query(
                    Product.id,
                    Product.description,
                    Product.llm_summarized_description,
                    Product.searchable_text_content,
                    Product.embedding
                ).filter(Product.id.in_(product_ids_for_db_lookup)).all()

                for entry in existing_db_entries:
                    existing_products_map[entry.id] = {
                        "description": entry.description,
                        "llm_summarized_description": entry.llm_summarized_description,
                        "searchable_text_content": entry.searchable_text_content,
                        "embedding": entry.embedding
                    }
            logger.info(f"Task {task_id}: Fetched existing data for {len(existing_products_map)} of "
                        f"{len(validated_items_for_processing)} validated products using lookup IDs.")
    except (SQLAlchemyOperationalError, CeleryOperationalError) as e:
        logger.error(f"Task {task_id}: Retriable DB/Broker error during batch read: {e}", exc_info=True)
        raise self.retry(exc=e)

    # --- STEP 3: PROCESS EACH ITEM (Summaries, Embeddings) ---
    db_ready_product_data_list = []
    for lookup_id, pydantic_product_obj, original_snake_case_data in validated_items_for_processing:
        try:
            existing_details = existing_products_map.get(lookup_id)
            
            llm_summary_to_use = existing_details.get("llm_summarized_description") if existing_details else None
            needs_new_summary = (
                (not existing_details and pydantic_product_obj.description) or
                (existing_details and pydantic_product_obj.description != existing_details.get("description")) or
                (existing_details and not existing_details.get("llm_summarized_description") and pydantic_product_obj.description)
            )

            if needs_new_summary and pydantic_product_obj.description:
                new_summary = llm_processing_service.generate_llm_product_summary(
                    html_description=pydantic_product_obj.description,
                    item_name=pydantic_product_obj.item_name
                )
                if new_summary:
                    llm_summary_to_use = new_summary
            
            product_data_dict_for_embedding = pydantic_product_obj.model_dump()
            
            text_to_embed = Product.prepare_text_for_embedding(
                damasco_product_data=product_data_dict_for_embedding,
                llm_generated_summary=llm_summary_to_use,
                raw_html_description_for_fallback=pydantic_product_obj.description
            )
            
            if not text_to_embed:
                logger.warning(f"Task {task_id}: No text for embedding for lookup_id {lookup_id}. Skipping item.")
                continue

            embedding_to_use = None
            generate_new_embedding = False
            
            if not existing_details:
                logger.info(f"Task {task_id}: Product {lookup_id} is new. Generating embedding.")
                generate_new_embedding = True
            else:
                existing_embedding_value = existing_details.get("embedding")
                if existing_embedding_value is None:
                    logger.info(f"Task {task_id}: Product {lookup_id} has no existing embedding data. Generating embedding.")
                    generate_new_embedding = True
                elif text_to_embed != existing_details.get("searchable_text_content"):
                    logger.info(f"Task {task_id}: Content changed for {lookup_id}. Regenerating embedding.")
                    generate_new_embedding = True
            
            if generate_new_embedding:
                # --- MODIFIED: Calling the correct service for embeddings ---
                embedding_to_use = openai_service.generate_product_embedding(text_to_embed)
                if embedding_to_use is None:
                    logger.error(f"Task {task_id}: Failed to generate new embedding for {lookup_id}. Skipping item.")
                    continue 
            else:
                embedding_to_use = existing_details["embedding"]
                logger.info(f"Task {task_id}: Reusing existing embedding for product {lookup_id}.")

            db_ready_product_data_list.append({
                "item_code": pydantic_product_obj.item_code,
                "item_name": pydantic_product_obj.item_name,
                "description": pydantic_product_obj.description,
                "llm_summarized_description": llm_summary_to_use,
                "specifitacion": pydantic_product_obj.specifitacion,
                "category": pydantic_product_obj.category,
                "sub_category": pydantic_product_obj.sub_category,
                "brand": pydantic_product_obj.brand,
                "line": pydantic_product_obj.line,
                "item_group_name": pydantic_product_obj.item_group_name,
                "warehouse_name": pydantic_product_obj.warehouse_name,
                "branch_name": pydantic_product_obj.branch_name,
                "store_address": pydantic_product_obj.store_address,
                "price": pydantic_product_obj.price,
                "price_bolivar": pydantic_product_obj.price_bolivar,
                "stock": pydantic_product_obj.stock,
                "searchable_text_content": text_to_embed,
                "embedding": embedding_to_use,
                "source_data_json": original_snake_case_data,
            })
        except Exception as item_proc_exc:
            logger.error(f"Task {task_id}: Failed to process item with lookup_id {lookup_id} due to: {item_proc_exc}. Skipping.", exc_info=True)
            continue

    # --- STEP 4: PERFORM THE ATOMIC BATCH UPSERT ---
    if not db_ready_product_data_list:
        logger.warning(f"Task {task_id}: No products ready for DB write after processing. Exiting.")
        return {"status": "success_nothing_to_write", "processed_count": 0}

    try:
        with db_utils.get_db_session() as session:
            product_service.upsert_products_batch(session, db_ready_product_data_list)
            session.commit()
            logger.info(f"Task {task_id}: Successfully upserted and COMMITTED {len(db_ready_product_data_list)} products.")
            return {"status": "success", "processed_count": len(db_ready_product_data_list)}
    except (SQLAlchemyOperationalError, CeleryOperationalError) as e_db_op:
        logger.error(f"Task {task_id}: Retriable DB/Broker error during final batch write: {e_db_op}", exc_info=True)
        raise self.retry(exc=e_db_op)
    except Exception as e_final:
        logger.critical(f"Task {task_id}: Error during final batch write: {e_final}", exc_info=True)
        raise self.retry(exc=e_final)


# =================================================================================================
# == DEPRECATED TASK - DO NOT USE =================================================================
# =================================================================================================
@celery_app.task(
    bind=True,
    base=FlaskTask,
    name='namwoo_app.celery_tasks.process_product_item_task_DEPRECATED',
    max_retries=Config.CELERY_TASK_MAX_RETRIES if hasattr(Config, 'CELERY_TASK_MAX_RETRIES') else 3,
    default_retry_delay=Config.CELERY_TASK_RETRY_DELAY if hasattr(Config, 'CELERY_TASK_RETRY_DELAY') else 300,
    acks_late=True
)
def process_product_item_task(self, product_data_dict_snake: Dict[str, Any]):
    logger.warning("DEPRECATED TASK 'process_product_item_task' was called. Please switch to 'process_products_batch_task'.")
    raise Ignore("Called a deprecated single-item processing task.")


# The `deactivate_product_task` is fine as it is.
@celery_app.task(
    bind=True,
    base=FlaskTask,
    name='namwoo_app.celery_tasks.deactivate_product_task',
    max_retries=Config.CELERY_TASK_MAX_RETRIES_SHORT if hasattr(Config, 'CELERY_TASK_MAX_RETRIES_SHORT') else 3,
    default_retry_delay=Config.CELERY_TASK_RETRY_DELAY_SHORT if hasattr(Config, 'CELERY_TASK_RETRY_DELAY_SHORT') else 60,
    acks_late=True
)
def deactivate_product_task(self, product_id: str):
    task_id = self.request.id    
    logger.info(f"Task {task_id}: Starting deactivation for product_id: {product_id}")
    try:
        with db_utils.get_db_session() as session:
            entry = session.query(Product).filter(Product.id == product_id).first()
            if entry:
                if entry.stock != 0:
                    entry.stock = 0
                    logger.info(f"Task {task_id}: Product_id: {product_id} stock set to 0 for deactivation.")
                    session.commit() 
                else:
                    logger.info(f"Task {task_id}: Product_id: {product_id} already has stock 0. No change needed.")
            else:
                logger.warning(f"Task {task_id}: Product_id {product_id} not found for deactivation. No action taken.")
    except (SQLAlchemyOperationalError, CeleryOperationalError) as e_op_deactivate:
        logger.error(f"Task {task_id}: Retriable OperationalError during deactivation of {product_id}: {e_op_deactivate}", exc_info=True)
        raise self.retry(exc=e_op_deactivate)
    except Exception as exc:
        logger.exception(f"Task {task_id}: Unexpected error during deactivation of product_id {product_id}: {exc}")
        raise self.retry(exc=exc)