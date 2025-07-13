# NAMWOO/services/product_service.py

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation as InvalidDecimalOperation
from datetime import datetime

import numpy as np
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import case, func, or_, and_
from sqlalchemy.dialects.postgresql import insert

from ..models.product import Product
from ..utils import db_utils, embedding_utils, text_utils
from ..config import Config

logger = logging.getLogger(__name__)

# --- Regex and Keywords for Intelligent Search ---
SPEC_REGEX = re.compile(r'\b(\d+)\s*(gb|tb)\s*(?:de\s*)?(ram|almacenamiento|storage|memoria|interno)?\b', re.IGNORECASE)
CATEGORY_KEYWORDS = {
    'CELULAR': ['celular', 'celulares', 'teléfono', 'telefonos', 'smartphone'],
    'TABLET': ['tablet', 'tablets'],
    'LAPTOP': ['laptop', 'laptops', 'portátil', 'portatiles'],
    'TELEVISOR': ['televisor', 'televisores', 'tv', 'pantalla'],
}
_SKU_PAT = re.compile(r'\b(SM-[A-Z0-9]+[A-Z]*|[A-Z0-9]{8,})\b')
_KNOWN_COLORS = {
    'negro', 'blanco', 'azul', 'rojo', 'verde', 'gris', 'plata', 'dorado',
    'rosado', 'violeta', 'morado', 'amarillo', 'naranja', 'marrón', 'beige',
    'celeste', 'turquesa', 'lila', 'crema', 'grafito', 'titanio', 'cobre',
    'negra', 'blanca', 'claro', 'oscuro', 'marino'
}

# ===========================================================================
# Helper Functions for Search Logic
# ===========================================================================

def _extract_base_name_and_color(item_name: str) -> Tuple[str, Optional[str]]:
    if not item_name: return "", None
    name_without_sku = _SKU_PAT.sub('', item_name).strip()
    words = name_without_sku.split()
    base_parts, color_parts = [], []
    found_color = False
    for w in reversed(words):
        if not found_color and w.lower() in _KNOWN_COLORS:
            color_parts.insert(0, w)
        else:
            found_color = True
            base_parts.insert(0, w)
    base = " ".join(base_parts).strip()
    color = " ".join(color_parts).strip()
    return (base or name_without_sku, color.capitalize() if color else None)

def _detect_sub_category(query: str) -> Optional[str]:
    """Detects a target sub_category from keywords in the user's query."""
    query_lower = query.lower()
    for sub_category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            return sub_category
    return None

def _group_product_results(product_rows: List[Product]) -> Dict[str, Any]:
    """Helper to group multiple product rows into a structured dictionary for the LLM."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for prod_row in product_rows:
        base, color = _extract_base_name_and_color(prod_row.item_name)
        if not base: base = prod_row.item_name

        if base not in grouped:
            desc = prod_row.llm_summarized_description or text_utils.strip_html_to_text(prod_row.description or "")
            specs = (prod_row.specifitacion or "").strip()
            grouped[base] = {
                "base_name": base, "brand": prod_row.brand, "category": prod_row.category,
                "sub_category": prod_row.sub_category, "marketing_description": desc.strip(),
                "technical_specs": specs, "variants": [], "locations": []
            }
        
        variant = {
            "color": color or "N/A", "price": float(prod_row.price) if prod_row.price else None,
            "price_bolivar": float(prod_row.price_bolivar) if prod_row.price_bolivar else None,
            "full_item_name": prod_row.item_name, "item_code": prod_row.item_code
        }
        if variant not in grouped[base]["variants"]:
            grouped[base]["variants"].append(variant)
        
        grouped[base]["locations"].append({
            "warehouse_name": prod_row.warehouse_name, "branch_name": prod_row.branch_name,
            "branch_address": prod_row.store_address, "stock": prod_row.stock
        })

    for product_group in grouped.values():
        merged_locations = {}
        for loc in product_group["locations"]:
            branch_name = loc["branch_name"]
            if branch_name and branch_name not in merged_locations:
                merged_locations[branch_name] = {"branch_name": branch_name, "total_stock": 0}
            if branch_name:
                merged_locations[branch_name]["total_stock"] += loc["stock"]
        product_group["locations"] = list(merged_locations.values())

    return {"status": "success", "products_grouped": list(grouped.values())}

def _format_sku_result(product_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Formats results for a single SKU or product name into the 'product_details' structure."""
    if not product_rows:
        return {}
    first = product_rows[0]
    base_name, _ = _extract_base_name_and_color(first.get("item_name", ""))
    
    variants = list({
        p['item_code']: {
            "color": _extract_base_name_and_color(p['item_name'])[1] or "N/A", 
            "price": p.get("price"),
            "price_bolivar": p.get("price_bolivar"), 
            "full_item_name": p.get("item_name"), 
            "item_code": p.get("item_code")
        } for p in product_rows
    }.values())

    locations = list({p['branch_name']: {"branch_name": p['branch_name']} for p in product_rows if p.get('branch_name')}.values())

    return {
        "status": "success",
        "product_details": {
            "item_name": base_name,
            "brand": first.get("brand"),
            "category": first.get("category"),
            "specifitacion": first.get("specifitacion"),
            "variants": variants,
            "locations": locations
        }
    }

# ===========================================================================
# Core Service Functions
# ===========================================================================

def get_available_brands_by_category(category: str = 'CELULAR') -> Optional[List[str]]:
    """Fetches a distinct list of in-stock brands for a given product sub-category."""
    if not category: return []
    logger.info(f"Fetching distinct, in-stock brands for sub_category: {category}")
    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for get_available_brands_by_category.")
            return None
        try:
            qry = (session.query(Product.brand)
                   .filter(Product.sub_category == category.upper(),
                           Product.item_group_name == "DAMASCO TECNO",
                           Product.stock > 0)
                   .distinct().order_by(Product.brand))
            brands = [b[0] for b in qry.all() if b[0] is not None]
            logger.info(f"Found {len(brands)} brands for {category}")
            return brands
        except Exception as e:
            logger.exception(f"Error fetching distinct brands: {e}")
            return None

def find_products(query: str, warehouse_names: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """
    Performs an intelligent, multi-stage search pipeline for products.
    1. Direct Match (SKU or Item Name) -> 2. Specific Spec Filter -> 3. Brand Match -> 4. Vector Search
    """
    if not query:
        logger.warning("find_products called with an empty query.")
        return {"status": "error", "message": "Query cannot be empty."}

    with db_utils.get_db_session() as session:
        # Step 1: Direct Match (SKU or Item Name)
        logger.debug(f"find_products [1/4]: Direct match for '{query}' on SKU or Item Name.")
        direct_match_results = get_live_product_details_by_direct_match(session, query)
        if direct_match_results:
            available_in_location = [p for p in direct_match_results if not warehouse_names or p.get("warehouse_name") in warehouse_names]
            if available_in_location:
                logger.info(f"find_products: Success [Direct Match] found {len(available_in_location)} results.")
                base_names = {_extract_base_name_and_color(p['item_name'])[0] for p in available_in_location}
                # If all results share the same base name, it's a single product with variants.
                if len(base_names) == 1:
                    formatted_result = _format_sku_result(available_in_location)
                    formatted_result['search_method'] = 'direct_match (single product)'
                    return formatted_result
                else: # Found multiple distinct products (e.g., query "honor")
                    product_objects = session.query(Product).filter(Product.id.in_([p['id'] for p in available_in_location])).all()
                    results = _group_product_results(product_objects)
                    results['search_method'] = 'direct_match (multiple products)'
                    return results

            elif warehouse_names: # Found but not in the requested city
                 return {"status": "not_found_in_city", "message": "Producto encontrado pero sin stock para retiro en la ciudad especificada."}

        # Step 2: Specific Spec Filter
        logger.debug(f"find_products [2/4]: Spec filter for '{query}'")
        spec_match = SPEC_REGEX.search(query)
        detected_category = _detect_sub_category(query)
        if spec_match and detected_category:
            value, unit = spec_match.group(1), spec_match.group(2)
            search_term = f"%{value}{unit}%"
            logger.info(f"Spec match found ('{search_term}'). Filtering by sub_category '{detected_category}'.")
            q = session.query(Product).filter(
                Product.stock > 0,
                Product.item_group_name == "DAMASCO TECNO",
                Product.sub_category == detected_category,
                or_(Product.specifitacion.ilike(search_term), Product.item_name.ilike(search_term))
            )
            if warehouse_names: q = q.filter(Product.warehouse_name.in_(warehouse_names))
            spec_rows = q.limit(300).all()
            if spec_rows:
                logger.info(f"find_products: Success [Spec Match] found {len(spec_rows)} results.")
                results = _group_product_results(spec_rows)
                results['search_method'] = 'spec_filter'
                return results

        # Step 3: Brand Match
        logger.debug(f"find_products [3/4]: Brand match for '{query}'")
        all_brands = get_available_brands_by_category() or []
        matched_brand = next((b for b in all_brands if b.lower() in query.lower()), None)
        if matched_brand:
            logger.info(f"Brand match found ('{matched_brand}').")
            q = session.query(Product).filter(
                Product.stock > 0,
                Product.item_group_name == "DAMASCO TECNO",
                Product.brand.ilike(f"%{matched_brand}%")
            )
            if warehouse_names: q = q.filter(Product.warehouse_name.in_(warehouse_names))
            brand_rows = q.limit(300).all()
            if brand_rows:
                logger.info(f"find_products: Success [Brand Match] found {len(brand_rows)} results.")
                results = _group_product_results(brand_rows)
                results['search_method'] = 'brand_match'
                return results
        
        # Step 4: Vector Search (Fallback)
        logger.debug(f"find_products [4/4]: Vector search for '{query}'")
        model = getattr(Config, 'OPENAI_EMBEDDING_MODEL', "text-embedding-3-small")
        q_emb = embedding_utils.get_embedding(query, model=model)
        if q_emb:
            q = session.query(Product).filter(Product.stock > 0, Product.item_group_name == "DAMASCO TECNO")
            if warehouse_names: q = q.filter(Product.warehouse_name.in_(warehouse_names))
            q = q.filter((1 - Product.embedding.cosine_distance(q_emb)) >= 0.10)
            q = q.order_by(Product.embedding.cosine_distance(q_emb)).limit(300)
            vector_rows = q.all()
            if vector_rows:
                logger.info(f"find_products: Success [Vector Search] found {len(vector_rows)} results.")
                results = _group_product_results(vector_rows)
                results['search_method'] = 'vector_search'
                return results

        # All methods failed
        logger.warning(f"find_products: Failure - All search methods failed for '{query}'")
        return {"status": "not_found", "message": "Lo siento, no pude encontrar productos que coincidan con tu búsqueda."}

def get_live_product_details_by_direct_match(session: Session, query: str) -> Optional[List[Dict[str, Any]]]:
    """
    Searches by exact SKU or for all words in the query within the item name.
    """
    if not (cleaned_query := query.strip()): 
        return []
    try:
        # Condition 1: Exact, case-insensitive match on item_code
        sku_condition = func.lower(Product.item_code) == func.lower(cleaned_query)

        # Condition 2: All words in the query must be present in the item_name (case-insensitive)
        query_words = cleaned_query.split()
        name_conditions = [Product.item_name.ilike(f"%{word}%") for word in query_words]
        
        rows = session.query(Product).filter(
            or_(
                sku_condition,
                and_(*name_conditions)
            )
        ).all()
        return [r.to_dict() for r in rows] if rows else []
    except Exception as e:
        logger.exception(f"DB error during direct match for query: {query}. Error: {e}")
        return None

def get_live_product_details_by_sku(item_code_query: str) -> Optional[List[Dict[str, Any]]]:
    """Legacy function, now delegates to the new direct match function."""
    if not (code := str(item_code_query or '').strip()): return []
    with db_utils.get_db_session() as session:
        if not session: return None
        return get_live_product_details_by_direct_match(session, code)


def get_branch_address(branch_name: str, city: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Fetches the address of a specific branch. The city parameter is optional and used for logging context.
    """
    if not branch_name:
        return {"status": "error", "message": "Branch name cannot be empty."}
    
    with db_utils.get_db_session() as session:
        if not session:
            return {"status": "error", "message": "Database session not available."}
        try:
            # Query for a product that is located at the specified branch and has an address.
            # We use ILIKE for case-insensitive matching and only need one result.
            result = (
                session.query(Product.branch_name, Product.store_address)
                .filter(
                    Product.branch_name.ilike(f"%{branch_name.strip()}%"),
                    Product.store_address.isnot(None),
                    Product.store_address != ""
                )
                .first()
            )
            
            if result:
                # result is a tuple: (branch_name_from_db, store_address_from_db)
                db_branch_name, db_store_address = result
                logger.info(f"Address found for branch '{branch_name}': {db_store_address}")
                return {
                    "status": "success",
                    "branch_name": db_branch_name,
                    "branch_address": db_store_address
                }
            
            logger.warning(f"Address not found for branch '{branch_name}' (city hint: '{city}').")
            return {"status": "not_found", "message": f"No se encontró la dirección para la sucursal '{branch_name}'."}

        except Exception as e:
            logger.exception(f"DB error fetching branch address for '{branch_name}': {e}")
            return {"status": "error", "message": f"Error interno buscando la dirección para '{branch_name}'."}

def query_accessories(main_product_item_code: str, city_warehouses: Optional[List[str]], limit: int = 3) -> Optional[List[str]]:
    """
    Finds relevant accessories for a given main product.

    The logic is to find accessories that are:
    1. From the same brand as the main product (highest priority).
    2. In the same general category (e.g., 'CELULAR').
    3. Filtered by the user's city if provided.
    """
    if not main_product_item_code:
        return []
        
    with db_utils.get_db_session() as session:
        if not session:
            return None
        
        try:
            main_product = session.query(Product).filter(Product.item_code == main_product_item_code).first()
            if not main_product or not main_product.brand:
                logger.warning(f"Could not find main product or its brand for item_code {main_product_item_code}")
                return []
            
            logger.info(f"Querying accessories for '{main_product.item_name}' (Brand: {main_product.brand})")

            # Base query for accessories
            query = session.query(Product).filter(
                Product.sub_category == "ACCESORIO",
                Product.stock > 0,
                Product.item_group_name == "DAMASCO TECNO",
                Product.brand.ilike(f"%{main_product.brand}%") # Stricter filter for the same brand
            )

            # If a city context exists, apply it.
            if city_warehouses:
                query = query.filter(Product.warehouse_name.in_(city_warehouses))
            
            # Order by stock and get the top results
            accessories = query.order_by(Product.stock.desc()).limit(limit).all()
            
            if not accessories:
                logger.info(f"No matching brand accessories found for {main_product.brand}. No fallback implemented.")
                return []

            # Format the results
            return [f"{_extract_base_name_and_color(acc.item_name)[0]} (${acc.price:.2f})" for acc in accessories]

        except Exception as e:
            logger.exception(f"Error in query_accessories for {main_product_item_code}: {e}")
            return None

# ===========================================================================
# Legacy Data Pipeline Functions (Unchanged)
# ===========================================================================

def upsert_products_batch(db_session: Session, products_data: List[Dict[str, Any]]):
    if not products_data:
        logger.info("upsert_products_batch called with an empty list.")
        return
    stmt = insert(Product).values(products_data)
    on_conflict_stmt = stmt.on_conflict_do_update(
        index_elements=['id'],
        set_={
            'item_code': stmt.excluded.item_code, 'item_name': stmt.excluded.item_name,
            'description': stmt.excluded.description, 'llm_summarized_description': stmt.excluded.llm_summarized_description,
            'specifitacion': stmt.excluded.specifitacion, 'category': stmt.excluded.category,
            'sub_category': stmt.excluded.sub_category, 'brand': stmt.excluded.brand,
            'line': stmt.excluded.line, 'item_group_name': stmt.excluded.item_group_name,
            'warehouse_name': stmt.excluded.warehouse_name, 'warehouse_name_canonical': stmt.excluded.warehouse_name_canonical,
            'branch_name': stmt.excluded.branch_name, 'store_address': stmt.excluded.store_address,
            'price': stmt.excluded.price, 'price_bolivar': stmt.excluded.price_bolivar,
            'stock': stmt.excluded.stock, 'searchable_text_content': stmt.excluded.searchable_text_content,
            'embedding': stmt.excluded.embedding, 'source_data_json': stmt.excluded.source_data_json,
        }
    )
    db_session.execute(on_conflict_stmt)
    logger.info(f"Executed batch upsert for {len(products_data)} products.")

def add_or_update_product_in_db(*args, **kwargs):
    # This function is part of a legacy data ingestion flow and is not called by the live agent.
    # It remains here for compatibility with other system components.
    pass

def get_product_by_id_from_db(db_session: Session, product_id: str) -> Optional[Product]:
    if not product_id: return None
    return db_session.query(Product).filter(Product.id == product_id).first()

def search_similar_products(item_code: str, warehouse_names: Optional[List[str]], limit: int = 5, min_score: float = 0.75) -> Optional[Dict[str, Any]]:
    if not item_code: return {}
    with db_utils.get_db_session() as session:
        if not session: return None
        source_product = session.query(Product).filter(Product.item_code == item_code).first()
        if not source_product or source_product.embedding is None: return {}
        
        q = session.query(Product).filter(
            Product.stock > 0,
            Product.item_group_name == "DAMASCO TECNO",
            Product.item_code != item_code,
            (1 - Product.embedding.cosine_distance(source_product.embedding)) >= min_score
        )
        if warehouse_names: q = q.filter(Product.warehouse_name.in_(warehouse_names))
        rows = q.order_by(Product.embedding.cosine_distance(source_product.embedding)).limit(limit).all()
        
        if not rows: return {"status": "not_found", "products_grouped": []}
        return _group_product_results(rows)