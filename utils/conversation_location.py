# namwoo_app/utils/conversation_location.py

import json
from pathlib import Path
import logging
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)

# In-memory cache for conversation locations and warehouse data
_conversation_cities: Dict[str, str] = {}
_warehouse_city_map: Optional[Dict[str, List[str]]] = None

# --- START OF MODIFICATION ---

# NEW DATA STRUCTURE: Maps specific branch names (lowercase) to their canonical city name.
# This allows the system to understand that a branch is inside a city.
BRANCH_TO_CITY_MAP: Dict[str, str] = {
    'san martin 1': 'caracas',
    'san martin 2': 'caracas',
    'san martin 4': 'caracas',
    'catia barbur': 'caracas',
    'catia antuan (gatonegro)': 'caracas',
    'catia muebleria': 'caracas',
    'la candelaria': 'caracas',
    'las mercedes': 'caracas',
    'ccct': 'caracas',
    'la trinidad': 'caracas',
    'sabana grande': 'caracas',
    'el paraíso': 'caracas', # Assuming El Paraiso is in Caracas
    'la california': 'caracas', # The key fix for the reported bug
    'guatire buenaventura': 'guatire',
    'los teques': 'los teques',
    'la guaira terminal': 'terminal la guaira',
    'valencia centro': 'valencia',
    'valencia 2 norte': 'valencia',
    'maracay': 'aragua',
    'cagua': 'cagua',
    'barquisimeto': 'barquisimeto',
    'barquisimeto ii': 'barquisimeto',
    'san cristobal': 'tachira',
    'maracaibo': 'maracaibo',
    'lecheria': 'lecherias',
    'lecherías': 'lecherias',
    'puerto ordaz': 'puerto ordaz',
    'maturín': 'maturin',
    'valera': 'trujillo',
    'puerto la cruz': 'puerto la cruz',
    'san felipe': 'san felipe',
}

# Synonyms for city names. The key is the user input, 
# the value is the canonical city name used in tiendas_data.json and the new map.
_city_synonyms: Dict[str, str] = {
    "caracas": "caracas",
    "ccs": "caracas",
    "maracaibo": "maracaibo",
    "mcbo": "maracaibo",
    "valencia": "valencia",
    "barquisimeto": "barquisimeto",
    "bqto": "barquisimeto",
    "maracay": "aragua", 
    "aragua": "aragua",
    "lecheria": "lecherias",
    "lecherías": "lecherias",
    "puerto la cruz": "puerto la cruz",
    "plc": "puerto la cruz",
    "san cristobal": "tachira",
    "tachira": "tachira",
    "maturin": "maturin",
    "puerto ordaz": "puerto ordaz",
    "pzo": "puerto ordaz",
    "la guaira": "terminal la guaira",
    "vargas": "terminal la guaira",
    "cagua": "cagua",
    "los teques": "los teques",
    "san felipe": "san felipe",
    "yaracuy": "san felipe",
    "trujillo": "trujillo",
    "valera": "trujillo",
    "guatire": "guatire" # Added Guatire as a city
}

# --- END OF MODIFICATION ---

def _load_and_process_tiendas_data() -> Dict[str, List[str]]:
    """
    Loads tienda data from JSON and creates a direct mapping from a canonical 
    city name to a list of exact `whsName` values for the database query.
    This version reads the 'city' and 'whsName' keys directly, no parsing needed.
    """
    global _warehouse_city_map
    if _warehouse_city_map is not None:
        return _warehouse_city_map

    logger.info("Initializing city-to-warehouse map from tiendas_data.json...")
    
    file_path = Path(__file__).parent.parent / 'data' / 'tiendas_data.json'
    
    if not file_path.exists():
        logger.error(f"FATAL: tiendas_data.json not found at {file_path}. Location features will fail.")
        _warehouse_city_map = {}
        return _warehouse_city_map

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tiendas = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading or parsing tiendas_data.json: {e}")
        _warehouse_city_map = {}
        return _warehouse_city_map

    processed_map: Dict[str, List[str]] = {}

    for tienda in tiendas:
        city = tienda.get("city")
        whs_name = tienda.get("whsName")
        
        if not city or not whs_name:
            logger.warning(f"Skipping store entry due to missing 'city' or 'whsName': {tienda}")
            continue
            
        canonical_city = city.lower()

        if canonical_city not in processed_map:
            processed_map[canonical_city] = []
        
        if whs_name not in processed_map[canonical_city]:
            processed_map[canonical_city].append(whs_name)

    _warehouse_city_map = processed_map
    logger.info(f"City-to-warehouse map initialized. Found mappings for {len(_warehouse_city_map)} cities.")
    logger.debug(f"Generated map: {_warehouse_city_map}")
    return _warehouse_city_map


def detect_city_from_text(text: str) -> Optional[str]:
    """
    Detects a known city from a text string, prioritizing specific branch names first.
    Returns the canonical city name if found.
    """
    text_lower = text.lower().strip()
    
    # --- START OF MODIFICATION ---
    # STEP 1: Prioritize matching a specific branch name first.
    if text_lower in BRANCH_TO_CITY_MAP:
        resolved_city = BRANCH_TO_CITY_MAP[text_lower]
        logger.info(f"Detected branch name '{text_lower}', resolved to canonical city '{resolved_city}'.")
        return resolved_city
    # --- END OF MODIFICATION ---
        
    # STEP 2: Fallback to checking for city names and synonyms.
    if text_lower in _city_synonyms:
        return _city_synonyms[text_lower]
        
    # STEP 3: Fallback to substring matching for robustness.
    for synonym, canonical_name in _city_synonyms.items():
        if synonym in text_lower:
            logger.info(f"Detected city '{canonical_name}' from text via substring synonym '{synonym}'.")
            return canonical_name
            
    return None

def set_conversation_city(conversation_id: str, city: str) -> None:
    """Stores the detected canonical city for a given conversation ID."""
    # We resolve the city name here to ensure we always store the canonical version
    resolved_city = detect_city_from_text(city) or city.lower()
    _conversation_cities[conversation_id] = resolved_city
    logger.info(f"Set city for conversation {conversation_id} to '{resolved_city}'.")


def get_conversation_city(conversation_id: str) -> Optional[str]:
    """Retrieves the stored city for a given conversation ID."""
    return _conversation_cities.get(conversation_id)


def get_city_warehouses(conversation_id: str) -> Optional[List[str]]:
    """
    Gets the list of exact warehouse names (`whsName`) associated with the city
    stored for the given conversation.
    """
    city = get_conversation_city(conversation_id)
    if not city:
        return None
    
    warehouse_map = _load_and_process_tiendas_data()
    # Ensure we use the final canonical name from our logic
    canonical_city = detect_city_from_text(city) or city
    warehouses = warehouse_map.get(canonical_city)
    
    if warehouses:
        logger.info(f"Found warehouses for city '{canonical_city}': {warehouses}")
    else:
        logger.warning(f"No warehouses found for city '{canonical_city}' in the map.")
        
    return warehouses

# Initialize the map on module load
_load_and_process_tiendas_data()