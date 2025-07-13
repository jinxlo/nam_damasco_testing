# Dependencies: requests, geopy
# Install them with: pip install requests geopy

import os
import json
import logging
import requests
from geopy.distance import geodesic
from typing import Union

logger = logging.getLogger(__name__)

# --- Configuration ---
# Uses the full, absolute path to the JSON file on your server.
STORE_LOCATIONS_FILE = '/home/ec2-user/namwoo_app/data/store_locations.json'


def load_store_locations():
    """
    Loads store locations from the JSON file into a more usable dictionary format.
    Returns a dictionary mapping branch names to their (lat, lon) coordinates.
    """
    try:
        with open(STORE_LOCATIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Transforms the list of stores into a dictionary for efficient lookups
            return {
                store['branch_name']: (store['latitude'], store['longitude'])
                for store in data.get('stores', [])
            }
    except FileNotFoundError:
        logger.error(f"FATAL: The store locations file was not found at '{STORE_LOCATIONS_FILE}'")
        return {}
    except json.JSONDecodeError:
        logger.error(f"FATAL: The store locations file '{STORE_LOCATIONS_FILE}' is not valid JSON.")
        return {}
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading stores: {e}")
        return {}

# --- Load store data once when the module is imported ---
STORE_LOCATIONS = load_store_locations()


def get_coords_from_address(address: str) -> dict:
    """
    Performs forward geocoding using the Google Maps Geocoding API.
    Converts a human-readable address string into GPS coordinates.

    Args:
        address: The address string to geocode (e.g., "Petare, Caracas, Venezuela").

    Returns:
        A dictionary with 'latitude', 'longitude', 'formatted_address', or an 'error'.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error("FATAL: GOOGLE_MAPS_API_KEY environment variable not set.")
        return {"error": "Google Maps API key is not configured on the server."}

    # Append ", Venezuela" to bias results towards the correct country
    if "venezuela" not in address.lower():
        address_with_country = f"{address}, Venezuela"
    else:
        address_with_country = address

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': address_with_country,
        'key': api_key,
        'language': 'es'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('status') == 'OK' and data.get('results'):
            location = data['results'][0]['geometry']['location']
            formatted_address = data['results'][0]['formatted_address']
            return {
                "latitude": location['lat'],
                "longitude": location['lng'],
                "formatted_address": formatted_address
            }
        elif data.get('status') == 'ZERO_RESULTS':
            logger.warning(f"Google Maps API could not find coordinates for address: '{address}'")
            return {"error": f"No se pudieron encontrar coordenadas para la dirección: '{address}'."}
        else:
            error_message = data.get('error_message', f"Google API Error: {data.get('status')}")
            logger.error(f"Google Maps API error: {error_message}")
            return {"error": error_message}

    except requests.exceptions.RequestException as e:
        logger.exception(f"ERROR calling Google Maps API for forward geocoding: {e}")
        return {"error": "Failed to connect to the Google Maps service."}


def get_address_from_coords(latitude: float, longitude: float) -> dict:
    """
    Performs reverse geocoding using the Google Maps Geocoding API.
    Converts GPS coordinates into a human-readable street address.
    
    Args:
        latitude: The latitude of the location.
        longitude: The longitude of the location.
        
    Returns:
        A dictionary containing the 'formatted_address' or an 'error'.
    """
    # Securely get the API key from environment variables.
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    
    if not api_key:
        logger.error("FATAL: GOOGLE_MAPS_API_KEY environment variable not set.")
        return {"error": "Google Maps API key is not configured on the server."}
        
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'latlng': f"{latitude},{longitude}",
        'key': api_key,
        'language': 'es' # Request the address in Spanish
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            # The first result is typically the most specific and accurate one.
            return {"formatted_address": data['results'][0]['formatted_address']}
        else:
            error_message = data.get('error_message', f"Google API Error: {data.get('status')}")
            logger.error(f"Google Maps API error: {error_message}")
            return {"error": error_message}
            
    except requests.exceptions.RequestException as e:
        logger.exception(f"ERROR calling Google Maps API: {e}")
        return {"error": "Failed to connect to the Google Maps service."}


def find_nearby_stores(latitude: float, longitude: float, limit: int = 3) -> dict:
    """
    Finds a sorted list of the closest stores to the given coordinates.

    Args:
        latitude: The user's latitude.
        longitude: The user's longitude.
        limit: The maximum number of nearby stores to return.

    Returns:
        A dictionary containing a list of nearby stores, sorted by distance.
    """
    if not STORE_LOCATIONS:
        logger.error("Could not find nearby stores because STORE_LOCATIONS is empty. Check file path and JSON content.")
        return {"nearby_stores": [], "error": "Store locations are not loaded on the server."}

    user_location = (latitude, longitude)
    
    distances = {
        name: geodesic(user_location, store_coords).km
        for name, store_coords in STORE_LOCATIONS.items()
    }
    
    # Sort the stores by distance (from closest to farthest)
    sorted_stores = sorted(distances.items(), key=lambda item: item[1])
    
    # Format the output into a list of dictionaries
    nearby_stores_list = [
        {"branch_name": name, "distance_km": round(dist, 2)}
        for name, dist in sorted_stores
    ]
    
    return {
        "nearby_stores": nearby_stores_list[:limit] # Return only the top N stores
    }


def get_location_details(latitude: float, longitude: float) -> dict:
    """
    This is the main public function for this service.
    It orchestrates the other functions to return a complete, structured
    set of details for a given GPS location.

    Args:
        latitude: The latitude to get details for.
        longitude: The longitude to get details for.

    Returns:
        A comprehensive dictionary with address and a list of nearby stores.
    """
    address_info = get_address_from_coords(latitude, longitude)
    store_info = find_nearby_stores(latitude, longitude, limit=3) # Get the top 3 stores
    
    # Combine all information into a single response object
    response = {
        "latitude": latitude,
        "longitude": longitude,
        "formatted_address": address_info.get("formatted_address"),
        "nearby_stores": store_info.get("nearby_stores", []),
        "error": address_info.get("error") or store_info.get("error")
    }
    return response


def get_location_details_from_address(address: str) -> dict:
    """
    This is a main public function for this service.
    It takes a text-based address, geocodes it, and then finds nearby stores.

    Args:
        address: The text-based address to get details for.

    Returns:
        A comprehensive dictionary with coordinates, address, and a list of nearby stores.
    """
    coords_info = get_coords_from_address(address)
    
    if "error" in coords_info:
        # Pass the error up
        return {
            "requested_address": address,
            "error": coords_info["error"]
        }

    latitude = coords_info["latitude"]
    longitude = coords_info["longitude"]
    
    store_info = find_nearby_stores(latitude, longitude, limit=3)
    
    # Combine all information into a single response object
    response = {
        "requested_address": address,
        "latitude": latitude,
        "longitude": longitude,
        "formatted_address": coords_info.get("formatted_address"),
        "nearby_stores": store_info.get("nearby_stores", []),
        "error": store_info.get("error") # In case stores fail to load
    }
    return response


# --- Example Usage Block for Testing ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not STORE_LOCATIONS:
        logger.error("\nHalting test because store locations could not be loaded.")
    else:
        logger.info(f"Successfully loaded {len(STORE_LOCATIONS)} store locations from '{STORE_LOCATIONS_FILE}'.")
        
        # Test case: The location from your logs
        test_latitude = 10.479374
        test_longitude = -66.810859
        
        logger.info(f"\n--- Testing with coordinates: ({test_latitude}, {test_longitude}) ---")
        
        location_details = get_location_details(test_latitude, test_longitude)
        
        print("\nService Response from Coordinates:")
        print(json.dumps(location_details, indent=2, ensure_ascii=False))

        # Test case: New forward geocoding function
        test_address = "Petare, Caracas"
        logger.info(f"\n--- Testing with address: '{test_address}' ---")
        location_details_from_address = get_location_details_from_address(test_address)
        print("\nService Response from Address:")
        print(json.dumps(location_details_from_address, indent=2, ensure_ascii=False))