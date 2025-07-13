# namwoo_app/utils/message_parser.py
import re
from typing import Optional, Dict

# Regex to find latitude and longitude in a Google Maps URL
# Example: https://www.google.com/maps/place/10.479382514954,-66.810897827148
# It looks for the pattern of two numbers separated by a comma.
LOCATION_URL_REGEX = re.compile(r'/@?([-]?\d+\.\d+),([-]?\d+\.\d+)')

def extract_location_from_text(text: str) -> Optional[Dict[str, float]]:
    """
    Parses a string to find and extract GPS coordinates from a map URL.
    
    Args:
        text: The user's message content.
        
    Returns:
        A dictionary with 'latitude' and 'longitude' if found, otherwise None.
    """
    if not text:
        return None

    match = LOCATION_URL_REGEX.search(text)
    if match:
        try:
            latitude = float(match.group(1))
            longitude = float(match.group(2))
            return {"latitude": latitude, "longitude": longitude}
        except (ValueError, IndexError):
            return None
    return None

# NOTE: WhatsApp also sends a native location object.
# The logic to handle that JSON object would typically be in your webhook handler
# before this text-based parser is even called. This parser is for text URLs only.