from dotenv import load_dotenv
import os
from global_services.defaults_config import DEFAULT_ENV_VARS

# Global dictionary to hold credentials
GLOBAL_CREDENTIALS = {}

def load_credentials():
    """Loads environment variables into GLOBAL_CREDENTIALS if not already set and returns the dictionary."""
    global GLOBAL_CREDENTIALS

    if not GLOBAL_CREDENTIALS:
        load_dotenv()  # Load from .env file
        for key, default in DEFAULT_ENV_VARS.items():
            value = os.getenv(key, default)
            GLOBAL_CREDENTIALS[key] = value
        print("[INFO] Environment variables loaded into GLOBAL_CREDENTIALS.")
    else:
        print("[INFO] GLOBAL_CREDENTIALS already loaded.")

    return GLOBAL_CREDENTIALS.copy()  # ✅ Return the dictionary
