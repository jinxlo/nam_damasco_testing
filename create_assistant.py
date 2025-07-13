# create_assistant.py
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# This import is correct and should remain.
try:
    from .services.tools_schema import tools_schema
except ImportError as e:
    print("\nERROR: Could not import 'tools_schema'.")
    print("Please make sure you are running this script from the parent directory of 'namwoo_app'.")
    print(f"Also ensure that 'namwoo_app/services/__init__.py' and 'namwoo_app/services/tools_schema.py' exist.")
    print(f"Detailed Error: {e}")
    exit()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_damasco_assistant():
    """
    Reads the system prompt and tools schema from local files to create
    a new OpenAI Assistant. This ensures the Assistant's configuration
    is version-controlled and reproducible.
    """
    load_dotenv()
    logging.info("Loaded environment variables from .env file...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("CRITICAL: OPENAI_API_KEY not found in .env file. Cannot proceed.")
        return

    client = OpenAI(api_key=api_key)
    logging.info("OpenAI client initialized.")

    # --- START OF MODIFICATION ---
    # Build a robust path to the prompt file.
    try:
        # Get the directory where this script itself is located.
        # When run with -m, __file__ gives the absolute path to the script.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the prompt file based on the project structure.
        prompt_file_path = os.path.join(script_dir, "data", "system_prompt.txt")
        
        logging.info(f"Attempting to read system prompt from: {prompt_file_path}")
        
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        logging.info("Successfully read 'system_prompt.txt'.")
        
    except FileNotFoundError:
        logging.error(f"CRITICAL: Could not find system prompt file at the expected path: {prompt_file_path}")
        return
    # --- END OF MODIFICATION ---

    logging.info("Sending request to OpenAI to create Assistant 'Tomás'...")
    try:
        assistant = client.beta.assistants.create(
            name="Tomás - Asistente Damasco",
            instructions=prompt_content,
            tools=tools_schema,
            model="gpt-4.1-mini"
        )
        logging.info(f"Assistant created with ID: {assistant.id}")

    except Exception as e:
        logging.error(f"Failed to create Assistant on OpenAI's servers. Error: {e}", exc_info=True)
        return

    print("\n" + "="*50)
    print("✅ Assistant Created Successfully!")
    print(f"   Assistant ID: {assistant.id}")
    print("="*50)
    print("\n>>> ACTION REQUIRED <<<\n")
    print("Copy the Assistant ID above and add it to your .env file as:")
    print(f"OPENAI_ASSISTANT_ID={assistant.id}\n")

if __name__ == "__main__":
    create_damasco_assistant()