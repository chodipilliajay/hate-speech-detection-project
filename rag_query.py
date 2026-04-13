from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get the API key from environment
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

from together import Together
client = Together(api_key=TOGETHER_API_KEY)
