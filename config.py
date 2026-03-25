import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    UPLOAD_FOLDER = "uploads"