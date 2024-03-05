import os
from dotenv import load_dotenv
from pydantic import BaseSettings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Settings(BaseSettings):
    BASE_PORT = os.getenv("BASE_PORT", 8900)
    BASE_HOST = os.getenv("BASE_HOST", "0.0.0.0")
    STATIC_URL = os.getenv("STATIC_URL", "static")
    LOG_FILE_PATH = os.getenv("STATIC_URL", "app.log")
    LOGGING_CONFIG_FILE = os.path.join(BASE_DIR, "logging.ini")
    STREAM_DELAY = float(os.getenv("STREAM_DELAY", 0.1))
    RETRY_TIMEOUT = int(os.getenv("RETRY_TIMEOUT", 15000))

    URL_OPEN_LLM = os.getenv("URL_OPEN_LLM", "0.0.0.0")
    URL_EMBED_MODEL = os.getenv("URL_EMBED_MODEL", "0.0.0.0")
    URL_VECTOR_STORE = os.getenv("URL_VECTOR_STORE", "0.0.0.0")

    OPENAI_KEY = os.getenv("OPENAI_KEY", "")
    GOOGLE_GEMINI_KEY = os.getenv("GOOGLE_GEMINI_KEY", "")

settings = Settings()
