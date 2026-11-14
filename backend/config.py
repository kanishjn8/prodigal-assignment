import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
load_dotenv(BACKEND_ROOT / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=False)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CANDIDATE_API_KEY = os.getenv("CANDIDATE_API_KEY", "")
CANDIDATE_API_BASE_URL = os.getenv(
    "CANDIDATE_API_BASE_URL",
    "https://candidate-db-api.service.external.usea2.aws.prodigaltech.com",
)
