"""Central settings. Everything else imports from here."""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")   # main answers
FAST_MODEL = os.getenv("FAST_MODEL", "openai/gpt-oss-20b")      # cheap helper tasks
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")

os.makedirs("data", exist_ok=True)
DATA_DIR = "data/chroma"              # vector DB (the persistent memory)
SUMMARY_FILE = "data/summaries.json"  # per-file summaries

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_IMAGES_PER_PAGE = 1
MIN_IMAGE_PX = 200
TOP_K = 8

# --- Vision limits (free tier: qwen allows ~1000 output tokens per MINUTE) ---
VISION_OTPM = int(os.getenv("VISION_OTPM", "1000"))   # raise in .env if you upgrade your plan
VISION_MAX_TOKENS = 450       # size of one figure caption
SCAN_MAX_TOKENS = 900         # size of a scanned-page transcription
DEFAULT_MAX_IMAGES = 6        # images captioned per file (UI slider overrides)                             # chunks sent to the LLM per question

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def reasoning_args(model: str) -> dict:
    """gpt-oss models 'think' first; keep it short so answers are fast and tokens aren't wasted."""
    return {"reasoning_effort": "low"} if "gpt-oss" in model else {}


# --- Local OCR (for image-only pages) ---
OCR_MIN_CHARS = 300   # pages with less real text than this get OCR'd
