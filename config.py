import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# claude-haiku-4-5 for bulk segment coding (cheap + fast via async)
# claude-opus-4-7 for theme synthesis (best quality for thesis output)
CODING_MODEL = "claude-haiku-4-5"
SYNTHESIS_MODEL = "claude-opus-4-7"

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
OUTPUTS_DIR = DATA_DIR / "outputs"
TEMPLATES_DIR = BASE_DIR / "templates"

MIN_SEGMENT_WORDS = 30
MAX_SEGMENT_WORDS = 400
MAX_CONCURRENT_REQUESTS = 20

TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
