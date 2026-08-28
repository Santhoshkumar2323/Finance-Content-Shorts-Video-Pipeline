import os
from dotenv import load_dotenv
load_dotenv()  

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")  
GROQ_MODEL = "openai/gpt-oss-120b"          
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt" 
POLLINATIONS_MODEL = "flux"  

TTS_MODEL = "kokoro-82m"                  
BEAT_DURATION_SECONDS = 4          
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

FINAL_VIDEO_WIDTH = 1080
FINAL_VIDEO_HEIGHT = 1920

RETRY_WAIT_MIN = 2      
RETRY_WAIT_MAX = 10     
RETRY_MAX_ATTEMPTS = 5

GROQ_DAILY_TOKEN_LIMIT = 100_000
GROQ_DAILY_TOKEN_WARN_THRESHOLD = 95_000  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
QUOTA_CONFIG_PATH = os.path.join(DATA_DIR, "quota_config.json")

LOGS_DIR = os.path.join(BASE_DIR, "logs")

for _dir in (DATA_DIR, CHECKPOINTS_DIR, OUTPUT_DIR, LOGS_DIR):
    os.makedirs(_dir, exist_ok=True)