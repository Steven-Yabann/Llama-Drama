import os
import instructor
from schemas import EarningsFeatures
from groq import Groq
from dotenv import load_dotenv
import time
from groq import RateLimitError

load_dotenv()

# LLM API
client = instructor.from_groq(
    Groq(
        api_key = os.getenv('GROQ_API_KEY')
    ),
    mode = instructor.Mode.JSON
)



def extract_features_from_transcript(transcript_summary: str, retries: int = 3) -> EarningsFeatures:
    if not os.getenv('GROQ_API_KEY'):
        raise RuntimeError("Missing GROQ_LLM_KEY in your environment configuration.")

    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model='llama-3.1-8b-instant',
                response_model=EarningsFeatures,
                max_tokens=500,
                messages=[
                    {'role': 'system', 'content': (
                        "You are a precise quantitative data extraction engine. "
                        "Your sole task is to analyze corporate earnings transcript summaries "
                        "and map linguistic and numeric indicators into the exact schema requested. "
                        "Be objective and ignore market hype."
                    )},
                    {'role': 'user', 'content': transcript_summary}
                ],
                temperature=0.0
            )
        except RateLimitError as e:
            wait = 15 * (attempt + 1)   # simple backoff: 15s, 30s, 45s
            print(f"⏳ Rate limited, waiting {wait}s (attempt {attempt+1}/{retries})...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} rate-limit retries.")