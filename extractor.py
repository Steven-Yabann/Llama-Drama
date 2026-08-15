import os
import time

import instructor
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from schemas import EarningsFeatures


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY in environment configuration."
    )

client = instructor.from_groq(
    Groq(api_key=GROQ_API_KEY),
    mode=instructor.Mode.JSON,
)


def extract_features_from_transcript(
    transcript_summary: str,
    retries: int = 2,
) -> EarningsFeatures:
    """
    Convert normalized earnings disclosure text into the structured
    EarningsFeatures schema.

    The competition provides a limited prediction window, so retries
    deliberately use short backoffs.
    """

    if not transcript_summary or not transcript_summary.strip():
        raise ValueError(
            "Cannot extract features from empty disclosure text."
        )

    for attempt in range(retries):
        try:
            result = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                response_model=EarningsFeatures,
                max_tokens=500,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise quantitative data extraction "
                            "engine. Your sole task is to analyze corporate "
                            "earnings disclosure information and map "
                            "linguistic and numeric indicators into the exact "
                            "schema requested. Be objective and ignore "
                            "market hype. Do not invent information that "
                            "is not present in the supplied disclosure."
                        ),
                    },
                    {
                        "role": "user",
                        "content": transcript_summary,
                    },
                ],
                temperature=0.0,
            )

            return result

        except RateLimitError as exc:

            if attempt == retries - 1:
                raise RuntimeError(
                    f"Groq rate limit persisted after "
                    f"{retries} attempts."
                ) from exc

            wait = 10 * (attempt + 1)

            print(
                f"⏳ Groq rate limited. "
                f"Waiting {wait}s "
                f"(attempt {attempt + 1}/{retries})..."
            )

            time.sleep(wait)

    raise RuntimeError(
        f"Feature extraction failed after {retries} attempts."
    )