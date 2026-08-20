import requests, gzip, json, os
from dotenv import load_dotenv
import random

load_dotenv()

HEADERS = {'X-API-Key' : os.getenv('EM_API_KEY')}

def fetch_manifest():
    r = requests.get("https://api.explainingmarkets.ai/v1/archive", headers=HEADERS)
    r.raise_for_status()
    return r.json()

def load_quarter(file_entry):
    r = requests.get(file_entry['url'], stream=True)
    r.raise_for_status()

    records = []
    with gzip.GzipFile(fileobj=r.raw) as gz:
        for line in gz:
            rec = json.loads(line)
            if rec.get("status") != "scored" or rec.get("return_status") != "ok":
                continue  # skip unscored/failed-return events
            records.append(rec)
            
    return records

def build_training_row(rec):
    items = rec.get("disclosure", {}).get("items", [])
    if not items:
        return None

    ticker = rec["focal_assets"][0]["identifier_value"]
    car1 = rec["event_returns"][ticker]["car1"]
    facts = items[0]["content"]
    summary_text = "\n".join(facts)
    return {
        "event_id": rec["event_id"],
        "ticker": ticker,
        "car1": car1,
        "true_percentile": rec.get("true_percentile"),   # <-- carried over from rank_to_percentile
        "surprise": rec.get("metrics", {}).get("earnings_surprise", {}).get("surprise"),
        "summary_text": summary_text,
        "baseline_gemini": rec.get("baseline_predictions", {}).get("gemini/ea-explain-contemp-summary", {}).get(ticker),
        "baseline_openai": rec.get("baseline_predictions", {}).get("openai/ea-explain-contemp-summary", {}).get(ticker),
    }


def rank_to_percentile(records_with_car1):
    """Convert car1 values within a quarter into true percentile labels."""
    sorted_vals = sorted(
        r["event_returns"][r["focal_assets"][0]["identifier_value"]]["car1"]
        for r in records_with_car1
    )
    n = len(sorted_vals)
    for r in records_with_car1:
        ticker = r["focal_assets"][0]["identifier_value"]
        this_car1 = r["event_returns"][ticker]["car1"]
        rank = sum(1 for v in sorted_vals if v < this_car1)
        r["true_percentile"] = rank / n
    return records_with_car1

def sample_balanced(records, n=250, seed=42):
    '''
    - random sampling increases chances of getting dominant tickers
    - Add stratification
    '''
    random.seed(seed)

    sorted_by_car1 = sorted(records, key=lambda r: r['event_returns'][r['focal_assets'][0]['identifier_value']]['car1'])
    n_buckets = 5
    bucket_size = len(sorted_by_car1) // n_buckets
    sample = []
    per_bucket = n // n_buckets

    for i in range(n_buckets):
        bucket = sorted_by_car1[i * bucket_size : (i + 1) * bucket_size]
        sample.extend(random.sample(bucket, min(per_bucket, len(bucket))))

    return sample