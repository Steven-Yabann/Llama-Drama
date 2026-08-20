from archive import fetch_manifest, load_quarter, sample_balanced, build_training_row, rank_to_percentile
from extractor import extract_features_from_transcript
from model import raw_score
from database import init_training_db, log_training_event


def run_backfill(sample_size: int = 250):
    init_training_db()
    manifest = fetch_manifest()
    sealed_files = [f for f in manifest["files"] if f["sealed"]]

    for file_entry in sealed_files:
        quarter = file_entry["quarter"]
        print(f"Loading {quarter} ({file_entry['events']} events)...")
        records = load_quarter(file_entry)
        records = rank_to_percentile(records)  # attaches true_percentile per event
        sampled = sample_balanced(records, n=sample_size)

        for rec in sampled:
            row = build_training_row(rec)
            if row is None:
                print(f"⏭ Skipping {rec.get('event_id')} — no disclosure content available.")
                continue
            try:
                features = extract_features_from_transcript(row["summary_text"])
                score = raw_score(features)
                # Note: predicted_percentile here should be computed against OTHER
                # sampled events' raw scores within this quarter -- for a first pass,
                # log raw_score and true_percentile, and derive predicted_percentile
                # in a separate evaluation pass once the full sample is in, so ranking
                # is computed against the complete sample rather than partial history.
                log_training_event(
                    event_id=row["event_id"],
                    quarter=quarter,
                    ticker=row["ticker"],
                    event_datetime=rec.get("event_datetime"),
                    raw_text=row["summary_text"],
                    features_dict=features.model_dump(),
                    raw_score_val=score,
                    predicted_percentile=None,  # filled in during evaluation pass
                    true_percentile=row["true_percentile"],
                    car1=row["car1"],
                    r_i=rec["event_returns"][row["ticker"]].get("r_i"),
                    r_m=rec["event_returns"][row["ticker"]].get("r_m"),
                    surprise=row["surprise"],
                    baseline_gemini=row["baseline_gemini"],
                    baseline_openai=row["baseline_openai"],
                )
            except Exception as e:
                print(f"Failed on {row['event_id']}: {e}")


if __name__ == "__main__":
    run_backfill(sample_size=250)