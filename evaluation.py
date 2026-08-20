from database import get_training_events, update_predicted_percentile
from model import raw_score_to_percentile


def compute_predicted_percentiles(quarter: str):
    events = get_training_events(quarter=quarter) # retrieve the quarter values from the DB
    scores = [e["raw_score"] for e in events] # Retrieve the raw score column from the table

    # Loop through the events
    for e in events:
        # Rank against the FULL sampled distribution, not a partial one
        others = [s for s in scores if s != e["raw_score"]] + [e["raw_score"]] # Add score
        pct = raw_score_to_percentile(e["raw_score"], others, min_history=0)
        update_predicted_percentile(event_id=e["event_id"], predicted_percentile=pct)

    print(f"Updated predicted_percentile for {len(events)} events in {quarter}.")


if __name__ == "__main__":
    compute_predicted_percentiles(quarter="2025Q4")