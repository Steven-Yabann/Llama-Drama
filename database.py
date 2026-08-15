# database.py
import sqlite3
import json
import os

# On Modal, this resolves to the mounted Volume path (see modal_app.py's
# volumes={"/data": volume}). Locally (e.g. running tests), it falls back
# to a file in the current directory.
DB_FILE = os.environ.get("LLAMADRAMA_DB_PATH", "/data/llamadrama.db")


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS earnings_events (
            event_id TEXT PRIMARY KEY,
            timestamp INTEGER,
            ticker TEXT,
            raw_transcript TEXT,
            extracted_features TEXT,
            raw_score REAL,
            predicted_percentile REAL,
            actual_percentile REAL DEFAULT NULL,
            error REAL DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"🗄️ Database initialized at {DB_FILE}")


def log_prediction(event_id, timestamp, ticker, raw_text, features_dict, raw_score_val, prediction):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO earnings_events
            (event_id, timestamp, ticker, raw_transcript, extracted_features, raw_score, predicted_percentile)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, timestamp, ticker, raw_text,
            json.dumps(features_dict), raw_score_val, prediction
        ))
        conn.commit()
        print(f"💾 [LIVE] Saved Event {event_id} to {DB_FILE}.")
    except Exception as e:
        print(f"❌ [LIVE] Database write failure: {e}")
    finally:
        conn.close()


def get_recent_raw_scores(limit: int = 200) -> list[float]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT raw_score FROM earnings_events
        WHERE raw_score IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


if __name__ == "__main__":
    init_db()