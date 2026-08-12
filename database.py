# database.py
import os
import sqlite3
import json

# Route to Modal volume directory /data if present, otherwise local file
DATA_DIR = "/data" if os.path.exists("/data") else "."
DB_FILE = os.path.join(DATA_DIR, "llamadrama.db")


def init_db():
    """Initializes both the live-prediction table and the historical training table."""
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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_events (
            event_id TEXT PRIMARY KEY,
            quarter TEXT,
            ticker TEXT,
            event_datetime TEXT,
            raw_transcript TEXT,
            extracted_features TEXT,
            raw_score REAL,
            predicted_percentile REAL,
            true_percentile REAL,
            car1 REAL,
            r_i REAL,
            r_m REAL,
            surprise REAL,
            baseline_gemini REAL,
            baseline_openai REAL
        )
    """)

    conn.commit()
    conn.close()
    print(f"🗄️ Database initialized at {DB_FILE}")


def log_prediction(event_id, timestamp, ticker, raw_text, features_dict, raw_score_val, prediction):
    """Logs a LIVE prediction submitted during the competition."""
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
    """Fetch recent LIVE raw scores, used for empirical percentile conversion during the competition."""
    if not os.path.exists(DB_FILE):
        return []

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


def log_training_event(event_id, quarter, ticker, event_datetime, raw_text, features_dict,
                        raw_score_val, predicted_percentile, true_percentile,
                        car1, r_i, r_m, surprise, baseline_gemini, baseline_openai):
    """Logs a HISTORICAL backfill event, including ground-truth outcome data."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO training_events
            (event_id, quarter, ticker, event_datetime, raw_transcript, extracted_features,
             raw_score, predicted_percentile, true_percentile, car1, r_i, r_m,
             surprise, baseline_gemini, baseline_openai)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, quarter, ticker, event_datetime, raw_text, json.dumps(features_dict),
            raw_score_val, predicted_percentile, true_percentile, car1, r_i, r_m,
            surprise, baseline_gemini, baseline_openai
        ))
        conn.commit()
        print(f"💾 [TRAIN] Saved Event {event_id} ({quarter}).")
    except Exception as e:
        print(f"❌ [TRAIN] Database write failure: {e}")
    finally:
        conn.close()

def get_training_events(quarter: str = None) -> list[dict]:
    """Fetch training events, optionally filtered by quarter, for evaluation/ablation work."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if quarter:
        cursor.execute("SELECT * FROM training_events WHERE quarter = ?", (quarter,))
    else:
        cursor.execute("SELECT * FROM training_events")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def update_predicted_percentile(event_id: str, predicted_percentile: float):
    """
    Updates ONLY the predicted_percentile column for a training event,
    without touching any other stored field (features, car1, etc.).
    Used in the evaluation pass, after raw_scores for a full quarter
    are known and can be ranked against each other.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE training_events
            SET predicted_percentile = ?
            WHERE event_id = ?
        """, (predicted_percentile, event_id))
        conn.commit()
    except Exception as e:
        print(f"❌ [TRAIN] Failed to update predicted_percentile for {event_id}: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()