# database.py
import sqlite3
import json
import os

# LIVE DB: synced with the Modal Volume, holds real competition predictions.
# Never write training/backfill data here.
LIVE_DB_FILE = os.environ.get("LLAMADRAMA_LIVE_DB_PATH", "/data/llamadrama.db")

# TRAINING DB: local-only, holds historical backfill/research data.
# Never synced to or from Modal -- regenerate via backfill.py if lost.
TRAINING_DB_FILE = os.environ.get("LLAMADRAMA_TRAINING_DB_PATH", "training.db")


# ------------------------------------------------------------------
# LIVE DB functions
# ------------------------------------------------------------------

def init_live_db():
    conn = sqlite3.connect(LIVE_DB_FILE)
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
    print(f"🗄️ Live DB initialized at {LIVE_DB_FILE}")


def log_prediction(event_id, timestamp, ticker, raw_text, features_dict, raw_score_val, prediction):
    conn = sqlite3.connect(LIVE_DB_FILE)
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
        print(f"💾 [LIVE] Saved Event {event_id} to {LIVE_DB_FILE}.")
    except Exception as e:
        print(f"❌ [LIVE] Database write failure: {e}")
    finally:
        conn.close()


def get_recent_raw_scores(limit: int = 200) -> list[float]:
    conn = sqlite3.connect(LIVE_DB_FILE)
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


# ------------------------------------------------------------------
# TRAINING DB functions
# ------------------------------------------------------------------

def init_training_db():
    conn = sqlite3.connect(TRAINING_DB_FILE)
    cursor = conn.cursor()
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
    print(f"🗄️ Training DB initialized at {TRAINING_DB_FILE}")


def log_training_event(event_id, quarter, ticker, event_datetime, raw_text, features_dict,
                        raw_score_val, predicted_percentile, true_percentile,
                        car1, r_i, r_m, surprise, baseline_gemini, baseline_openai):
    conn = sqlite3.connect(TRAINING_DB_FILE)
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


def update_predicted_percentile(event_id: str, predicted_percentile: float):
    conn = sqlite3.connect(TRAINING_DB_FILE)
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


def get_training_events(quarter: str = None) -> list[dict]:
    conn = sqlite3.connect(TRAINING_DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if quarter:
        cursor.execute("SELECT * FROM training_events WHERE quarter = ?", (quarter,))
    else:
        cursor.execute("SELECT * FROM training_events")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


if __name__ == "__main__":
    # Running this file directly initializes BOTH locally, for convenience
    # during local dev/testing. In production on Modal, only init_live_db()
    # is ever called (via modal_app.py's init_remote_db function).
    init_live_db()
    init_training_db()