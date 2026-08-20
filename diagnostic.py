"""
diagnostics.py — Full prediction diagnostic suite.

Run against either:
  - training_events   (historical backfill: true_percentile, car1, surprise all populated today)
  - earnings_events    (live Q3 predictions: actual_percentile/car1/surprise are currently NULL
                         until you backfill them from the platform's realized-outcome API)

Usage:
    python3 diagnostics.py --table training_events --db training.db
    python3 diagnostics.py --table earnings_events --db llamadrama.db

Requires: pandas, scipy, numpy (pip install pandas scipy numpy --break-system-packages)
"""

import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------
# Table-specific column mapping. Both tables conceptually have the same
# four things we need -- predicted percentile, realized percentile,
# realized abnormal return, and earnings surprise -- but the column
# names differ between the live and historical tables.
# ----------------------------------------------------------------------
TABLE_CONFIG = {
    "training_events": {
        "predicted_col": "predicted_percentile",
        "actual_pct_col": "true_percentile",
        "car1_col": "car1",
        "surprise_col": "surprise",
    },
    "earnings_events": {
        "predicted_col": "predicted_percentile",
        "actual_pct_col": "actual_percentile",
        "car1_col": None,   # not present in this table's schema at all
        "surprise_col": None,
    },
}


def load_data(db_path: str, table: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()
    return df


def run_diagnostics(df: pd.DataFrame, config: dict, table_name: str):
    predicted_col = config["predicted_col"]
    actual_col = config["actual_pct_col"]
    car1_col = config["car1_col"]
    surprise_col = config["surprise_col"]

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC REPORT — table: {table_name}")
    print(f"{'='*70}")

    print(f"\nTotal rows in table: {len(df)}")

    if predicted_col not in df.columns or actual_col not in df.columns:
        print(f"❌ Required columns missing. Have: {list(df.columns)}")
        sys.exit(1)

    # --- Filter to rows with BOTH prediction and realized outcome ---
    scored = df.dropna(subset=[predicted_col, actual_col]).copy()
    n_total = len(df)
    n_scored = len(scored)
    n_missing = n_total - n_scored

    print(f"Rows with both prediction and realized outcome: {n_scored} / {n_total} ({n_missing} missing outcome — likely unresolved/awaiting-outcome events)")

    if n_scored < 5:
        print(f"\n⚠️  Only {n_scored} scored rows available — most of these diagnostics will be unstable or meaningless at this sample size.")
        print("   (This is expected if you're running against earnings_events before backfilling actual outcomes.)")
        if n_scored == 0:
            return

    pred = scored[predicted_col].astype(float)
    actual = scored[actual_col].astype(float)
    residuals = actual - pred

    # ------------------------------------------------------------------
    # 1. Prediction vs actual distribution
    # ------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("1. PREDICTION vs ACTUAL DISTRIBUTION")
    print(f"{'-'*70}")
    print(f"{'':20}{'Predicted':>15}{'Actual':>15}")
    print(f"{'Mean':20}{pred.mean():>15.4f}{actual.mean():>15.4f}")
    print(f"{'Std Dev':20}{pred.std():>15.4f}{actual.std():>15.4f}")
    print(f"{'Min':20}{pred.min():>15.4f}{actual.min():>15.4f}")
    print(f"{'25th pct':20}{pred.quantile(0.25):>15.4f}{actual.quantile(0.25):>15.4f}")
    print(f"{'Median':20}{pred.median():>15.4f}{actual.median():>15.4f}")
    print(f"{'75th pct':20}{pred.quantile(0.75):>15.4f}{actual.quantile(0.75):>15.4f}")
    print(f"{'Max':20}{pred.max():>15.4f}{actual.max():>15.4f}")

    # A well-calibrated predictor should have predicted mean/std close to actual's.
    # Since 'actual' is itself a within-quarter rank, its true distribution should be
    # roughly uniform on [0,1] with mean ~0.5, std ~0.289 (uniform std = 1/sqrt(12)).
    uniform_std = 1 / np.sqrt(12)
    print(f"\n(Reference: a uniform [0,1] distribution has mean=0.500, std={uniform_std:.4f})")
    if abs(pred.std() - uniform_std) > 0.08:
        spread_note = "predictions are notably " + ("more concentrated (underconfident spread)" if pred.std() < uniform_std else "more dispersed than expected")
        print(f"⚠️  {spread_note} — predicted std={pred.std():.4f} vs expected ~{uniform_std:.4f}")

    # ------------------------------------------------------------------
    # 2. Residuals
    # ------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("2. RESIDUALS (actual - predicted)")
    print(f"{'-'*70}")
    print(f"Mean residual (bias):     {residuals.mean():+.4f}   (0 = unbiased; positive = you underpredict on average)")
    print(f"Std of residuals:         {residuals.std():.4f}")
    print(f"MAE:                      {residuals.abs().mean():.4f}")
    print(f"RMSE:                     {np.sqrt((residuals**2).mean()):.4f}")

    skew = stats.skew(residuals)
    kurt = stats.kurtosis(residuals)  # excess kurtosis; 0 = normal
    print(f"Residual skewness:        {skew:+.4f}  (0 = symmetric)")
    print(f"Residual excess kurtosis: {kurt:+.4f}  (0 = normal-tailed; positive = fat-tailed / more extreme errors than normal predicts)")

    # ------------------------------------------------------------------
    # 3. Spearman / Pearson correlation
    # ------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("3. CORRELATION: predicted vs actual")
    print(f"{'-'*70}")
    if n_scored >= 3:
        spearman_r, spearman_p = stats.spearmanr(pred, actual)
        pearson_r, pearson_p = stats.pearsonr(pred, actual)
        print(f"Spearman ρ = {spearman_r:+.4f}  (p = {spearman_p:.4f})")
        print(f"Pearson  r = {pearson_r:+.4f}  (p = {pearson_p:.4f})")
        sig_threshold = 1.96 / np.sqrt(n_scored - 1)
        print(f"(Rough significance threshold at n={n_scored}: |ρ| > {sig_threshold:.3f})")
    else:
        print("Not enough rows for a meaningful correlation.")

    # ------------------------------------------------------------------
    # 4. Tail-event frequency
    # ------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("4. TAIL-EVENT FREQUENCY")
    print(f"{'-'*70}")
    # "Tail" here = actual percentile in the extreme deciles (<0.1 or >0.9)
    tail_actual = scored[(actual < 0.10) | (actual > 0.90)]
    tail_pred = scored[(pred < 0.10) | (pred > 0.90)]
    print(f"Events with ACTUAL in tail (<0.10 or >0.90):    {len(tail_actual)} / {n_scored} ({100*len(tail_actual)/n_scored:.1f}%)")
    print(f"Events with PREDICTED in tail (<0.10 or >0.90): {len(tail_pred)} / {n_scored} ({100*len(tail_pred)/n_scored:.1f}%)")
    print("(A uniform [0,1] target should have ~20% of events in the tails by construction.)")

    if len(tail_actual) > 0:
        # Of the events that were ACTUALLY tail events, how many did we predict were in the tail too?
        caught = tail_actual[(tail_actual[predicted_col] < 0.10) | (tail_actual[predicted_col] > 0.90)]
        print(f"Of actual tail events, predicted also in tail: {len(caught)} / {len(tail_actual)} ({100*len(caught)/len(tail_actual):.1f}%) — tail-event 'recall'")

    # ------------------------------------------------------------------
    # 5. Calibration by prediction decile
    # ------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("5. CALIBRATION BY PREDICTION DECILE")
    print(f"{'-'*70}")
    if n_scored >= 10:
        scored["decile"] = pd.qcut(pred, q=min(10, scored[predicted_col].nunique()), duplicates="drop")
        calib = scored.groupby("decile", observed=True).agg(
            n=(actual_col, "size"),
            mean_predicted=(predicted_col, "mean"),
            mean_actual=(actual_col, "mean"),
        )
        calib["gap"] = calib["mean_actual"] - calib["mean_predicted"]
        print(calib.to_string(float_format=lambda x: f"{x:.4f}"))
        print("\n(Well-calibrated: mean_actual tracks mean_predicted closely within each decile — 'gap' near 0.)")
    else:
        print(f"Only {n_scored} scored rows — too few for decile-level calibration (need >=10, ideally 50+).")

    # ------------------------------------------------------------------
    # 6. Large errors vs large SURPRISE / CAR1
    # ------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("6. LARGE ERRORS vs LARGE SURPRISE / CAR1")
    print(f"{'-'*70}")

    abs_resid = residuals.abs()
    error_threshold = abs_resid.quantile(0.75)  # top-quartile errors = "large"
    large_error_mask = abs_resid >= error_threshold

    if car1_col and car1_col in scored.columns and scored[car1_col].notna().any():
        car1_vals = scored[car1_col].astype(float)
        abs_car1 = car1_vals.abs()
        corr_err_car1, p_err_car1 = stats.spearmanr(abs_resid, abs_car1, nan_policy="omit")
        print(f"Spearman(|residual|, |car1|) = {corr_err_car1:+.4f}  (p = {p_err_car1:.4f})")
        print(f"  -> {'Large errors ARE associated with large realized moves (harder events are where you miss).' if corr_err_car1 > 0.15 else 'No strong association detected between error size and realized move size.'}")

        mean_car1_large_err = abs_car1[large_error_mask].mean()
        mean_car1_small_err = abs_car1[~large_error_mask].mean()
        print(f"  Mean |car1| for top-quartile-error events: {mean_car1_large_err:.4f}")
        print(f"  Mean |car1| for rest:                       {mean_car1_small_err:.4f}")
    else:
        print("car1 column not available/populated in this table — skipping.")

    if surprise_col and surprise_col in scored.columns and scored[surprise_col].notna().any():
        surprise_vals = scored[surprise_col].astype(float)
        abs_surprise = surprise_vals.abs()
        corr_err_surprise, p_err_surprise = stats.spearmanr(abs_resid, abs_surprise, nan_policy="omit")
        print(f"\nSpearman(|residual|, |surprise|) = {corr_err_surprise:+.4f}  (p = {p_err_surprise:.4f})")
        print(f"  -> {'Large errors ARE associated with large earnings surprises.' if corr_err_surprise > 0.15 else 'No strong association detected between error size and surprise magnitude.'}")

        mean_surprise_large_err = abs_surprise[large_error_mask].mean()
        mean_surprise_small_err = abs_surprise[~large_error_mask].mean()
        print(f"  Mean |surprise| for top-quartile-error events: {mean_surprise_large_err:.4f}")
        print(f"  Mean |surprise| for rest:                        {mean_surprise_small_err:.4f}")
    else:
        print("surprise column not available/populated in this table — skipping.")

    print(f"\n{'='*70}")
    print("END OF REPORT")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Run prediction diagnostics against a LlamaDrama SQLite table.")
    parser.add_argument("--db", default="llamadrama.db", help="Path to the local SQLite DB file")
    parser.add_argument("--table", default="training_events", choices=["training_events", "earnings_events"],
                         help="Which table to diagnose (default: training_events, since it has populated outcomes today)")
    args = parser.parse_args()

    config = TABLE_CONFIG[args.table]
    df = load_data(args.db, args.table)
    run_diagnostics(df, config, args.table)


if __name__ == "__main__":
    main()