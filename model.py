# model.py
from schemas import EarningsFeatures

def raw_score(features: EarningsFeatures) -> float:
    """
    Computes an unbounded, per-event strength score.
    """
    score = 0.50

    guidance_map = {
        'raised': 0.15,
        'maintained': 0.00,
        'lowered': -0.20,
        'not_given': -0.05
    }
    score += guidance_map.get(features.guidance_direction, 0.0)

    surprise_map = {
        "significant_beat": 0.12,
        "slight_beat": 0.05,
        "inline": 0.00,
        "miss": -0.15,
        'not_given': 0.00
    }
    score += surprise_map.get(features.revenue_surprise_tone, 0.0)
    score += surprise_map.get(features.eps_surprise_tone, 0.0)

    if features.guidance_magnitude_basis_points:
        magnitude_adj = (features.guidance_magnitude_basis_points / 100) * 0.02
        score += max(min(magnitude_adj, 0.08), -0.08)

    confidence_adj = (features.management_confidence_score - 0.5) * 0.05
    headwinds_adj = features.macro_headwinds_prominence * -0.05
    score += confidence_adj
    score += headwinds_adj

    if getattr(features, "one_off_items_present", False):
        score -= 0.02

    return round(score, 4)


def raw_score_to_percentile(new_score: float, historical_scores: list[float], min_history: int = 20) -> float:
    """
    Converts a raw score into an empirical percentile (0.0-1.0) relative
    to the distribution of past raw scores.
    """
    if len(historical_scores) < min_history:
        return round(max(min(new_score, 1.0), 0.0), 4)

    rank = sum(1 for s in historical_scores if s < new_score)
    percentile = rank / len(historical_scores)
    return round(percentile, 4)