from pydantic import BaseModel, Field
from typing import Literal, Optional

class EarningsFeatures(BaseModel):
    """
    Structured feature matrix row extracted from unstructured earnings text.
    """
    # Guidance Signals
    guidance_direction: Literal["raised", "maintained", "lowered", "not_given"] = Field(
        description="The forward-looking direction of management's future performance guidance."
    )
    guidance_magnitude_basis_points: Optional[int] = Field(
        default=0,
        description="Stated basis point shift in guidance or revenue margins if explicitly given, else 0."
    )
    
    # Financial Surprise Tone
    revenue_surprise_tone: Literal["significant_beat", "slight_beat", "inline", "miss", "not_given"] = Field(
    description="How management describes reported revenue relative to consensus/analyst expectations. Use 'not_given' if the transcript does not provide enough information to compare against expectations."
    )
    eps_surprise_tone: Literal["significant_beat", "slight_beat", "inline", "miss", "not_given"] = Field(
        description="How management describes reported Earnings Per Share (EPS) relative to expectations. Use 'not_given' if the transcript does not provide enough information to compare against expectations."
    )
    
    # Linguistic & Macro Risk Factors
    management_confidence_score: float = Field(
        description="Score from 0.0 (highly uncertain, heavy hedging/cautious language) to 1.0 (extremely confident)."
    )
    macro_headwinds_prominence: float = Field(
        description="Density score from 0.0 to 1.0 evaluating how heavily systemic macro issues (inflation, supply chain) are stressed."
    )
    one_off_items_present: bool = Field(
        description="True if unusual, non-recurring charges, legal settlements, or write-downs are mentioned as skewing numbers."
    )