from typing import Literal, Optional

from pydantic import BaseModel, Field


class EarningsFeatures(BaseModel):
    """
    Structured feature matrix row extracted from unstructured earnings text.
    """

    # Guidance Signals
    guidance_direction: Literal[
        "raised",
        "maintained",
        "lowered",
        "not_given",
    ] = Field(
        description=(
            "The forward-looking direction of management's future "
            "performance guidance."
        )
    )

    guidance_magnitude_basis_points: Optional[int] = Field(
        description=(
            "Stated basis-point shift in guidance or revenue margins "
            "if explicitly given. Use null if no explicit basis-point "
            "change is provided."
        )
    )

    # Financial Surprise Tone
    revenue_surprise_tone: Literal[
        "significant_beat",
        "slight_beat",
        "inline",
        "miss",
        "not_given",
    ] = Field(
        description=(
            "How reported revenue compares with consensus or analyst "
            "expectations. Use 'not_given' when the disclosure does not "
            "provide enough information to determine this."
        )
    )

    eps_surprise_tone: Literal[
        "significant_beat",
        "slight_beat",
        "inline",
        "miss",
        "not_given",
    ] = Field(
        description=(
            "How reported EPS compares with expectations. Use "
            "'not_given' when the disclosure does not provide enough "
            "information to determine this."
        )
    )

    # Linguistic & Macro Risk Factors
    management_confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Score from 0.0 to 1.0 measuring management confidence. "
            "0.0 means highly uncertain, heavily hedged or cautious "
            "language. 1.0 means extremely confident language."
        ),
    )

    macro_headwinds_prominence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Score from 0.0 to 1.0 measuring how prominently systemic "
            "macro issues such as inflation, interest rates, demand "
            "conditions, or supply-chain problems are emphasized."
        ),
    )

    one_off_items_present: bool = Field(
        description=(
            "True if unusual non-recurring charges, legal settlements, "
            "write-downs, restructuring items, or similar one-off items "
            "are mentioned as materially affecting reported results."
        )
    )