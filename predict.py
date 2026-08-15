"""
Prediction pipeline for the Explaining Markets competition.

Pipeline:

    webhook event
        |
        v
    information_url
        |
        v
    DisclosureBundle
        |
        v
    normalize_disclosures()
        |
        v
    text representation
        |
        v
    LLM feature extraction
        |
        v
    raw quantitative score
        |
        v
    empirical percentile
        |
        v
    one prediction per focal asset

The Explaining Markets API specifies that information_url returns:

    {
        "schema_version": "...",
        "event_id": "...",
        "generated_at": "...",
        "items": [...]
    }

Each DisclosureItem contains either:
    - inline "content"
    OR
    - a by-reference "url"

Never assume a top-level "summary" field exists.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import httpx

from database import get_recent_raw_scores, log_prediction
from extractor import extract_features_from_transcript
from model import raw_score, raw_score_to_percentile


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INFORMATION_TIMEOUT_SECONDS = 20.0
REFERENCE_TIMEOUT_SECONDS = 20.0

MAX_REFERENCE_BYTES = 25 * 1024 * 1024  # 25 MB safety limit
MAX_TEXT_CHARS = 100_000               # Prevent accidentally huge LLM input


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _short(value: Any, max_chars: int = 500) -> str:
    """
    Convert a value to a compact string suitable for logs.

    We deliberately avoid logging entire disclosures because the disclosure
    may contain a large amount of event information.
    """
    text = str(value)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "...[truncated]"


def _content_to_text(content: Any) -> str:
    """
    Convert DisclosureItem.content into text.

    The API currently documents `facts` as an array of strings, but future
    disclosure kinds may use other JSON shapes. We therefore normalize
    structured content without assuming every kind is a string.
    """

    if content is None:
        return ""

    # facts -> ["fact 1", "fact 2", ...]
    if isinstance(content, list):
        parts = []

        for item in content:
            if item is None:
                continue

            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )

        return "\n".join(parts)

    # Plain text
    if isinstance(content, str):
        return content

    # Structured JSON object
    if isinstance(content, dict):
        return json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )

    # Numbers / booleans / other JSON primitives
    return str(content)


def _validate_sha256(raw_bytes: bytes, expected_sha256: str) -> None:
    """
    Verify the SHA-256 digest supplied by the API for a by-reference item.
    """

    if not expected_sha256:
        return

    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "SHA-256 mismatch for referenced disclosure: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )


# ---------------------------------------------------------------------------
# By-reference disclosure handling
# ---------------------------------------------------------------------------

def _fetch_referenced_content(
    client: httpx.Client,
    url: str,
    expected_sha256: str | None = None,
) -> str:
    """
    Fetch a DisclosureItem whose content is supplied through `url`.

    The API specifies that large disclosure items may be supplied this way
    instead of inline `content`.
    """

    if not url:
        raise ValueError("Referenced disclosure has an empty URL")

    response = client.get(
        url,
        timeout=REFERENCE_TIMEOUT_SECONDS,
        follow_redirects=True,
    )

    response.raise_for_status()

    raw_bytes = response.content

    if len(raw_bytes) > MAX_REFERENCE_BYTES:
        raise ValueError(
            "Referenced disclosure exceeds safety limit: "
            f"{len(raw_bytes):,} bytes > {MAX_REFERENCE_BYTES:,}"
        )

    # Verify integrity when the API supplied sha256.
    if expected_sha256:
        _validate_sha256(raw_bytes, expected_sha256)

    # First try JSON because disclosure content may itself be structured.
    content_type = (
        response.headers.get("content-type", "")
        .lower()
    )

    if "json" in content_type:
        try:
            payload = response.json()
            return _content_to_text(payload)
        except ValueError:
            # Fall through to UTF-8 decoding.
            pass

    # Otherwise treat it as textual content.
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            "Referenced disclosure is not valid UTF-8 text and does not "
            "appear to be JSON"
        )


# ---------------------------------------------------------------------------
# DisclosureBundle parsing
# ---------------------------------------------------------------------------

def extract_disclosure_text(
    information_url: str,
    expected_event_id: str | None = None,
) -> str:
    """
    Download and normalize the DisclosureBundle returned by information_url.

    Expected API structure:

        {
            "schema_version": "1.0",
            "event_id": "...",
            "generated_at": "...",
            "items": [
                {
                    "id": "...",
                    "kind": "...",
                    "source": "...",
                    "content": ...
                }
            ]
        }

    or:

        {
            ...
            "items": [
                {
                    "id": "...",
                    "kind": "...",
                    "source": "...",
                    "url": "...",
                    "bytes": ...,
                    "sha256": "..."
                }
            ]
        }
    """

    if not information_url:
        raise ValueError("Event is missing information_url")

    with httpx.Client(
        timeout=INFORMATION_TIMEOUT_SECONDS,
        follow_redirects=True,
    ) as client:

        response = client.get(information_url)
        response.raise_for_status()

        try:
            bundle = response.json()
        except ValueError as exc:
            raise ValueError(
                "information_url did not return valid JSON"
            ) from exc

        # ---------------------------------------------------------------
        # Top-level bundle validation
        # ---------------------------------------------------------------

        if not isinstance(bundle, dict):
            raise ValueError(
                "information_url returned "
                f"{type(bundle).__name__}, expected JSON object"
            )

        schema_version = bundle.get("schema_version")
        bundle_event_id = bundle.get("event_id")
        items = bundle.get("items")

        print(
            "[INFO] DisclosureBundle:"
            f" schema_version={schema_version!r},"
            f" event_id={bundle_event_id!r},"
            f" items={len(items) if isinstance(items, list) else 'INVALID'}"
        )

        # event_id consistency check
        if expected_event_id and bundle_event_id:
            if bundle_event_id != expected_event_id:
                raise ValueError(
                    "DisclosureBundle event_id mismatch: "
                    f"webhook={expected_event_id}, "
                    f"bundle={bundle_event_id}"
                )

        # API requires items.
        if not isinstance(items, list):
            raise ValueError(
                "information_url payload is missing a valid 'items' array"
            )

        if not items:
            raise ValueError(
                f"DisclosureBundle for event {expected_event_id} "
                "contains zero disclosure items"
            )

        # ---------------------------------------------------------------
        # Extract every disclosure item
        # ---------------------------------------------------------------

        text_parts: list[str] = []

        inline_count = 0
        reference_count = 0
        skipped_count = 0

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                print(
                    f"[WARN] Disclosure item {index} is not an object; "
                    "skipping."
                )
                skipped_count += 1
                continue

            item_id = item.get("id")
            kind = item.get("kind")
            source = item.get("source")

            content = item.get("content")
            url = item.get("url")

            has_content = content is not None
            has_url = bool(url)

            # API says an item is either inline OR by-reference.
            if has_content and has_url:
                raise ValueError(
                    f"Disclosure item {item_id!r} contains both "
                    "'content' and 'url'. API contract says these are "
                    "mutually exclusive."
                )

            # -----------------------------------------------------------
            # Inline content
            # -----------------------------------------------------------

            if has_content:
                inline_count += 1

                text = _content_to_text(content)

                if text.strip():
                    text_parts.append(
                        f"[SOURCE: {source or 'unknown'} | "
                        f"KIND: {kind or 'unknown'}]\n"
                        f"{text}"
                    )
                else:
                    print(
                        f"[WARN] Disclosure item {item_id!r} has empty "
                        "inline content."
                    )

                continue

            # -----------------------------------------------------------
            # By-reference content
            # -----------------------------------------------------------

            if has_url:
                reference_count += 1

                try:
                    referenced_text = _fetch_referenced_content(
                        client=client,
                        url=url,
                        expected_sha256=item.get("sha256"),
                    )

                    if referenced_text.strip():
                        text_parts.append(
                            f"[SOURCE: {source or 'unknown'} | "
                            f"KIND: {kind or 'unknown'}]\n"
                            f"{referenced_text}"
                        )
                    else:
                        print(
                            f"[WARN] Referenced disclosure item "
                            f"{item_id!r} returned empty content."
                        )

                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to fetch referenced disclosure "
                        f"item {item_id!r}: {exc}"
                    ) from exc

                continue

            # -----------------------------------------------------------
            # Neither content nor URL
            # -----------------------------------------------------------

            skipped_count += 1

            print(
                f"[WARN] Disclosure item {item_id!r} has neither "
                "'content' nor 'url'; skipping."
            )

        # ---------------------------------------------------------------
        # Final validation
        # ---------------------------------------------------------------

        if not text_parts:
            raise ValueError(
                "DisclosureBundle contained no usable textual content "
                f"(items={len(items)}, inline={inline_count}, "
                f"references={reference_count}, skipped={skipped_count})"
            )

        combined_text = "\n\n".join(text_parts).strip()

        if not combined_text:
            raise ValueError(
                "Disclosure normalization produced an empty string"
            )

        # Prevent accidentally feeding an enormous document to the LLM.
        if len(combined_text) > MAX_TEXT_CHARS:
            print(
                "[WARN] Normalized disclosure is "
                f"{len(combined_text):,} characters; "
                f"truncating to {MAX_TEXT_CHARS:,}."
            )

            combined_text = combined_text[:MAX_TEXT_CHARS]

        print(
            "[INFO] Disclosure normalized successfully:"
            f" inline={inline_count},"
            f" references={reference_count},"
            f" skipped={skipped_count},"
            f" text_chars={len(combined_text):,}"
        )

        return combined_text


# ---------------------------------------------------------------------------
# Event validation
# ---------------------------------------------------------------------------

def _validate_event(event: dict) -> None:
    """
    Validate the minimum WebhookPayload fields required by predict().
    """

    if not isinstance(event, dict):
        raise ValueError(
            f"Event must be a dict, got {type(event).__name__}"
        )

    required_fields = [
        "event_id",
        "event_type",
        "focal_assets",
        "information_url",
        "prediction_deadline",
    ]

    missing = [
        field
        for field in required_fields
        if not event.get(field)
    ]

    if missing:
        raise ValueError(
            "Webhook event is missing required fields: "
            + ", ".join(missing)
        )

    focal_assets = event["focal_assets"]

    if not isinstance(focal_assets, list) or not focal_assets:
        raise ValueError(
            "Webhook event must contain at least one focal asset"
        )

    for index, asset in enumerate(focal_assets):

        if not isinstance(asset, dict):
            raise ValueError(
                f"focal_assets[{index}] is not an object"
            )

        identifier = asset.get("identifier_value")

        if not identifier:
            raise ValueError(
                f"focal_assets[{index}] is missing identifier_value"
            )


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict(event: dict) -> list[dict]:
    """
    Generate one prediction for every focal asset in an Explaining Markets
    event.

    Important:

    The disclosure/LLM extraction happens ONCE per event.

    The current feature model does not contain asset-specific inputs, so
    every focal asset receives the same event-level score. We still return
    one prediction object per focal asset as required by the API.
    """

    started_at = time.time()

    # ------------------------------------------------------------------
    # 1. Validate webhook payload
    # ------------------------------------------------------------------

    _validate_event(event)

    event_id = event["event_id"]

    print(
        f"\n[event {event_id}] "
        f"Starting prediction pipeline"
    )

    print(
        f"[event {event_id}] "
        f"type={event.get('event_type')!r}, "
        f"assets={len(event['focal_assets'])}, "
        f"deadline={event.get('prediction_deadline')}"
    )

    # ------------------------------------------------------------------
    # 2. Fetch + normalize disclosure
    # ------------------------------------------------------------------

    transcript_text = extract_disclosure_text(
        information_url=event["information_url"],
        expected_event_id=event_id,
    )

    elapsed = time.time() - started_at

    print(
        f"[event {event_id}] "
        f"Disclosure fetched and normalized "
        f"in {elapsed:.2f}s"
    )

    # ------------------------------------------------------------------
    # 3. LLM feature extraction — ONCE per event
    # ------------------------------------------------------------------

    features = extract_features_from_transcript(
        transcript_text
    )

    elapsed = time.time() - started_at

    print(
        f"[event {event_id}] "
        f"LLM feature extraction completed "
        f"in {elapsed:.2f}s"
    )

    print(
        f"[event {event_id}] "
        f"Extracted features: {features.model_dump()}"
    )

    # ------------------------------------------------------------------
    # 4. Convert features → quantitative raw score
    # ------------------------------------------------------------------

    score = raw_score(features)

    print(
        f"[event {event_id}] "
        f"Raw model score = {score:+.4f}"
    )

    # ------------------------------------------------------------------
    # 5. Obtain historical calibration distribution
    # ------------------------------------------------------------------

    historical_scores = get_recent_raw_scores()

    print(
        f"[event {event_id}] "
        f"Historical score count = {len(historical_scores)}"
    )

    # ------------------------------------------------------------------
    # 6. Raw score → empirical percentile
    # ------------------------------------------------------------------

    predicted_percentile = raw_score_to_percentile(
        new_score=score,
        historical_scores=historical_scores,
    )

    # ------------------------------------------------------------------
    # 7. Validate percentile
    # ------------------------------------------------------------------

    if not isinstance(predicted_percentile, (int, float)):
        raise ValueError(
            f"Model returned non-numeric percentile: "
            f"{predicted_percentile!r}"
        )

    predicted_percentile = float(predicted_percentile)

    if not 0.0 <= predicted_percentile <= 1.0:
        raise ValueError(
            f"Predicted percentile outside [0,1]: "
            f"{predicted_percentile}"
        )

    predicted_percentile = round(
        predicted_percentile,
        4,
    )

    print(
        f"[event {event_id}] "
        f"Predicted percentile = {predicted_percentile:.4f}"
    )

    # ------------------------------------------------------------------
    # 8. Create predictions for every focal asset
    # ------------------------------------------------------------------

    predictions: list[dict] = []

    timestamp = int(time.time())

    for asset in event["focal_assets"]:

        ticker = asset["identifier_value"]

        prediction = {
            "identifier_value": ticker,
            "predicted_percentile": predicted_percentile,
        }

        predictions.append(prediction)

        # --------------------------------------------------------------
        # 9. Persist local/Modal-volume prediction record
        # --------------------------------------------------------------

        log_prediction(
            event_id=event_id,
            timestamp=timestamp,
            ticker=ticker,
            raw_text=transcript_text,
            features_dict=features.model_dump(),
            raw_score_val=score,
            prediction=predicted_percentile,
        )

        print(
            f"[event {event_id}] "
            f"{ticker}: percentile={predicted_percentile:.4f}"
        )

    # ------------------------------------------------------------------
    # 10. Final structural validation
    # ------------------------------------------------------------------

    if not predictions:
        raise ValueError(
            f"No predictions generated for event {event_id}"
        )

    for prediction in predictions:

        if not prediction.get("identifier_value"):
            raise ValueError(
                f"Prediction missing identifier_value: "
                f"{prediction}"
            )

        percentile = prediction.get(
            "predicted_percentile"
        )

        if not isinstance(percentile, (int, float)):
            raise ValueError(
                f"Prediction percentile is not numeric: "
                f"{prediction}"
            )

        if not 0.0 <= float(percentile) <= 1.0:
            raise ValueError(
                f"Prediction percentile outside [0,1]: "
                f"{prediction}"
            )

    elapsed = time.time() - started_at

    print(
        f"[event {event_id}] "
        f"Prediction pipeline completed in {elapsed:.2f}s"
    )

    print(
        f"[event {event_id}] "
        f"Predictions: {predictions}\n"
    )

    return predictions