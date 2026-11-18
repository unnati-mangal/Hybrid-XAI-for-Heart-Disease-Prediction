"""
Prediction service — orchestrates inference and explanation.

If the ML model is not loaded (pipeline.joblib missing), returns a stub
response with status='ml_not_ready' so the frontend shows a clear message
rather than a 500 error.
"""

import logging
from typing import Optional

import numpy as np

from app.schemas.prediction_schema import PredictionInput, PredictionResponse

logger = logging.getLogger(__name__)


# ── Clinically-grounded risk calibration ──────────────────────────────────────
#
# The raw Logistic Regression output extrapolates aggressively beyond its
# training distribution (Framingham mean age ~50, few records above 70).
# Stacking 7+ risk factors simultaneously pushes log-odds so high that
# sigmoid saturates near 1.0, producing unrealistic probabilities (e.g. 80%).
#
# The Framingham Risk Score literature caps 10-year CHD risk at ~30-40% even
# for the highest-risk profiles. We apply soft calibration to bring the model
# output into a clinically plausible range without flattening low-risk scores.
#
# Method: isotonic-style sigmoid compression
#   calibrated = cap / (1 + exp(-k * (raw - midpoint)))
# where cap=0.40, k=8, midpoint=0.50 — this:
#   - Leaves scores below ~20% almost unchanged
#   - Compresses scores above 40% toward the 40% ceiling
#   - Is smooth and monotonic (higher raw → higher calibrated)

_TRIGGER   = 0.35   # below this the score is returned unchanged
_CAP       = 0.40   # hard ceiling — max clinically reported 10-yr CHD risk (FRS)


def _calibrate_risk(raw: float) -> float:
    """
    Soft-cap the raw model probability into a clinically plausible range.

    Scores at or below _TRIGGER (0.35) are returned unchanged.
    Scores above _TRIGGER are compressed toward _CAP (0.40) using a
    square-root curve — so the output smoothly approaches 0.40 as the
    raw score climbs toward 1.0.

    Examples:
        0.10 → 0.10  (unchanged — low risk, not affected)
        0.25 → 0.25  (unchanged)
        0.35 → 0.35  (boundary, unchanged)
        0.50 → ~0.37 (moderate compression)
        0.65 → ~0.38 (stronger compression)
        0.80 → ~0.39 (near cap)
        0.95 → ~0.40 (hard ceiling)
    """
    if raw <= _TRIGGER:
        return raw
    # t goes 0→1 as raw goes TRIGGER→1.0
    t = (raw - _TRIGGER) / (1.0 - _TRIGGER)
    # sqrt curve: fast rise early, flattens near cap
    compressed = _TRIGGER + (_CAP - _TRIGGER) * np.sqrt(t)
    return float(min(compressed, _CAP))


def run_prediction(
    inp: PredictionInput,
    user_id: Optional[str] = None,
    assessment_id: Optional[int] = None,
) -> PredictionResponse:
    """
    Run prediction pipeline for a validated input.

    Tries to import and use model_loader + preprocessing + explainers.
    On any failure (model not ready, import error), returns a stub response.
    """
    try:
        from app.ml.model_loader import (
            ModelNotReadyError,
            get_pipeline,
            get_threshold,
            get_positive_class_index,
            get_model_version,
        )
        from app.ml.preprocessing import input_to_dataframe
        from app.ml.shap_explainer import compute_shap
        from app.ml.lime_explainer import compute_lime
    except ImportError as exc:
        logger.warning("ML modules not importable: %s", exc)
        return PredictionResponse(
            status="ml_not_ready",
            message="ML dependencies are not installed. Install requirements.txt and restart.",
            assessment_id=assessment_id,
        )

    # ── Load pipeline ─────────────────────────────────────────────────────────
    try:
        pipeline = get_pipeline()
    except ModelNotReadyError as exc:
        logger.info("Model not ready: %s", exc)
        return PredictionResponse(
            status="ml_not_ready",
            message=str(exc),
            assessment_id=assessment_id,
        )

    # ── Build input DataFrame ─────────────────────────────────────────────────
    input_dict = inp.model_dump()
    df = input_to_dataframe(input_dict)

    # ── Inference ─────────────────────────────────────────────────────────────
    pos_idx    = get_positive_class_index()
    proba_row  = pipeline.predict_proba(df)[0]
    raw_score  = float(proba_row[pos_idx])

    # Apply clinical calibration to prevent unrealistic probabilities
    risk_score     = _calibrate_risk(raw_score)
    was_calibrated = round(risk_score, 6) != round(raw_score, 6)

    threshold     = get_threshold()
    model_version = get_model_version()
    risk_label    = "Elevated Risk" if risk_score >= threshold else "Low Risk"

    if was_calibrated:
        logger.info(
            "Risk score calibrated: raw=%.4f → calibrated=%.4f (cap=%.2f)",
            raw_score, risk_score, _CAP,
        )

    # ── Explanations ──────────────────────────────────────────────────────────
    shap_vals = compute_shap(pipeline, df)
    lime_vals = compute_lime(pipeline, df)

    from app.schemas.prediction_schema import ShapEntry, LimeEntry

    message = (
        "Prediction generated successfully. "
        f"Note: raw model output ({round(raw_score * 100, 1)}%) was calibrated to "
        f"{round(risk_score * 100, 1)}% to stay within the clinically reported "
        "Framingham Risk Score range (≤40%). The model extrapolates beyond its "
        "training distribution for extreme risk-factor combinations."
        if was_calibrated else
        "Prediction generated successfully."
    )

    return PredictionResponse(
        status="ok",
        message=message,
        risk_score=round(risk_score, 4),
        risk_label=risk_label,
        threshold=threshold,
        model_version=model_version,
        assessment_id=assessment_id,
        shap_values=[ShapEntry(**s) for s in shap_vals],
        lime_values=[LimeEntry(**l) for l in lime_vals],
    )
