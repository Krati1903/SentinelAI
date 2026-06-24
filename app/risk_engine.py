"""
Converts a list of warning codes into a weighted risk score and a
discrete risk level (SAFE / MODERATE / HIGH / CRITICAL).
"""

from app import config


def calculate_risk(warning_codes: list) -> dict:
    score = 0
    for code in warning_codes:
        score += config.RISK_WEIGHTS.get(code, 0)

    score = min(score, 100)

    level = "SAFE"
    for level_name, (low, high) in config.RISK_LEVELS.items():
        if low <= score <= high:
            level = level_name
            break

    return {"score": score, "level": level}
