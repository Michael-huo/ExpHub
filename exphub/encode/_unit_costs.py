from __future__ import annotations


_RISK_SPAN_POLICY = {
    "low": {"min": 24, "target": 56, "max": 72},
    "medium": {"min": 16, "target": 36, "max": 48},
    "high": {"min": 12, "target": 20, "max": 28},
}


def span_policy_for_risk_level(risk_level):
    return dict(_RISK_SPAN_POLICY.get(str(risk_level or "medium"), _RISK_SPAN_POLICY["medium"]))


def boundary_choice_cost(start_idx, candidate_idx, policy, candidate_strength):
    policy = dict(policy or {})
    target_idx = int(start_idx) + int(policy.get("target", 32) or 32)
    distance = abs(int(candidate_idx) - int(target_idx))
    strength_bonus = float(candidate_strength) * 8.0
    return float(distance) - float(strength_bonus)
