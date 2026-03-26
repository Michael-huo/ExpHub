#!/usr/bin/env python3
# -*- coding: utf-8 -*-

OFFICIAL_POLICY_NAMES = ("uniform", "state")
_DISPLAY_NAMES = {
    "uniform": "Uniform",
    "state": "State",
}


def normalize_policy_name(policy_name):
    name = str(policy_name or "uniform").strip().lower()
    if not name:
        name = "uniform"
    return str(name)


def is_supported_policy_name(policy_name):
    name = normalize_policy_name(policy_name)
    if not name:
        return True
    return name in OFFICIAL_POLICY_NAMES


def policy_display_name(policy_name):
    canonical_name = normalize_policy_name(policy_name)
    return _DISPLAY_NAMES.get(canonical_name, str(canonical_name))
