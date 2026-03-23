#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib

from .naming import OFFICIAL_POLICY_NAMES, normalize_policy_name

_POLICY_MODULES = {
    "motion": "_segment.policies.motion",
    "uniform": "_segment.policies.uniform",
    "semantic": "_segment.policies.semantic",
    "risk": "_segment.policies.risk",
}


def list_policy_names():
    return sorted(OFFICIAL_POLICY_NAMES)


def get_policy_builder(policy_name):
    name = normalize_policy_name(policy_name)
    if name not in _POLICY_MODULES:
        raise ValueError("unsupported segment policy: {}".format(name))
    module = importlib.import_module(_POLICY_MODULES[name])
    return getattr(module, "build_policy_plan")
