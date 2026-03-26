#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib

from .naming import OFFICIAL_POLICY_NAMES, is_supported_policy_name, normalize_policy_name

_POLICY_MODULES = {
    "uniform": "scripts._segment.policies.uniform",
    "state": "scripts._segment.policies.state",
}


def list_policy_names():
    return sorted(OFFICIAL_POLICY_NAMES)


def get_policy_builder(policy_name):
    name = normalize_policy_name(policy_name)
    if not is_supported_policy_name(name):
        raise ValueError("unsupported segment policy: {}".format(name))
    if name not in _POLICY_MODULES:
        raise ValueError("unsupported segment policy: {}".format(name))
    module = importlib.import_module(_POLICY_MODULES[name])
    return getattr(module, "build_policy_plan")
