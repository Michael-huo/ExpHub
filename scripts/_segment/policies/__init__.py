#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib


_POLICY_MODULES = {
    "uniform": "_segment.policies.uniform",
    "sks_v1": "_segment.policies.sks_v1",
    "semantic_guarded_v1": "_segment.policies.semantic_guarded_v1",
    "semantic_guarded_v2": "_segment.policies.semantic_guarded_v2",
}


def list_policy_names():
    return sorted(_POLICY_MODULES.keys())


def get_policy_builder(policy_name):
    name = str(policy_name or "uniform")
    if name not in _POLICY_MODULES:
        raise ValueError("unsupported segment policy: {}".format(name))
    module = importlib.import_module(_POLICY_MODULES[name])
    return getattr(module, "build_policy_plan")
