#!/usr/bin/env python3
"""
holdout.py
The held-out 20% of failure codes.

Lives here rather than under tests/ so that fixture builders exclude it BY
CONSTRUCTION instead of by remembering to. A holdout that a fixture can reach by
accident is not a holdout, and the accident is silent -- the metric still
reports a number, it just stops meaning generalisation.

Deterministic and content-addressed: the same 20% on every machine and every
run. A random split would make train-vs-holdout comparisons noise.
"""
from __future__ import annotations

import hashlib
from typing import List, Set, Tuple

HOLDOUT_FRACTION = 0.20
SPLIT_SALT = "komatsu-holdout-v1"


def split(codes) -> Tuple[List[str], List[str]]:
    train, holdout = [], []
    for c in sorted(codes):
        h = hashlib.sha256(f"{SPLIT_SALT}:{c}".encode()).hexdigest()
        (holdout if int(h[:8], 16) / 0xFFFFFFFF < HOLDOUT_FRACTION
         else train).append(c)
    return train, holdout


_cache = None


def train_codes() -> List[str]:
    return _split_cached()[0]


def holdout_codes() -> List[str]:
    return _split_cached()[1]


def holdout_set() -> Set[str]:
    return set(_split_cached()[1])


def _split_cached():
    global _cache
    if _cache is None:
        from agent import tools
        _cache = split(tools.records())
    return _cache
