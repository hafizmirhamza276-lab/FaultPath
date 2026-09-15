#!/usr/bin/env python3
"""
config.py
Settings read at CALL TIME, from the real environment first and .env second.

WHY NOT A MODULE-LEVEL CONSTANT
-------------------------------
`API_KEY = os.environ["..."]` at module scope fails at import for every process
that does not need a model -- including tests/test_extraction.py, which has no
business knowing a key exists. That is the same defect as extract_golden.py's
PDF guard, which ran at import and made the offline agent require the manual
(see 102600a). settings() is the require_pdf() of this file: the check happens
where the value is used, so importing costs nothing.

WHY NOT python-dotenv
---------------------
It would be a new pinned dependency in a requirements.txt that treats pins as
load-bearing, to parse KEY=VALUE. The parser below is the whole feature.

THE REAL ENVIRONMENT WINS over .env, so CI and a container can supply
credentials without a file, and a developer's local .env cannot silently
override what an operator set deliberately.

Nothing here prints, logs or returns a key alongside anything that gets
recorded. `describe()` exists so a run record can name the model and deployment
without the caller reaching for the settings dict.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(REPO_ROOT, ".env")

SECRET_SUFFIXES = ("_KEY", "_SECRET", "_TOKEN", "_PASSWORD")


def _parse_env_file(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def settings(path: Optional[str] = None) -> Dict[str, str]:
    """Environment overlaid on .env. Read fresh; never cached at import."""
    merged = _parse_env_file(path or ENV_PATH)
    merged.update({k: v for k, v in os.environ.items() if k in merged or
                   k.startswith("AZURE_OPENAI_")})
    return merged


def get(name: str, default: Optional[str] = None,
        path: Optional[str] = None) -> Optional[str]:
    return settings(path).get(name, default)


def require(name: str, path: Optional[str] = None) -> str:
    """The value, or a clear exit naming what is missing and where to put it.

    Never echoes the value, so a misconfiguration message cannot leak a key.
    """
    v = settings(path).get(name)
    if not v:
        raise SystemExit(
            f"ERROR: {name} is not set.\n"
            f"  Put it in {ENV_PATH} as {name}=... or export it.\n"
            "  .env is gitignored and must stay that way.")
    return v


def is_secret(name: str) -> bool:
    return any(name.upper().endswith(s) for s in SECRET_SUFFIXES)


def describe(path: Optional[str] = None) -> Dict[str, str]:
    """Non-secret settings, safe to put in a run record or a log line."""
    return {k: v for k, v in settings(path).items() if not is_secret(k)}


def available(*names: str, path: Optional[str] = None) -> bool:
    """True when every named setting is present. For skipping, not for gating."""
    s = settings(path)
    return all(s.get(n) for n in names)
