#!/usr/bin/env python3
"""
model_system.py
A real model as a SYSTEM UNDER TEST, alongside the synthetic ceiling and floor.

GoodSystem and WeakSystem are unchanged and stay. They are the reference and
the floor; the whole point of this file is that a real model can now be placed
between them and the distance measured in both directions.

THE THREE CATEGORIES, and why a system must declare one
--------------------------------------------------------
  REFERENCE   answers out of the ground truth. Its perfection is ASSERTED --
              the ceiling guard requires 1.0000 in every bucket, because a
              bucket where the reference is imperfect is a bucket where the
              metric has no headroom.
  FLOOR       deliberately broken. Its failure is asserted: if it passes a
              gate, that gate has a hole.
  UNDER_TEST  a real system. Its score is the MEASUREMENT. Nothing about it is
              asserted, and applying the ceiling guard to it would be asserting
              that the thing being measured has already succeeded.

Declared on the class, checked as a union against SYSTEMS -- a system in no
category fails naming itself, the same rule as CEILING/NOT_CEILING.

THE CACHE, and the one rule that matters
----------------------------------------
A STALE ENTRY IS A MISS, NEVER A HIT. The key covers everything that can change
a response: the case id, the exact prompt text, the model string, and the
ordered ids of the retrieved chunks. Change the prompt, change the retrieval,
change the model behind the deployment -- the key changes and the old answer is
not served.

An undetectably-stale cache is reports/fact_verification.json again, and worse:
that one under-reported verification, this one would report a model that no
longer exists in that configuration as though it had just answered. Hits and
misses go into the run record, and a run served entirely from cache says so,
because a number that looks like a fresh measurement and is not is the thing
being avoided.

DEPLOYMENT AND MODEL ARE DIFFERENT THINGS. The deployment is what was called;
the model string is what answered, and it can change under a stable deployment
name. Both go in the record, and the CACHE KEYS ON THE MODEL -- so the day
Azure moves gpt-4.1 to a new build, every entry misses and the numbers are
re-measured rather than inherited.

No tuning lives here. Prompts are built from the retrieved context and the
question, and nothing is special-cased per case type.
"""
from __future__ import annotations

import hashlib
import re
import json
import os
import time
from typing import Dict, List, Optional

from eval.refusal import looks_refused as _looks_refused

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(REPO_ROOT, "eval_out", "model_cache.jsonl")

REFERENCE, FLOOR, UNDER_TEST = "reference", "floor", "under_test"

MAX_CONTEXT_CHUNKS = 6
MAX_CHUNK_CHARS = 1800
# Azure rejects max_tokens on this API version; the parameter is
# max_completion_tokens. Kept as a named constant so the difference is visible
# rather than buried in a dict literal.
MAX_COMPLETION_TOKENS = 700

SYSTEM_PROMPT = (
    "You are a diagnostic assistant for Komatsu excavator technicians. "
    "Answer ONLY from the manual extracts provided. "
    "Quote measurement values, standard values and branch outcomes exactly as "
    "written, including units and braces. "
    "If the extracts do not contain the answer, say you cannot find it and do "
    "not guess. "
    "If the question names a failure code or machine model that does not "
    "appear in the extracts, refuse and say why. "
    "Cite the manual page you used as 'Page: <page>'. "
    "Do not invent values, pages, part numbers or procedures."
)


# ------------------------------------------------------------------ prompt

def build_prompt(case: dict, contexts: List[dict]) -> str:
    """The exact user-message text. Deterministic, and part of the cache key."""
    lines = []
    for c in contexts[:MAX_CONTEXT_CHUNKS]:
        page = c.get("manual_page") or "?"
        lines.append(f"--- extract (page {page}, record {c.get('code')}) ---\n"
                     f"{(c.get('text') or '')[:MAX_CHUNK_CHARS]}")
    filt = case.get("filters") or {}
    return (f"Machine: {filt.get('model', 'unspecified')}   "
            f"Manual: {filt.get('manual_id', 'unspecified')}\n\n"
            + "\n\n".join(lines)
            + f"\n\n--- question ---\n{case['question']}")


# ------------------------------------------------------------------- cache

def cache_key(case_id: str, prompt: str, model: str,
              chunk_ids: List[str]) -> str:
    """Everything that can change the response, and nothing that cannot.

    chunk_ids are ORDERED: a reranking that changes which extract comes first
    changes the prompt and must change the key.
    """
    payload = json.dumps({"case": case_id, "prompt": prompt, "model": model,
                          "chunks": list(chunk_ids)},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Append-only JSONL. Last write for a key wins."""

    def __init__(self, path: str = CACHE_PATH, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._by_key: Dict[str, dict] = {}
        if enabled and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._by_key[rec["key"]] = rec

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        hit = self._by_key.get(key)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return hit["response"]

    def put(self, key: str, response: dict, meta: dict) -> None:
        if not self.enabled:
            return
        rec = {"key": key, "response": response, **meta}
        self._by_key[key] = rec
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "lookups": total,
                "hit_rate": (self.hits / total) if total else 0.0,
                "entries_on_disk": len(self._by_key),
                # A run with no miss answered no question. Recorded so a fully
                # cached run cannot be read as a fresh measurement.
                "fully_cached": total > 0 and self.misses == 0,
                "enabled": self.enabled}


# ------------------------------------------------------------------ client

class AzureChatClient:
    """Minimal Azure OpenAI chat client over httpx. No SDK, no new pin.

    Credentials are read at CALL TIME via core.config, never at import, so a
    process that does not use a model imports this file without a key present.
    """

    def __init__(self):
        self._settings = None

    def _s(self) -> dict:
        if self._settings is None:
            from core import config
            self._settings = {
                "key": config.require("AZURE_OPENAI_API_KEY"),
                "endpoint": config.require("AZURE_OPENAI_ENDPOINT").rstrip("/"),
                "deployment": config.require("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                "version": config.require("AZURE_OPENAI_CHAT_API_VERSION"),
            }
        return self._settings

    def deployment(self) -> str:
        return self._s()["deployment"]

    def complete(self, system: str, user: str, retries: int = 3) -> dict:
        """-> {"text", "model", "usage"}. Raises on repeated failure."""
        import httpx
        s = self._s()
        url = (f"{s['endpoint']}/openai/deployments/{s['deployment']}"
               f"/chat/completions?api-version={s['version']}")
        body = {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "temperature": 0}
        last = None
        for attempt in range(retries):
            try:
                r = httpx.post(url, headers={"api-key": s["key"],
                                             "content-type": "application/json"},
                               json=body, timeout=120)
                if r.status_code == 200:
                    d = r.json()
                    return {"text": d["choices"][0]["message"]["content"] or "",
                            "model": d.get("model", ""),
                            "usage": d.get("usage", {})}
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}"
                    time.sleep(2 ** attempt)
                    continue
                # Never surface the body: Azure error payloads can echo headers.
                raise RuntimeError(f"Azure returned HTTP {r.status_code}")
            except Exception as exc:                       # noqa: BLE001
                last = f"{type(exc).__name__}"
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Azure call failed after {retries} attempts: {last}")


# ------------------------------------------------------------------ system

# The refusal detector is imported at the top of this file from eval/refusal.py
# and is shared with eval/metrics/safety.py. It used to live HERE, which left
# safety.py running a SECOND, weaker, English-only detector as its fallback --
# two implementations of one judgement, the untested one waiting for the first
# system that reports no explicit flag. See that module for the rule and for
# what it misses.


PAGE_RE = None


def _pages(text: str) -> List[str]:
    global PAGE_RE
    if PAGE_RE is None:
        import re
        PAGE_RE = re.compile(r"\b(\d{2}-\d{1,4})\b")
    seen, out = set(), []
    for p in PAGE_RE.findall(text or ""):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


class ModelSystem:
    """The real model, scored exactly like any other system."""

    name = "model"
    CATEGORY = UNDER_TEST

    def __init__(self, records, client=None, cache=None):
        self.records = records
        self.client = client or AzureChatClient()
        self.cache = cache if cache is not None else ResponseCache()
        self.model_seen = set()

    # Retrieval is the harness's deterministic half and is NOT the model's
    # doing. Same call the good system makes, so a good/model comparison is a
    # comparison of answers rather than of two different retrievals.
    def retrieve(self, case, retriever, k):
        return retriever.search(case["question"], k, case.get("filters"))

    def answer(self, case, contexts):
        prompt = build_prompt(case, contexts)
        chunk_ids = [c.get("id") for c in contexts[:MAX_CONTEXT_CHUNKS]]
        # The model string is part of the key, so it has to be known before the
        # lookup. The last observed string is used, and the first call of a run
        # establishes it; a changed build therefore misses exactly once and
        # then keys consistently.
        model_hint = sorted(self.model_seen)[0] if self.model_seen else \
            self.client.deployment()
        key = cache_key(case["id"], prompt, model_hint, chunk_ids)
        hit = self.cache.get(key)
        if hit is not None:
            return self._derive(case, hit["answer"])

        out = self.client.complete(SYSTEM_PROMPT, prompt)
        self.model_seen.add(out["model"])
        text = out["text"]
        # ONLY WHAT THE MODEL RETURNED is cached. Everything else is derived on
        # the way out, on hits and misses alike -- see _derive.
        self.cache.put(key, {"answer": text},
                       {"case_id": case["id"], "model": out["model"],
                        "deployment": self.client.deployment(),
                        "usage": out.get("usage", {}),
                        "n_chunks": len(chunk_ids)})
        return self._derive(case, text)

    @staticmethod
    def _derive(case, text):
        """The response, rebuilt from the model's text every time.

        THE CACHE STORES THE MODEL'S OUTPUT, NOT OUR CONCLUSIONS ABOUT IT.
        `refused` and `citations` are ours, computed by code in this repo. An
        earlier version cached them alongside the answer, which froze them: the
        English-only refusal detector was fixed and every cache HIT kept
        returning the old verdict, so refusal_correctness would not have moved
        and the fix would have looked ineffective rather than uncached.

        That is the same defect as a frozen verification artefact and it is the
        general rule that prevents it: cache the EVIDENCE, derive the JUDGEMENT.
        Nothing here calls the network, so re-deriving is free.
        """
        return {"answer": text,
                "citations": _pages(text),
                "refused": _looks_refused(case, text),
                # The model names no fact ids. It cannot: it never sees them.
                # So citation_span_precision and the fact-id path simply do not
                # apply to it, which is a true statement about this system
                # rather than a gap to paper over.
                "fact_ids": []}

    def converse(self, scen, contexts):
        """Multi-turn. One call per turn, same cache discipline."""
        turns = []
        for i, _ in enumerate(scen.get("turns", [])):
            sub = dict(scen, id=f"{scen['id']}#turn{i}",
                       question=scen["turns"][i])
            turns.append(self.answer(sub, contexts)["answer"])
        return turns

    def describe(self) -> dict:
        """What produced the numbers. Deployment and model are both recorded."""
        return {"system": self.name, "category": self.CATEGORY,
                "deployment": self.client.deployment(),
                "model": sorted(self.model_seen) or None,
                "cache": self.cache.stats()}
