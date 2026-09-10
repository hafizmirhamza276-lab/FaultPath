#!/usr/bin/env python3
"""
adapters.py
The two seams between this harness and a real system.

Nothing downstream imports a concrete backend. Metrics receive plain dicts, so
the same metric code scores a local BM25 run and a hosted Azure run without
modification -- which is the only way a run-to-run comparison across backends
means anything.

    Retriever.search(query, k, filters) -> [{id, text, score, **metadata}]
    Generator.generate(question, contexts, history) -> {answer, citations, refused}

LocalBM25Retriever ships working, pure python, no dependencies and no network,
so the baseline runs today. AzureSearchRetriever and the generators are
documented stubs that raise on use rather than silently returning empty results.
"""
import math
import re
import collections


class Retriever:
    """Interface. `filters` is the model/manual scope the caller must honour."""

    name = "abstract"

    def search(self, query, k=10, filters=None):
        raise NotImplementedError


class Generator:
    """Interface.

    contexts  the retrieved chunks, in rank order
    history   prior turns as [{"role": "technician"|"assistant", "text": ...}]
    returns   {"answer": str, "citations": [str], "refused": bool}
    """

    name = "abstract"

    def generate(self, question, contexts, history=None):
        raise NotImplementedError


# ------------------------------------------------------------------ tokenising

# Keep alphanumerics together so connector names (CK06, P68) and codes (CA451)
# survive as single tokens. Splitting them destroys the strongest retrieval
# signal in the corpus.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOKEN_RE.findall(str(text or "").lower())


# ------------------------------------------------------------------ local BM25

class LocalBM25Retriever(Retriever):
    """Okapi BM25. Pure python, deterministic, no dependencies.

    Present so the harness has a real retriever on day one. Its absolute scores
    are not the point -- holding it fixed while the chunking changes is, because
    then the spread between chunkings is attributable to the chunking alone.
    """

    name = "bm25"

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._docs = [tokenize(c["text"]) for c in self.chunks]
        self._len = [len(d) for d in self._docs]
        self._avglen = (sum(self._len) / len(self._len)) if self._len else 0.0
        self._tf = [collections.Counter(d) for d in self._docs]
        df = collections.Counter()
        for d in self._docs:
            df.update(set(d))
        n = len(self._docs)
        self._idf = {
            t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def _score(self, q_tokens, i):
        tf, dl, s = self._tf[i], self._len[i], 0.0
        for t in q_tokens:
            f = tf.get(t)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avglen or 1))
            s += self._idf.get(t, 0.0) * f * (self.k1 + 1) / denom
        return s

    def search(self, query, k=10, filters=None):
        q = tokenize(query)
        hits = []
        for i, c in enumerate(self.chunks):
            if filters and not self._passes(c, filters):
                continue
            hits.append((self._score(q, i), i))
        # Sort by score desc, then chunk id asc. The tiebreak is what makes the
        # run byte-reproducible -- without it, equal-scoring chunks come back in
        # dict order and two runs of the same config can disagree.
        hits.sort(key=lambda x: (-x[0], self.chunks[x[1]]["id"]))
        out = []
        for score, i in hits[:k]:
            c = dict(self.chunks[i])
            c["score"] = round(score, 6)
            out.append(c)
        return out

    @staticmethod
    def _passes(chunk, filters):
        for key, want in filters.items():
            if want is None:
                continue
            if chunk.get(key) != want:
                return False
        return True


class AzureSearchRetriever(Retriever):
    """STUB -- not implemented.

    Intended shape: hybrid semantic + vector query against an Azure AI Search
    index, with `filters` pushed down as an OData filter expression rather than
    applied client-side, so recall is measured against what the service actually
    returns.

    Two things must hold before a run against this is comparable to a local one:
    the filter must be enforced server-side (otherwise filter_correctness is
    measuring the harness, not the system), and the chunking must be the same
    corpus this repo builds -- not whatever the ingestion pipeline happened to
    produce.
    """

    name = "azure_search"

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "AzureSearchRetriever is a documented stub. Use LocalBM25Retriever "
            "for the local baseline, or implement search() against your index."
        )

    def search(self, query, k=10, filters=None):
        raise NotImplementedError


class StubGenerator(Generator):
    """STUB -- not implemented.

    A real generator goes here: prompt assembly, model call, citation
    extraction. It must return `refused` as an explicit boolean rather than
    leaving the harness to infer refusal from prose, because clean_refusal and
    over_refusal both depend on knowing what the system intended.
    """

    name = "stub"

    def generate(self, question, contexts, history=None):
        raise NotImplementedError(
            "StubGenerator is a documented stub. The synthetic generators in "
            "eval/synthetic.py exercise the harness; a real one plugs in here."
        )
