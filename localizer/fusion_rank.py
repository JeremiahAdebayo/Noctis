"""
Localization fusion for Noctis.

Combines three signal sources into one final ranked file list:
  1. Stack trace (stack_trace.py)  - highest confidence, when present
  2. Lexical/BM25 search (lexical_index.py) - exact-match strength
  3. Vector search (your existing Qdrant leg) - paraphrase/semantic strength

Design decision, and why it's NOT plain equal-weighted RRF:

A stack trace frame is categorically different evidence from a retrieval
hit. A retrieval hit says "this chunk is topically related to the query."
A stack trace frame says "this exact line executed during the failure."
Treating those as one-vote-each in a fusion formula would let two weak
topical matches outvote a single piece of near-ground-truth evidence,
which is the wrong behavior. So fusion here is tiered, not flat:

  - If stack trace candidates exist: they ARE the primary ranking.
    Vector/lexical results are appended after, deduplicated, as secondary
    candidates only (useful if the trace alone doesn't pin down enough
    files for a multi-file fix).
  - If no stack trace: fall back to RRF(vector, lexical), which IS a fair
    fight between two genuinely comparable signal types - both are
    "estimated relevance from retrieval," just via different mechanisms.

This keeps determinism intact at each tier: which tier activates is a
fixed, explicit rule (stack trace present or not), not a learned weight.
"""

from __future__ import annotations

from dataclasses import dataclass

from stack_trace import TraceFrame, select_fix_candidates, _normalize_to_repo_relative


@dataclass(frozen=True)
class LocalizationCandidate:
    file_path: str
    confidence_tier: str   # "stack_trace" | "fused_retrieval"
    score: float            # meaningful only within a tier, not across tiers
    source_detail: str      # human-readable provenance, for debugging/critic feedback


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """
    Fuses multiple ranked lists of file/chunk IDs into one ranking via
    Reciprocal Rank Fusion. k=60 is the standard damping constant from
    the original RRF paper - rarely needs tuning, intentionally not
    exposed as a knob here to avoid an unnecessary source of variance.

    Fully deterministic given fixed input rankings: same lists in, same
    fused ranking out, every time.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def localize(
    bug_report_text: str,
    repo_path: str,
    vector_results: list[tuple[str, str, float]],
    lexical_results: list[tuple[str, str, float]],
    max_results: int = 10,
) -> list[LocalizationCandidate]:
    """
    Main entry point. Combine your three legs into one final candidate
    list, ready to hand to the planner.

    vector_results / lexical_results: both expected as
        [(chunk_id, file_path, score), ...], already ranked best-first -
        i.e. exactly what LexicalIndex.search() returns, and whatever
        shape your existing Qdrant query wrapper returns (adapt the
        tuple shape at the call site if it currently differs).

    Stack trace extraction happens internally from bug_report_text - the
    caller doesn't need to pre-check for a traceback.
    """
    trace_candidates = select_fix_candidates(bug_report_text, repo_path)

    results: list[LocalizationCandidate] = []
    seen_files: set[str] = set()

    if trace_candidates:
        for frame in trace_candidates:
            normalized = _normalize_to_repo_relative(frame.file_path, repo_path)
            if normalized in seen_files:
                continue
            seen_files.add(normalized)
            results.append(
                LocalizationCandidate(
                    file_path=normalized,
                    confidence_tier="stack_trace",
                    score=1.0,  # all trace frames are top-tier; order is the signal
                    source_detail=(
                        f"line {frame.line_number} in {frame.function_name}"
                        f"{f': {frame.code_line}' if frame.code_line else ''}"
                    ),
                )
            )

    # Secondary tier: fused retrieval, always computed (even if trace
    # candidates exist), so multi-file fixes aren't starved down to only
    # the files visible in the traceback. Appended after, deduplicated
    # against anything the trace already surfaced.
    vector_chunk_ids = [chunk_id for chunk_id, _, _ in vector_results]
    lexical_chunk_ids = [chunk_id for chunk_id, _, _ in lexical_results]
    file_lookup = {
        chunk_id: file_path
        for chunk_id, file_path, _ in [*vector_results, *lexical_results]
    }

    fused = reciprocal_rank_fusion([vector_chunk_ids, lexical_chunk_ids])

    for chunk_id, fused_score in fused:
        file_path = file_lookup.get(chunk_id)
        if file_path is None or file_path in seen_files:
            continue
        seen_files.add(file_path)
        results.append(
            LocalizationCandidate(
                file_path=file_path,
                confidence_tier="fused_retrieval",
                score=fused_score,
                source_detail=f"RRF(vector, lexical) chunk={chunk_id}",
            )
        )
        if len(results) >= max_results:
            break

    return results[:max_results]