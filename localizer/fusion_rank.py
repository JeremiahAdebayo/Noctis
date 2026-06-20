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
from pathlib import Path

from localizer.stack_trace import TraceFrame, select_fix_candidates, _normalize_to_repo_relative
from localizer.lexical_index import LexicalIndex
from localizer.qdrant_utils import retrieve_chunks


@dataclass(frozen=True)
class LocalizationCandidate:
    file_path: str
    confidence_tier: str   # "stack_trace" | "fused_retrieval"
    score: float            # meaningful only within a tier, not across tiers
    source_detail: str      # human-readable provenance, for debugging/critic feedback
    chunk_metadata: dict | None = None  # for retrieval results: {name, type, source, start_line, end_line, ...}


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
    repo_path: str,
    vector_results: list[tuple[str, str, float]],
    lexical_results: list[tuple[str, str, float]],
    chunk_metadata: dict | None = None,
    lexical_index: LexicalIndex | None = None,
    bug_report_text: str = "",
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

    chunk_metadata: dict mapping chunk_id -> {name, type, source, start_line, end_line, ...}
        If provided, chunk details are attached to candidates for downstream retrieval.
        
    lexical_index: LexicalIndex instance to fetch full chunk details for lexical results.
        If provided with chunk_metadata=None, will populate metadata by querying the index.

    Stack trace extraction happens internally from bug_report_text - the
    caller doesn't need to pre-check for a traceback.
    """
    if chunk_metadata is None:
        chunk_metadata = {}

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
                    chunk_metadata=None,  # stack trace frames don't carry chunk metadata
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

    # If chunk_metadata wasn't provided, build entries from the results
    if not chunk_metadata:
        for chunk_id, file_path, _ in [*vector_results, *lexical_results]:
            if chunk_id not in chunk_metadata:
                # Try to fetch full chunk from lexical_index if available
                if lexical_index is not None:
                    full_chunk = lexical_index.get_chunk_by_id(chunk_id)
                    if full_chunk:
                        chunk_metadata[chunk_id] = full_chunk
                        continue
                
                # Fallback to minimal data if not in index
                chunk_metadata[chunk_id] = {
                    "chunk_id": chunk_id,
                    "path": file_path,
                    "name": chunk_id.split("::")[-1] if "::" in chunk_id else chunk_id,
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
                chunk_metadata=chunk_metadata.get(chunk_id),
            )
        )
        if len(results) >= max_results:
            break

    return results[:max_results]


def localize_full(
    bug_report_text: str,
    query: str,
    repo_path: str,
    lexical_db_path: str | Path = ":memory:",
    max_results: int = 10,
    vector_top_k: int = 10,
    lexical_top_k: int = 10,
) -> list[LocalizationCandidate]:
    """
    End-to-end localization: integrate all three legs (stack trace, lexical,
    vector) into one ranked candidate list.

    Performs:
      1. Stack trace extraction from bug_report_text (raw issue/traceback)
      2. Lexical search (BM25) using the semantic query
      3. Vector search (Qdrant embeddings) using the semantic query
      4. Tiered fusion: stack trace primary, RRF(lexical, vector) secondary

    Args:
        bug_report_text: the raw bug report / error message (may contain traceback)
            Used for stack trace extraction only.
        query: semantic search query (typically f"{issue_title} {issue_body}")
            Used for both lexical and vector search. Can be refined in subsequent
            calls with critic feedback or test output.
        repo_path: path to target repository (for trace normalization)
        lexical_db_path: path to SQLite FTS5 database (default: in-memory)
        max_results: cap on final candidate list
        vector_top_k: how many vector results to pull before fusion
        lexical_top_k: how many lexical results to pull before fusion

    Returns:
        List of LocalizationCandidate sorted by tier and score.
        Each candidate carries chunk_metadata for retrieval results.
    """
    # Lexical search using semantic query
    lexical_index = LexicalIndex(db_path=lexical_db_path)
    lexical_results = lexical_index.search(query, top_k=lexical_top_k)
    if not lexical_results:
        print("Lexical search returned empty")
    
    # Vector search using semantic query: convert qdrant payload format to (chunk_id, file_path, score)
    vector_payloads = retrieve_chunks(query, top_k=vector_top_k)
    vector_results = [
        (
            f"{p.get('path')}::{p.get('name')}",  # chunk_id
            p.get("path"),  # file_path
            1.0,  # placeholder score (Qdrant returns distances, not scores directly)
        )
        for p in vector_payloads
    ]
    
    # Build chunk metadata mapping for both vector and lexical results
    chunk_metadata: dict = {}
    
    # Vector payloads as metadata
    for p in vector_payloads:
        chunk_id = f"{p.get('path')}::{p.get('name')}"
        chunk_metadata[chunk_id] = {
            "name": p.get("name"),
            "type": p.get("type"),
            "path": p.get("path"),
            "source": p.get("source"),
            "start_line": p.get("start_line"),
            "end_line": p.get("end_line"),
        }
    
    # Lexical results — retrieve full chunk content from index
    for chunk_id, file_path, score in lexical_results:
        if chunk_id not in chunk_metadata:
            # Query the index to get full chunk details including source
            full_chunk = lexical_index.get_chunk_by_id(chunk_id)
            if full_chunk:
                chunk_metadata[chunk_id] = full_chunk
            else:
                # Fallback if chunk not found in index
                chunk_metadata[chunk_id] = {
                    "chunk_id": chunk_id,
                    "path": file_path,
                    "name": chunk_id.split("::")[-1] if "::" in chunk_id else chunk_id,
                }
    
    # Fused localization using the tiered strategy
    # Stack trace extraction uses bug_report_text (raw for traceback parsing)
    return localize(
        bug_report_text=bug_report_text,
        repo_path=repo_path,
        vector_results=vector_results,
        lexical_results=lexical_results,
        chunk_metadata=chunk_metadata,
        lexical_index=lexical_index,
        max_results=max_results,
    )


def get_chunk_summaries(candidates: list[LocalizationCandidate]) -> str:
    """
    Converts ranked LocalizationCandidates into a formatted string summary
    suitable for the planner (mimics the existing retrieve_chunks format).

    Useful for integrating fusion results into the agent pipeline:
        candidates = localize_full(...)
        context = get_chunk_summaries(candidates)
        # pass context to planner_chain
    """
    lines = []
    for i, candidate in enumerate(candidates):
        if candidate.chunk_metadata:
            meta = candidate.chunk_metadata
            lines.append(
                f"[{candidate.confidence_tier.upper()}] "
                f"=== {meta.get('path')} | {meta.get('type', 'unknown')} "
                f"'{meta.get('name', 'unnamed')}' "
                f"(lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')}) ==="
            )
            if meta.get("source"):
                lines.append(meta["source"])
        else:
            # Stack trace candidate without chunk metadata
            lines.append(
                f"[{candidate.confidence_tier.upper()}] "
                f"=== {candidate.file_path} ===\n"
                f"{candidate.source_detail}"
            )
        lines.append("")  # blank line between chunks
    
    return "\n".join(lines)