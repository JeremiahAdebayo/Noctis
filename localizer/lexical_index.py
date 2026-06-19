"""
Lexical retrieval leg for Noctis localization.

Uses SQLite's built-in FTS5 extension (BM25 ranking, no extra service,
no extra memory footprint) as the deterministic counterpart to Qdrant's
vector search. Indexes the same chunk units your CodeMapVisitor already
produces, so this is additive, not a re-chunking effort.

Determinism note: BM25 over a fixed corpus and fixed query string always
returns the same ranking. The only non-determinism in this leg is upstream
(what query string you feed it), not in the scoring itself.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChunkRecord:
    """One indexable unit. Maps 1:1 to whatever CodeMapVisitor already
    produces per file (e.g. a function, method, or class chunk)."""

    chunk_id: str          # stable id, e.g. "click/types.py::Choice.normalize_choice"
    file_path: str         # relative path, e.g. "src/click/types.py"
    symbol_name: str       # e.g. "normalize_choice" or "Choice" - used for exact-match boost
    content: str           # the chunk's source text, used for full-text search


class LexicalIndex:
    """
    Thin wrapper around a SQLite FTS5 virtual table.

    Usage:
        index = LexicalIndex(db_path="repo_lexical.db")
        index.rebuild(chunks)               # full rebuild from current chunk set
        results = index.search("os.access", top_k=10)
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # contentless-adjacent design: we store metadata in a normal table
        # and only the searchable text in the FTS5 virtual table, joined by
        # rowid. This keeps file_path/symbol_name out of the tokenizer's
        # reach (we don't want "types.py" itself treated as search tokens).
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunk_meta (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                content,
                content='',
                contentless_delete=1
            );
            """
        )
        self._conn.commit()

    def rebuild(self, chunks: list[ChunkRecord]) -> None:
        """
        Full rebuild. Call this whenever the reassembler re-indexes a file
        (i.e. the same trigger point as your existing CodeMapVisitor
        re-index step) — keeps this index from drifting out of sync with
        the file_registry.
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM chunk_fts")
        cur.execute("DELETE FROM chunk_meta")

        for chunk in chunks:
            cur.execute(
                "INSERT INTO chunk_meta (chunk_id, file_path, symbol_name) "
                "VALUES (?, ?, ?)",
                (chunk.chunk_id, chunk.file_path, chunk.symbol_name),
            )
            rowid = cur.lastrowid
            cur.execute(
                "INSERT INTO chunk_fts (rowid, content) VALUES (?, ?)",
                (rowid, chunk.content),
            )

        self._conn.commit()

    def upsert_file(self, file_path: str, chunks: list[ChunkRecord]) -> None:
        """
        Partial update for a single file's chunks — use this on the
        reassembler's per-file re-index path instead of a full rebuild,
        so a multi-file patch run doesn't pay O(whole repo) cost per file.
        """
        cur = self._conn.cursor()
        # remove this file's existing chunks first
        cur.execute(
            "SELECT rowid FROM chunk_meta WHERE file_path = ?", (file_path,)
        )
        old_rowids = [r[0] for r in cur.fetchall()]
        for rowid in old_rowids:
            cur.execute("DELETE FROM chunk_fts WHERE rowid = ?", (rowid,))
            cur.execute("DELETE FROM chunk_meta WHERE rowid = ?", (rowid,))

        for chunk in chunks:
            cur.execute(
                "INSERT INTO chunk_meta (chunk_id, file_path, symbol_name) "
                "VALUES (?, ?, ?)",
                (chunk.chunk_id, chunk.file_path, chunk.symbol_name),
            )
            rowid = cur.lastrowid
            cur.execute(
                "INSERT INTO chunk_fts (rowid, content) VALUES (?, ?)",
                (rowid, chunk.content),
            )

        self._conn.commit()

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, str, float]]:
        """
        Returns [(chunk_id, file_path, bm25_score), ...] ranked best-first.
        Lower bm25() return value = better match (SQLite convention), so we
        negate it before returning, keeping the convention "higher = better"
        consistent with the vector search leg for fusion downstream.
        """
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            print("Query not safe")
            return []

        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT chunk_meta.chunk_id, chunk_meta.file_path, bm25(chunk_fts) AS score
            FROM chunk_fts
            JOIN chunk_meta ON chunk_meta.rowid = chunk_fts.rowid
            WHERE chunk_fts.content MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (safe_query, top_k),
        )
        return [(row[0], row[1], -row[2]) for row in cur.fetchall()]

    def search_scoped(
        self, symbol_name: str, scope_symbol: str | None, top_k: int = 10
    ) -> list[tuple[str, str, float]]:
        """
        Dotted-path-aware search: when a bug report references something
        like "Choice.normalize_choice", scope_symbol="Choice" and
        symbol_name="normalize_choice". Chunks whose symbol_name matches
        AND whose containing class matches (if your chunk metadata tracks
        a parent/class field — see note below) get a deterministic ranking
        boost over a same-named symbol in an unrelated class.

        This is the one piece of "symbol-aware" retrieval that's genuinely
        not redundant with plain BM25: FTS5 alone can't distinguish
        "normalize_choice in Choice" from "normalize_choice in SomeOtherClass"
        without this scoping, since both chunks would just contain the
        token "normalize_choice".

        Falls back to a plain symbol_name search if scope_symbol is None
        or doesn't match anything — never returns fewer results than a
        plain search would.
        """
        cur = self._conn.cursor()

        if scope_symbol:
            # Exact match on symbol_name AND file contains scope_symbol
            # as a chunk too (i.e. the class is indexed in the same file).
            # This is intentionally conservative: a true AND, not a boost,
            # because a wrong-class match is worse than no match at all
            # for this fast path - it should fall through to fusion instead.
            cur.execute(
                """
                SELECT m.chunk_id, m.file_path, bm25(chunk_fts) AS score
                FROM chunk_fts
                JOIN chunk_meta m ON m.rowid = chunk_fts.rowid
                WHERE chunk_fts.content MATCH ?
                  AND m.symbol_name = ?
                  AND m.file_path IN (
                      SELECT file_path FROM chunk_meta WHERE symbol_name = ?
                  )
                ORDER BY score
                LIMIT ?
                """,
                (self._sanitize_fts_query(symbol_name), symbol_name, scope_symbol, top_k),
            )
            scoped = [(row[0], row[1], -row[2]) for row in cur.fetchall()]
            if scoped:
                return scoped

        # Fallback: plain symbol_name search, no class scoping applied.
        return self.search(symbol_name, top_k=top_k)

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """
        FTS5 query syntax treats certain characters specially (quotes,
        hyphens, asterisks, colons). Bug reports and code snippets are
        full of these, so we tokenize defensively: keep only word-like
        tokens and OR them together, rather than passing raw text through
        and risking a syntax error mid-pipeline.
        """
        import re

        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
        if not tokens:
            print("No tokens match")
            return ""
        # de-duplicate while preserving order, cap length defensively
        seen = set()
        deduped = []
        for tok in tokens:
            if tok.lower() not in seen:
                seen.add(tok.lower())
                deduped.append(tok)
        return " OR ".join(deduped[:64])

    def close(self) -> None:
        self._conn.close()