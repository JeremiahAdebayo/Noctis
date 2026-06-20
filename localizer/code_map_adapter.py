"""
Adapter: CodeMapVisitor output -> ChunkRecord (for lexical_index.py).

CodeMapVisitor produces a flat list of dicts: {name, start_line, end_line, type}.
Three gaps relative to what lexical_index.py needs, bridged here rather
than by modifying CodeMapVisitor itself:

  1. No chunk_id / no uniqueness guarantee - names collide constantly
     (e.g. "convert" appears once per ParamType subclass). Built here as
     f"{file_path}::{name}::{start_line}", since start_line is always
     unique within a file.

  2. No class-membership tracking - because CodeMapVisitor's
     visit_ClassDef calls generic_visit(node), every method gets
     recorded with type="function", indistinguishable from a true
     top-level function. Reconstructed here by checking which class's
     [start_line, end_line] range each function falls inside.

  3. No source content - only line ranges. Sliced here from the full
     source text using start_line/end_line.

If you later extend CodeMapVisitor to track parent class directly (the
more correct long-term fix), this adapter's class-membership inference
step becomes unnecessary and can be deleted - everything else still
applies.
"""

from __future__ import annotations

from localizer.lexical_index import ChunkRecord


def chunks_from_code_map(
    visitor_chunks: list[dict],
    source: str,
    file_path: str,
) -> list[ChunkRecord]:
    """
    visitor_chunks: the .chunks list straight off a CodeMapVisitor instance.
    source: the full file source text (needed since CodeMapVisitor only
        gives line ranges, not content).
    file_path: repo-relative path, used to build chunk_id and tag each
        record's file_path field.
    """
    lines = source.splitlines()

    classes = [c for c in visitor_chunks if c["type"] == "class"]
    functions = [c for c in visitor_chunks if c["type"] in ("function", "async_function")]

    records: list[ChunkRecord] = []

    for cls in classes:
        chunk_id = f"{file_path}::{cls['name']}::{cls['start_line']}"
        content = "\n".join(lines[cls["start_line"] - 1 : cls["end_line"]])
        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                file_path=file_path,
                symbol_name=cls["name"],
                content=content,
            )
        )

    for fn in functions:
        owner = _find_owning_class(fn, classes)
        # symbol_name stays as the bare function/method name (matches
        # search_scoped's expectation of symbol_name == method name,
        # with the class looked up separately via scope_symbol).
        chunk_id = f"{file_path}::{owner + '.' if owner else ''}{fn['name']}::{fn['start_line']}"
        content = "\n".join(lines[fn["start_line"] - 1 : fn["end_line"]])
        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                file_path=file_path,
                symbol_name=fn["name"],
                content=content,
            )
        )

    return records


def _find_owning_class(function_chunk: dict, class_chunks: list[dict]) -> str | None:
    """
    A function's chunk is "owned" by a class if its line range falls
    entirely inside that class's line range. Picks the SMALLEST enclosing
    class range (i.e. the innermost), in case of nested classes - the
    immediate parent is the more useful scope for search_scoped, not an
    outer ancestor several levels up.
    """
    candidates = [
        c for c in class_chunks
        if c["start_line"] <= function_chunk["start_line"]
        and function_chunk["end_line"] <= c["end_line"]
    ]
    if not candidates:
        return None

    innermost = min(candidates, key=lambda c: c["end_line"] - c["start_line"])
    return innermost["name"]