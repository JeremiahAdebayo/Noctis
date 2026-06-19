"""
Stack-trace fast path for Noctis localization.

When a bug report includes a Python traceback, it names the file (and
often the exact line) where the failure occurred — no retrieval needed
at all. This is the strongest possible signal: not "probably relevant"
but "this frame executed during the failure."

Determinism: parsing a fixed traceback string always yields the same
frames. The only judgment call is which frame(s) to trust as the fix
location (see _select_relevant_frames below) — that policy is explicit
and fixed, not learned or sampled.

Use as a pre-filter: if this returns non-empty, treat it as a strong
prior and feed it into fusion as a heavily-weighted candidate set, or
short-circuit straight to it if your repo's own files appear in the
trace (see is_repo_internal_frame).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TraceFrame:
    file_path: str       # as it appeared in the traceback (may be absolute or relative)
    line_number: int
    function_name: str | None
    code_line: str | None    # the source line shown under the frame, if present


# Matches standard CPython traceback frame lines:
#   File "src/click/types.py", line 1161, in convert
_FRAME_PATTERN = re.compile(
    r'File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>\S+))?'
)

# Matches the optional source-line echo directly under a frame, e.g.:
#     if self.executable and not os.access(value, os.X_OK):
_CODE_LINE_PATTERN = re.compile(r"^\s{4}(?P<code>\S.*)$")


def extract_trace_frames(text: str) -> list[TraceFrame]:
    """
    Parses every Python-style traceback frame found in free text. Returns
    frames in the order they appear (which, for a real traceback, is
    outermost-call-first to innermost-failure-last — i.e. the LAST frame
    is almost always where the actual fix belongs).
    """
    lines = text.splitlines()
    frames: list[TraceFrame] = []

    for i, line in enumerate(lines):
        match = _FRAME_PATTERN.search(line)
        if not match:
            continue

        code_line = None
        if i + 1 < len(lines):
            code_match = _CODE_LINE_PATTERN.match(lines[i + 1])
            if code_match:
                code_line = code_match.group("code")

        frames.append(
            TraceFrame(
                file_path=match.group("file"),
                line_number=int(match.group("line")),
                function_name=match.group("func"),
                code_line=code_line,
            )
        )

    return frames


def is_repo_internal_frame(frame: TraceFrame, repo_path: str) -> bool:
    """
    Distinguishes frames inside the target repo from frames in stdlib,
    site-packages, unrelated dependencies, or unrelated calling code
    outside the repo entirely (e.g. the user's own script that invoked
    a library function inside the repo).

    Policy: default to EXCLUDED. A frame is only treated as repo-internal
    when its path positively contains the repo_path segment. This is
    deliberately conservative — a frame from some other project's file
    that doesn't happen to match a known stdlib/site-packages pattern
    must not be treated as "probably ours" by default, since that
    produces false localization targets outside the actual repo.
    """
    normalized_repo = repo_path.rstrip("/").replace("\\", "/")
    normalized_frame = frame.file_path.replace("\\", "/")
    return normalized_repo in normalized_frame


def _normalize_to_repo_relative(trace_file_path: str, repo_path: str) -> str:
    """
    Stack trace file paths can be absolute (e.g.
    "/home/user/proj/src/click/types.py") or already repo-relative (e.g.
    "src/click/types.py", which is what file_registry keys use).

    If the repo_path segment appears in the trace path, return everything
    FROM that segment onward (inclusive) — this is the repo-relative path
    matching file_registry's convention. If repo_path isn't found at all,
    return the trace path unchanged (caller should already have filtered
    via is_repo_internal_frame before reaching this point, so this is a
    defensive fallback, not the primary path).
    """
    normalized_repo = repo_path.rstrip("/").replace("\\", "/")
    normalized_trace = trace_file_path.replace("\\", "/")

    idx = normalized_trace.find(normalized_repo)
    if idx == -1:
        return normalized_trace

    return normalized_trace[idx:]


def select_fix_candidates(
    text: str, repo_path: str, max_candidates: int = 3
) -> list[TraceFrame]:
    """
    Returns the most likely fix-location frames, innermost-first (i.e.
    closest to where the exception actually originated), filtered to
    repo-internal files only.

    Policy, explicit and fixed:
      1. Parse all frames.
      2. Keep only repo-internal frames (drop stdlib/site-packages noise).
      3. Reverse order so the deepest/most-recent frame comes first -
         this is usually where the actual fix belongs, since it's the
         frame that was executing when things went wrong.
      4. Cap at max_candidates; the planner should still verify by reading
         the actual file content, this is a prioritized candidate list,
         not an auto-commit.
    """
    all_frames = extract_trace_frames(text)
    internal_frames = [f for f in all_frames if is_repo_internal_frame(f, repo_path)]
    return list(reversed(internal_frames))[:max_candidates]