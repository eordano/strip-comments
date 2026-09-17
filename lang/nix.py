"""Nix language module. KEEP is None: the `# shellcheck` lines that matter here
live inside embedded shell scripts (''...'' string content), never comments.

    #  ... EOL      line comment          /* ... */   block comment (no nesting)
    //               NOT a comment -- attribute-update operator
    "..."            string; ${...} interpolation holds code
    ''...''          indented string; ${...} interpolation holds code

Interpolations are tracked with a mode stack, so a '#' inside ${...} inside a
string is a real comment while a '#' in the surrounding string text is not.

normal form: `nix-instantiate --parse` (AST, comments already discarded). It
resolves relative paths against the file's own directory, so before/after
comparisons must be taken at the same path.
"""

from __future__ import annotations

import bisect
import subprocess
from functools import lru_cache

EXTS = ('.nix',)

KEEP = None

class ParseError(ValueError):
    """The file does not parse; no normal form can be trusted for it."""

def _line_starts(src: str) -> list[int]:
    starts = [0]
    i = src.find('\n')
    while i != -1:
        starts.append(i + 1)
        i = src.find('\n', i + 1)
    return starts

def _locate(starts: list[int], off: int) -> tuple[int, int]:
    row = bisect.bisect_right(starts, off) - 1
    return row + 1, off - starts[row]

@lru_cache(maxsize=8)
def _scan(src: str) -> tuple[tuple[tuple[int, int, int, str], ...], frozenset[int]]:
    starts = _line_starts(src)
    found = []
    spans: list[tuple[int, int]] = []
    stack = ['code']
    open_at = None
    i, n = 0, len(src)

    def record(a: int, b: int) -> None:
        line, col = _locate(starts, a)
        found.append((line, col, _locate(starts, b)[1], src[a:b]))

    while i < n:
        ctx, c = stack[-1], src[i]

        if ctx == 'code':
            if c == '}' and len(stack) > 1:
                stack.pop()
                i += 1
                open_at = i
                continue
            if c == '#':
                j = i
                while j < n and src[j] != '\n':
                    j += 1
                record(i, j)
                i = j
                continue
            if c == '/' and i + 1 < n and src[i + 1] == '*':
                j = i + 2
                while j + 1 < n and not (src[j] == '*' and src[j + 1] == '/'):
                    j += 1
                j = min(j + 2, n)
                record(i, j)
                i = j
                continue
            if c == "'" and i + 1 < n and src[i + 1] == "'":
                stack.append('ind')
                i += 2
                open_at = i
                continue
            if c == '"':
                stack.append('str')
                i += 1
                open_at = i
                continue
            i += 1
            continue

        if ctx == 'str':
            if c == '\\' and i + 1 < n:
                i += 2
                continue
            if c == '"':
                spans.append((open_at, i))
                stack.pop()
                i += 1
                open_at = None
                continue
        else:
            if c == "'" and i + 2 < n and src[i + 1] == "'" and src[i + 2] in "'$\\":
                i += 3
                continue
            if c == "'" and i + 1 < n and src[i + 1] == "'":
                spans.append((open_at, i))
                stack.pop()
                i += 2
                open_at = None
                continue

        if c == '$' and i + 1 < n and src[i + 1] == '{':
            spans.append((open_at, i))
            stack.append('code')
            i += 2
            continue
        i += 1

    if open_at is not None:
        spans.append((open_at, n))

    protected = set()
    for a, b in spans:
        k = src.find('\n', a)
        while k != -1 and k < b:
            protected.add(_locate(starts, k)[0])
            k = src.find('\n', k + 1)
    return tuple(found), frozenset(protected)

def comments(src: str) -> list[tuple[int, int, int, str]]:
    return list(_scan(src)[0])

def protected_lines(src: str) -> set[int]:
    """Lines whose blankness is string content, not formatting.

    A blank line inside a ''...'' (or a multi-line "...") is string CONTENT:
    dropping it rewrites the embedded script the string carries. Lines inside a
    ${...} interpolation are code again, and are not protected.
    """
    return set(_scan(src)[1])

def normalize(path: str) -> str:
    """Comment-free normal form; raises ParseError if `path` does not parse.

    No fallback: a file we cannot prove is a file we do not strip.
    """
    proc = subprocess.run(['nix-instantiate', '--parse', path],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise ParseError('nix-instantiate could not parse %s: %s'
                         % (path, (proc.stderr.strip() or 'exit %d' % proc.returncode)))
    return proc.stdout
