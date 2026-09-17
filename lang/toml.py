"""TOML language module. A comment is '#' to end of line, never spans lines.
normalize() RAISES if the file does not parse.

Four string forms can hide a '#', so detection is a whole-document scanner,
not a regex or line loop: basic `"..."` (newline ends it -- can't cross
lines), literal `'...'` (no escapes), multi-line basic `\"\"\"...\"\"\"` and
multi-line literal `'''...'''` (both span lines; closing delimiter is found
by measuring a whole run of quotes, since up to two adjacent quotes can be
content: `\"\"\"a\"\"\"\"` is the string `a"`). Bare keys are `[A-Za-z0-9_-]`
and so contain neither '#' nor a quote, which is what makes the scanner
total: outside a string, '#' always opens a comment. Blank lines inside a
multi-line string are reported by protected_lines() so --collapse-blanks
leaves them alone.

normal form: the parsed document via tomllib, canonicalised (scalars tagged
with their type so `"1979-05-27"` the string can't equal the date; tables
key-sorted). Anything tomllib rejects raises and the file is skipped.
"""

from __future__ import annotations

import datetime as _datetime
import json
import re
import tomllib
from bisect import bisect_right
from functools import lru_cache

EXTS = ('.toml',)

KEEP = re.compile(
    r"""
    \#
    (?:
        :schema                                   # taplo / SchemaStore binding
      | \s* renovate \s* :                        # renovate custom-manager directive
      | \s* (?: vim | vi | ex ) :                 # editor modelines
      | \s* -\*-
      | .* (?:                                    # banners: anywhere in the line
              SPDX-[A-Za-z-]+
            | @generated
            | [Aa]utomatically \s+ [Gg]enerated
            | AUTOMATICALLY \s+ GENERATED
            | DO \s+ NOT \s+ EDIT
           )
    )
    """,
    re.VERBOSE,
)

class ParseError(ValueError):
    """The file is not a valid TOML document; no normal form can be trusted for it."""

def _line_starts(src: str) -> list[int]:
    starts = [0]
    i = src.find('\n')
    while i != -1:
        starts.append(i + 1)
        i = src.find('\n', i + 1)
    return starts

def _locate(starts: list[int], off: int) -> tuple[int, int]:
    row = bisect_right(starts, off) - 1
    return row + 1, off - starts[row]

def _skip_string(src: str, i: int, quote: str) -> tuple[int, bool]:
    """Index just past the string opening at `i`, and whether it was multi-line.

    `i` points at the opening quote character. Recovery is deliberate rather than
    strict: an unterminated single-line string stops at its newline and an
    unterminated multi-line string stops at EOF, so a malformed file yields a
    bounded misreading instead of swallowing the document. It is not stripped
    either way -- normalize() rejects it.
    """
    n = len(src)
    escapes = quote == '"'
    multi = src.startswith(quote * 3, i)
    j = i + (3 if multi else 1)

    while j < n:
        c = src[j]
        if not multi and c in '\r\n':
            return j, False
        if escapes and c == '\\':
            if not multi and j + 1 < n and src[j + 1] in '\r\n':
                return j + 1, False
            j += 2
            continue
        if c == quote:
            if not multi:
                return j + 1, False
            run = 1
            while j + run < n and src[j + run] == quote:
                run += 1
            if run >= 3:
                return j + (run if run <= 5 else 3), True
            j += run
            continue
        j += 1
    return n, multi

@lru_cache(maxsize=8)
def _scan(src: str) -> tuple[tuple[tuple[int, int, int, str], ...], frozenset[int]]:
    starts = _line_starts(src)
    found: list[tuple[int, int, int, str]] = []
    protected: set[int] = set()
    i, n = 0, len(src)

    while i < n:
        c = src[i]
        if c == '#':
            j = i
            while j < n and src[j] != '\n':
                j += 1
            end = j - 1 if j > i and src[j - 1] == '\r' else j
            text = src[i:end]
            line, col = _locate(starts, i)
            found.append((line, col, col + len(text), text))
            i = j
            continue
        if c in '"\'':
            end, multi = _skip_string(src, i, c)
            if multi:
                first, _ = _locate(starts, i)
                last, _ = _locate(starts, end - 1)
                protected.update(range(first, last + 1))
            i = max(end, i + 1)
            continue
        i += 1

    return tuple(found), frozenset(protected)

def comments(src: str) -> list[tuple[int, int, int, str]]:
    return list(_scan(src)[0])

def protected_lines(src: str) -> set[int]:
    """Lines whose blankness is string content, not formatting.

    A blank line inside a \"\"\"...\"\"\" or '''...''' string is part of that
    string's value -- an embedded script, a description, a key blob -- so
    collapsing it rewrites data rather than layout.
    """
    return set(_scan(src)[1])

def _canonical(value) -> str:
    """Deterministic rendering of a parsed TOML value, type included.

    Injective by construction: a table renders as `{...}`, an array as `[...]`,
    and every scalar as `<type>:<payload>`, so no two differently-typed or
    differently-shaped values can render alike. Table keys are sorted because
    TOML tables are unordered; array order is meaning and is kept.
    """
    if isinstance(value, dict):
        return '{%s}' % ','.join(
            '%s:%s' % (json.dumps(k, ensure_ascii=True), _canonical(v))
            for k, v in sorted(value.items())
        )
    if isinstance(value, list):
        return '[%s]' % ','.join(_canonical(v) for v in value)
    if isinstance(value, bool):
        return 'bool:%s' % ('true' if value else 'false')
    if isinstance(value, int):
        return 'int:%d' % value
    if isinstance(value, float):
        return 'float:%r' % value
    if isinstance(value, str):
        return 'str:%s' % json.dumps(value, ensure_ascii=True)
    if isinstance(value, (_datetime.datetime, _datetime.date, _datetime.time)):
        return '%s:%s' % (type(value).__name__, value.isoformat())
    raise ParseError('unexpected value type from tomllib: %s' % type(value).__name__)

def normalize(path: str) -> str:
    """Comment-free normal form; raises ParseError if `path` is not valid TOML.

    No fallback: a file we cannot prove is a file we do not strip. Fragments,
    templates and anything with a duplicate key are rejected here rather than
    stripped against a weaker form.
    """
    try:
        with open(path, 'rb') as fh:
            data = tomllib.load(fh)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ParseError('%s is not a valid TOML document: %s' % (path, exc)) from exc
    return _canonical(data)
