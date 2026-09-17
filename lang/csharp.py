"""C# language module: hand-rolled scanner (no C# parser/dotnet available here).

Comment forms: `//`, `/* */` (non-nesting), doc forms `///` and `/** */`.

NOT comments: preprocessor directives (`#region #if #define #pragma` etc. --
recognised only when `#` is first non-whitespace on the line) consume their
whole line opaquely, including any `//` in a `pp-message`. Any `//`/`/*` inside
a string form (`"..."`, `'...'`, verbatim `@"..."`, raw `\"\"\"...\"\"\"`, or
an interpolated `$"..."`) is content, not a comment -- interpolated `{...}` holes
are walked opaquely so nothing inside one is ever reported.

normalize() re-emits a record stream, one record per line: `C<code>` (tokens
space-joined), `S<json>` (one string/char literal, JSON-quoted), `P<text>`
(one preprocessor line, verbatim). Tokens not characters, because a comment
can only corrupt the file by fusing its neighbours (`a/*c*/b` -> `ab`); a
directive is always its own `P` record so it can't absorb surrounding code.
Line endings and the BOM are not part of the form (both whitespace here).
normalize() RAISES on anything that doesn't lex; the driver then skips the
file rather than strip it unproven.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from functools import lru_cache

EXTS = ('.cs',)

_KEEP_ANCHORED = (
    r"///(?!/)",
    r"/\*\*(?![*/])",
)

_KEEP_ANYWHERE = (
    r"</?auto-?generated",
    r"@generated",
    r"DO\s+NOT\s+EDIT",
    r"ReSharper\b",
    r"@formatter\s*:",
    r"\bnoinspection\b",
    r"\bncrunch\s*:",
    r"\blanguage\s*=",
    r"SPDX-[A-Za-z-]+",
    r"@license|@preserve|@copyright",
    r"\bcopyright\b",
    r"\blicen[cs]ed\s+under\b",
    r"^//+\s*(?:vim|vi|ex)\s*:",
    r"-\*-",
)

KEEP = re.compile(
    r"(?:%s)|[\s\S]*?(?:%s)"
    % ("|".join(_KEEP_ANCHORED), "|".join(_KEEP_ANYWHERE)),
    re.IGNORECASE,
)

_BOM = '\ufeff'

_NEWLINES = '\n\u2028\u2029\u0085'
_LINE_END = _NEWLINES + '\r'

_PSEUDO_BREAKS = '\v\f\x1c\x1d\x1e'

class ParseError(ValueError):
    """The file does not lex as C#; no normal form can be trusted for it."""

def _line_starts(src: str) -> list[int]:
    """Offsets of each line start, numbered exactly as the driver numbers lines.

    Derived from str.splitlines() rather than from `\\n` alone so that the
    reported (line, col) can never disagree with the driver's excision, which
    indexes into `src.splitlines(keepends=True)`.
    """
    starts, at = [0], 0
    for piece in src.splitlines(keepends=True):
        at += len(piece)
        starts.append(at)
    return starts

def _locate(starts: list[int], off: int) -> tuple[int, int]:
    row = bisect_right(starts, off) - 1
    return row + 1, off - starts[row]

def _end_of_line(src: str, i: int) -> int:
    """Offset of the first C# line terminator at or after `i`, else len(src)."""
    n = i
    while n < len(src) and src[n] not in _LINE_END:
        n += 1
    return n

def _run_of(src: str, i: int, ch: str) -> int:
    """Length of the run of `ch` starting at `i`."""
    n, j = len(src), i
    while j < n and src[j] == ch:
        j += 1
    return j - i

def _prefix(src: str, i: int):
    """Parse a string literal's prefix at `i` -> (quote_pos, dollars, verbatim), or None.

    `@` is also the identifier escape (`@class`, `@this`), and `$`/`@` in any
    other position is not legal C#, so anything not ending at a quote is code.
    """
    n, j = len(src), i
    dollars, verbatim = 0, False
    while j < n:
        ch = src[j]
        if ch == '$':
            dollars += 1
        elif ch == '@' and not verbatim:
            verbatim = True
        else:
            break
        j += 1
    if j < n and src[j] == '"':
        return j, dollars, verbatim
    return None

def _read_char(src: str, i: int) -> int:
    """End offset (exclusive) of the char literal opening at `i`."""
    n, j = len(src), i + 1
    while j < n:
        c = src[j]
        if c == '\\':
            j += 2
            continue
        if c in _LINE_END:
            raise ParseError('unterminated character literal')
        if c == "'":
            return j + 1
        j += 1
    raise ParseError('unterminated character literal')

def _read_hole(src: str, j: int, dollars: int, raw: bool) -> int:
    """End offset (exclusive) of an interpolation hole whose brace(s) end at `j`.

    A hole is code: it can nest parentheses, brackets, braces, strings, char
    literals and comments. A `:` at the hole's own top level opens the format
    specifier, which is literal text to the closing brace -- that is why the C#
    conditional operator has to be parenthesised inside an interpolation, and
    why scanning it as code would misread `$"{d:yyyy//MM}"`.
    """
    n = len(src)
    depth = 0
    fmt = False
    while j < n:
        c = src[j]

        if c == '}':
            if not fmt and depth > 0:
                depth -= 1
                j += 1
                continue
            run = _run_of(src, j, '}')
            if raw:
                if run >= dollars:
                    return j + dollars
                j += run
                continue
            if fmt and run >= 2:
                j += 2
                continue
            return j + 1

        if fmt:
            j += 1
            continue

        if c in '([{':
            depth += 1
            j += 1
            continue
        if c in ')]':
            depth = max(0, depth - 1)
            j += 1
            continue
        if c == ':' and depth == 0:
            if j + 1 < n and src[j + 1] == ':':
                j += 2
                continue
            fmt = True
            j += 1
            continue
        if c == '/' and j + 1 < n and src[j + 1] == '/':
            j = _end_of_line(src, j)
            continue
        if c == '/' and j + 1 < n and src[j + 1] == '*':
            end = src.find('*/', j + 2)
            if end == -1:
                raise ParseError('unterminated /* comment inside an interpolation')
            j = end + 2
            continue
        if c == "'":
            j = _read_char(src, j)
            continue
        if c in '"$@' and _prefix(src, j) is not None:
            j = _read_string(src, j)
            continue
        j += 1
    raise ParseError('unterminated interpolation hole')

def _read_string(src: str, i: int) -> int:
    """End offset (exclusive) of the string literal starting at `i`."""
    pre = _prefix(src, i)
    if pre is None:
        raise ParseError('not a string literal')
    q, dollars, verbatim = pre
    n = len(src)
    run = _run_of(src, q, '"')

    if not verbatim and run >= 3:
        return _read_raw(src, q + run, run, dollars)
    if verbatim:
        return _read_verbatim(src, q + 1, dollars)
    if run >= 2:
        return q + 2
    return _read_regular(src, q + 1, dollars)

def _read_regular(src: str, j: int, dollars: int) -> int:
    n = len(src)
    while j < n:
        c = src[j]
        if c == '\\':
            j += 2
            continue
        if c in _LINE_END:
            raise ParseError('unterminated string literal')
        if c == '"':
            return j + 1
        if dollars and c in '{}':
            if j + 1 < n and src[j + 1] == c:
                j += 2
                continue
            if c == '{':
                j = _read_hole(src, j + 1, dollars, raw=False)
                continue
        j += 1
    raise ParseError('unterminated string literal')

def _read_verbatim(src: str, j: int, dollars: int) -> int:
    """`@"..."`: no backslash escapes, `""` is one quote, newlines are legal."""
    n = len(src)
    while j < n:
        c = src[j]
        if c == '"':
            if j + 1 < n and src[j + 1] == '"':
                j += 2
                continue
            return j + 1
        if dollars and c in '{}':
            if j + 1 < n and src[j + 1] == c:
                j += 2
                continue
            if c == '{':
                j = _read_hole(src, j + 1, dollars, raw=False)
                continue
        j += 1
    raise ParseError('unterminated verbatim string literal')

def _read_raw(src: str, j: int, fence: int, dollars: int) -> int:
    """`\"\"\"...\"\"\"` with a fence of `fence` quotes; no escapes at all.

    Content may contain a run of quotes shorter than the fence. A run longer
    than the fence is a compile error (the delimiter may not be followed by a
    quote), so it raises rather than guess which quotes were meant to close.
    """
    n = len(src)
    while j < n:
        c = src[j]
        if c == '"':
            run = _run_of(src, j, '"')
            if run > fence:
                raise ParseError('raw string literal closed by %d quotes, fence is %d'
                                 % (run, fence))
            if run == fence:
                return j + fence
            j += run
            continue
        if dollars and c == '{':
            run = _run_of(src, j, '{')
            if run >= dollars:
                j = _read_hole(src, j + run, dollars, raw=True)
                continue
            j += run
            continue
        j += 1
    raise ParseError('unterminated raw string literal')

_OPERATORS = frozenset((
    '>>>=',
    '<<=', '>>=', '>>>', '??=',
    '++', '--', '&&', '||', '->', '==', '!=', '<=', '>=', '+=', '-=', '*=',
    '/=', '%=', '&=', '|=', '^=', '<<', '>>', '??', '?.', '::', '=>', '..',
    '{', '}', '[', ']', '(', ')', '.', ',', ':', ';', '+', '-', '*', '/',
    '%', '&', '|', '^', '!', '~', '=', '<', '>', '?', '@', '$', '#', '\\',
))

def _read_token(src: str, i: int) -> int:
    """End offset (exclusive) of the ordinary token starting at `i`.

    Ordinary means: not a comment, string, char literal or directive -- those
    are recognised by the caller first. A token can never span a comment,
    because no character of one (`/`, `*`) continues an identifier or a number
    and no C# operator but `/` and `/=` contains a slash, both of which the
    caller has already ruled out.
    """
    n = len(src)
    c = src[i]

    if c.isdigit():
        j, hexish = i, src[i:i + 2].lower() in ('0x', '0b')
        while j < n:
            ch = src[j]
            if ch.isalnum() or ch == '_':
                j += 1
            elif ch == '.' and j + 1 < n and src[j + 1].isdigit():
                j += 1
            elif ch in '+-' and not hexish and j > i and src[j - 1] in 'eE':
                j += 1
            else:
                break
        return j

    if c.isalpha() or c == '_' or (c == '@' and i + 1 < n and (src[i + 1].isalpha() or src[i + 1] == '_')):
        j = i + 1
        while j < n and (src[j].isalnum() or src[j] == '_'):
            j += 1
        return j

    for width in (4, 3, 2, 1):
        if src[i:i + width] in _OPERATORS:
            return i + width
    return i + 1

@lru_cache(maxsize=8)
def _scan(src: str):
    """One pass -> (comments, protected lines, normal-form records).

    Every consumer goes through here, so the comment inventory, the protected
    set and the proof can never be computed from disagreeing views of the file.
    """
    n = len(src)
    for ch in _PSEUDO_BREAKS:
        if ch in src:
            raise ParseError('file contains %r, which the driver counts as a line '
                             'break and C# does not' % ch)
    starts = _line_starts(src)
    found: list[tuple[int, int, int, str]] = []
    protected: set[int] = set()
    records: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            records.append('C' + ' '.join(buf))
            del buf[:]

    i = 0
    line_start = True
    while i < n:
        c = src[i]

        if c in _NEWLINES:
            i += 1
            line_start = True
            continue
        if c.isspace() or c == _BOM:
            i += 1
            continue

        if c == '#' and line_start:
            j = _end_of_line(src, i)
            flush()
            records.append('P' + src[i:j].rstrip())
            i = j
            continue

        line_start = False

        if c == '/' and i + 1 < n and src[i + 1] == '/':
            j = _end_of_line(src, i)
            text = src[i:j]
            line, col = _locate(starts, i)
            found.append((line, col, col + len(text), text))
            i = j
            continue

        if c == '/' and i + 1 < n and src[i + 1] == '*':
            end = src.find('*/', i + 2)
            if end == -1:
                raise ParseError('unterminated /* comment')
            end += 2
            line, col = _locate(starts, i)
            _, end_col = _locate(starts, end)
            found.append((line, col, end_col, src[i:end]))
            i = end
            continue

        if c == "'":
            end = _read_char(src, i)
        elif c in '"$@' and _prefix(src, i) is not None:
            end = _read_string(src, i)
        else:
            j = _read_token(src, i)
            buf.append(src[i:j])
            i = j
            continue

        flush()
        records.append('S' + json.dumps(src[i:end]))
        first, _ = _locate(starts, i)
        last, _ = _locate(starts, end - 1)
        if last > first:
            protected.update(range(first, last + 1))
        i = end

    flush()
    return tuple(found), frozenset(protected), '\n'.join(records)

def comments(src: str) -> list[tuple[int, int, int, str]]:
    return list(_scan(src)[0])

def protected_lines(src: str) -> set[int]:
    """Lines whose blankness is string content, not formatting.

    A blank line inside a verbatim `@"..."` or a raw `\"\"\"...\"\"\"` literal
    -- including their interpolated forms -- is part of the string's value: an
    embedded JSON blob, a shader, a query. Collapsing it rewrites data. Regular
    and interpolated non-verbatim strings cannot span lines at all (the scanner
    rejects the file if one appears to), so they contribute nothing here.
    """
    return set(_scan(src)[1])

def normalize(path: str) -> str:
    """Comment-free normal form; raises ParseError if `path` does not lex.

    No fallback: a file we cannot prove is a file we do not strip.
    """
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    return _scan(src)[2]
