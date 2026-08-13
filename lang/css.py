"""CSS language module (CSS Syntax Level 3 comment/token rules).

`/* ... */` is the ONLY comment form, non-nesting. `//` is NOT a comment (two
delim tokens) -- it's also just the middle of `https://`. A `url(...)` whose
ident is `url` (case-insensitive, escapes decoded) and not immediately quoted
is ONE url-token up to `)`, so a `/*` inside `url(http://h/a/*.png)` or inside
base64 data is content, never a comment; same for `/*` inside a `"..."` string.

normalize() re-emits the source with comments dropped and whitespace runs
outside a string/url collapsed to ONE SPACE, not nothing: a comment is a token
separator in CSS (`a/**/b` is two idents, `ab` is one), so a strip that fused
two tokens must still show a moved normal form. It errs toward refusing: a
comment touching code on only one side (`margin: 0/* why */;`) loses a
separator and reverts rather than risk a wrong call. normalize() RAISES on an
unterminated comment/string/url-token or unbalanced braces; the driver then
skips the file rather than strip it unproven.
"""

from __future__ import annotations

import bisect
import re
from functools import lru_cache

EXTS = ('.css',)

# Directives, not prose: cutting these changes minifier output, source maps or
# lint results. `/*!` is the bang comment every minifier preserves; `/*#` carries
# the source map link. The rest are matched anywhere in the body.
_KEEP_SIGNALS = (
    r"stylelint-(?:disable|enable)",         # incl. -line / -next-line variants
    r"postcss-[a-z0-9-]+",                   # postcss-nested, postcss-preset-env, ...
    r"\bpostcss\s*:",                        # /* postcss: ... */
    r"autoprefixer\s*[a-z-]*\s*:",           # autoprefixer: off / autoprefixer grid: on
    r"prettier-ignore",
    r"\bcsslint\b",
    r"source(?:Mapping)?URL",
    r"@(?:license|preserve|copyright|cc_on)\b",
    r"SPDX-[A-Za-z-]+",
)
# KEEP is applied by the driver as `KEEP.match(text.strip())`, i.e. anchored at the
# comment's opening `/*`, so every alternative spells out how it gets there.
KEEP = re.compile(
    r"^/\*(?:"
    r"!"                                     # /*! bang comment
    r"|\#"                                   # /*# sourceMappingURL=... */
    r"|(?:(?!\*/).)*?(?:%s)" % "|".join(_KEEP_SIGNALS) +
    r")",
    re.DOTALL | re.IGNORECASE,
)

_WS = ' \t\n\r\f'
_HEX = '0123456789abcdefABCDEF'

# The driver addresses comments as (line, col) into `src.splitlines(keepends=True)`
# and derives the last line as `line + text.count('\n')`. Both of those are only
# true while the file's line structure IS its `\n` structure. `str.splitlines`
# also breaks on these, so one of them anywhere in the file shifts every later
# line number out from under `excise()` -- which then cuts the named columns out
# of the WRONG line: seen either as a revert, or (when the mis-cut lands in a
# whitespace run) as a silent "stripped" that deleted a blank line and left the
# comment in place. CSS itself calls \f and \r newlines (Syntax L3 §3.3
# preprocessing), so this is not merely exotic. No coordinate system can be
# agreed here, so the file is refused.
_ALIEN_BREAK = re.compile(r'[\v\f\x1c\x1d\x1e\x85  ]|\r(?!\n)')


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


def _ident_start(src: str, i: int) -> bool:
    """True if a name token can begin at `i`: letter, '_', '-', non-ASCII, or an escape."""
    c = src[i]
    if c.isalpha() or c == '_' or c == '-' or ord(c) >= 0x80:
        return True
    return c == '\\' and i + 1 < len(src) and src[i + 1] != '\n'


def _end_of_escape(src: str, i: int) -> int:
    """Offset just past the valid escape starting at `src[i] == '\\\\'`.

    A hex escape swallows up to six hex digits AND one optional trailing
    whitespace character, which is part of the escape rather than a separator:
    `\\41 B` is the two-character ident "AB" while `\\41  B` is "A" then "B". That
    single space must therefore survive verbatim into the normal form.
    """
    n = len(src)
    j = i + 1
    if j >= n:
        return n
    if src[j] in _HEX:
        k = j
        while k < n and k - j < 6 and src[k] in _HEX:
            k += 1
        if k < n and src[k] in _WS:
            k += 2 if src[k] == '\r' and k + 1 < n and src[k + 1] == '\n' else 1
        return k
    return j + 1


def _ident_value(name: str) -> str:
    """The ident's VALUE, escapes decoded, as "consume an ident sequence" yields it.

    Only `url` needs this, and it needs it badly: the spec compares the DECODED
    value, so `\\75 rl(http://h/a/*.png)` is a url-token whose `/*` is content.
    Reading the raw spelling instead saw a comment there, cut from inside the URL
    to the next `*/` anywhere in the file, and -- because the same misreading
    produced the normal form -- passed the equality gate.
    """
    if '\\' not in name:
        return name
    out: list[str] = []
    i, n = 0, len(name)
    while i < n:
        c = name[i]
        if c != '\\':
            out.append(c)
            i += 1
            continue
        j = i + 1
        if j >= n:
            out.append('�')
            break
        if name[j] in _HEX:
            k = j
            while k < n and k - j < 6 and name[k] in _HEX:
                k += 1
            cp = int(name[j:k], 16)
            out.append(chr(cp) if 0 < cp <= 0x10FFFF and not 0xD800 <= cp <= 0xDFFF
                       else '�')
            if k < n and name[k] in _WS:
                k += 2 if name[k] == '\r' and k + 1 < n and name[k + 1] == '\n' else 1
            i = k
        else:
            out.append(name[j])
            i = j + 1
    return ''.join(out)


def _end_of_name(src: str, i: int) -> int:
    """Offset just past the ident/name token starting at `i`."""
    n = len(src)
    while i < n:
        c = src[i]
        if c == '\\' and i + 1 < n and src[i + 1] != '\n':
            i = _end_of_escape(src, i)
        elif c.isalnum() or c in '_-' or ord(c) >= 0x80:
            i += 1
        else:
            break
    return i


def _end_of_string(src: str, i: int, strict: bool) -> int:
    """Offset just past the string starting at the quote `src[i]`.

    A raw newline ends a CSS string (the spec's bad-string-token); a backslash
    before it is a line continuation and keeps the string open.
    """
    quote, n = src[i], len(src)
    j = i + 1
    while j < n:
        c = src[j]
        if c == '\\':
            if j + 1 >= n:
                break
            j += 3 if src[j + 1] == '\r' and j + 2 < n and src[j + 2] == '\n' else 2
            continue
        if c == quote:
            return j + 1
        if c == '\n':
            break
        j += 1
    if strict:
        raise ParseError('unterminated string opened at offset %d' % i)
    return min(j + 1, n)


@lru_cache(maxsize=8)
def _scan(src: str, strict: bool) -> tuple[tuple, frozenset, str]:
    """One pass over `src` yielding (comments, protected lines, normal form).

    `strict` decides only what happens on malformed input -- raise (normalize, so
    the file is skipped) or salvage (comments/protected_lines, so `list` still
    shows what it can). The tokenisation itself is identical either way, which is
    what keeps the inventory and the proof from ever disagreeing.
    """
    if strict:
        m = _ALIEN_BREAK.search(src)
        if m:
            raise ParseError('line break %r at offset %d that str.splitlines() '
                             'counts and this module cannot address'
                             % (m.group(), m.start()))
    starts = _line_starts(src)
    found: list[tuple[int, int, int, str]] = []
    protected: set[int] = set()
    out: list[str] = []
    depth = 0
    i, n = 0, len(src)

    def emit(text: str) -> None:
        out.append(text)

    def sep() -> None:
        # A comment and the whitespace around it collapse to ONE separator.
        if out and out[-1] != ' ':
            out.append(' ')

    def span_lines(a: int, b: int) -> None:
        k = src.find('\n', a)
        while k != -1 and k < b:
            protected.add(_locate(starts, k)[0] + 1)
            k = src.find('\n', k + 1)

    while i < n:
        c = src[i]

        if c == '/' and src[i + 1:i + 2] == '*':
            end = src.find('*/', i + 2)
            if end == -1:
                if strict:
                    raise ParseError('unterminated /* opened at line %d'
                                     % _locate(starts, i)[0])
                end = n
            else:
                end += 2
            line, col = _locate(starts, i)
            found.append((line, col, _locate(starts, end)[1], src[i:end]))
            sep()
            i = end
            continue

        if c in _WS:
            while i < n and src[i] in _WS:
                i += 1
            sep()
            continue

        if c == '"' or c == "'":
            end = _end_of_string(src, i, strict)
            emit(src[i:end])
            span_lines(i, end)
            i = end
            continue

        if _ident_start(src, i):
            end = _end_of_name(src, i)
            name = src[i:end]
            if _ident_value(name).lower() == 'url' and src[end:end + 1] == '(':
                k = end + 1
                while k < n and src[k] in _WS:
                    k += 1
                if k < n and src[k] in '"\'':
                    emit(name)          # url("...") is an ordinary function call
                    emit('(')
                    i = end + 1
                    continue
                while k < n and src[k] != ')':
                    k = _end_of_escape(src, k) if src[k] == '\\' else k + 1
                if k >= n:
                    if strict:
                        raise ParseError('unterminated url( at line %d'
                                         % _locate(starts, i)[0])
                    k = n - 1
                emit(src[i:k + 1])      # verbatim: a url-token has no inner syntax
                span_lines(i, k + 1)
                i = k + 1
                continue
            emit(name)
            i = end
            continue

        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth < 0:
                if strict:
                    raise ParseError('unbalanced "}" at line %d' % _locate(starts, i)[0])
                depth = 0
        emit(c)
        i += 1

    if strict and depth != 0:
        raise ParseError('%d unclosed "{"' % depth)
    if out and out[-1] == ' ':
        out.pop()
    if out and out[0] == ' ':
        out.pop(0)
    return tuple(found), frozenset(protected), ''.join(out)


def comments(src: str) -> list[tuple[int, int, int, str]]:
    return list(_scan(src, False)[0])


def protected_lines(src: str) -> set[int]:
    """Interior lines of a multi-line string or url-token.

    In VALID CSS this set can never actually contain a blank line, and saying so
    is more useful than implying otherwise: a raw newline ends a string, so the
    only multi-line string is the `\\<newline>` continuation form, and every line
    of one carries a trailing `\\` -- never blank. A url-token likewise may not
    contain whitespace.

    It is still reported, because `--collapse-blanks` is the trap this contract
    exists for and the lenient scan (used by `list`) does accept the malformed
    shapes a strict CSS parser rejects. Costing nothing and being right if CSS
    ever grows a multi-line literal beats being empty on purpose.
    """
    return set(_scan(src, False)[1])


def normalize(path: str) -> str:
    """Comment-free normal form; raises ParseError if `path` does not parse.

    No fallback: a file we cannot prove is a file we do not strip.
    """
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    return _scan(src, True)[2]
