"""Go language module. normalize() RAISES if the file does not lex.

    // ... EOL              line comment
    /* ... */               general comment -- does NOT nest
    "..."                   interpreted string, backslash escapes, no newline
    `...`                   raw string, no escapes, may span lines
    'x'  '\\n'  '\\u12ab'     rune literal, backslash escapes, no newline

NEWLINES ARE TOKENS. Go inserts a semicolon after a line's final token when
that token could end a statement, so `x++\\n)` and `x++)` are different
programs and so are `return\\nx` and `return x`. A general comment that spans
lines acts as a newline; one that does not acts as a space. The normal form
therefore keeps ONE newline for every whitespace/comment run that contained a
newline and one space otherwise, so a strip that removed a line break shows up
as a moved normal form. Spaces are dropped next to `( ) [ ] { }` since no Go
token contains those, which lets `f(/*c*/x)` strip to `f(x)`.

CGO PREAMBLES ARE C SOURCE. The comment group directly above `import "C"` is
compiled by the C toolchain; to a Go lexer it is just a comment. Those comments
are not reported by comments() at all (so nothing can name them) and are kept
byte for byte in the normal form, so a scanner bug that lost one still reverts.
For the grouped form `import (\n "C"\n)` cgo takes the comment above `"C"`, or
the one above `import (` when "C" is the only spec; both are protected (for a
multi-spec group the second is merely conservative).

Directives (`//go:build`, `//go:embed`, `//go:generate`, `//line`, `//export`,
`// +build`, `//nolint`, `//sys`) and example `// Output:` blocks are KEEP and
also emitted verbatim in the normal form as a second line of defence.

normal form: own scanner re-emits the source with plain comments dropped,
strings/runes/KEEP comments kept byte for byte, and whitespace runs collapsed
to one newline or one space as above.

CRLF files: reads are universal-newline, so a CRLF file and its LF twin
normalize identically; the driver writes a uniformly-CRLF file back as CRLF.
"""

from __future__ import annotations

import bisect
import re
from functools import lru_cache

EXTS = ('.go',)

_KEEP_ALTERNATIVES = (
    r"//go:",
    r"//line ",
    r"//export ",
    r"//extern ",
    r"//sys(?:nb)?\b",
    r"//nolint",
    r"//lint:",
    r"//\s*\+build",
    r"//\s*(?:Unordered output|Output):",
    r"//\s*Code generated .*DO NOT EDIT",
    r"/[/*][^\n]*SPDX-[A-Za-z-]+",
    r"/[/*][^\n]*@(?:license|preserve|copyright|generated)\b",
    r"//\s*(?:vim?|ex|emacs)\s*:",
    r"//\s*-\*-",
)
KEEP = re.compile("|".join("(?:%s)" % p for p in _KEEP_ALTERNATIVES), re.IGNORECASE)

_WS = frozenset(' \t\n\r')
_INTERESTING = re.compile(r'[ \t\n\r"\'`/]')
_DELIMS = frozenset('()[]{}')
_IMPORT_BEFORE = re.compile(r'(?:(?:^|[^\w])import\s*$)|^\s*$')
_IMPORT_GROUP = re.compile(r'(?:^|[^\w])(import)\s*\(\s*\Z')


class ParseError(ValueError):
    """The file does not lex; no normal form can be trusted for it."""


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


def _block_end(src: str, i: int) -> int:
    j = src.find('*/', i + 2)
    if j < 0:
        raise ParseError('unterminated general comment opened at offset %d' % i)
    return j + 2


def _quoted_end(src: str, i: int, quote: str) -> int:
    """End offset of the `"` string or `'` rune opening at `i`; neither may span a line."""
    n = len(src)
    j = i + 1
    while j < n:
        c = src[j]
        if c == '\\':
            j += 2
            continue
        if c == quote:
            return j + 1
        if c == '\n':
            break
        j += 1
    raise ParseError('unterminated %s literal at offset %d'
                     % ('string' if quote == '"' else 'rune', i))


def _raw_end(src: str, i: int) -> int:
    j = src.find('`', i + 1)
    if j < 0:
        raise ParseError('unterminated raw string literal at offset %d' % i)
    return j + 1


def _preamble(cmts: list[tuple[int, int, int, int, str]], c_lines: list[int]) -> set[int]:
    """Indices into cmts of every comment in a group ending right above `import "C"`.

    A group is comments on consecutive lines with nothing but whitespace between
    them; a blank line or code ends it. cgo takes the whole group as C source.
    """
    keep: set[int] = set()
    for c_line in c_lines:
        idx = [k for k, c in enumerate(cmts) if c[3] == c_line - 1]
        if not idx:
            continue
        k = idx[-1]
        keep.add(k)
        while k > 0 and cmts[k - 1][3] >= cmts[k][2] - 1:
            k -= 1
            keep.add(k)
    return keep


@lru_cache(maxsize=8)
def _scan(src: str) -> tuple[tuple[tuple[int, int, int, str], ...], frozenset[int], str]:
    """One pass yielding (comments, protected lines, normal form).

    The comment list and the normal form come from the same walk, so the gate
    can never be proving a different set of spans than the driver is cutting.
    """
    starts = _line_starts(src)
    n = len(src)
    # (off, end, line, endLine, text) for every comment, then filtered for cgo.
    raw_cmts: list[tuple[int, int, int, int, str]] = []
    # Every piece of the normal form in order: ('code', text) | ('sep', ' '|'\n')
    # | ('cmt', index) -- comments are resolved to drop/verbatim once cgo is known.
    parts: list[tuple[str, object]] = []
    verbatim: list[tuple[int, int]] = []
    c_lines: list[int] = []
    i = 0
    pending = 0

    def flush(k: int) -> None:
        if k > pending:
            parts.append(('code', src[pending:k]))

    while True:
        m = _INTERESTING.search(src, i)
        if m is None:
            break
        k = m.start()
        c = src[k]

        if c in _WS:
            j = k + 1
            while j < n and src[j] in _WS:
                j += 1
            flush(k)
            parts.append(('sep', '\n' if '\n' in src[k:j] else ' '))
            pending = i = j
            continue

        if c == '/' and k + 1 < n and src[k + 1] in '/*':
            if src[k + 1] == '/':
                j = src.find('\n', k)
                j = n if j < 0 else j
                nl = True
            else:
                j = _block_end(src, k)
                nl = '\n' in src[k:j]
            text = src[k:j]
            flush(k)
            raw_cmts.append((k, j, _locate(starts, k)[0], _locate(starts, j)[0], text))
            parts.append(('cmt', len(raw_cmts) - 1))
            parts.append(('sep', '\n' if nl else ' '))
            pending = i = j
            continue

        if c == '"':
            j = _quoted_end(src, k, '"')
            if src[k:j] == '"C"':
                line_start = starts[_locate(starts, k)[0] - 1]
                if _IMPORT_BEFORE.search(src[line_start:k]):
                    c_lines.append(_locate(starts, k)[0])
                    g = _IMPORT_GROUP.search(src, 0, line_start)
                    if g:
                        c_lines.append(_locate(starts, g.start(1))[0])
            verbatim.append((k, j))
            i = j
            continue

        if c == "'":
            i = _quoted_end(src, k, "'")
            continue

        if c == '`':
            j = _raw_end(src, k)
            verbatim.append((k, j))
            i = j
            continue

        i = k + 1

    flush(n)

    preamble = _preamble(raw_cmts, c_lines)
    for k in preamble:
        verbatim.append((raw_cmts[k][0], raw_cmts[k][1]))

    out: list[str] = []
    last = ''
    sep = ''

    def push(text: str) -> None:
        nonlocal last, sep
        if not text:
            return
        if sep == '\n':
            out.append('\n')
        elif sep == ' ' and last not in _DELIMS and text[0] not in _DELIMS:
            out.append(' ')
        sep = ''
        out.append(text)
        last = text[-1]

    for kind, val in parts:
        if kind == 'code':
            push(val)  # type: ignore[arg-type]
        elif kind == 'sep':
            if last and (sep != '\n'):
                sep = val  # type: ignore[assignment]
        else:
            cm = raw_cmts[val]  # type: ignore[index]
            if val in preamble or KEEP.match(cm[4].strip()):
                push(cm[4])

    found = tuple(
        (line, _locate(starts, off)[1], _locate(starts, end)[1], text)
        for k, (off, end, line, _endl, text) in enumerate(raw_cmts)
        if k not in preamble
    )

    protected: set[int] = set()
    for a, b in verbatim:
        opens = _locate(starts, a)[0]
        closes = _locate(starts, b)[0]
        protected.update(range(opens + 1, closes + 1))

    return found, frozenset(protected), ''.join(out)


def comments(src: str) -> list[tuple[int, int, int, str]]:
    return list(_scan(src)[0])


def protected_lines(src: str) -> set[int]:
    """Lines whose blankness is content: inside a raw string, or inside a cgo
    preamble block, where a blank line is C source layout the gate holds verbatim.
    """
    return set(_scan(src)[1])


def normalize(path: str) -> str:
    """Comment-free normal form; raises ParseError if `path` does not lex.

    No fallback: a file we cannot prove is a file we do not strip.
    """
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    return _scan(src)[2]
