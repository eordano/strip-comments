"""Rust language module. normalize() RAISES if the file does not parse. "Comment"
means the trivia kind only -- a doc comment is an attribute and stays (see below).

    // ... EOL              line comment
    /* ... */               block comment, and it NESTS -- depth is counted,
                            never "find the first */"
    /// ... //! ...         line doc comments (see below)
    /** ... */  /*! ... */  block doc comments
    "..."  b"..."  c"..."   strings
    r"..." r#"..."# r##...  raw strings, any hash count, no escapes
    r#type                  RAW IDENTIFIER (not a raw string) -- `r#loop`/`r#type`
                            are real identifiers when not followed by a quote
    'a  'static  'outer:    lifetime or loop label
    'c'  '\\n'  b'x'         character literals

`'` is the one genuine ambiguity between a char literal and a lifetime;
mis-lexing `'a` as an open char literal swallows source to the next quote and
can hide or fabricate a comment.

DOC COMMENTS ARE NOT TRIVIA. `///`, `//!`, `/**`, `/*!` desugar to `#[doc]`
attributes -- a fenced code block inside one is a DOCTEST `cargo test` runs, so
stripping one deletes a test. They are KEEP, and also preserved VERBATIM in
the normal form as a second line of defence:

    ///x  //!x   are docs        ////x  /////x   are plain comments
    /**x*/  /*!x*/  are docs     /**/  /***/  /***x*/  are plain comments

normal form: own scanner re-emits the source with plain comments dropped, doc
comments and string/char literals kept byte for byte, and whitespace/dropped-
comment runs collapsed to ONE space (not zero) -- Rust is whitespace-
insensitive outside literals, so a space preserves every token boundary,
exposing a strip that fused two tokens (`x/*c*/as i32` -> `x as i32` must still
differ from a fused `xas i32`). Bracket-adjacent comments (`any(/*c*/)`) are
exempt since no multi-char token contains `( ) [ ] { }`.

CRLF files: reads are universal-newline, so a CRLF file and its LF twin
normalize identically -- the gate is blind to line-ending changes there, which
is only safe because the driver writes a uniformly-CRLF file back as CRLF.
"""

from __future__ import annotations

import bisect
import re
from functools import lru_cache

EXTS = ('.rs',)

_KEEP_ALTERNATIVES = (
    r"///(?!/)",
    r"//!",
    r"/\*\*(?![*/])",
    r"/\*!",
    r"//@",
    r"//~",
    r"/[/*][^\n]*\bcbindgen\s*:",
    r"/[/*][^\n]*\b(?:GRCOV|LCOV)_EXCL_",
    r"/[/*][^\n]*\bcoverage\s*:\s*(?:ignore|off|on)",
    r"/[/*][^\n]*\b(?:ignore-tidy|tidy-alphabetical)-",
    r"/[/*][/*!]?\s*(?:SAFETY|INVARIANT)\s*:",
    r"/[/*][^\n]*SPDX-[A-Za-z-]+",
    r"/[/*][^\n]*@(?:license|preserve|copyright|generated)\b",
    r"//\s*(?:vim?|ex|emacs)\s*:",
    r"//\s*-\*-",
)
KEEP = re.compile("|".join("(?:%s)" % p for p in _KEEP_ALTERNATIVES), re.IGNORECASE)

_WS = frozenset(' \t\n\r\x0b\x0c\x85\u200e\u200f\u2028\u2029')

_INTERESTING = re.compile(r'[\s‎‏"\'/]|[bcr]{1,2}#*"')

_RAW_STRING = re.compile(r'(?:b|c)?r(#*)"')
_ESC_STRING = re.compile(r'(?:b|c)"')
_IDENT_CHAR = re.compile(r'\w', re.UNICODE)

_DELIMS = frozenset('()[]{}')

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

def is_doc(text: str) -> bool:
    """Does this comment desugar to a #[doc] attribute?

    Exact per the reference grammar (and checked against proc-macro2): a fourth
    slash or a third star demotes a doc comment back to a plain one, and `/**/`
    is the empty block comment rather than an empty doc.
    """
    if text.startswith('//'):
        return text.startswith('//!') or (
            text.startswith('///') and not text.startswith('////'))
    return text.startswith('/*!') or (
        text.startswith('/**') and not text.startswith(('/***', '/**/')))

def _block_end(src: str, i: int) -> int:
    """End offset of the NESTING block comment opening at `i`.

    Nesting is purely textual: inside a comment a quote is just a character, so
    `/* "*/` ends at that `*/` exactly as rustc says it does.
    """
    n = len(src)
    depth = 0
    j = i
    while j + 1 < n:
        pair = src[j:j + 2]
        if pair == '/*':
            depth += 1
            j += 2
            continue
        if pair == '*/':
            depth -= 1
            j += 2
            if depth == 0:
                return j
            continue
        j += 1
    raise ParseError('unterminated block comment opened at offset %d' % i)

def _escaped_string_end(src: str, k: int) -> int:
    """End offset of a `"`-terminated string whose body starts at `k`."""
    n = len(src)
    j = k
    while j < n:
        c = src[j]
        if c == '\\':
            j += 2
            continue
        if c == '"':
            return j + 1
        j += 1
    raise ParseError('unterminated string literal ending at offset %d' % k)

def _quote_end(src: str, i: int) -> int:
    """End offset of the char literal OR lifetime/label starting at `i`.

    `'\\n'` and `'c'` are literals whose body may contain anything, including a
    quote or a comment opener; `'a` is a lifetime and consumes only its name.
    Always returns > i, so the caller cannot stall.
    """
    n = len(src)
    if i + 1 >= n:
        raise ParseError('dangling quote at offset %d' % i)
    if src[i + 1] == '\\':
        j = i + 2
        if j >= n:
            raise ParseError('unterminated character literal at offset %d' % i)
        if src[j] == 'u' and j + 1 < n and src[j + 1] == '{':
            close = src.find('}', j + 2)
            if close < 0 or '\n' in src[j:close]:
                raise ParseError('unterminated \\u escape at offset %d' % i)
            j = close + 1
        elif src[j] == 'x':
            j += 3
        else:
            j += 1
        if j < n and src[j] == "'":
            return j + 1
        raise ParseError('unterminated character literal at offset %d' % i)
    if i + 2 < n and src[i + 2] == "'":
        return i + 3
    j = i + 1
    while j < n and (src[j].isalnum() or src[j] == '_'):
        j += 1
    if j == i + 1:
        raise ParseError('unterminated character literal at offset %d' % i)
    return j

@lru_cache(maxsize=8)
def _scan(src: str) -> tuple[tuple[tuple[int, int, int, str], ...], frozenset[int], str]:
    """One pass yielding (comments, protected lines, normal form).

    Single source of truth: the comment list and the normal form come from the
    same walk, so the gate can never be proving a different set of spans than the
    driver is cutting.
    """
    starts = _line_starts(src)
    n = len(src)
    found: list[tuple[int, int, int, str]] = []
    verbatim: list[tuple[int, int]] = []
    out: list[str] = []
    last = ''
    sep = False
    i = 0
    pending = 0

    def push(text: str) -> None:
        """Emit code, resolving any separator owed to the run before it."""
        nonlocal last, sep
        if not text:
            return
        if sep:
            if last not in _DELIMS and text[0] not in _DELIMS:
                out.append(' ')
            sep = False
        out.append(text)
        last = text[-1]

    def separator() -> None:
        nonlocal sep
        if last:
            sep = True

    if src.startswith('#!'):
        eol = src.find('\n')
        head = src[2:] if eol < 0 else src[2:eol]
        if not head.lstrip().startswith('['):
            i = n if eol < 0 else eol

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
            push(src[pending:k])
            separator()
            pending = i = j
            continue

        if c == '/' and k + 1 < n and src[k + 1] in '/*':
            if src[k + 1] == '/':
                j = src.find('\n', k)
                j = n if j < 0 else j
            else:
                j = _block_end(src, k)
            text = src[k:j]
            line, col = _locate(starts, k)
            found.append((line, col, _locate(starts, j)[1], text))
            push(src[pending:k])
            separator()
            if is_doc(text):
                push(text)
                separator()
                verbatim.append((k, j))
            pending = i = j
            continue

        if c == '"':
            j = _escaped_string_end(src, k + 1)
            verbatim.append((k, j))
            i = j
            continue

        if c in 'bcr':
            if k and _IDENT_CHAR.match(src[k - 1]):
                i = k + 1
                continue
            raw = _RAW_STRING.match(src, k)
            if raw:
                close = '"' + raw.group(1)
                end = src.find(close, raw.end())
                if end < 0:
                    raise ParseError('unterminated raw string at offset %d' % k)
                j = end + len(close)
            else:
                esc = _ESC_STRING.match(src, k)
                if not esc:
                    i = k + 1
                    continue
                j = _escaped_string_end(src, esc.end())
            verbatim.append((k, j))
            i = j
            continue

        if c == "'":
            i = _quote_end(src, k)
            continue

        i = k + 1

    push(src[pending:])

    protected: set[int] = set()
    for a, b in verbatim:
        opens = _locate(starts, a)[0]
        closes = _locate(starts, b)[0]
        protected.update(range(opens + 1, closes + 1))

    return tuple(found), frozenset(protected), ''.join(out)

def comments(src: str) -> list[tuple[int, int, int, str]]:
    return list(_scan(src)[0])

def protected_lines(src: str) -> set[int]:
    """Lines whose blankness is content, not formatting.

    A blank line inside a `"..."` or `r#"..."#` is part of the literal's value --
    an embedded SQL statement, shader source or GraphQL document -- so collapsing
    it rewrites data, not layout.

    The body of a `/** */` or `/*! */` doc comment counts for the same reason:
    it is the value of a `#[doc]` attribute, so a blank line in it is markdown
    (and, inside a ``` fence, doctest source). Without this, `--collapse-blanks`
    rewrote one and the gate had to revert the whole file.
    """
    return set(_scan(src)[1])

def normalize(path: str) -> str:
    """Comment-free normal form; raises ParseError if `path` does not lex.

    No fallback: a file we cannot prove is a file we do not strip.
    """
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    return _scan(src)[2]
