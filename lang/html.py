"""HTML language module: a WHATWG-style tokeniser, never a regex.

Only comment syntax is ``<!-- ... -->``. NOT comments: ``<script>``/``<style>``
bodies and the other RAWTEXT/RCDATA elements (``textarea title iframe noembed
noframes noscript xmp plaintext``) -- emitted as ONE opaque token, verbatim,
never scanned (their own ``//``/``/* */`` belong to the js/css modules);
attribute values (``<div style="<!--SSR:panel-style-->">`` is real, tag-aware
scanning only); doctypes and bogus comments (``<!DOCTYPE>``, ``<!foo>``,
``<?php``, CDATA). Downlevel-revealed/hidden conditional comments
(``<!--[if IE]>...``) are KEEP -- they're load-bearing markup, not prose.

Two cases are silently not reported rather than risked: a comment inside
``<pre>``/``<listing>`` (whitespace there is rendered, so cutting it changes
output), and a multi-line comment with code on both its first and last line
(splitting it across lines changes rendering, e.g. ``a<!--\\n-->b`` -> "ab" vs
``a\\nb`` -> "a b").

normalize() collapses whitespace runs in character data to one space; tags,
attributes, doctypes, and raw-text/preformatted bodies are copied byte for
byte. Before trusting it, the module also verifies its token stream
reassembles the source exactly, and cross-checks every comment it found
against stdlib ``html.parser`` (which sees more, e.g. inside ``<textarea>`` --
those extra ones are correctly kept, but a comment ONLY this module misses
means its tokeniser desynced). Any check failing, or an unsafe line-ending
character, raises and the driver skips the file.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from functools import lru_cache
from html.parser import HTMLParser

EXTS = ('.html', '.htm')


class ParseError(ValueError):
    """The file does not tokenise cleanly; no normal form can be trusted for it."""


# Directives and placeholders, not prose. Removing any of these changes what a
# browser renders, what a server substitutes, or what a build step emits.
#
# `<!--SSR:name-->` markers are substituted by literal string replace in
# real server-side rendering code, so cutting one silently empties a page
# region. `GENERATED` is a marker some generator scripts write. The rest are
# the standard tool vocabularies.
_KEEP_PREFIX = (                     # immediately after the `<!--`
    r"-->",                          # <!----> an empty comment is a real DOM text-node separator
    r"SSR:",                         # server-side substitution slots
    r"#(?:include|echo|exec|config|set|printenv|flastmod|fsize|if|elif|else|endif)\b",  # SSI
    r"(?:/)?\$[!?]?-->",             # React / Next suspense boundaries: <!--$--> <!--/$-->
    r"[\[\]]-->",                    # Vue 3 fragment anchors: <!--[--> <!--]-->
    r"teleport\b",                   # Vue teleport anchors
    r"(?:end)?build:|endbuild\b|/build\b",   # usemin blocks: <!-- build:js app.js -->
    r"/?\s*ko\b",                    # knockout containerless bindings: <!-- ko if: x --> <!-- /ko -->
    r"htmlmin:|htmlhint\b|stylelint|prettier-ignore|eslint|vue-ignore|lit-",
    r"google(?:off|on):|/?noindex\s*-->",
    r"more\s*-->",                   # <!--more--> splits summary from body: Hugo, WordPress,
                                     # Jekyll, Ghost. Anchored to the whole comment, so
                                     # `<!-- more stuff -->` and `<!--moreover-->` still go.
    r"\{\{|\{%|<%|\$\{|<\?|%\}|\[\[",  # a template engine's placeholder inside the marker
)
_KEEP_ANYWHERE = (                   # anywhere inside the comment
    r"\[if\b",                       # <!--[if lt IE 9]> ... <![endif]-->
    r"<!\[endif\]",                  # <!--<![endif]--> (downlevel-revealed close)
    r"SPDX-[A-Za-z0-9.-]+",
    r"@(?:license|preserve|copyright|generated)\b",
    r"\bGENERATED\b",
)
KEEP = re.compile(
    r"<!--(?:\s*(?:%s)|[\s\S]*?(?:%s))"
    % ("|".join(_KEEP_PREFIX), "|".join(_KEEP_ANYWHERE)),
    re.IGNORECASE,
)

# HTML whitespace, per the spec: TAB LF FF CR SPACE. Not \v, not NBSP.
_WS = ' \t\n\r\f'
_WS_RUN = re.compile(r'[ \t\n\r\f]+')

# Bodies that are not markup. RCDATA still expands entities, RAWTEXT does not;
# neither can contain a comment, which is all this module needs to know.
_RAWTEXT = frozenset(('style', 'xmp', 'iframe', 'noembed', 'noframes', 'noscript'))
_RCDATA = frozenset(('title', 'textarea'))
# `white-space: pre` in the UA stylesheet: a newline here is rendered.
_PREFORMATTED = frozenset(('pre', 'listing'))

_END_SCRIPT = re.compile(r'</script(?=[ \t\n\r\f/>])', re.IGNORECASE)
_OPEN_SCRIPT = re.compile(r'<script(?=[ \t\n\r\f/>])', re.IGNORECASE)
# An end tag only closes a raw-text element when its name is followed by
# whitespace, '/' or '>' -- `</styles>` does not close a `<style>`.
_END_RAW = {name: re.compile(r'</%s(?=[ \t\n\r\f/>])' % name, re.IGNORECASE)
            for name in _RAWTEXT | _RCDATA}

# `<!-->` / `<!--->`: the spec's abrupt-closing empty comments. Carrying no prose
# and sometimes load-bearing as DOM text-node separators, they are never offered.
_ABRUPT = re.compile(r'^<!---?>$')

# Characters whose line semantics differ between this module and the driver that
# cuts the lines it names. `comments.py` slices with `str.splitlines()`, which
# ends a line on any of these; a browser -- and `_line_starts` here, and the
# `text.count('\n')` the driver uses to find a multi-line comment's last line --
# counts only '\n'. One of them anywhere in a file shifts every later line number
# on the way to the cut, so the driver deletes a line nobody named, and when the
# line it lands on is blank the normal form cannot see the difference and blesses
# it. '\r' is a separate hazard with the same remedy: the driver reads and
# rewrites with universal newlines, so stripping one comment from a CRLF file
# also silently retypes every line ending in it -- invisible in character data,
# a diff inside every script, style, <pre> and attribute value.
#
# Both are rare enough that refusing is far cheaper than a wrong cut:
# normalize() raises, so the driver reports `skip` and never writes.
_UNSAFE_BREAKS = '\r\v\f\x1c\x1d\x1e\x85\u2028\u2029'


def _is_alpha(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _tag(src: str, lt: int) -> tuple[int, str, bool, list[tuple[int, int]]]:
    """Consume a start/end tag at `lt`; return (end, name, self_closing, attr_value_spans).

    The attribute states are walked explicitly because a quote only opens an
    attribute value after an `=`: in `<div a=b'c>` the apostrophe is part of the
    unquoted value, and guessing otherwise swallows the rest of the document.
    """
    n = len(src)
    i = lt + 1
    if src[i:i + 1] == '/':
        i += 1
    j = i
    while j < n and src[j] not in _WS + '/>':
        j += 1
    name = src[i:j].lower()
    i = j
    selfclose = False
    values: list[tuple[int, int]] = []
    while i < n:
        c = src[i]
        if c in _WS:
            i += 1
            continue
        if c == '>':
            return i + 1, name, selfclose, values
        if c == '/':
            selfclose = src[i + 1:i + 2] == '>'
            i += 1
            continue
        selfclose = False
        while i < n and src[i] not in _WS + '/>=':      # attribute name
            i += 1
        while i < n and src[i] in _WS:
            i += 1
        if i < n and src[i] == '=':
            i += 1
            while i < n and src[i] in _WS:
                i += 1
            if i < n and src[i] in '"\'':
                quote = src[i]
                end = src.find(quote, i + 1)
                if end < 0:
                    raise ParseError('unterminated attribute value in tag at offset %d' % lt)
                values.append((i + 1, end))
                i = end + 1
            else:
                while i < n and src[i] not in _WS + '>':
                    i += 1
    raise ParseError('unterminated tag at offset %d' % lt)


def _comment_end(src: str, lt: int) -> int:
    """End offset of the `<!--` comment opening at `lt` (one past its final `>`)."""
    i = lt + 4
    if src.startswith('>', i):          # <!-->      abrupt close, empty comment
        return i + 1
    if src.startswith('->', i):         # <!--->     abrupt close, empty comment
        return i + 2
    while True:
        k = src.find('--', i)
        if k < 0:
            raise ParseError('unterminated comment at offset %d' % lt)
        if src.startswith('-->', k):
            return k + 3
        if src.startswith('--!>', k):   # the spec's tolerated `--!>` close
            return k + 4
        i = k + 1                       # so `---->` still closes on its last two dashes


def _raw_end(src: str, start: int, name: str) -> int:
    """Offset of the `<` that opens the end tag closing a RAWTEXT/RCDATA element."""
    end = _END_RAW[name].search(src, start)
    if end is None:
        raise ParseError('unterminated <%s> starting at offset %d' % (name, start))
    return end.start()


def _script_end(src: str, start: int) -> int:
    """Offset of the `<` that closes a `<script>`, honouring the escape states.

    `<!--` inside script data enters the escaped state and `<script` inside THAT
    enters the double-escaped state, where `</script>` does not end the element --
    the legacy cloak `<!-- document.write('<script>...</script>') -->`. `-->`
    leaves both. Without this, the element ends early and the remaining JavaScript
    gets scanned as markup.
    """
    n = len(src)
    i = start
    escaped = False
    double = False
    while i < n:
        if src[i] == '<':
            hit = _END_SCRIPT.match(src, i)
            if hit is not None:
                if double:
                    double = False
                    i = hit.end()
                    continue
                return i
            if not escaped and src.startswith('<!--', i):
                escaped = True
                i += 4
                continue
            if escaped and not double and _OPEN_SCRIPT.match(src, i) is not None:
                double = True
                i += len('<script')
                continue
            i += 1
            continue
        if escaped and src.startswith('-->', i):
            escaped = False
            double = False
            i += 3
            continue
        i += 1
    raise ParseError('unterminated <script> starting at offset %d' % start)


@lru_cache(maxsize=8)
def _scan(src: str) -> tuple[tuple[tuple[str, int, int], ...], tuple[tuple[int, int], ...]]:
    """Tokenise `src` into contiguous spans covering every character.

    Kinds: 'text' (character data), 'pretext' (character data inside <pre>),
    'comment', 'precomment' (a comment inside <pre>), 'tag', 'decl' (doctype,
    bogus comment, CDATA section), 'raw' (script/style/RCDATA body).
    """
    n = len(src)
    toks: list[tuple[str, int, int]] = []
    values: list[tuple[int, int]] = []
    i = 0
    pre = 0

    def text(start: int, stop: int) -> None:
        if stop > start:
            toks.append(('pretext' if pre else 'text', start, stop))

    while i < n:
        lt = src.find('<', i)
        if lt < 0:
            text(i, n)
            break
        text(i, lt)
        nxt = src[lt + 1:lt + 2]

        if nxt == '!':
            if src.startswith('<!--', lt):
                end = _comment_end(src, lt)
                toks.append(('precomment' if pre else 'comment', lt, end))
            elif src[lt + 2:lt + 9].upper() == 'DOCTYPE':
                end = src.find('>', lt + 2)
                if end < 0:
                    raise ParseError('unterminated <!DOCTYPE at offset %d' % lt)
                end += 1
                toks.append(('decl', lt, end))
            elif src.startswith('<![CDATA[', lt):
                # Only legal in foreign content; in HTML content a browser reads it
                # as a bogus comment ending at the first '>'. Bracketing to ']]>' is
                # the conservative read either way: the span stays opaque, so nothing
                # inside it is ever offered for stripping.
                end = src.find(']]>', lt + 9)
                if end < 0:
                    raise ParseError('unterminated CDATA section at offset %d' % lt)
                end += 3
                toks.append(('decl', lt, end))
            else:
                end = src.find('>', lt + 1)     # bogus comment: <!foo>, <![if !IE]>
                if end < 0:
                    raise ParseError('unterminated markup declaration at offset %d' % lt)
                end += 1
                toks.append(('decl', lt, end))
            i = end
            continue

        if nxt == '?':                          # bogus comment: <?php, <?xml
            end = src.find('>', lt + 1)
            if end < 0:
                raise ParseError('unterminated processing instruction at offset %d' % lt)
            toks.append(('decl', lt, end + 1))
            i = end + 1
            continue

        if nxt == '/' and not _is_alpha(src[lt + 2:lt + 3]):
            end = src.find('>', lt + 1)         # bogus comment: `</ >`, `</>`
            if end < 0:
                raise ParseError('unterminated end tag at offset %d' % lt)
            toks.append(('decl', lt, end + 1))
            i = end + 1
            continue

        if nxt == '/' or _is_alpha(nxt):
            closing = nxt == '/'
            end, name, _selfclose, attrs = _tag(src, lt)
            toks.append(('tag', lt, end))
            values.extend(attrs)
            i = end
            if closing:
                if name in _PREFORMATTED and pre:
                    pre -= 1
                continue
            if name == 'plaintext':             # never ends: the rest of the file is text
                if n > i:
                    toks.append(('raw', i, n))
                i = n
            elif name == 'script':
                stop = _script_end(src, i)
                if stop > i:
                    toks.append(('raw', i, stop))
                i = stop
            elif name in _RAWTEXT or name in _RCDATA:
                stop = _raw_end(src, i, name)
                if stop > i:
                    toks.append(('raw', i, stop))
                i = stop
            elif name in _PREFORMATTED:
                # HTML ignores a trailing slash on a non-void element, so `<pre/>`
                # OPENS a pre. Honouring the slash made every comment after it look
                # like ordinary markup, and cutting one there drops a rendered
                # newline that the normal form -- which collapses whitespace outside
                # pre -- cannot see.
                pre += 1
            continue

        text(lt, lt + 1)                        # a bare '<' is character data
        i = lt + 1

    return tuple(toks), tuple(values)


def _line_starts(src: str) -> list[int]:
    starts = [0]
    at = src.find('\n')
    while at != -1:
        starts.append(at + 1)
        at = src.find('\n', at + 1)
    return starts


def _locate(starts: list[int], off: int) -> tuple[int, int]:
    row = bisect_right(starts, off) - 1
    return row + 1, off - starts[row]


def comments(src: str) -> list[tuple[int, int, int, str]]:
    """Every strippable HTML comment, in source order.

    Comments inside `<pre>`/`<listing>`, and multi-line comments with code on both
    sides, are omitted -- see the module docstring: excising either changes what is
    rendered, so they are not comment strips at all.
    """
    toks, _ = _scan(src)
    starts = _line_starts(src)
    lines = src.split('\n')
    out: list[tuple[int, int, int, str]] = []
    for kind, begin, end in toks:
        if kind != 'comment':
            continue
        text = src[begin:end]
        if _ABRUPT.match(text):
            continue
        line, col = _locate(starts, begin)
        last, col_end = _locate(starts, end)
        if last > line and lines[line - 1][:col].strip() and lines[last - 1][col_end:].strip():
            continue
        out.append((line, col, col_end, text))
    return out


def protected_lines(src: str) -> set[int]:
    """Lines whose blankness is content, not formatting.

    A blank line inside a `<script>`/`<style>`/`<textarea>` body belongs to the
    JavaScript, CSS or default text it carries; inside `<pre>` it is a rendered
    newline; inside a quoted attribute value it is part of the value. Collapsing
    any of them rewrites the page rather than tidying it.
    """
    toks, values = _scan(src)
    starts = _line_starts(src)
    out: set[int] = set()
    spans = [(begin, end) for kind, begin, end in toks if kind in ('raw', 'pretext')]
    spans.extend(values)
    for begin, end in spans:
        first, _ = _locate(starts, begin)
        last, _ = _locate(starts, max(begin, end - 1))
        out.update(range(first, last + 1))
    return out


class _CommentFinder(HTMLParser):
    """Independent second opinion on where the comments are.

    Its verdict is only ever compared with this module's; it never drives an edit.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.starts: list[tuple[int, int]] = []

    def handle_comment(self, data: str) -> None:
        self.starts.append(self.getpos())


def _second_opinion(src: str) -> list[tuple[int, int]]:
    finder = _CommentFinder()
    try:
        finder.feed(src)
        finder.close()
    except Exception as exc:                     # html.parser gave up; we cannot cross-check
        raise ParseError('html.parser could not read the document: %s' % exc) from exc
    return finder.starts


def _normal_form(src: str) -> str:
    bad = [c for c in _UNSAFE_BREAKS if c in src]
    if bad:
        raise ParseError('line break the driver counts differently: %r'
                         % [hex(ord(c)) for c in bad])

    toks, _ = _scan(src)
    if ''.join(src[begin:end] for _, begin, end in toks) != src:
        raise ParseError('tokeniser did not cover the source')

    # Every comment this module offers for stripping must also be a comment to an
    # independent implementation. Only that direction matters: html.parser reports
    # more (bogus comments such as `<!foo>` and `<![if !IE]>`, comments inside
    # <textarea>) and this module keeps all of those verbatim, but a comment IT
    # cannot see is a sign the tokeniser has fallen out of sync -- which is exactly
    # how a `<!--` inside a script body would come to be cut.
    offered = {(line, col) for line, col, _end, _text in comments(src)}
    theirs = set(_second_opinion(src))
    stray = sorted(offered - theirs)
    if stray:
        raise ParseError('html.parser sees no comment at %r' % (stray[:6],))

    pieces: list[tuple[bool, list[str]]] = []
    for kind, begin, end in toks:
        if kind in ('comment', 'precomment'):
            continue                             # dropped, so its neighbours merge
        loose = kind == 'text'
        if pieces and loose and pieces[-1][0]:
            pieces[-1][1].append(src[begin:end])
        else:
            pieces.append((loose, [src[begin:end]]))

    out = [_WS_RUN.sub(' ', ''.join(parts)) if loose else ''.join(parts)
           for loose, parts in pieces]
    # Whitespace at the very edges of the document is never rendered, and excision
    # can remove it, so it must not be part of the normal form.
    if out and pieces[0][0]:
        out[0] = out[0].lstrip(_WS)
    if out and pieces[-1][0]:
        out[-1] = out[-1].rstrip(_WS)
    return ''.join(out)


def normalize(path: str) -> str:
    """Comment-free normal form of `path`; raises ParseError if it cannot be trusted.

    No fallback: a file we cannot prove is a file we do not strip.
    """
    try:
        with open(path, encoding='utf-8', newline='') as fh:
            src = fh.read()
    except UnicodeDecodeError as exc:
        raise ParseError('%s is not UTF-8: %s' % (path, exc)) from exc
    return _normal_form(src)
