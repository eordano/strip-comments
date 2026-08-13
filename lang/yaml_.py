"""YAML language module. A comment is '#' to end of line and always single-line."""

import re

import yaml

EXTS = ('.yml', '.yaml')

# Every branch is anchored on '#' because the driver tests KEEP with .match()
# against text that always starts at the comment marker: an unanchored
# alternative can never fire, however well it describes the directive.
KEEP = re.compile(
    r"""
    ^\# (?:
        !                               # shebang
      | \s* -\*-                        # emacs file-local variables
      | \s* vim:                        # editor modeline
      | \s* yaml-language-server \s* :  # schema association
      | \s* yamllint \s+ (?: disable | enable )
      | .* \b noqa \b
      | .* SPDX-License-Identifier
      | .* @generated
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# '|', '>', with any combination of chomping indicator and explicit indentation.
_BLOCK_HEADER = re.compile(r'^[|>](?:[+-]\d*|\d*[+-]?)$')

# Characters after which a quote can begin a scalar rather than sit inside one.
_SCALAR_INTRO = ':-,[{'


def _is_doc_marker(line):
    return line[:3] in ('---', '...') and (len(line) == 3 or line[3] in ' \t')


def _opens_scalar(line, i):
    """Can the quote at `i` begin a scalar, rather than sit inside a plain one?

    Only a quote introducing a value, a key or a flow entry can open a scalar
    that spans lines. An apostrophe further into a plain scalar (`msg: it 's`)
    cannot, and carrying its state onward would silence every comment below it.
    """
    head = line[:i].rstrip()
    return not head or head[-1] in _SCALAR_INTRO


def _comment_col(line, quote=None, flow=0):
    """(column of the comment-opening '#' or None, quote open at line end, flow).

    A '#' only opens a comment at line start or after a space/tab -- 'a#b' is
    the plain scalar 'a#b'. Quotes only open a quoted scalar at a token start,
    otherwise the apostrophe in `msg: don't panic` would swallow the rest.

    `quote` carries an unterminated quoted scalar in from the previous line: a
    quoted scalar may span lines, and a '#' inside one is content on every line
    it covers, not only on the line that opened it.

    `flow` is the '['/'{' nesting depth carried in the same way, because inside
    a flow collection JSON syntax is legal and a quote may follow ':' with no
    space: `{"k":"v # w"}` holds the value 'v # w'. In block context the very
    same text is the plain scalar `k:"v` and the '#' does open a comment, so the
    depth is what tells the two apart. Miscounting depth (an unbalanced brace in
    a plain scalar) can only make a real comment look quoted -- it under-reports,
    which costs a comment rather than corrupting a file.
    """
    i, n = 0, len(line)
    carry = quote is not None
    while i < n:
        c = line[i]
        if quote == "'":
            if c == "'":
                if i + 1 < n and line[i + 1] == "'":
                    i += 2
                    continue
                quote = None
            i += 1
        elif quote == '"':
            if c == '\\':
                i += 2
                continue
            if c == '"':
                quote = None
            i += 1
        elif c == '#' and (i == 0 or line[i - 1] in ' \t'):
            return i, None, flow    # rest of the line is comment; nothing is open
        elif c in '"\'' and (i == 0 or line[i - 1] in ' \t,[{'
                             or (flow and line[i - 1] == ':')):
            quote = c
            carry = _opens_scalar(line, i)
            i += 1
        else:
            if c in '[{':
                flow += 1
            elif c in ']}' and flow:
                flow -= 1
            i += 1
    return None, (quote if carry else None), flow


def _block_header_parent(code):
    """(indent a block scalar's body must exceed, keeps trailing blanks?) or None.

    The second element is the '+' chomping indicator: with it the blank lines
    that trail the body are part of the scalar's value instead of being clipped.
    """
    body = code.rstrip()
    if not body:
        return None
    tok = body.rsplit(None, 1)[-1]
    if not _BLOCK_HEADER.match(tok):
        return None
    indent = len(body) - len(body.lstrip(' '))
    # A header alone on its line belongs to the mapping key on the line above,
    # so its body may sit at the same indent as the header itself.
    if body.lstrip(' ') == tok:
        indent -= 1
    return indent, '+' in tok


def _scan(src):
    """Single pass yielding (comments, lines whose blankness is content).

    Block-scalar state is tracked explicitly because everything inside a `|` or
    `>` body is string CONTENT: a '#' there is a shell comment the job actually
    runs, and a blank line there is part of the value.

    Block scalars are not the only multi-line scalars, though. A quoted or plain
    scalar may run over several lines too, and there a run of line breaks folds
    to one fewer '\\n' -- so a blank line inside one is just as load-bearing as a
    blank inside a `|` body, and closing it up rewrites the value.
    """
    found = []
    protected = set()
    pending = []          # blanks that count as body only if the body resumes
    block = None          # (parent indent, keeps trailing blanks) inside a body
    quote = None          # quoted scalar left open by the previous line
    flow = 0              # '['/'{' nesting depth left open by the previous line
    gap = []              # blanks outside a block, awaiting the next real line
    prev_indent = None    # indent of the last non-blank line outside a block

    # splitlines(), not split('\n'): the driver addresses lines with
    # splitlines(keepends=True), which also breaks on a lone '\r' and on
    # U+0085 / U+2028 / U+2029 -- all of which PyYAML likewise reads as line
    # breaks. Numbering lines any other way desynchronises module and driver,
    # and the excision lands on the wrong line.
    for idx, line in enumerate(src.splitlines(), 1):

        if block is not None:
            if not line.strip():
                pending.append(idx)
                continue
            indent = len(line) - len(line.lstrip(' '))
            if indent > block[0] and not (indent == 0 and _is_doc_marker(line)):
                protected.update(pending)
                pending.clear()
                protected.add(idx)
                continue
            # Body over. Its trailing blanks are value only under '+' chomping;
            # otherwise they are clipped and their blankness carries nothing.
            if block[1]:
                protected.update(pending)
            block = None
            pending.clear()

        if not line.strip():
            # Inside a quoted scalar the carried quote settles it outright; a
            # continuation may even sit at column 0, so indentation cannot.
            if quote is not None:
                protected.add(idx)
            else:
                gap.append(idx)
            continue

        indent = len(line) - len(line.lstrip(' '))
        if gap:
            # A multi-line scalar's continuation is indented further than the
            # line that opened it, while the next sibling of a node never is.
            # Deeper text after a blank run therefore means the run sits inside
            # a scalar. Over-protecting here only leaves a blank run standing
            # under --collapse-blanks; under-protecting rewrites a value.
            if prev_indent is not None and indent > prev_indent:
                protected.update(gap)
            gap.clear()
        prev_indent = indent

        col, quote, flow = _comment_col(line, quote, flow)
        if col is not None:
            found.append((idx, col, len(line), line[col:]))

        # A trailing '|' inside an unterminated quoted scalar is text, not a header.
        if quote is None:
            block = _block_header_parent(line if col is None else line[:col])

    if block is not None and block[1]:
        protected.update(pending)
    return found, protected


def comments(src):
    return _scan(src)[0]


def protected_lines(src):
    return _scan(src)[1]


def normalize(path):
    """Comment-free normal form; raises if `path` does not parse.

    The form is the loaded data re-emitted canonically. Comments are not data,
    so they cannot reach it, while every scalar -- including a block scalar's
    body, blank lines and all -- is compared by value: cutting a '#' out of a
    `run: |` script moves the form and the driver restores the file.

    No fallback: a file we cannot prove is a file we do not strip.
    """
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    docs = list(yaml.safe_load_all(src))
    return yaml.safe_dump_all(
        docs,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        width=1 << 20,
    )
