"""Python language module. A comment never spans lines, so col_end is always
on `line`. Detection is `tokenize`, never a regex, so a '#' inside a string
literal or f-/t-string is never mistaken for a comment. Docstrings are NOT
comments -- they're string expressions in the AST -- and are never reported.

normal form: `ast.dump(ast.parse(...))`; comments are absent from the AST.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from functools import lru_cache

EXTS = ('.py', '.pyi')

# Directives, not prose: removing these changes behaviour or tooling results.
# `!` is the shebang and `-*-` the PEP 263 encoding declaration; both are only
# meaningful on lines 1-2, but a `#!`/`# -*-` further down is prose so rare that
# keeping it costs one stray line and guarantees the real ones survive.
KEEP = re.compile(
    r"^#\s*(type:|noqa|pylint:|mypy:|ruff:|fmt:|isort:|nosec|pragma:|shellcheck|"
    r"-\*-|!)"
)

# Token types whose text is literal string CONTENT. FSTRING_MIDDLE / TSTRING_MIDDLE
# exist from 3.12 / 3.14 on and cover only the literal chunks, so a blank line in a
# {...} replacement field stays unprotected -- there it really is formatting.
_STRING_TOKENS = frozenset(
    getattr(tokenize, name)
    for name in ('STRING', 'FSTRING_MIDDLE', 'TSTRING_MIDDLE')
    if hasattr(tokenize, name)
)


@lru_cache(maxsize=8)
def _scan(src: str) -> tuple[list[tuple[int, int, int, str]], frozenset[int]]:
    found = []
    protected: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            found.append((tok.start[0], tok.start[1], tok.end[1], tok.string))
        elif tok.type in _STRING_TOKENS and tok.end[0] > tok.start[0]:
            protected.update(range(tok.start[0] + 1, tok.end[0] + 1))
    return found, frozenset(protected)


def comments(src: str) -> list[tuple[int, int, int, str]]:
    return list(_scan(src)[0])


def protected_lines(src: str) -> set[int]:
    """Lines whose blankness is string content, not formatting.

    A blank line inside a triple-quoted string is part of the string's value:
    collapsing it rewrites a docstring or an embedded script, which is a semantic
    change, not a cosmetic one.
    """
    return set(_scan(src)[1])


def normalize(path: str) -> str:
    """Comment-free normal form; raises SyntaxError/ValueError if `path` does not parse.

    No fallback: a file we cannot prove is a file we do not strip.
    """
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    return ast.dump(ast.parse(src))
