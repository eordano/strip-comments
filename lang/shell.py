"""Shell language module. Deliberately CONSERVATIVE, leaning on the external
gate rather than a complete shell lexer:

  * only WHOLE-LINE comments are reported (first non-blank char is '#'). A
    trailing comment is never reported -- deciding whether a mid-line '#' opens
    one needs full quoting/expansion context (`${x#pat}`, `$#`, `a#b`).
  * here-document bodies are never scanned; a '#' there is data.

Residual false positive: a '#' line inside a multi-line quoted string (e.g. an
embedded jq/awk program) is left to the gate to catch.

normal form: `shfmt -mn` (parses and reprints minified, comments dropped).
Here-doc bodies survive minification verbatim.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache

EXTS = ('.sh', '.bash', '.ksh')

# Directives and file-local variables, not prose: shebangs, shellcheck/nixfmt
# pragmas, and vim/emacs mode lines.
KEEP = re.compile(r"^#\s*(!|shellcheck|nixfmt|vi:|vim:|ex:|emacs:|-\*-|=+\s*$)")

# `<<` / `<<-` followed by an optionally quoted delimiter, ignoring `<<<`.
_HEREDOC = re.compile(r"<<(-?)\s*(?!<)(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")

_NIXPKGS = 'nixpkgs=/persist/nixpkgs'


class ToolchainError(RuntimeError):
    """shfmt could not be found or run."""


class ParseError(ValueError):
    """The file does not parse; no normal form can be trusted for it."""


def _scan(src: str) -> tuple[list[tuple[int, int, int, str]], set[int]]:
    found = []
    protected: set[int] = set()
    pending: list[tuple[str, bool]] = []   # delimiters still queued on one line
    active: tuple[str, bool] | None = None

    for idx, raw in enumerate(src.split('\n'), 1):
        line = raw[:-1] if raw.endswith('\r') else raw

        if active is not None:
            protected.add(idx)
            delim, dash = active
            probe = line.lstrip('\t') if dash else line
            if probe.strip() == delim:
                active = pending.pop(0) if pending else None
            continue

        stripped = line.lstrip()
        if stripped.startswith('#'):
            found.append((idx, len(line) - len(stripped), len(line), stripped))
            continue

        opened = [(m.group(3), bool(m.group(1))) for m in _HEREDOC.finditer(line)]
        if opened:
            active, pending = opened[0], opened[1:]

    return found, protected


def comments(src: str) -> list[tuple[int, int, int, str]]:
    return _scan(src)[0]


def protected_lines(src: str) -> set[int]:
    """Here-document body lines (plus the terminator, which is never blank).

    A blank line in a here-doc is part of the data the script feeds downstream;
    collapsing it rewrites that payload.
    """
    return _scan(src)[1]


@lru_cache(maxsize=1)
def _shfmt() -> str:
    found = shutil.which('shfmt')
    if found:
        return found
    argv = ['nix-shell', '-p', 'shfmt', '--run', 'command -v shfmt']
    try:
        proc = subprocess.run(argv, env=dict(os.environ, NIX_PATH=_NIXPKGS),
                              capture_output=True, text=True, timeout=1800)
    except OSError as exc:
        raise ToolchainError(
            'shfmt is not on PATH and nix-shell could not provide it: %s' % exc) from exc
    path = (proc.stdout.strip().splitlines() or [''])[-1].strip()
    if proc.returncode != 0 or not path or not os.path.exists(path):
        raise ToolchainError(
            'cannot resolve shfmt, so no shell file can be verified.\n'
            '  tried: PATH, then `NIX_PATH=%s %s`\n'
            '  stderr: %s' % (_NIXPKGS, ' '.join(argv), (proc.stderr.strip() or 'none')[:400]))
    return path


def normalize(path: str) -> str:
    """Comment-free normal form; raises ParseError if `path` does not parse.

    No fallback: a file we cannot prove is a file we do not strip.
    """
    proc = subprocess.run([_shfmt(), '-mn', path],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise ParseError('shfmt could not parse %s: %s'
                         % (path, (proc.stderr.strip() or 'exit %d' % proc.returncode)))
    return proc.stdout
