"""Elixir language module (.ex/.exs). Detection and the normal form both come
from the Elixir compiler itself, never a regex: `Code.string_to_quoted_with_comments!`
reports every `#` comment with its 1-based line and column, and skips `#` inside
strings, charlists, heredocs and sigils (an `~H` HEEx template is one sigil
token, so `<%# %>` and CSS `#fff` inside it are never reported).

Comments never span lines, so col_end is always on `line`.

normal form: `Code.string_to_quoted!` with positional metadata dropped
(`line`, `column`, `end_line`, ...): comments are not AST nodes, and removing
one shifts the line numbers of everything below it, which must not count as a
change. Value-bearing metadata (`delimiter`, `format`, `token`) is kept.

protected_lines: every line from a string / heredoc / sigil token to the next
token, so a blank line inside a heredoc or `~H` template is never collapsed.
Over-protection is harmless: `mix format` folds any blank run left behind.

Needs an `elixir` on PATH; without one every
call raises and the driver skips the file rather than strip it unproven.
"""

from __future__ import annotations

import base64
import re
import subprocess
from functools import lru_cache

EXTS = ('.ex', '.exs')

KEEP = re.compile(r"^#\s*(!|credo:|coveralls-ignore|dialyzer|sobelow_skip)")

_SCAN = r'''
src = IO.read(:stdio, :eof)
{_q, comments} = Code.string_to_quoted_with_comments!(src, columns: true)
for c <- comments do
  IO.write("C\t#{c.line}\t#{c.column}\t#{Base.encode64(c.text)}\n")
end
{:ok, _l, _c, _w, tokens, _t} = :elixir_tokenizer.tokenize(String.to_charlist(src), 1, 1, [])
pos = fn t -> {l, c, _} = elem(t, 1); {l, c} end
sorted = tokens |> Enum.filter(&(tuple_size(&1) >= 2 and is_tuple(elem(&1, 1)))) |> Enum.sort_by(pos)
stringy = [:bin_string, :list_string, :bin_heredoc, :list_heredoc, :sigil, :atom_quoted, :kw_identifier_unsafe, :identifier_unsafe]
Enum.chunk_every(sorted, 2, 1, [nil])
|> Enum.each(fn [t, nxt] ->
  if elem(t, 0) in stringy do
    {l1, _} = pos.(t)
    l2 = if nxt, do: elem(pos.(nxt), 0), else: length(String.split(src, "\n"))
    IO.write("P\t#{l1}\t#{l2}\n")
  end
end)
'''

_NORM = r'''
src = IO.read(:stdio, :eof)
q = Code.string_to_quoted!(src, columns: false, token_metadata: false)
drop = [:line, :column, :end_line, :end_of_expression, :newlines, :closing, :do, :end, :last, :leading_comments, :trailing_comments]
q = Macro.prewalk(q, fn
  {a, meta, args} when is_list(meta) -> {a, Keyword.drop(meta, drop), args}
  other -> other
end)
IO.write(inspect(q, limit: :infinity, printable_limit: :infinity, width: :infinity))
'''


def _elixir(script: str, src: str) -> str:
    r = subprocess.run(
        ["elixir", "-e", script], input=src, capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "elixir failed")
    return r.stdout


@lru_cache(maxsize=8)
def _scan(src: str) -> tuple[list[tuple[int, int, int, str]], frozenset[int]]:
    found: list[tuple[int, int, int, str]] = []
    protected: set[int] = set()
    for row in _elixir(_SCAN, src).splitlines():
        kind, *rest = row.split("\t")
        if kind == "C":
            line, col, b64 = rest
            text = base64.b64decode(b64).decode("utf-8")
            start = int(col) - 1
            found.append((int(line), start, start + len(text), text))
        elif kind == "P":
            l1, l2 = (int(x) for x in rest)
            protected.update(range(l1, l2 + 1))
    return found, frozenset(protected)


def comments(src: str) -> list[tuple[int, int, int, str]]:
    return list(_scan(src)[0])


def protected_lines(src: str) -> set[int]:
    return set(_scan(src)[1])


def normalize(path: str) -> str:
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    return _elixir(_NORM, src)
