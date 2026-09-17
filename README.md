# comments

Inventory and strip code comments, proving every strip is semantics-free.

A reader decides WHICH comments earn their place (cheap — a list of line
numbers); this tool owns the cutting and the proof. `list` emits one compact
line per comment so the decision needs no file reads; `strip` removes exactly
the lines named and reverts any file whose normal form moves.

Supported: Python, shell, Nix, TypeScript/JavaScript, C#, Rust, Go, CSS,
HTML, SQL, protobuf, YAML, TOML, Elixir.

## The gate

For every file: normalise, edit, normalise again. Each language's normaliser
discards comments and preserves everything else, so an identical normal form
before and after proves the edit removed comments and nothing else. A file
that stops parsing, or whose normal form moves, is restored and reported — a
bad strip cannot reach the tree. A **REVERTED** file is a language-module bug,
not an acceptable loss (non-zero exit); diff it and fix the module.

The per-language normal forms, the KEEP-protected directive patterns, the
`selftest` that attacks the scanners themselves, and the operator's guide all
live in [SKILL.md](SKILL.md).

## Usage

```sh
comments list [PATH...]                  # path:line:col: text — read this, decide
comments strip FILE:LINE[,LINE...] ...   # cut exactly those, proved safe
comments strip --all [PATH...]           # the zero-comment sweep
```

`comments-list` and `comments-strip` are shorthands for the two subcommands.

`list` prints one line per comment and nothing else — a whole directory costs
a few hundred tokens to review. `strip` removes only the comments **starting**
on the lines you name (a multi-line block needs just its first line) and
refuses KEEP-protected directives even when named. `--check` reports without
writing; `--collapse-blanks` closes blank runs a removal leaves behind, never
inside a string, here-doc, or block-scalar body.

With nix:

```sh
nix run github:eordano/strip-comments -- list src/
```

## Runtime dependencies

Python 3.11+ with PyYAML. Per language, resolved at runtime: `.sh` needs
`shfmt` (bundled in the nix package), `.nix` needs `nix-instantiate`, and
`.ts`/`.js` need `node` plus a `typescript` package resolvable upward from
the file's own directory (or via `NODE_PATH` / `TS_FALLBACK_DIR`), and
`.ex`/`.exs` need `elixir` — with none reachable, those files are skipped
rather than stripped unproven.

## Adding a language

Drop a module in `lang/`; the driver owns all excision, the module only
supplies knowledge. The module protocol is specified in
[SKILL.md](SKILL.md#adding-a-language).

## History

This repository previously held a lexer-only stripper (two comment grammars,
no proof of safety). This tool supersedes it: per-language scanners, a
semantics-equality gate on every edit, and a selftest that attacks the
scanners themselves.
