#!/usr/bin/env python3
"""Inventory and strip code comments, proving every strip is semantics-free.

A reader decides WHICH comments earn their place (cheap -- a list of line
numbers); this owns the cutting and the proof. `list` emits one compact line
per comment so the decision needs no file reads; `strip` removes exactly the
lines named and reverts any file whose normal form moves.

The gate is the point. For every file: normalise, edit, normalise again. Each
language's normaliser discards comments and preserves everything else, so an
identical normal form before and after means the edit removed comments and
nothing else. A file that stops parsing, or whose normal form moves, is
restored and reported -- a bad strip cannot reach the tree.

  .py                     ast.dump(ast.parse(...))     comments are not nodes
  .sh                     shfmt -mn                    here-doc bodies verbatim
  .nix                    nix-instantiate --parse       comments dropped at parse
  .ts .tsx .js .jsx .mjs  TypeScript scanner tokens    trivia excluded by construction
  .yml .yaml              safe_load_all -> sorted json  comments are not data

Usage:
  comments.py list  [PATH...] [--json] [--all]
  comments.py strip FILE:LINE[,LINE...] [FILE:LINE...] [--check]
  comments.py strip --all [PATH...] [--collapse-blanks] [--check]

  list          every strippable comment as `path:line:col: text`; --all also
                shows KEEP-protected directives, marked, which strip refuses
  strip         remove exactly the comments starting on those lines
  strip --all   remove every strippable comment under PATH (the zero-comment
                sweep); positionals become paths, not FILE:LINE

  --collapse-blanks  also close a blank run left behind by a removal, never
                     inside a string/here-doc/block-scalar body
  --check            report what would change, write nothing

Exit is non-zero if any file was reverted, i.e. a lexer bug. A revert is never
an acceptable loss: diff the file and fix the language module.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LANGDIR = os.path.join(HERE, "lang")

SKIP_DIRS = {".git", "node_modules", "__pycache__", "vendor", "dist", "build", ".venv", "target"}


def _load_langs() -> dict[str, object]:
    by_ext: dict[str, object] = {}
    for fn in sorted(os.listdir(LANGDIR)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        path = os.path.join(LANGDIR, fn)
        spec = importlib.util.spec_from_file_location(f"_lang_{fn[:-3]}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for ext in mod.EXTS:
            by_ext[ext] = mod
    return by_ext


LANGS = _load_langs()


def _walk(paths: list[str]) -> list[str]:
    """Tracked files under paths, or the paths themselves when they are files."""
    out: list[str] = []
    for p in paths:
        if os.path.isfile(p):
            out.append(p)
            continue
        try:
            listed = subprocess.run(
                ["git", "ls-files", "-z", p],
                capture_output=True, text=True, check=True,
            ).stdout.split("\0")
            out.extend(f for f in listed if f)
            continue
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            out.extend(os.path.join(root, f) for f in files)
    return [f for f in out if os.path.splitext(f)[1] in LANGS and os.path.isfile(f)]


# excise() addresses lines via str.splitlines(), which breaks on more characters
# than any of these languages call a line terminator. A module that counts lines
# by \n therefore hands excise() coordinates from a different frame, and excise()
# cuts the wrong line -- observed reporting "stripped" while the named comment
# survived and an unrelated line was deleted instead. The gate still catches any
# semantic damage, but the report is a lie, so refuse the file outright. \r is
# excluded here because CRLF is handled on read and lone \r is caught by (?!\n).
_ALIEN_BREAK = re.compile(r"[\v\f\x1c\x1d\x1e\x85  ]|\r(?!\n)")


def _kept(mod, text: str) -> bool:
    keep = getattr(mod, "KEEP", None)
    return bool(keep and keep.match(text.strip()))


def _find(mod, src: str):
    try:
        return mod.comments(src)
    except Exception:
        return []


def excise(src: str, spans, protected: set[int], collapse: bool) -> str:
    """Remove (line, col_start, col_end, text) spans. Cols are 0-based, line 1-based.

    A span covering a whole line takes the line with it; a trailing one leaves
    the code and drops the dangling whitespace.
    """
    lines = src.splitlines(keepends=True)
    drop: set[int] = set()
    edits: dict[int, list[tuple[int, int]]] = {}
    for line, c0, c1, text in spans:
        last = line + text.count("\n")
        if last == line:
            edits.setdefault(line, []).append((c0, c1))
            continue
        head = lines[line - 1][:c0] if line - 1 < len(lines) else ""
        tail = lines[last - 1][c1:] if last - 1 < len(lines) else ""
        if not head.strip() and not tail.strip():
            drop.update(range(line, last + 1))
        else:
            edits.setdefault(line, []).append((c0, len(lines[line - 1])))
            drop.update(range(line + 1, last))
            edits.setdefault(last, []).append((0, c1))

    out: list[tuple[str, int]] = []
    for i, raw in enumerate(lines, 1):
        if i in drop:
            continue
        if i not in edits:
            out.append((raw, i))
            continue
        nl = "\n" if raw.endswith("\n") else ""
        body = raw[:-1] if nl else raw
        for c0, c1 in sorted(edits[i], reverse=True):
            body = body[:c0] + body[c1:]
        if not body.strip():
            continue
        out.append((body.rstrip() + nl, i))

    if collapse:
        kept: list[tuple[str, int]] = []
        for text_, lineno in out:
            blank = text_.strip() == "" and lineno not in protected
            if blank and kept and kept[-1][0].strip() == "" and kept[-1][1] not in protected:
                continue
            kept.append((text_, lineno))
        out = kept
    return "".join(t for t, _ in out)


def _strip_file(path: str, want: set[int] | None, collapse: bool, check: bool) -> str:
    """Returns 'unchanged' | 'stripped' | 'reverted' | 'skip:<why>'."""
    mod = LANGS[os.path.splitext(path)[1]]
    try:
        fh = open(path, encoding="utf-8")
    except OSError as e:
        return f"skip:cannot open ({type(e).__name__})"
    with fh:
        try:
            src = fh.read()
        except (UnicodeDecodeError, OSError) as e:
            # `list` already tolerates these; a sweep must too. Letting it raise
            # aborted the whole run part-written on the first odd byte.
            return f"skip:cannot read ({type(e).__name__})"
        # Reading is universal-newline, so `src` is LF whatever the file used.
        # Write it back the way we found it: a CRLF file rewritten as LF changes
        # every line, and no normaliser can see that -- they read universal-
        # newline too, so the gate would pass a whole-file rewrite as a comment
        # strip. Only a uniformly-CRLF file is translated back; mixed or lone-CR
        # files have no "the way we found it" to restore.
        newline = "\r\n" if fh.newlines == "\r\n" else None
    if _ALIEN_BREAK.search(src):
        return "skip:line terminator str.splitlines() splits on but the language does not"
    spans = [c for c in _find(mod, src) if not _kept(mod, c[3])]
    if want is not None:
        spans = [c for c in spans if c[0] in want]
    if not spans:
        return "unchanged"

    try:
        before = mod.normalize(path)
    except Exception as e:
        return f"skip:does not parse ({type(e).__name__})"

    try:
        protected = mod.protected_lines(src)
    except Exception:
        protected = set()
    new = excise(src, spans, protected, collapse)
    if new == src:
        return "unchanged"
    if check:
        # Run the same gate a write would, on a copy. Returning "stripped" here
        # without normalising made --check unable to predict a revert, which is
        # the one thing it is for. The copy sits beside the original so modules
        # that resolve a toolchain from the file's directory (JS -> typescript)
        # and dispatch on extension behave identically.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                   suffix=os.path.splitext(path)[1])
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as fh:
                fh.write(new)
            try:
                ok = mod.normalize(tmp) == before
            except Exception:
                ok = False
        finally:
            os.unlink(tmp)
        return "stripped" if ok else "reverted"

    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(new)
    try:
        after = mod.normalize(path)
        ok = after == before
    except Exception:
        ok = False
    if not ok:
        with open(path, "w", encoding="utf-8", newline=newline) as fh:
            fh.write(src)
        return "reverted"
    return "stripped"


def cmd_list(args) -> int:
    rows = []
    for path in _walk(args.paths or ["."]):
        mod = LANGS[os.path.splitext(path)[1]]
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        for line, c0, _c1, text in _find(mod, src):
            keep = _kept(mod, text)
            if keep and not args.all:
                continue
            rows.append({
                "path": path, "line": line, "col": c0,
                "text": text.strip().replace("\n", " ")[:160],
                "keep": keep,
            })
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            mark = "  [KEEP] " if r["keep"] else " "
            print(f"{r['path']}:{r['line']}:{r['col']}:{mark}{r['text']}")
        print(f"\n{len(rows)} comment(s) in {len({r['path'] for r in rows})} file(s)", file=sys.stderr)
    return 0


def _report(results: dict[str, str]) -> int:
    counts: dict[str, int] = {}
    for path, verdict in sorted(results.items()):
        key = verdict.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        if key in ("stripped", "reverted") or key == "skip":
            print(f"{verdict:<40} {path}")
    print("\n" + "   ".join(f"{k}: {v}" for k, v in sorted(counts.items())), file=sys.stderr)
    if counts.get("reverted"):
        print("REVERTED files are a language-module bug, not an acceptable loss.", file=sys.stderr)
    return 1 if counts.get("reverted") else 0


def cmd_strip(args) -> int:
    if args.all:
        for a in args.targets:
            if ":" in a and not os.path.exists(a):
                print(f"--all takes paths, not FILE:LINE (got {a!r})", file=sys.stderr)
                return 2
        targets = {f: None for f in _walk(args.targets or ["."])}
    else:
        if not args.targets:
            print("expected FILE:LINE[,LINE...], or --all to take every comment", file=sys.stderr)
            return 2
        targets = {}
        for spec in args.targets:
            path, _, lines = spec.rpartition(":")
            if not path or not lines:
                print(f"expected FILE:LINE[,LINE...], got {spec!r}", file=sys.stderr)
                return 2
            if os.path.splitext(path)[1] not in LANGS:
                print(f"no language module for {path}", file=sys.stderr)
                return 2
            try:
                want = {int(n) for n in lines.split(",") if n}
            except ValueError:
                print(f"expected FILE:LINE[,LINE...], got {spec!r}", file=sys.stderr)
                return 2
            targets.setdefault(path, set()).update(want)
    results = {p: _strip_file(p, want, args.collapse_blanks, args.check)
               for p, want in targets.items()}
    return _report(results)


def _keep_alternatives(mod) -> list[str]:
    """The KEEP branches, so each can be tested for reachability on its own."""
    alts = getattr(mod, "_KEEP_ALTERNATIVES", None)
    if alts:
        return list(alts)
    keep = getattr(mod, "KEEP", None)
    if not keep:
        return []
    pat = keep.pattern
    if pat.startswith("(?:") and pat.endswith(")"):
        return [a for a in pat[3:-1].split(")|(?:") if a]
    return [pat]


def _check_coords(mod, path, src, found):
    """Do the module's (line, col) land on the text it reported, in the driver's frame?

    excise() addresses src.splitlines(keepends=True). A module that counts lines
    any other way hands over coordinates from a different frame and the driver
    cuts the wrong line while reporting success.
    """
    lines = src.splitlines(keepends=True)
    bad = []
    for line, c0, c1, text in found:
        last = line + text.count("\n")
        if line < 1 or last > len(lines):
            bad.append(f"{path}:{line} span runs past EOF ({last} > {len(lines)})")
            continue
        head = text.split("\n", 1)[0]
        got = lines[line - 1][c0:c0 + len(head)]
        if got != head:
            bad.append(f"{path}:{line}:{c0} reported {head[:40]!r} but frame holds {got[:40]!r}")
        if last != line and lines[last - 1][:c1] != text.rsplit("\n", 1)[1]:
            bad.append(f"{path}:{line} multi-line span end column disagrees")
    return bad


def _check_keep_reachable(mod, comments_seen):
    """Every KEEP branch must be able to fire. A branch that cannot is a silent hole.

    javascript.py listed eslint / @__PURE__ / prettier-ignore as KEEP, but the
    driver matches with re.match against the comment INCLUDING its opener, so the
    bare branches never fired and a sweep stripped live directives.
    """
    dead = []
    for alt in _keep_alternatives(mod):
        try:
            loose = re.compile(alt, re.IGNORECASE)
        except re.error:
            continue
        hits = [c for c in comments_seen if loose.search(c)]
        if hits and not any(_kept(mod, c) for c in hits):
            dead.append(f"{alt!r} matches {len(hits)} comment(s) but KEEP never fires "
                        f"(e.g. {hits[0].strip()[:60]!r})")
    return dead


def _check_gate(mod, path, src, rounds, rng, found):
    """Mutate non-comment, non-whitespace bytes; the normal form must move or raise.

    A normal form too lossy to notice is a normal form that will certify a
    corrupting edit.
    """
    starts, acc = [0], 0
    for l in src.splitlines(keepends=True):
        acc += len(l)
        starts.append(acc)
    spans = set()
    for line, c0, c1, text in found:
        if line - 1 < len(starts):
            off = starts[line - 1] + c0
            spans.update(range(off, off + len(text)))
    cands = [i for i, ch in enumerate(src) if not ch.isspace() and i not in spans]
    if not cands:
        return 0, 0
    try:
        base = mod.normalize(path)
    except Exception:
        return 0, 0
    blind = tried = 0
    tmpdir = os.path.dirname(path) or "."
    for i in rng.sample(cands, min(rounds, len(cands))):
        for mutant in (src[:i] + src[i + 1:], src[:i] + ("Z" if src[i] != "Z" else "Q") + src[i + 1:]):
            fd, tmp = tempfile.mkstemp(dir=tmpdir, suffix=os.path.splitext(path)[1])
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(mutant)
                tried += 1
                try:
                    if mod.normalize(tmp) == base:
                        blind += 1
                except Exception:
                    pass
            finally:
                os.unlink(tmp)
    return tried, blind


def cmd_selftest(args) -> int:
    rng = __import__("random").Random(args.seed)
    files = _walk(args.paths or ["."])
    by_mod: dict[str, list[str]] = {}
    for f in files:
        by_mod.setdefault(LANGS[os.path.splitext(f)[1]].__name__, []).append(f)

    failures = 0
    for name in sorted(by_mod):
        paths = by_mod[name]
        mod = LANGS[os.path.splitext(paths[0])[1]]
        if args.lang and ("." + args.lang.lstrip(".")) not in mod.EXTS:
            continue
        coord_bad, seen, tried, blind = [], [], 0, 0
        sample = set(rng.sample(paths, min(args.files, len(paths))))
        for p in paths:
            try:
                src = open(p, encoding="utf-8").read()
            except Exception:
                continue
            if _ALIEN_BREAK.search(src):
                continue
            found = _find(mod, src)          # one call: for JS this spawns node
            coord_bad.extend(_check_coords(mod, p, src, found))
            seen.extend(c[3] for c in found)
            if p in sample:
                t, b = _check_gate(mod, p, src, args.mutations, rng, found)
                tried += t
                blind += b
        dead = _check_keep_reachable(mod, seen)

        ok = not coord_bad and not dead and not blind
        failures += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {name:16} files={len(paths):5} comments={len(seen):6} "
              f"coords={'ok' if not coord_bad else str(len(coord_bad)) + ' BAD'} "
              f"keep={'ok' if not dead else str(len(dead)) + ' DEAD'} "
              f"gate={tried - blind}/{tried} moved{'' if not blind else '  ' + str(blind) + ' BLIND'}")
        for m in coord_bad[:5]:
            print(f"       coord: {m}")
        for m in dead[:5]:
            print(f"       keep:  {m}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="comments.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("selftest", help="prove each language module against a real corpus")
    p.add_argument("paths", nargs="*")
    p.add_argument("--lang", help="only this extension")
    p.add_argument("--files", type=int, default=6, help="files per language to mutate")
    p.add_argument("--mutations", type=int, default=6, help="offsets mutated per file")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_selftest)

    p = sub.add_parser("list", help="inventory comments as path:line:col: text")
    p.add_argument("paths", nargs="*")
    p.add_argument("--json", action="store_true")
    p.add_argument("--all", action="store_true", help="include KEEP-protected directives")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("strip", help="strip the comments starting on the named lines")
    p.add_argument("targets", nargs="*", metavar="FILE:LINE[,LINE...] | PATH (with --all)")
    p.add_argument("--all", action="store_true",
                   help="take every strippable comment under PATHs (default: .) instead")
    p.add_argument("--collapse-blanks", action="store_true")
    p.add_argument("--check", action="store_true", help="report what would change, write nothing")
    p.set_defaults(fn=cmd_strip)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
