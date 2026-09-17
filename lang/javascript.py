"""JavaScript / TypeScript language module: delegates to the real TypeScript
compiler via ``node`` rather than hand-rolling JS lexing, because ``//`` inside
a string/template/regex/JSX text is not a comment and ``/`` is regex-or-divide
only by parser context.

The normal form is a token stream (one line per token, not printer output or
an AST dump -- a printer reformats and an AST dump is version-fragile), each
line ``<SyntaxKind name> \\t <0|1 preceded-by-newline> \\t <JSON-quoted text>``.
The newline flag matters because a line terminator between tokens is semantic
in JS (ASI): deleting ``a /*\\n*/ ++b``'s comment removes the only newline and
changes the program, and the flag makes that visible.
"""

from __future__ import annotations

import bisect
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache

EXTS = ('.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs')

_KEEP_ALTERNATIVES = (
    r"^\s*/[/*]+\s*<",
    r"^\s*/\*+\s*globals?\b",
)

_KEEP_ANYWHERE = (
    r"eslint",
    r"tslint|jshint|jslint",
    r"@ts-[a-z-]+",
    r"prettier-ignore",
    r"(?:istanbul|c8|v8|node|deno)\s+ignore",
    r"biome-ignore",
    r"webpack[A-Za-z]*",
    r"@vite-ignore",
    r"[@\#]__[A-Za-z0-9_]+__",
    r"sourceMappingURL|sourceURL",
    r"@license|@preserve|@copyright|@internal",
    r"SPDX-[A-Za-z-]+",
    r"@jsx[A-Za-z]*",
    r"@flow",
)
_KEEP_ALTERNATIVES += tuple(r"[\s\S]*?" + p for p in _KEEP_ANYWHERE)
KEEP = re.compile("|".join("(?:%s)" % p for p in _KEEP_ALTERNATIVES), re.IGNORECASE)

_FALLBACK_TS_DIR = os.environ.get('TS_FALLBACK_DIR', '')
_NIX_TS = 'nixpkgs#nodePackages.typescript'

class ToolchainError(RuntimeError):
    """node or the typescript package could not be used."""

class ParseError(ValueError):
    """The file does not parse; no normal form can be trusted for it."""

_RESOLVE_PROBE = r"""
let p = '';
const from = process.env.STRIP_FROM_DIR;
if (from) { try { p = require.resolve('typescript', { paths: [from] }); } catch (e) {} }
if (!p) { try { p = require.resolve('typescript'); } catch (e) {} }
if (!p) process.exit(3);
process.stdout.write(p);
"""

_NODE_PROGRAM = r"""
'use strict';
let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', function (d) { buf += d; });
process.stdin.on('end', function () {
  try {
    process.stdout.write(JSON.stringify(main(JSON.parse(buf))));
  } catch (e) {
    process.stderr.write(String((e && e.stack) || e));
    process.exit(1);
  }
});

function main(req) {
  const ts = require(req.tsPath);
  const want = {};
  for (const w of req.want) want[w] = true;
  const results = {};
  for (const kind of req.kinds) results[kind] = analyze(ts, req.text, kind, want);
  return { tsVersion: ts.version, results: results };
}

function byteOffsets(text, offsets) {
  const uniq = Array.from(new Set(offsets)).sort(function (a, b) { return a - b; });
  const map = new Map();
  let prev = 0, acc = 0;
  for (const o of uniq) {
    acc += Buffer.byteLength(text.slice(prev, o), 'utf8');
    prev = o;
    map.set(o, acc);
  }
  return map;
}

function analyze(ts, text, kindName, want) {
  const names = { TS: 'input.ts', TSX: 'input.tsx', JSX: 'input.jsx', JS: 'input.js' };
  const sf = ts.createSourceFile(names[kindName] || 'input.ts', text,
                                 ts.ScriptTarget.Latest, true, ts.ScriptKind[kindName]);
  const out = { errors: [], comments: [], templates: [], tokens: '' };

  const diags = sf.parseDiagnostics || [];
  for (const d of diags) {
    const at = typeof d.start === 'number' ? d.start : 0;
    const lc = sf.getLineAndCharacterOfPosition(at);
    out.errors.push({
      line: lc.line + 1, col: lc.character + 1,
      message: ts.flattenDiagnosticMessageText(d.messageText, ' ')
    });
  }

  const shebang = /^#![^\n]*/.exec(text);
  const shebangEnd = shebang ? shebang[0].length : 0;

  const isJSDoc = function (n) {
    return n.kind >= ts.SyntaxKind.FirstJSDocNode && n.kind <= ts.SyntaxKind.LastJSDocNode;
  };
  const isToken = function (n) { return n.kind < ts.SyntaxKind.FirstNode; };
  const templateKinds = {};
  templateKinds[ts.SyntaxKind.NoSubstitutionTemplateLiteral] = 1;
  templateKinds[ts.SyntaxKind.TemplateHead] = 1;
  templateKinds[ts.SyntaxKind.TemplateMiddle] = 1;
  templateKinds[ts.SyntaxKind.TemplateTail] = 1;

  const seen = new Set();
  const comments = [];
  const templates = [];
  const tokenLines = [];
  let prevEnd = shebangEnd;
  let firstToken = true;
  if (want.tokens && shebang) tokenLines.push('Shebang\t0\t' + JSON.stringify(shebang[0]));

  const visit = function (node) {
    const full = node.getFullStart();
    const start = node.getStart(sf, false);
    // Leading ranges start collecting only after a line break; trailing ranges stop at
    // the first one. Union of both == every comment in this token's trivia. The
    // end <= start bound matters for JsxText, whose own text starts at its full start
    // and would otherwise be scanned as if it were trivia.
    const ranges = [].concat(ts.getTrailingCommentRanges(text, full) || [],
                             ts.getLeadingCommentRanges(text, full) || []);
    for (const r of ranges) {
      if (r.pos < shebangEnd || r.end > start) continue;
      const key = r.pos + ':' + r.end;
      if (seen.has(key)) continue;
      seen.add(key);
      comments.push([r.pos, r.end]);
    }
    if (!isToken(node)) return;
    if (templateKinds[node.kind]) templates.push([start, node.end]);
    if (want.tokens) {
      // The newline flag is an ASI guard: it records whether the gap before this
      // token could end a statement. Nothing precedes the first token, so ASI
      // cannot apply to it, and its flag must not depend on a leading comment
      // header -- excise() drops a whole-line comment together with its newline,
      // which would otherwise flip 1 -> 0 and revert the strip.
      const gap = text.slice(prevEnd, start);
      const nl = firstToken ? '0' : (/[\r\n\u2028\u2029]/.test(gap) ? '1' : '0');
      firstToken = false;
      tokenLines.push(ts.SyntaxKind[node.kind] + '\t' + nl +
                      '\t' + JSON.stringify(node.getText(sf)));
    }
    prevEnd = node.end;
  };

  (function walk(node) {
    if (isJSDoc(node)) return;   // TS hangs parsed JSDoc off getChildren(); it is trivia, not tokens
    const kids = node.getChildren(sf);
    if (kids.length === 0) { visit(node); return; }
    for (const kid of kids) walk(kid);
  })(sf);

  comments.sort(function (a, b) { return a[0] - b[0]; });
  const offsets = [];
  for (const c of comments) { offsets.push(c[0], c[1]); }
  for (const t of templates) { offsets.push(t[0], t[1]); }
  const bytes = byteOffsets(text, offsets);

  for (const c of comments) out.comments.push([bytes.get(c[0]), bytes.get(c[1]), text.slice(c[0], c[1])]);
  for (const t of templates) out.templates.push([bytes.get(t[0]), bytes.get(t[1])]);
  if (want.tokens) out.tokens = tokenLines.join('\n');
  return out;
}
"""

def _node_argv() -> list[str]:
    if shutil.which('node'):
        return ['node']
    if shutil.which('nix'):
        return ['nix', 'shell', 'nixpkgs#nodejs', '--command', 'node']
    raise ToolchainError("node is not on PATH and nix is unavailable to provide it")

@lru_cache(maxsize=32)
def _resolve_typescript(start_dir: str) -> str:
    """Absolute path of the typescript module to require.

    Order: a repo-local / NODE_PATH-visible install found from ``start_dir`` upward,
    then the checkout named by ``TS_FALLBACK_DIR`` if it is set, then a nix shell.
    Local wins so a repo is analyzed by the very compiler it builds with.
    """
    node = _node_argv()
    env = dict(os.environ)
    attempts: list[tuple[list[str], dict[str, str], str]] = []

    local_env = dict(env, STRIP_FROM_DIR=start_dir or os.getcwd())
    attempts.append((node + ['-e', _RESOLVE_PROBE], local_env, 'local resolution'))

    if _FALLBACK_TS_DIR:
        fb_env = dict(env, STRIP_FROM_DIR=_FALLBACK_TS_DIR)
        attempts.append((node + ['-e', _RESOLVE_PROBE], fb_env, 'fallback ' + _FALLBACK_TS_DIR))

    if shutil.which('nix'):
        nix_env = dict(env)
        nix_env.pop('STRIP_FROM_DIR', None)
        attempts.append((['nix', 'shell', _NIX_TS, '--command'] + node + ['-e', _RESOLVE_PROBE],
                         nix_env, 'nix shell ' + _NIX_TS))

    problems = []
    for argv, penv, label in attempts:
        try:
            proc = subprocess.run(argv, env=penv, capture_output=True, text=True, timeout=300)
        except OSError as exc:
            problems.append('%s: %s' % (label, exc))
            continue
        path = proc.stdout.strip()
        if proc.returncode == 0 and path and os.path.exists(path):
            return path
        problems.append('%s: %s' % (label, (proc.stderr.strip() or 'not found')[:200]))

    raise ToolchainError(
        "cannot resolve the 'typescript' package, so no JS/TS file can be verified.\n"
        "  tried:\n    " + "\n    ".join(problems) +
        "\n  fix: npm install typescript in the repo, set NODE_PATH, or make nix available")

def _run(request: dict, start_dir: str) -> dict:
    request = dict(request, tsPath=_resolve_typescript(start_dir))
    proc = subprocess.run(_node_argv() + ['-e', _NODE_PROGRAM],
                          input=json.dumps(request), capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise ToolchainError('typescript scan failed: ' + (proc.stderr.strip() or 'exit %d' % proc.returncode))
    return json.loads(proc.stdout)

@lru_cache(maxsize=8)
def _analyze(src: str, kinds: tuple[str, ...], start_dir: str) -> dict:
    want = ['comments', 'templates', 'tokens']
    return _run({'text': src, 'kinds': list(kinds), 'want': want}, start_dir)['results']

def _script_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.ts', '.mts', '.cts'):
        return 'TS'
    if ext == '.tsx':
        return 'TSX'
    if ext == '.jsx':
        return 'JSX'
    return 'JS'

def _kinds_for(ext: str | None) -> tuple[str, ...]:
    if not ext:
        return ('TS', 'TSX')
    return (_script_kind('x' + ext),)

def _line_starts(data: bytes) -> list[int]:
    starts = [0]
    i = data.find(b'\n')
    while i != -1:
        starts.append(i + 1)
        i = data.find(b'\n', i + 1)
    return starts

def _locate(starts: list[int], off: int) -> tuple[int, int]:
    row = bisect.bisect_right(starts, off) - 1
    return row + 1, off - starts[row]

def comments(src: str, ext: str | None = None) -> list[tuple[int, int, int, str]]:
    """Every comment in ``src``, in source order.

    One tuple per comment: ``(line, col_start, col_end, text)``. ``line`` is 1-based.
    ``col_start`` / ``col_end`` are 0-based **byte** columns (UTF-8 bytes, not
    characters) within a line.

    A block comment spanning several lines is still exactly ONE tuple: ``line`` and
    ``col_start`` locate its first character, and ``col_end`` is the end column **on
    the comment's last line**. The driver reconstructs that last line as
    ``line + text.count('\\n')`` -- ``text`` is raw source, newlines included, so the
    identity always holds.

    A ``#!`` shebang on line 1 is not a comment token and is never reported.

    ``ext`` (e.g. ``'.tsx'``) picks the parse mode; without it both TS and TSX are
    parsed and only comments both agree on are returned.
    """
    results = _analyze(src, _kinds_for(ext), os.getcwd())
    agreed = None
    for kind in results:
        found = {(c[0], c[1]): c[2] for c in results[kind]['comments']}
        agreed = found if agreed is None else {k: v for k, v in agreed.items() if k in found}
    starts = _line_starts(src.encode('utf-8'))
    out = []
    for (bs, be), text in sorted((agreed or {}).items()):
        line, col_start = _locate(starts, bs)
        _, col_end = _locate(starts, be)
        out.append((line, col_start, col_end, text))
    return out

def protected_lines(src: str, ext: str | None = None) -> set[int]:
    """1-based lines that are blank *as content*, not as formatting.

    In JS that means blank lines inside a template literal chunk: they are part of the
    string's value, so collapsing them rewrites the program. Chunks are taken per
    template token, so blank lines inside a ``${...}`` substitution -- which really are
    formatting -- are correctly left unprotected.
    """
    results = _analyze(src, _kinds_for(ext), os.getcwd())
    starts = _line_starts(src.encode('utf-8'))
    lines = src.splitlines()
    out: set[int] = set()
    for kind in results:
        for bs, be in results[kind]['templates']:
            first, _ = _locate(starts, bs)
            last, _ = _locate(starts, be)
            for row in range(first, last + 1):
                if row <= len(lines) and not lines[row - 1].strip():
                    out.add(row)
    return out

def normalize(path: str, src: str | None = None) -> str:
    """Comment-free normal form of ``path``; raises ``ParseError`` if it does not parse.

    Two files whose normal forms are equal differ only in comments and in whitespace
    that carries no line-terminator meaning -- which is exactly the claim a comment
    strip has to prove. See the module docstring for the token-stream format.
    """
    if src is None:
        with open(path, 'r', encoding='utf-8', newline='') as fh:
            src = fh.read()
    kind = _script_kind(path)
    start_dir = os.path.dirname(os.path.abspath(path)) or os.getcwd()
    result = _analyze(src, (kind,), start_dir)[kind]
    if result['errors']:
        first = result['errors'][0]
        raise ParseError('%s:%d:%d: %s (%d parse error%s)' % (
            path, first['line'], first['col'], first['message'],
            len(result['errors']), '' if len(result['errors']) == 1 else 's'))
    return result['tokens']
