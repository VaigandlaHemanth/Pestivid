#!/usr/bin/env python3
"""
Regression guards for the defects the August 2026 audit found.

Run locally or from CI:

    python tools/guards.py            # check
    python tools/guards.py --list     # show what each guard defends against

WHY THIS IS NOT A GREP
    The first version of these checks was a set of `grep -v` lines in the CI
    workflow, and every one of them failed -- on the comments that *describe* the
    bug being prevented. A guard that cannot coexist with its own documentation
    is a guard people delete.

    So: comments are always stripped. String literals are stripped too UNLESS the
    guard sets keep_strings -- because for some defects (a fabrication prompt, a
    hardcoded model id, a bind address) the offending text *is* a string literal.
    Getting this wrong made two guards silently never fire.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "artifacts", "tools"}


# ── comment / string stripping ──────────────────────────────────────────────
def strip_python(src: str, keep_strings: bool = False) -> str:
    """Remove comments (and string literals unless keep_strings), keeping line numbers."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            drop = (tok.type == tokenize.COMMENT
                    or (tok.type == tokenize.STRING and not keep_strings))
            if drop:
                out.append((tok.start[0], ""))
            elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                  tokenize.DEDENT, tokenize.ENDMARKER):
                out.append((tok.start[0], tok.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src                                    # fail open; parse job catches it
    lines: dict[int, list[str]] = {}
    for ln, text in out:
        lines.setdefault(ln, []).append(text)
    n = src.count("\n") + 1
    return "\n".join(" ".join(lines.get(i, [])) for i in range(1, n + 1))

_JS_TOKEN = re.compile(
    r"""(?P<block>/\*.*?\*/)      # /* ... */
      | (?P<line>//[^\n]*)        # // ...
      | (?P<tpl>`(?:\\.|[^`\\])*`)
      | (?P<dq>"(?:\\.|[^"\\\n])*")
      | (?P<sq>'(?:\\.|[^'\\\n])*')
      | (?P<html><!--.*?-->)
    """, re.X | re.S)


def strip_js(src: str, keep_strings: bool = False) -> str:
    def repl(m):
        if keep_strings and m.lastgroup in ("tpl", "dq", "sq"):
            return m.group(0)          # a prompt IS a string literal
        # preserve newline count so reported line numbers stay right
        return "\n" * m.group(0).count("\n")
    return _JS_TOKEN.sub(repl, src)


def code_of(path: Path, keep_strings: bool = False) -> str:
    src = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".py":
        return strip_python(src, keep_strings)
    return strip_js(src, keep_strings)


def walk(*suffixes: str):
    for p in ROOT.rglob("*"):
        if p.suffix not in suffixes or not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


# ── guards ──────────────────────────────────────────────────────────────────
GUARDS: list[dict] = [
    dict(
        name="no-password-in-client-storage",
        defends=("Signup used to write the user's CLEARTEXT password into "
                 "localStorage as `passwordHash: this.signupForm.password` -- a "
                 "field name implying it was hashed, next to the real user record. "
                 "Any XSS on the page, or anyone with the device, could read it, "
                 "and users reuse passwords. The client never needs the password "
                 "after login; the JWT is the credential."),
        suffixes=(".html", ".js"),
        roots=("public",),
        patterns=[r"passwordHash\s*:\s*this\.",
                  r"passwordHash\s*:\s*['\"][^'\"]+['\"]",
                  r"setItem\([^)]*password"],
        keep_strings=True,
    ),
    dict(
        name="no-overclaiming-verification-copy",
        defends=("A hash and a Bitcoin anchor prove the FILE has not changed and "
                 "existed by a certain time. They prove nothing about where or when "
                 "the video was filmed, or that it shows this farmer's land -- which "
                 "is the thing an investor actually needs. Copy like 'blockchain "
                 "verified' invites people to risk money on a guarantee that was "
                 "never made. State what was checked, not a badge."),
        suffixes=(".html", ".js"),
        roots=("public",),
        patterns=[r"blockchain[\s-]?verified",
                  r"secure\s+blockchain",
                  r"blockchain[\s-]?linked\s+video"],
        keep_strings=True,
    ),
    dict(
        name="no-storage-credential-in-frontend",
        defends=("index.html once hardcoded the Pinata JWT and POSTed the video "
                 "straight to api.pinata.cloud. That shipped the storage credential "
                 "to every visitor, and because the server never saw the file it had "
                 "to accept `cid` and `videoFileHash` from req.body -- so the "
                 "'integrity hash' was chosen by the uploader and proved nothing. "
                 "Uploads go through POST /api/videos/upload, which hashes and pins "
                 "server-side with a key held only in backend/.env."),
        suffixes=(".html", ".js"),
        roots=("public",),
        patterns=[r"api\.pinata\.cloud",
                  r"pinataJwt\s*[:=]\s*['\"]",
                  r"PINATA_JWT"],
        keep_strings=True,   # the URL and the key name are string literals
    ),
    dict(
        name="no-client-hash-as-verified",
        defends=("A videoFileHash from req.body must never be stored as if the "
                 "server computed it, because the blockchain anchor only means "
                 "something for hashes we derived from the actual bytes."),
        suffixes=(".js",),
        roots=("backend/routes",),
        patterns=[r"videoFileHash:\s*videoFileHash"],
        keep_strings=True,
    ),
    dict(
        name="no-q8-backbone",
        defends=("dtype 'q8' does not degrade the leaf classifier, it destroys it. "
                 "Measured with preprocessing held bit-exact, mean cosine distance of "
                 "the 768-d feature vector against the Python reference was 8.78e-1 "
                 "for q8, versus 3.15e-4 for fp16 and 4.12e-2 for q4f16. At 0.878 the "
                 "features are essentially unrelated to what the head was fitted to, "
                 "while the model still returns confident-looking probabilities -- so "
                 "nothing errors and the UI shows a plausible wrong diagnosis. It is "
                 "the obvious size compromise, which is exactly why it needs a guard."),
        suffixes=(".py", ".js", ".html"),
        patterns=[r"""dtype\s*[:=]\s*["']q8["']""",
                  r"""default\s*=\s*["']q8["']""",
                  r"""dtype\s*[:=]\s*["']uint8["']"""],
        keep_strings=True,   # the dtype is a string literal
    ),
    dict(
        name="no-canvas-resize-for-features",
        defends=("The browser must not feed images to the backbone through the "
                 "transformers.js image pipeline. Its canvas resize does not antialias "
                 "a downscale, which shifted features by 5.4e-2..1.4e-1 cosine distance "
                 "on 1500px photos -- for comparison, choosing between proper filters "
                 "(bicubic/bilinear/Lanczos/box) costs under 2.4e-3. Preprocess with "
                 "pil-resize.js and call the model directly."),
        suffixes=(".js", ".html"),
        patterns=[r"""pipeline\s*\(\s*["']image-feature-extraction["']"""],
        keep_strings=True,
    ),
    dict(
        name="no-fabricated-diagnosis",
        defends=("Three code paths once fabricated a plant disease. One asked an LLM to "
                 "'make a realistic, randomised disease prediction' with no image attached."),
        suffixes=(".py", ".js"),
        patterns=[r"randomi[sz]ed disease", r"Vary the predicted disease",
                  r"simulates disease detection",
                  r"Potential Fungal or Pest Issue"],
        # A fabrication prompt IS a string literal, so strings must be kept.
        # Comments are still stripped, so the notes explaining this bug do not
        # trip the guard that prevents it.
        keep_strings=True,
    ),
    dict(
        name="no-hardcoded-model-id",
        defends=("llama3-70b-8192 was decommissioned 2025-08-30 and its replacement was "
                 "deprecated 2026-06-17. Any inline id goes stale; use GROQ_MODEL."),
        suffixes=(".py", ".js"),
        patterns=[r"""model\s*[:=]\s*["']llama""", r"""model_name\s*=\s*["']llama"""],
        keep_strings=True,   # the id is a string literal
    ),
    dict(
        name="no-debug-server",
        defends="Werkzeug's debugger is remote code execution; 0.0.0.0 exposes it to the LAN.",
        suffixes=(".py",),
        patterns=[r"debug\s*=\s*True", r"""host\s*=\s*["']0\.0\.0\.0["']"""],
        keep_strings=True,   # the bind address is a string literal
    ),
    dict(
        name="no-credential-literal",
        defends="A live Pinecone key, three Pinata JWTs and a Supabase key were committed.",
        suffixes=(".py", ".js", ".json", ".html", ".md", ".txt", ".yml", ".bat"),
        patterns=[r"pcsk_[A-Za-z0-9_]{20}", r"gsk_[A-Za-z0-9]{20}",
                  r"sb_publishable_[A-Za-z0-9_-]{10}",
                  r"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\.eyJ1c2"],
        raw=True,   # secrets in comments are still secrets: do NOT strip
    ),
    dict(
        name="no-label-leak",
        defends=("The reported 84.10% was invalid because the ground-truth label reached the "
                 "model through a frozen text branch. Structural guard against its return."),
        suffixes=(".py",),
        patterns=[r"text_prompts\s*\[\s*(label|row\[|y\b)",
                  r"image_features\s*\*\s*[0-9.]+\s*\+\s*text_features"],
    ),
    dict(
        name="no-chained-document-populate",
        defends=("Mongoose 8 Document.populate() returns a Promise, so chaining throws "
                 "AFTER the write commits -- money taken, then a 500."),
        suffixes=(".js",),
        patterns=[r"await\s+\w+\.populate\([^)]*\)\s*\n\s*\.populate\("],
        multiline=True,
    ),
]


def check(guard: dict) -> list[str]:
    hits = []
    pats = [re.compile(p, re.I | (re.S if guard.get("multiline") else 0))
            for p in guard["patterns"]]
    # Optional directory scoping. Some rules are about WHERE code lives, not
    # whether it exists: a Pinata credential is correct in backend/ and a
    # vulnerability in public/. Without this the guard would flag the legitimate
    # server-side use and get switched off, which is how guards die.
    roots = guard.get("roots")
    for path in walk(*guard["suffixes"]):
        if roots:
            rel = path.relative_to(ROOT).as_posix()
            if not any(rel.startswith(r.rstrip("/") + "/") or rel == r for r in roots):
                continue
        text = (path.read_text(encoding="utf-8", errors="replace")
                if guard.get("raw")
                else code_of(path, keep_strings=guard.get("keep_strings", False)))
        for pat in pats:
            for m in pat.finditer(text):
                line = text[:m.start()].count("\n") + 1
                hits.append(f"{path.relative_to(ROOT)}:{line}: {m.group(0)[:70].strip()}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for g in GUARDS:
            print(f"\n{g['name']}\n  {g['defends']}")
        return 0

    failed = 0
    for g in GUARDS:
        hits = check(g)
        status = "PASS" if not hits else "FAIL"
        print(f"  {status}  {g['name']}")
        if hits:
            failed += 1
            print(f"        {g['defends']}")
            for h in hits[:10]:
                print(f"        {h}")
            if len(hits) > 10:
                print(f"        ... and {len(hits) - 10} more")

    print()
    if failed:
        print(f"{failed}/{len(GUARDS)} guard(s) FAILED")
        return 1
    print(f"all {len(GUARDS)} guards pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
