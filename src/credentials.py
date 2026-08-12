"""Shared CADBUSI credentials loader.

Reads a `KEY=value` credentials file into `os.environ`. Deliberately
self-contained — stdlib only, no imports from this repo — so it can be copied
verbatim into any other CADBUSI repo that needs the same secrets. Keep it that
way; the moment it depends on something local it stops being portable.

Do NOT quote values and put no trailing comment on a value line: `docker run
--env-file` does no quote processing at all, so HF_TOKEN="hf_abc" would send
the quotes upstream and 401. Compose v2 and the shell DO strip them, which is
what makes this bite only sometimes. `parse` below strips quotes and trailing
whitespace, but the other two consumers are not under our control.

`KEY=value` rather than JSON because Docker Compose's `env_file:` and the shell
(`set -a; . hf-credentials.env`) both consume that format with no parser at all.
A containerised consumer needs zero code; this module is for everything else.

The file lives in GCS alongside the Mayo CA cert
(`gs://shared-aif-bucket-87d1/Keys/`) so it can be rotated centrally and is
never committed. `setup.sh` fetches it at deploy time.

Search order (first hit wins):
  1. $CADBUSI_CREDENTIALS                  explicit path
  2. ./hf-credentials.env                  repo root / CWD
  3. <this repo>/hf-credentials.env
  4. ~/.config/cadbusi/hf-credentials.env  per-user
  5. /etc/cadbusi/hf-credentials.env       system-wide

Values ALREADY present in the environment always win — a platform-injected
secret (Cloud Run `--set-secrets`, a compose `environment:` entry) must not be
silently overwritten by a stale file left on the box.

Usage:
    from src.credentials import load_credentials
    load_credentials()                 # idempotent, safe to call repeatedly
    token = os.environ.get("HF_TOKEN")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

FILENAME = "hf-credentials.env"

# Set once loaded so repeated calls are free. Holds the path we loaded from, or
# "" when nothing was found (which is not an error — the platform may have
# injected the vars directly).
_loaded_from: Optional[str] = None


def candidate_paths() -> List[Path]:
    """Every location searched, in priority order.

    Walks UP from this file rather than assuming a fixed depth, because the
    repos that share this module put it at different levels
    (`backend/src/credentials.py` in Production, `src/credentials.py` in
    Database). Hard-coding `parent.parent.parent` would silently find nothing
    in the shallower one.
    """
    out = []
    explicit = os.environ.get("CADBUSI_CREDENTIALS")
    if explicit:
        out.append(Path(explicit))
    out.append(Path.cwd() / FILENAME)

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        out.append(ancestor / FILENAME)
        # Stop at the repo root — going further would reach shared parent dirs
        # (F:/CODE/CADBUSI/) where one repo could pick up another's file.
        if (ancestor / ".git").exists():
            break

    out += [
        Path.home() / ".config" / "cadbusi" / FILENAME,
        Path(f"/etc/cadbusi/{FILENAME}"),
    ]
    return out


def parse(text: str) -> Dict[str, str]:
    """Parse `KEY=value` lines.

    Supports `#` comments, blank lines, a leading `export `, and surrounding
    single/double quotes. Does NOT do variable interpolation or multi-line
    values — a credentials file holding opaque tokens needs neither, and both
    are a common source of silently-wrong secrets.
    """
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


def load_credentials(path: Optional[str] = None, force: bool = False) -> Optional[str]:
    """Load the credentials file into `os.environ`. Returns the path used, or
    None when no file was found.

    Idempotent: after the first call this is a no-op unless `force`. Never
    overwrites a variable that is already set.
    """
    global _loaded_from
    if _loaded_from is not None and not force:
        return _loaded_from or None

    paths = [Path(path)] if path else candidate_paths()
    for p in paths:
        try:
            if not p.is_file():
                continue
            values = parse(p.read_text(encoding="utf-8"))
        except OSError:
            continue
        for k, v in values.items():
            os.environ.setdefault(k, v)
        _loaded_from = str(p)
        return _loaded_from

    _loaded_from = ""
    return None


def describe() -> str:
    """One-line status for startup logs. Never prints a secret VALUE — only
    which keys were found, so a misconfigured deploy is diagnosable from logs
    without leaking anything into them."""
    load_credentials()
    if not _loaded_from:
        return f"[credentials] no {FILENAME} found (relying on the environment)"
    return f"[credentials] loaded {_loaded_from}"
