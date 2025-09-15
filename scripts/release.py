#!/usr/bin/env python3
"""
Release helper for GitHub Actions.

Functions:
- detect: emit version, prev_version, release_needed to GITHUB_OUTPUT or stdout
- build: run `uv sync` + `uv build`
- publish: run `uv publish --token $PYPI_TOKEN`

Usage in CI:
  python scripts/release.py detect
  python scripts/release.py build
  PYPI_TOKEN=... python scripts/release.py publish
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except Exception as e:  # pragma: no cover
    print(f"tomllib import failed: {e}", file=sys.stderr)
    raise


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def read_version_from_bytes(data: bytes) -> str:
    cfg = tomllib.loads(data.decode("utf-8"))
    return cfg["project"]["version"]


def current_version() -> str:
    with PYPROJECT.open("rb") as f:
        return read_version_from_bytes(f.read())


def previous_version() -> str | None:
    try:
        prev_bytes = subprocess.check_output(["git", "show", "HEAD^:pyproject.toml"], cwd=ROOT)
        return read_version_from_bytes(prev_bytes)
    except subprocess.CalledProcessError:
        return None


def write_output(name: str, value: str) -> None:
    out_file = os.environ.get("GITHUB_OUTPUT")
    line = f"{name}={value}\n"
    if out_file:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(line)
    else:
        # Fallback to stdout for local runs
        sys.stdout.write(line)


def cmd_detect() -> int:
    curr = current_version()
    prev = previous_version() or ""
    release_needed = "true" if curr != prev else "false"
    write_output("version", curr)
    write_output("prev_version", prev)
    write_output("release_needed", release_needed)
    return 0


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def cmd_build() -> int:
    run(["uv", "sync"])
    run(["uv", "build"])
    return 0


def cmd_publish() -> int:
    token = os.environ.get("PYPI_TOKEN") or os.environ.get("UV_PUBLISH_TOKEN")
    if not token:
        print("PYPI_TOKEN not set; refusing to publish", file=sys.stderr)
        return 2
    run(["uv", "publish", "--token", token])
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: release.py [detect|build|publish]", file=sys.stderr)
        return 64
    cmd = argv[1]
    if cmd == "detect":
        return cmd_detect()
    if cmd == "build":
        return cmd_build()
    if cmd == "publish":
        return cmd_publish()
    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

