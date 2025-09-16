# Repository Guidelines

## Project Structure & Module Organization
- `mcp_web_tools/` — core package
  - `__init__.py`: FastMCP server setup, tools (`search_web`, `fetch_url`).
  - `search.py`: Brave → Google → DuckDuckGo search pipeline.
  - `loaders.py`: universal loader for webpages, PDFs, images.
- `tests/`: pytest suite (unit + async tests).
- `pyproject.toml`: metadata, dependencies, console script `mcp-web-tools`.
- `dist/`: build artifacts (`uv build`).

## Build, Test, and Development Commands
- Install/sync deps: `uv sync`
- Run all tests: `uv run pytest -q`
- Run a specific test: `uv run pytest tests/test_search.py::TestWebSearch::test_web_search_all_fail`
- Start MCP server locally: `uv run mcp-web-tools`
- Build package: `uv build`

## Coding Style & Naming Conventions
- Python 3.12, PEP 8, 4‑space indentation.
- Use type hints (`dict | None`, `list[str]`); avoid `Any` where possible.
- Import order: stdlib → third‑party → local.
- Keep functions small and focused; prefer composable helpers.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio` for async.
- Tests live in `tests/`, files `test_*.py`, functions `test_*`.
- Mock external HTTP/browsers; cover provider fallbacks and timeouts.
- Run fast and deterministically; avoid network by default.

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scoped (e.g., "search: improve DDG fallback").
- Reference issues (`Fixes #123`) when applicable.
- PRs include a clear rationale, behavior changes, test updates, and any config/env notes.
- When you ship any new feature or bug fix, bump the `version` in `pyproject.toml` using semantic versioning that matches the scope of the change.

## Security & Configuration Tips
- MCP stdio compliance: stdout must contain only newline‑delimited JSON‑RPC; write logs to stderr only. Avoid `print()`.
- Transport: run with `transport="stdio"` (default here via `mcp_web_tools.__init__`).
- Brave Search: set `BRAVE_SEARCH_API_KEY` for best results. Persistent logging is disabled by default via `BRAVE_SEARCH_PYTHON_CLIENT_LOG_FILE_NAME = os.devnull`.
- Quick check: `echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | uv run mcp-web-tools` — ensure stdout is pure JSON.
