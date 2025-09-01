# Repository Guidelines

## Project Structure & Modules
- `mcp_web_tools/`: core package
  - `__init__.py`: FastMCP server setup, tool entrypoints (`search_web`, `fetch_url`).
  - `search.py`: Brave→Google→DDG search pipeline.
  - `loaders.py`: universal loader for webpages, PDFs, images.
- `tests/`: pytest suite (incl. async tests).
- `pyproject.toml`: package metadata, deps, `project.scripts` (`mcp-web-tools`).
- `dist/`: build artifacts (created by `uv build`).

## Build, Test, and Dev Commands
- Install/sync deps: `uv sync`
- Run tests: `uv run pytest -q`
- Run a specific test: `uv run pytest tests/test_search.py::TestWebSearch::test_web_search_all_fail`
- Start MCP server locally: `uv run mcp-web-tools`
- Build distribution: `uv build`

## Coding Style & Conventions
- Python 3.12, 4-space indentation, PEP 8.
- Use type hints throughout (e.g., `dict | None`, `list[str]`).
- Import order: stdlib → third-party → local.
- Keep modules focused; prefer small, composable functions.
- Log at info/warn/error in network and extraction paths.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`.
- Test files in `tests/` named `test_*.py`; functions `test_*`.
- Cover fallbacks (Brave→Google→DDG) and error/timeouts.
- For async code, use `pytest.mark.asyncio` and `AsyncMock`.
- Run fast, deterministic tests; mock external HTTP and browsers.

## Commit & Pull Requests
- Commits: imperative, concise, scoped (e.g., "deps update", "tests: fallback order").
- Reference issues when applicable (`Fixes #123`).
- PRs must include: clear description, rationale, before/after notes, test updates, and any config/env changes.

## Configuration & Security
- Searching: optional `BRAVE_SEARCH_API_KEY` improves results; without it, falls back to Google→DDG.
- Network timeouts are enforced; avoid introducing long-running requests.
- Do not commit secrets; use env vars and document them in PRs.

## Agent/Tooling Notes
- Entry point: console script `mcp-web-tools` (defined in `pyproject.toml`).
- For Claude: `claude mcp add web-tools uvx mcp-web-tools` (optionally add `-e BRAVE_SEARCH_API_KEY=<key>`).
