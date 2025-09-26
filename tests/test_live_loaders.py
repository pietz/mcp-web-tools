import os
from pathlib import Path

import pytest

from mcp.server.fastmcp import Image
from mcp_web_tools.loaders import (
    load_content,
    load_image_file,
    load_pdf_document,
    load_webpage,
)


def _load_dotenv_if_present():
    for filename in (".env", "local.env"):
        env_path = Path.cwd() / filename
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


RUN_LIVE_LOADER_TESTS = os.getenv("RUN_LIVE_LOADER_TESTS") == "1"


def _xfail_loader(target: str, error: Exception | str) -> None:
    pytest.xfail(f"{target} live loader check failed: {error}")


@pytest.mark.asyncio
@pytest.mark.live_loader
@pytest.mark.skipif(
    not RUN_LIVE_LOADER_TESTS,
    reason="Live loader tests disabled; set RUN_LIVE_LOADER_TESTS=1 to enable",
)
async def test_live_load_webpage_returns_content():
    _load_dotenv_if_present()
    try:
        result = await load_webpage("https://example.com", limit=2000)
    except Exception as exc:  # pragma: no cover
        _xfail_loader("load_webpage", exc)
        return

    if not isinstance(result, str) or result.startswith("Error"):
        _xfail_loader("load_webpage", "unexpected result content")

    assert "Example Domain" in result


@pytest.mark.asyncio
@pytest.mark.live_loader
@pytest.mark.skipif(
    not RUN_LIVE_LOADER_TESTS,
    reason="Live loader tests disabled; set RUN_LIVE_LOADER_TESTS=1 to enable",
)
async def test_live_load_pdf_document_returns_text():
    _load_dotenv_if_present()
    try:
        result = await load_pdf_document(
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            limit=2000,
        )
    except Exception as exc:  # pragma: no cover
        _xfail_loader("load_pdf_document", exc)
        return

    if not isinstance(result, str) or result.startswith("Error"):
        _xfail_loader("load_pdf_document", "unexpected result content")

    assert "Dummy PDF file" in result


@pytest.mark.asyncio
@pytest.mark.live_loader
@pytest.mark.skipif(
    not RUN_LIVE_LOADER_TESTS,
    reason="Live loader tests disabled; set RUN_LIVE_LOADER_TESTS=1 to enable",
)
async def test_live_load_image_file_returns_image():
    _load_dotenv_if_present()
    try:
        image = await load_image_file("https://placekitten.com/200/300")
    except Exception as exc:  # pragma: no cover
        _xfail_loader("load_image_file", exc)
        return

    if not isinstance(image, Image) or not isinstance(image.data, (bytes, bytearray)):
        _xfail_loader("load_image_file", "unexpected image payload")

    assert len(image.data) > 0


@pytest.mark.asyncio
@pytest.mark.live_loader
@pytest.mark.skipif(
    not RUN_LIVE_LOADER_TESTS,
    reason="Live loader tests disabled; set RUN_LIVE_LOADER_TESTS=1 to enable",
)
async def test_live_load_content_auto_detects():
    _load_dotenv_if_present()
    try:
        result = await load_content("https://example.com", limit=2000)
    except Exception as exc:  # pragma: no cover
        _xfail_loader("load_content", exc)
        return

    if not isinstance(result, str) or result.startswith("Error"):
        _xfail_loader("load_content", "unexpected result content")

    assert "Example Domain" in result
