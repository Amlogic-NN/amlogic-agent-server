# -*- coding: utf-8 -*-
"""Tests for URL shortener functionality in parse_search_results_to_json."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy_server.tools_hook import (
    encode_url,
    get_url,
    set_base_url,
    parse_search_results_to_json,
)

# ---------------------------------------------------------------------------
# 设置测试用的 base URL，使 encode_url 返回完整的可访问地址
# ---------------------------------------------------------------------------
TEST_BASE_URL = "http://test-server:8000"
set_base_url(TEST_BASE_URL)


def _extract_short_code(full_url: str) -> str:
    """从完整的重定向 URL 中提取短代码部分（/r/ 之后的 8 位 hex）。"""
    m = re.search(r"/r/([a-zA-Z0-9]+)$", full_url)
    assert m is not None, f"Cannot extract short_code from: {full_url}"
    return m.group(1)


# ============================================================================
# 1.  encode_url / get_url
# ============================================================================


def test_encode_url_is_deterministic():
    """Same URL should always produce the same full redirect URL."""
    url = "https://example.com/very/long/path?query=1"
    result1 = encode_url(url)
    result2 = encode_url(url)
    assert result1 == result2
    assert result1.startswith(TEST_BASE_URL + "/r/")


def test_encode_url_returns_full_fastapi_url():
    """encode_url must return a full URL pointing to the FastAPI redirect endpoint."""
    result = encode_url("https://example.com/path")
    assert result.startswith("http://"), f"Not a full URL: {result}"
    assert re.match(
        rf"{re.escape(TEST_BASE_URL)}/r/[a-zA-Z0-9]+$", result
    ), f"Unexpected URL format: {result}"


def test_encode_url_short_code_part_is_8_chars():
    """The short code part in the full URL should be 8 alphanumeric characters."""
    result = encode_url("https://example.com/path")
    short_code = _extract_short_code(result)
    assert len(short_code) == 8
    assert re.fullmatch(r'[a-zA-Z0-9]+', short_code)


def test_get_url_returns_original():
    """get_url should return the original URL for a valid short code."""
    url = "https://example.com/page"
    full = encode_url(url)
    short_code = _extract_short_code(full)
    assert get_url(short_code) == url


def test_get_url_unknown_code_returns_none():
    """get_url should return None for an unknown short code."""
    assert get_url("nonexistent") is None


def test_different_urls_have_different_codes():
    """Different URLs should produce different short codes."""
    code1 = _extract_short_code(encode_url("https://example.com/a"))
    code2 = _extract_short_code(encode_url("https://example.com/b"))
    assert code1 != code2


# ============================================================================
# 2.  parse_search_results_to_json with URL shortening
# ============================================================================


def test_parse_search_results_replaces_urls_with_full_redirect_urls():
    """URLs in search results should be replaced with full redirect URLs."""
    text = (
        "Results for: test query (via Google)\n"
        "1. Example Title\n"
        "   https://www.example.com/very-long-path/article?id=12345\n"
        "   This is a summary of the result.\n"
        "2. Another Title\n"
        "   https://another.example.com/page\n"
        "   Another summary line.\n"
    )
    result = json.loads(parse_search_results_to_json(text))

    assert result["count"] == 2
    assert result["engine"] == "Google"

    for item in result["results"]:
        full_url = item["url"]
        assert full_url.startswith(TEST_BASE_URL + "/r/"), f"Not a full redirect URL: {full_url}"
        short_code = _extract_short_code(full_url)
        original = get_url(short_code)
        assert original is not None
        assert original.startswith("https://")


def test_parse_search_results_preserves_url_in_store():
    """Short codes in results should map back to the correct original URLs."""
    url1 = "https://www.example.com/page1"
    url2 = "https://www.example.com/page2"

    text = (
        "Results for: test (via Bing)\n"
        f"1. Title One\n"
        f"   {url1}\n"
        f"   Summary one.\n"
        f"2. Title Two\n"
        f"   {url2}\n"
        f"   Summary two.\n"
    )
    result = json.loads(parse_search_results_to_json(text))

    assert get_url(_extract_short_code(result["results"][0]["url"])) == url1
    assert get_url(_extract_short_code(result["results"][1]["url"])) == url2


def test_parse_search_results_no_urls():
    """Results with no URLs should have empty url field."""
    text = (
        "Results for: query (via DuckDuckGo)\n"
        "1. Title Only\n"
        "   Just a summary with no URL.\n"
    )
    result = json.loads(parse_search_results_to_json(text))

    assert result["count"] == 1
    assert result["results"][0]["url"] == ""


def test_parse_search_results_empty_input():
    """Empty input should return empty results."""
    result = json.loads(parse_search_results_to_json(""))
    assert result == {"results": [], "count": 0, "engine": ""}


def test_parse_search_results_unknown_engine():
    """When engine cannot be parsed, it should be empty string."""
    text = (
        "Results for: something\n"
        "1. Title\n"
        "   https://example.com\n"
        "   Summary.\n"
    )
    result = json.loads(parse_search_results_to_json(text))
    assert result["engine"] == ""
    assert result["results"][0]["url"].startswith(TEST_BASE_URL + "/r/")


def test_parse_search_results_title_and_summary_preserved():
    """Title and summary should be preserved after URL shortening."""
    text = (
        "Results for: search (via Google)\n"
        "1. Hello World\n"
        "   https://example.com/hello\n"
        "   This is a multi-line\n"
        "   summary of the result.\n"
    )
    result = json.loads(parse_search_results_to_json(text))

    item = result["results"][0]
    assert item["title"] == "Hello World"
    assert item["summary"] == "This is a multi-line\nsummary of the result."
    assert item["url"].startswith(TEST_BASE_URL + "/r/")


# ============================================================================
# 3.  Full URL format validation
# ============================================================================


def test_encode_url_characters_in_alphanumeric_range():
    """All characters in the short code part must be within a-zA-Z0-9."""
    test_urls = [
        "https://example.com",
        "https://foo.bar/baz?q=1&x=2",
        "https://very.long.domain.name/with/deep/nested/path/and?many=params&more=values",
    ]
    for url in test_urls:
        full = encode_url(url)
        short_code = _extract_short_code(full)
        assert re.fullmatch(r'[a-zA-Z0-9]+', short_code), \
            f"Invalid short code in URL: {full}"


def test_encode_url_format_is_valid_http_url():
    """The returned URL must be a syntactically valid HTTP URL."""
    result = encode_url("https://example.com")
    assert re.match(r'^https?://[^/]+/r/[a-zA-Z0-9]+$', result), \
        f"Invalid URL format: {result}"
