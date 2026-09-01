"""Minimal internet-browsing support for inference-time responses.

Off by default. When a `LoadedModel` is asked to generate with
``internet="connect"``, this module performs a lightweight web search for
the prompt, fetches a couple of top results, and hands back short
text snippets that get folded into the prompt as extra context before
the model generates -- letting a Tensorless-trained model ground its
reply in something more current than its training data.

Deliberately dependency-free (uses only `urllib` + `html.parser` from the
standard library) so `pip install tensorless-pytorch` doesn't have to pull
in a scraping/requests stack just for this optional feature. It is not a
general-purpose scraper or a substitute for a real search API -- if it
can't reach the network or the page shape changes upstream, it fails
soft (returns nothing) rather than raising, so a training/inference run
never breaks because the internet was flaky or unreachable.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import List, Optional

_USER_AGENT = (
    "Mozilla/5.0 (compatible; TensorlessPyTorch/1.0; "
    "+https://github.com/) TextGenerationClient"
)
_SEARCH_URL = "https://html.duckduckgo.com/html/"
_DEFAULT_TIMEOUT = 6.0
_ALLOWED_SCHEMES = {"http", "https"}
# Cap how much of a response we read: a short context snippet never needs
# more than a fraction of this, and without a cap a malicious/misbehaving
# server could send an unbounded response and exhaust memory.
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MB


class _UnsafeUrlError(Exception):
    """Internal: a URL failed the scheme/SSRF safety checks in `_get`.
    Every caller catches this alongside ordinary network errors, so an
    unsafe URL fails the same soft, no-exception way a timeout would --
    this is a "don't browse there" outcome, not something that should
    ever propagate out of this optional, best-effort feature.
    """


def _check_url_is_safe(url: str) -> None:
    """Reject anything that isn't a plain http(s) fetch of a public host.

    Search-result URLs (and any URL that reaches this function generally)
    are not trusted input:
      - without a scheme check, `file://`, `ftp://`, or similar would let
        `urlopen` read local files or reach unintended protocols;
      - without resolving the hostname and checking the IP, a hostname
        that resolves to loopback/private/link-local addresses (e.g.
        127.0.0.1, 169.254.169.254 -- the AWS/GCP/Azure metadata service
        IP -- or an internal 10.x/192.168.x host) would let this
        "browse the web" feature be turned into a way to probe or reach
        services that were never meant to be internet-facing (SSRF).
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise _UnsafeUrlError(f"unsupported URL scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise _UnsafeUrlError("URL has no host")
    try:
        addrinfo = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise _UnsafeUrlError(f"could not resolve host {host!r}: {e}") from e
    for _family, _type, _proto, _canon, sockaddr in addrinfo:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified
        ):
            raise _UnsafeUrlError(f"host {host!r} resolves to a non-public address ({ip})")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validates every redirect target with `_check_url_is_safe` before
    following it. Without this, a request that starts at a safe, public
    URL could still be redirected server-side to a `file://` URL or an
    internal address, bypassing the check in `_get` entirely.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url_is_safe(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirectHandler)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def __str__(self) -> str:
        return f"{self.title}\n{self.snippet}\nSource: {self.url}"


class _TextExtractor(HTMLParser):
    """Strips tags and script/style content, keeping just visible text."""

    _SKIP_TAGS = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.chunks: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self.chunks.append(data.strip())

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.chunks)).strip()


class _ResultLinkParser(HTMLParser):
    """Parses DuckDuckGo's HTML-only search results page
    (html.duckduckgo.com/html/), which has no JS and a stable-enough
    structure: each result is an `<a class="result__a" href="...">title</a>`
    followed by a `<a class="result__snippet">snippet</a>`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: List[SearchResult] = []
        self._in_title = False
        self._in_snippet = False
        self._cur_title = ""
        self._cur_url = ""
        self._cur_snippet = ""

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        cls = attrs_d.get("class", "") or ""
        if tag == "a" and "result__a" in cls:
            self._in_title = True
            self._cur_title = ""
            self._cur_url = attrs_d.get("href", "") or ""
        elif tag == "a" and "result__snippet" in cls:
            self._in_snippet = True
            self._cur_snippet = ""

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
        elif tag == "a" and self._in_snippet:
            self._in_snippet = False
            if self._cur_title.strip() and self._cur_url.strip():
                self.results.append(
                    SearchResult(
                        title=re.sub(r"\s+", " ", self._cur_title).strip(),
                        url=self._cur_url.strip(),
                        snippet=re.sub(r"\s+", " ", self._cur_snippet).strip(),
                    )
                )

    def handle_data(self, data):
        if self._in_title:
            self._cur_title += data
        elif self._in_snippet:
            self._cur_snippet += data


def _get(url: str, timeout: float, data: Optional[bytes] = None) -> str:
    _check_url_is_safe(url)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": _USER_AGENT})
    with _opener.open(req, timeout=timeout) as resp:
        raw = resp.read(_MAX_RESPONSE_BYTES + 1)[:_MAX_RESPONSE_BYTES]
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def web_search(
    query: str, max_results: int = 3, timeout: float = _DEFAULT_TIMEOUT
) -> List[SearchResult]:
    """Search the web for `query`. Returns up to `max_results` results,
    or an empty list if the network is unreachable / the request fails
    for any reason -- callers should treat "no results" as a normal,
    expected outcome, not an error.
    """
    if not query or not query.strip():
        return []
    try:
        body = urllib.parse.urlencode({"q": query}).encode("utf-8")
        html = _get(_SEARCH_URL, timeout=timeout, data=body)
    except (urllib.error.URLError, socket.timeout, OSError, ValueError, _UnsafeUrlError):
        return []
    parser = _ResultLinkParser()
    try:
        parser.feed(html)
    except Exception:
        return []
    return parser.results[:max_results]


def fetch_page_text(url: str, timeout: float = _DEFAULT_TIMEOUT, max_chars: int = 1500) -> str:
    """Fetch `url` and return its visible text, truncated to `max_chars`.
    Returns "" on any failure (unreachable host, non-HTML content, an
    unsafe/non-public URL, etc.).
    """
    try:
        html = _get(url, timeout=timeout)
    except (urllib.error.URLError, socket.timeout, OSError, ValueError, _UnsafeUrlError):
        return ""
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        return ""
    text = extractor.text()
    return text[:max_chars]


def browse_for_context(
    query: str, max_results: int = 3, fetch_top_page: bool = False, timeout: float = _DEFAULT_TIMEOUT
) -> str:
    """High-level helper: search the web for `query` and return a single
    formatted text block of context suitable for prepending to a prompt.
    Returns "" if nothing could be found (network down, no results, ...).
    """
    results = web_search(query, max_results=max_results, timeout=timeout)
    if not results:
        return ""
    lines = [f"Web search results for: {query}"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title} -- {r.snippet} (source: {r.url})")
    if fetch_top_page:
        extra = fetch_page_text(results[0].url, timeout=timeout)
        if extra:
            lines.append(f"\nExcerpt from top result ({results[0].url}):\n{extra}")
    return "\n".join(lines)
