import pytest

import tensorless as tl
from tensorless.web.browser import _check_url_is_safe, _UnsafeUrlError, fetch_page_text

from .conftest import TINY_TEXT_KWARGS


def test_tl_file_loads_with_weights_only(text_corpus, workdir):
    """`.tl` files must load via `torch.load(..., weights_only=True)`,
    since they're explicitly designed to be shared -- weights_only=False
    would let a crafted .tl file execute arbitrary code on load."""
    import torch

    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    # If load_tl internally used weights_only=False this would still
    # succeed, so directly assert the safe path works standalone too.
    payload = torch.load("model.tl", map_location="cpu", weights_only=True)
    assert "model_state_dict" in payload


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "javascript:alert(1)",
    ],
)
def test_unsafe_url_schemes_rejected(url):
    with pytest.raises(_UnsafeUrlError):
        _check_url_is_safe(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata service
        "http://10.0.0.5/",
        "http://192.168.1.1/",
    ],
)
def test_private_and_loopback_hosts_rejected(url):
    with pytest.raises(_UnsafeUrlError):
        _check_url_is_safe(url)


def test_fetch_page_text_fails_soft_on_unsafe_url():
    """fetch_page_text is a best-effort, optional feature -- an unsafe
    URL must return "" rather than raising out of user-facing code."""
    assert fetch_page_text("file:///etc/passwd") == ""
