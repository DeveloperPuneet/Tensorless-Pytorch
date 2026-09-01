from tensorless.data.fingerprint import fingerprint_path


def test_fingerprint_stable(text_corpus):
    fp1 = fingerprint_path(text_corpus)
    fp2 = fingerprint_path(text_corpus)
    assert fp1 == fp2


def test_fingerprint_changes_on_content_change(workdir):
    path = workdir / "a.txt"
    path.write_text("hello world")
    fp1 = fingerprint_path(str(path))
    path.write_text("hello world!!!")
    fp2 = fingerprint_path(str(path))
    assert fp1 != fp2


def test_fingerprint_changes_on_new_file(workdir):
    d = workdir / "data"
    d.mkdir()
    (d / "a.txt").write_text("hello")
    fp1 = fingerprint_path(str(d))
    (d / "b.txt").write_text("world")
    fp2 = fingerprint_path(str(d))
    assert fp1 != fp2


def test_fingerprint_unaffected_by_mtime(workdir):
    import os
    import time

    path = workdir / "a.txt"
    path.write_text("hello world")
    fp1 = fingerprint_path(str(path))
    # Touch the file (content unchanged) -- fingerprint must be stable.
    time.sleep(0.01)
    os.utime(path, None)
    fp2 = fingerprint_path(str(path))
    assert fp1 == fp2


def test_fingerprint_ignores_hidden_directories(workdir):
    """A directory-wide fingerprint must only reflect files the loader
    would actually read -- churn inside a hidden directory like `.git`
    must never change the fingerprint (previously it did, since
    fingerprinting walked *every* file, hidden dirs included)."""
    d = workdir / "data"
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("hello")
    fp1 = fingerprint_path(str(d))

    hidden = d / ".git"
    hidden.mkdir()
    (hidden / "internal_object").write_text("some git internals")
    fp2 = fingerprint_path(str(d))
    assert fp1 == fp2


def test_fingerprint_ignores_unsupported_extensions(workdir):
    """A file the loader would silently skip (unsupported extension)
    must not perturb the fingerprint either."""
    d = workdir / "data"
    d.mkdir()
    (d / "a.txt").write_text("hello")
    fp1 = fingerprint_path(str(d))

    (d / "notes.pdf").write_bytes(b"%PDF-1.4 not a real pdf")
    fp2 = fingerprint_path(str(d))
    assert fp1 == fp2


def test_fingerprint_still_changes_on_supported_file_edit(workdir):
    """Sanity check that the extension filter doesn't accidentally make
    the fingerprint insensitive to real data changes."""
    d = workdir / "data"
    d.mkdir()
    (d / "a.txt").write_text("hello")
    fp1 = fingerprint_path(str(d))
    (d / "a.txt").write_text("hello, changed")
    fp2 = fingerprint_path(str(d))
    assert fp1 != fp2
