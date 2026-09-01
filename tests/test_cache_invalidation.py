import tensorless as tl

from .conftest import TINY_TEXT_KWARGS


def test_matching_config_reuses_cached_model(text_corpus, workdir, capsys):
    """Calling train() twice with the exact same config should reuse the
    existing .tl rather than retraining."""
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    mtime_before = (workdir / "model.tl").stat().st_mtime_ns

    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    mtime_after = (workdir / "model.tl").stat().st_mtime_ns

    assert mtime_after == mtime_before


def test_conflicting_architecture_override_forces_retrain(text_corpus, workdir):
    """A cached, *finished* model trained with one d_model must not be
    silently handed back when a later call explicitly asks for a
    different d_model -- the cache must be config-aware, not just
    dataset-fingerprint-aware."""
    kwargs = dict(TINY_TEXT_KWARGS)
    kwargs["d_model"] = 16
    m1 = tl.train(text_corpus, out="model.tl", **kwargs)
    assert m1.config["d_model"] == 16

    kwargs2 = dict(TINY_TEXT_KWARGS)
    kwargs2["d_model"] = 32
    m2 = tl.train(text_corpus, out="model.tl", **kwargs2)
    assert m2.config["d_model"] == 32


def test_conflicting_hyperparameter_override_forces_retrain(tabular_regression_csv, workdir):
    """Same as above but for a non-architecture field (learning_rate),
    on the tabular/regression path."""
    from .conftest import TINY_TABULAR_KWARGS

    kwargs = dict(TINY_TABULAR_KWARGS)
    kwargs["learning_rate"] = 1e-2
    m1 = tl.train(tabular_regression_csv, out="model.tl", task="regression", **kwargs)
    assert m1.config["learning_rate"] == 1e-2

    kwargs2 = dict(TINY_TABULAR_KWARGS)
    kwargs2["learning_rate"] = 1e-3
    m2 = tl.train(tabular_regression_csv, out="model.tl", task="regression", **kwargs2)
    assert m2.config["learning_rate"] == 1e-3


def test_non_conflicting_call_still_uses_cache(text_corpus, workdir):
    """Passing `verbose` (a lifecycle knob, not a training-config field)
    should never be treated as a conflict."""
    tl.train(text_corpus, out="model.tl", verbose=False, **TINY_TEXT_KWARGS)
    mtime_before = (workdir / "model.tl").stat().st_mtime_ns

    tl.train(text_corpus, out="model.tl", verbose=True, **TINY_TEXT_KWARGS)
    mtime_after = (workdir / "model.tl").stat().st_mtime_ns

    assert mtime_after == mtime_before
