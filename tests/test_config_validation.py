import pytest

import tensorless as tl
from tensorless.config import TrainConfig
from tensorless.errors import ConfigError


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(optimizer="rmsprop"),
        dict(task="regressoin"),
        dict(precision="fp8"),
        dict(device="gpu"),
        dict(tokenizer="wordpiece"),
        dict(model_type="cnn"),
        dict(architecture="v3"),
    ],
)
def test_invalid_enum_field_raises_config_error(kwargs):
    with pytest.raises(ConfigError):
        TrainConfig(**kwargs).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(epochs=-1),
        dict(epochs=0),
        dict(batch_size=0),
        dict(d_model=-16),
        dict(learning_rate=-0.1),
        dict(learning_rate=0),
        dict(patience=0),
        dict(warmup_steps=-1),
    ],
)
def test_invalid_numeric_field_raises_config_error(kwargs):
    with pytest.raises(ConfigError):
        TrainConfig(**kwargs).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dropout=-0.1),
        dict(dropout=1.0),
        dict(dropout=1.5),
        dict(val_split=1.0),
        dict(val_split=-0.1),
    ],
)
def test_invalid_fraction_field_raises_config_error(kwargs):
    with pytest.raises(ConfigError):
        TrainConfig(**kwargs).validate()


def test_heads_must_divide_d_model():
    with pytest.raises(ConfigError, match="divisible"):
        TrainConfig(d_model=10, heads=3).validate()
    # Should not raise:
    TrainConfig(d_model=32, heads=4).validate()


def test_valid_config_does_not_raise():
    TrainConfig(
        task="text-generation", optimizer="adamw", precision="fp32",
        epochs=5, batch_size=16, learning_rate=1e-3, dropout=0.1, val_split=0.1,
    ).validate()


def test_all_none_config_is_valid():
    """Fields left unset (None) are filled in later by auto-config and
    must never fail validation themselves."""
    TrainConfig().validate()


def test_validation_runs_before_any_file_access(workdir):
    """A bad config must be rejected immediately, before train() ever
    tries to touch the (nonexistent) dataset path."""
    with pytest.raises(ConfigError):
        tl.train("/this/path/does/not/exist/at/all", optimizer="not-a-real-optimizer")
