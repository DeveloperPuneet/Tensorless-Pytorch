"""Configuration objects.

`TrainConfig` holds every knob a user can override in `tl.train(...)`.
Any field left as `None` means "let Tensorless PyTorch decide automatically".
`ResolvedConfig` is what the auto-configuration system produces after
filling in every `None` with a concrete value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

_VALID_TASKS = {"text-generation", "text-classification", "classification", "regression"}
_VALID_MODEL_TYPES = {"transformer", "mlp"}
_VALID_ARCHITECTURES = {"v1", "v2"}
_VALID_TOKENIZERS = {"char", "bpe"}
_VALID_OPTIMIZERS = {"adamw", "adam", "sgd"}
_VALID_PRECISIONS = {"fp32", "fp16", "bf16"}
_VALID_DEVICES = {"cpu", "cuda", "mps", "tpu"}


@dataclass
class TrainConfig:
    """User-facing training configuration. Every field is optional;
    unset fields are chosen automatically by the auto-config system.
    """

    # --- output / lifecycle ---
    out: Optional[str] = None                 # output .tl path, default "model.tl"
    force: bool = False                        # force retraining even if unchanged
    resume: Optional[bool] = None               # force/forbid resume (None = auto)
    ask_on_data_change: bool = False            # raise instead of auto-retrain on data change

    # --- task / architecture ---
    task: Optional[str] = None                  # "text-generation", "classification", "regression"
    model_type: Optional[str] = None            # "transformer", "mlp"
    architecture: Optional[str] = None          # "v1" (legacy) or "v2" (RoPE/RMSNorm/SwiGLU); default "v2"
    d_model: Optional[int] = None
    layers: Optional[int] = None
    heads: Optional[int] = None
    ff_mult: Optional[int] = None
    dropout: Optional[float] = None
    max_seq_len: Optional[int] = None
    tokenizer: Optional[str] = None           # "char" or "bpe"
    bpe_vocab_size: Optional[int] = None
    gradient_checkpointing: Optional[bool] = None  # trade compute for memory on big models
    compile: Optional[bool] = None                  # torch.compile() on CUDA when available
    num_workers: Optional[int] = None               # DataLoader worker processes
    multi_gpu: Optional[bool] = None                # auto-wrap with DataParallel when >1 GPU visible
                                                      # in a single process (e.g. a notebook), so a
                                                      # second GPU doesn't just sit idle
    pretrained: Optional[str] = None                 # path to a .tl checkpoint to fine-tune from --
                                                      # initializes weights (backbone only, if the
                                                      # task head differs) instead of training from
                                                      # scratch; architecture + tokenizer are locked
                                                      # to match the pretrained model

    # --- optimization ---
    optimizer: Optional[str] = None             # "adamw", "adam", "sgd"
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    max_steps: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    grad_clip: Optional[float] = None
    warmup_steps: Optional[int] = None

    # --- validation / early stopping ---
    val_split: Optional[float] = None
    patience: Optional[int] = None
    min_delta: Optional[float] = None

    # --- hardware ---
    device: Optional[str] = None                # "cpu", "cuda", "tpu", or None = auto
    precision: Optional[str] = None              # "fp32", "fp16", "bf16"

    # --- checkpointing ---
    checkpoint_every: Optional[int] = None       # steps between checkpoints
    checkpoint_dir: Optional[str] = None

    # --- misc ---
    seed: int = 42
    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def overrides(self) -> Dict[str, Any]:
        """Return only the fields the user explicitly set (non-None)."""
        d = asdict(self)
        d.pop("extra", None)
        return {k: v for k, v in d.items() if v is not None and v is not False}

    def validate(self) -> None:
        """Validate every explicitly-set field, centrally and up front.

        Previously, bad values (an unknown optimizer, a negative
        `epochs`, an out-of-range `dropout`, ...) were only caught deep
        inside the training pipeline -- sometimes as an inconsistent mix
        of `ValueError`/`AssertionError`/`ZeroDivisionError`, sometimes
        only after the dataset had already been loaded and preprocessed.
        Validating here (called immediately in `_build_train_config`,
        before any file I/O) fails fast with one consistent error type
        (`ConfigError`) and a message that names the offending field.

        Fields left as `None` are untouched -- they'll be filled in by
        the auto-config system, whose own choices are always valid by
        construction.
        """
        from .errors import ConfigError

        def fail(field_name: str, msg: str) -> None:
            raise ConfigError(f"Invalid {field_name}: {msg}")

        def check_choice(field_name: str, value: Any, choices: set) -> None:
            if value is not None and value not in choices:
                fail(field_name, f"must be one of {sorted(choices)}, got {value!r}")

        def check_positive_int(field_name: str, value: Any, allow_zero: bool = False) -> None:
            if value is None:
                return
            if not isinstance(value, int) or isinstance(value, bool):
                fail(field_name, f"must be an int, got {value!r}")
            elif value < 0 or (value == 0 and not allow_zero):
                bound = ">= 0" if allow_zero else "> 0"
                fail(field_name, f"must be {bound}, got {value!r}")

        def check_positive_float(field_name: str, value: Any, allow_zero: bool = False) -> None:
            if value is None:
                return
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                fail(field_name, f"must be a number, got {value!r}")
            elif value < 0 or (value == 0 and not allow_zero):
                bound = ">= 0" if allow_zero else "> 0"
                fail(field_name, f"must be {bound}, got {value!r}")

        def check_fraction(field_name: str, value: Any, low: float = 0.0, high: float = 1.0) -> None:
            if value is None:
                return
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                fail(field_name, f"must be a number, got {value!r}")
            elif not (low <= value < high):
                fail(field_name, f"must be in [{low}, {high}), got {value!r}")

        check_choice("task", self.task, _VALID_TASKS)
        check_choice("model_type", self.model_type, _VALID_MODEL_TYPES)
        check_choice("architecture", self.architecture, _VALID_ARCHITECTURES)
        check_choice("tokenizer", self.tokenizer, _VALID_TOKENIZERS)
        check_choice("optimizer", self.optimizer, _VALID_OPTIMIZERS)
        check_choice("precision", self.precision, _VALID_PRECISIONS)
        check_choice("device", self.device, _VALID_DEVICES)

        check_positive_int("d_model", self.d_model)
        check_positive_int("layers", self.layers)
        check_positive_int("heads", self.heads)
        check_positive_int("ff_mult", self.ff_mult)
        check_positive_int("max_seq_len", self.max_seq_len)
        check_positive_int("bpe_vocab_size", self.bpe_vocab_size)
        check_positive_int("num_workers", self.num_workers, allow_zero=True)
        check_positive_int("batch_size", self.batch_size)
        check_positive_int("epochs", self.epochs)
        check_positive_int("max_steps", self.max_steps)
        check_positive_int("gradient_accumulation_steps", self.gradient_accumulation_steps)
        check_positive_int("warmup_steps", self.warmup_steps, allow_zero=True)
        check_positive_int("patience", self.patience)
        check_positive_int("checkpoint_every", self.checkpoint_every)

        check_positive_float("learning_rate", self.learning_rate)
        check_positive_float("weight_decay", self.weight_decay, allow_zero=True)
        check_positive_float("grad_clip", self.grad_clip, allow_zero=True)
        check_positive_float("min_delta", self.min_delta, allow_zero=True)

        check_fraction("dropout", self.dropout, low=0.0, high=1.0)
        check_fraction("val_split", self.val_split, low=0.0, high=1.0)

        if self.d_model is not None and self.heads is not None and self.d_model % self.heads != 0:
            fail("heads", f"d_model ({self.d_model}) must be divisible by heads ({self.heads})")

        if self.pretrained is not None and not isinstance(self.pretrained, str):
            fail("pretrained", f"must be a path string, got {self.pretrained!r}")


@dataclass
class ResolvedConfig:
    """Fully resolved configuration -- every field has a concrete value.
    This is what actually gets used for training and what gets embedded
    in checkpoints and the final `.tl` file.
    """

    out: str
    force: bool
    resume: Optional[bool]
    ask_on_data_change: bool

    task: str
    model_type: str
    architecture: str
    d_model: int
    layers: int
    heads: int
    ff_mult: int
    dropout: float
    max_seq_len: int
    tokenizer: str
    bpe_vocab_size: int
    gradient_checkpointing: bool
    compile: bool
    num_workers: int
    multi_gpu: bool

    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    max_steps: Optional[int]
    gradient_accumulation_steps: int
    grad_clip: float
    warmup_steps: int

    val_split: float
    patience: int
    min_delta: float

    device: str
    precision: str

    checkpoint_every: int
    checkpoint_dir: str

    seed: int
    verbose: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResolvedConfig":
        known = {f for f in cls.__dataclass_fields__.keys()}
        return cls(**{k: v for k, v in d.items() if k in known})
