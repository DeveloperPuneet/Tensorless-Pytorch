# API Reference

## `tensorless.train(path, **kwargs) -> LoadedModel`

Train a model on the dataset at `path`. See
[configuration.md](configuration.md) for every valid keyword argument.
Implements the Smart Auto Check (see [automatic_mode.md](automatic_mode.md)).

Raises:
- `tensorless.DataError` — dataset missing, empty, malformed, or unsupported format
- `tensorless.ConfigError` — invalid/unknown config argument, or dataset changed with `ask_on_data_change=True`
- `tensorless.CheckpointError` — an existing checkpoint is corrupt or unreadable
- `tensorless.SerializationError` — an existing `.tl` file is corrupt

## `tensorless.inspect(path) -> InspectionReport`

Load and analyze a dataset without training. Prints a human-readable
report and returns a structured object.

```python
@dataclass
class InspectionReport:
    path: str
    fingerprint: str
    kind: str              # "text" | "text_labeled" | "tabular"
    task: str               # "text-generation" | "text-classification" | "classification" | "regression"
    n_examples: int
    n_files: int
    columns: List[str]
    sample: Any
    warnings: List[str]
    recommendations: List[str]
    stats: Dict[str, Any]
```

## `tensorless.pretrain(out="english_pretrained.tl", language="english", **kwargs) -> LoadedModel`

Train a text-generation model on the packaged English starter corpus. The
corpus is intended for demos and smoke tests, not as a general-purpose
language model. `language` currently accepts only `"english"`; use `train()`
with your own text corpus for real pretraining.

```python
model = tl.pretrain(out="english.tl", epochs=20, max_seq_len=128)
```

## `tensorless.load(path, device=None, internet="off") -> LoadedModel`

Load a trained `.tl` file for inference. `device` overrides the device
recorded at training time (e.g. load a GPU-trained model on a CPU-only
machine with `device="cpu"`).

Raises `tensorless.SerializationError` if the file is missing, corrupt,
or from an unsupported future format version.

`internet="connect"` enables optional web search for generation calls;
browsing is off by default. See [inference.md](inference.md#internet-browsing-opt-in-off-by-default).

## `tensorless.run(path, prompt=None, internet="off") -> Any`

Convenience wrapper around `load()` for quick command-line-style usage.
See [inference.md](inference.md#tlrun--the-cli-friendly-shortcut).

## `class tensorless.runtime.LoadedModel`

Returned by both `train()` and `load()`.

| Method | Applies to | Description |
|---|---|---|
| `.generate(prompt="", max_new_tokens=200, temperature=0.8, top_k=40, top_p=None, repetition_penalty=1.0, internet=None, internet_max_results=3)` | `text-generation` | Generate a text continuation, optionally using web context |
| `.chat(internet=None)` | `text-generation` | Interactive terminal chat loop |
| `.predict(x, internet=None)` | all tasks | Unified prediction API; `x` is a string (text tasks) or dict/list-of-dicts (tabular tasks) |
| `.info()` | all | Dict summary: task, model type, versions, config, metrics, param count |

Attributes: `.task`, `.model_type`, `.config`, `.meta`, `.metrics`,
`.dataset_fingerprint`, `.device`, `.model` (the underlying
`torch.nn.Module`), `.tokenizer` (the trained tokenizer or `None`),
`.preprocessor` (`TabularPreprocessor` or `None`), and
`.last_web_sources` (the `SearchResult` objects from the most recent browsing
call). Use `.set_internet("connect")` or `.set_internet("off")` to change the
per-model default.

## `class tensorless.TrainConfig`

The dataclass of every trainable override; see
[configuration.md](configuration.md) for field-by-field defaults.

## Errors — `tensorless.errors`

All Tensorless PyTorch exceptions inherit from `TensorlessError`:

```python
try:
    tl.train("./data")
except tl.TensorlessError as e:
    ...
```

| Class | Raised when |
|---|---|
| `DataError` | Dataset can't be read, is empty, or is malformed |
| `ConfigError` | Invalid/unknown configuration, or a Smart-Auto-Check conflict |
| `ModelError` | Unsupported task/model combination, or a runtime prediction-API misuse |
| `CheckpointError` | Checkpoint missing, corrupt, or incompatible |
| `SerializationError` | `.tl` file can't be written or read |

## Lower-level modules

These aren't part of the stable public API but are documented here for
contributors — see [architecture.md](architecture.md) for how they fit
together:

- `tensorless.data.loader.load_dataset(path) -> Dataset`
- `tensorless.data.fingerprint.fingerprint_path(path) -> str`
- `tensorless.auto.detector.detect_task(ds) -> str`
- `tensorless.auto.config.resolve_config(ds, TrainConfig) -> ResolvedConfig`
- `tensorless.models.registry.build_model(task, model_type, cfg, meta) -> nn.Module`
- `tensorless.training.trainer.run_training(...) -> dict`
- `tensorless.checkpoint.manager.CheckpointManager`
- `tensorless.serialization.tl_format.save_tl(path, payload)` / `load_tl(path) -> dict`
