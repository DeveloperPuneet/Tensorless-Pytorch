# Command-Line Interface

Installing Tensorless PyTorch (`pip install tensorless-pytorch`) puts a `tensorless` command
on your `PATH`.

## `tensorless train`

```bash
tensorless train <path> [options]
```

| Option | Description |
|---|---|
| `--out PATH` | Output `.tl` path (default: `model.tl`) |
| `--force` | Retrain even if a matching model/checkpoint exists |
| `--d-model N` | Hidden dimension |
| `--layers N` | Number of layers |
| `--heads N` | Attention heads |
| `--batch-size N` | Batch size |
| `--gradient-accumulation-steps N` | Micro-batches per optimizer update |
| `--epochs N` | Max epochs |
| `--learning-rate F` | Learning rate |
| `--device {cpu,cuda,tpu,mps}` | Force a device |
| `--pretrained PATH` | Initialize from an existing `.tl` model |
| `--quiet` | Suppress training logs |

For any configuration option not exposed as a CLI flag, use the Python
API — the CLI covers the common cases; `tl.train()` covers everything in
[configuration.md](configuration.md).

Example:

```bash
tensorless train ./data --out sentiment.tl --epochs 20 --batch-size 32
```

Fine-tune from the CLI with the same lifecycle rules as the Python API:

```bash
tensorless train ./support-chats.jsonl \
  --pretrained base.tl --out support.tl --epochs 5
```

## `tensorless run`

```bash
tensorless run <model.tl> [--prompt TEXT] [--internet {off,connect}]
```

- Text-generation model, no `--prompt`: starts an interactive chat.
- Text-generation model, with `--prompt`: prints one generated
  continuation.
- Text-classification model, with `--prompt`: prints the predicted
  class.
- Tabular (classification/regression) model: the CLI can't accept
  structured input as a single string, so it prints a pointer to the
  Python API (`tl.load(path).predict({...})`).
- `--internet connect`: lets a text-generation model search the web for
  extra context before answering (off by default). In interactive chat,
  you can also type `internet on` / `internet off` to toggle it.

## `tensorless inspect`

```bash
tensorless inspect <path>
```

Loads the dataset, runs task detection, and prints a report: detected
kind and task, example count, columns (for tabular data), and any
warnings/recommendations — without training anything.

## `tensorless info`

```bash
tensorless info <model.tl>
```

Prints a JSON summary of a trained model: task, model type, versions,
whether training completed, final metrics, and a truncated dataset
fingerprint.

```json
{
  "task": "classification",
  "model_type": "mlp",
  "tensorless_version": "0.9.0",
  "tl_format_version": 1,
  "training_complete": true,
  "metrics": {"final_train_loss": 0.42, "final_val_loss": 0.51, "global_step": 96, "elapsed_seconds": 0.14},
  "dataset_fingerprint": "de779ad33e2dc1d4"
}
```

## Exit codes

All commands return `0` on success. Errors raised as `TensorlessError`
subclasses (bad data, bad config, corrupt checkpoint/model, etc.) are
caught, printed to stderr as `tensorless: error: <message>`, and result
in exit code `1`.
