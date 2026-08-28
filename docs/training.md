# Training

## Basic usage

```python
import tensorless as tl

tl.train("./data")
```

Returns a `LoadedModel` (see [inference.md](inference.md)) ready for
predictions, and writes `model.tl` (plus a `model.tl.ckpt/` checkpoint
directory) to the current directory.

## What gets printed at startup

Before training begins (when `verbose=True`, the default), Tensorless
prints every resolved configuration value -- architecture, optimization,
validation, hardware, checkpointing -- tagged with how it was decided:

- **`(manual)`** — you passed it explicitly, as a `tl.train(...)` keyword
  argument or a CLI flag.
- **`(auto)`** — chosen automatically from dataset size and detected
  hardware (see [automatic_mode.md](automatic_mode.md)).
- **`(locked)`** — forced to match a `pretrained=` checkpoint's
  architecture/tokenizer when fine-tuning.

```
[tensorless] ===== training configuration =====
[tensorless] -- task / architecture --
[tensorless]   task                         = 'text-generation' (auto)
[tensorless]   d_model                      = 512          (manual)
[tensorless]   ff_mult                      = 4            (auto)
...
[tensorless] 3 manual, 34 auto -- 37 parameters total
[tensorless] ===================================
```

This makes the "fully automatic by default" behavior inspectable rather
than a black box: you can see at a glance exactly which numbers Tensorless
picked for you and which ones you're overriding. Pass `verbose=False` to
suppress this (and all other) training output. Resumed runs print a
shorter note instead, since a resumed run keeps the exact configuration
the checkpoint was originally created with.

## Supported data formats

| Format | Notes |
|---|---|
| `.txt`, `.md` | Treated as raw text for language modeling |
| `.csv`, `.tsv` | Tabular; first row is the header |
| `.json` | A list of records, or `{"data": [...]}` / `{"records": [...]}` |
| `.jsonl`, `.ndjson` | One JSON object per line |
| `.yaml`, `.yml` | Same record shapes as `.json` (requires `pip install pyyaml`) |
| A directory of the above | Merged; see below |
| A directory of class subfolders of `.txt`/`.md` files | Text classification, folder name = label |

For JSON/JSONL/YAML records, Tensorless PyTorch tries several
interpretations, in order, and uses the first one that fits:

1. **A single text field.** A field named `text`, `content`, `body`,
   `document`, or `sentence` is treated as the input text, and `label`,
   `target`, `class`, `category`, or `y` as the label if present. Text
   field only -> text-generation. Text + label -> text-classification.
2. **Chat / turn-list records**, e.g. OpenAI-style
   `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
   or ShareGPT-style
   `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`.
   Also recognized under the aliases `turns`, `dialogue`, `dialog`, `chat`,
   and role names `user`/`human`/`question`/`prompt`/`instruction`/`input`
   and `assistant`/`bot`/`ai`/`gpt`/`answer`/`response`/`output`/`completion`/`reply`.
3. **Flat conversational pairs**, e.g. `{"user": "...", "bot": "..."}`,
   `{"human": "...", "gpt": "..."}`, `{"question": "...", "answer": "..."}`,
   `{"prompt": "...", "completion": "..."}`, or Alpaca-style
   `{"instruction": "...", "input": "...", "output": "..."}`. Any of the
   role aliases from (2) work here too, and an optional `system`/`context`
   field is included as a leading system turn. These are all normalized
   into `"User: ...\nAssistant: ..."`-style text and trained as
   text-generation (chat fine-tuning).
4. **Anything else that looks like free text** (records whose string
   fields read like prose rather than short table cells) is flattened
   into readable `"Key: value"` text and trained as text-generation,
   rather than being forced into the tabular pipeline or rejected. A
   one-line notice is printed when this fallback is used, since it's a
   best-effort guess at an unfamiliar format.

Only if none of the above apply -- i.e. the records genuinely look like a
structured table (short/numeric values, no recognizable text or chat
fields) -- are they treated as tabular data (see
[automatic_mode.md](automatic_mode.md) for how the target column and task
type are chosen).

A directory can mix multiple files of the *same* format (e.g. several
`.csv` files, or several `.txt` files) — they're concatenated. Mixing
plain text files with structured (JSON/CSV/YAML) files in the same
directory raises a `DataError` asking you to separate them.

For tabular data, numeric columns are robustly scaled and missing values use
the training median. ISO-8601 date and datetime columns are converted to
numeric timestamps. Categorical columns are frequency-ranked and capped at
1,000 learned values; rare or unseen values use the `<unk>` category.

Tensorless PyTorch never modifies, moves, or deletes files in your dataset
directory. It only ever reads from `path`; all output goes to the `out`
file and `checkpoint_dir`.

## Overriding auto-configuration

Every field of `TrainConfig` can be passed as a keyword argument. A few
common ones:

```python
tl.train(
    "./data",
    task="text-generation",       # skip auto-detection
    d_model=512, layers=6, heads=8,
    batch_size=32, epochs=20,
    learning_rate=3e-4,
    val_split=0.15,
    device="cuda",
    out="my_model.tl",
)
```

See [configuration.md](configuration.md) for the full list.

## Validation and early stopping

By default, Tensorless PyTorch holds out `val_split` of the data (10% for
datasets with 50+ examples, 0% for smaller ones where a held-out split
wouldn't be meaningful) and tracks validation loss after each epoch. If
validation loss doesn't improve by at least `min_delta` for `patience`
consecutive epochs, training stops early. The automatic default is 3
consecutive epochs, and the completed model is still written to the `.tl`
output file.

## Mixed precision

On CUDA, `precision` defaults to `bf16` when the GPU supports it and otherwise
to `fp16`. Tensorless PyTorch uses PyTorch autocast for forward and validation
passes, gradient scaling for `fp16`, and checkpointed scaler state for
consistent resumption. Use `precision="fp32"` for maximum compatibility or
explicitly select `precision="bf16"` / `precision="fp16"` on CUDA.

## Gradient accumulation

Set `gradient_accumulation_steps` when a full batch does not fit in memory.
For example, `batch_size=8` and `gradient_accumulation_steps=4` performs one
optimizer update for every 32 examples while keeping each device batch at 8.
Learning-rate scheduling, `max_steps`, and checkpoint intervals count actual
optimizer updates.

## Distributed and multi-GPU training

Use PyTorch's launcher with one process per GPU. Tensorless PyTorch initializes
DDP, assigns each process its local CUDA device, shards map-style and streaming
datasets, synchronizes validation loss, and lets rank zero write artifacts:

```bash
torchrun --standalone --nproc-per-node=2 -m tensorless.cli.main train ./data \
    --device cuda
```

## Checkpointing during training

A checkpoint is written every `checkpoint_every` steps (default: 50)
and at the end of every epoch, to `<out>.ckpt/checkpoint.pt`. See
[checkpointing.md](checkpointing.md) for what's in it and how it's used
for resumption.

## What each task trains

| Task | Model | What's learned |
|---|---|---|
| `text-generation` | Small GPT-style decoder transformer, BPE tokenizer by default | Next-token prediction over your text |
| `text-classification` | Same transformer backbone, BPE tokenizer by default, classification head on the final token | Text → one of your labeled classes |
| `classification` | MLP with per-column categorical embeddings | Row of features → one of your labeled classes |
| `regression` | Same MLP, single continuous output | Row of features → a number |

## Resuming and force-retraining

```python
tl.train("./data")               # resumes an interrupted run automatically
tl.train("./data", force=True)   # always retrain from scratch
```

See [automatic_mode.md](automatic_mode.md#5-the-smart-auto-check) for
the exact decision logic.

## Command-line equivalent

```bash
tensorless train ./data --d-model 512 --layers 6 --batch-size 32
```

See [cli.md](cli.md) for the full CLI reference.
