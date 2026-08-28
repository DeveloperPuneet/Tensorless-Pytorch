# Tensorless PyTorch

Tensorless PyTorch is a lightweight toolkit for turning ordinary text and
tabular data into portable PyTorch models with minimal setup. It supports text
generation, text classification, tabular classification, and regression.

The distribution is installed as `tensorless-pytorch`; the stable Python import
remains `tensorless` for compatibility.

## Install

```bash
pip install tensorless-pytorch
```

## Train on your data

```python
import tensorless as tl

model = tl.train("./corpus.txt", task="text-generation")
print(model.generate("The", max_new_tokens=40))
```

Text files are trained as next-token language models. BPE is the default
tokenizer; use `tokenizer="char"` for a character-level model. Tensorless PyTorch
derives model size, batch size, epochs, validation, device, and BPE vocabulary
size from the data, while every setting can be overridden.

Text is tokenized once up front (not re-tokenized every epoch) and streamed
through PyTorch in fixed-size batches. Model size is auto-scaled with corpus
size across four tiers -- small, lower-mid, upper-mid, large -- capped at
"upper-mid" (roughly 50M-350M parameters) unless you pass explicit
`d_model=`/`layers=`/etc. yourself. New training runs use the `architecture="v2"`
backbone (RoPE + RMSNorm + SwiGLU + KV-cached generation); older `.tl`
checkpoints keep loading and running on the original `"v1"` backbone
automatically. CUDA training automatically uses `fp16` or `bf16` when
supported (with gradient scaling and checkpointed scaler state), TPU (XLA)
training is supported via `device="tpu"`, and large auto-sized models enable
gradient checkpointing automatically to fit in memory. Reduce `batch_size` if
memory is limited. The model that actually gets saved is always the **best**
checkpoint seen during training (lowest validation loss, or lowest train loss
when there's no validation split), not just whatever the final epoch happened
to land on -- this protects against late-training instability (e.g. a run
that looks fine for a while and then diverges) silently producing a broken
saved model. This applies to both `tl.train()` and `tl.pretrain()`.

## English starter pretraining

```python
import tensorless as tl

model = tl.pretrain(out="english.tl", epochs=20, max_seq_len=128)
print(model.generate("A complete sentence", max_new_tokens=30))
```

This offline starter corpus contains English prose and grammar examples. It is
for demos and smoke tests, not a replacement for a large language dataset. For
real pretraining, pass your own `.txt` corpus to `tl.train()` and increase the
training settings as your hardware allows.

## Pretrain, then fine-tune

```python
import tensorless as tl

# 1. Pretrain a base model on a large general corpus
base = tl.train("./big_corpus.txt", task="text-generation", out="base.tl", epochs=20)

# 2. Fine-tune it on a smaller, task-specific dataset
tuned = tl.train("./my_conversations.json", task="text-generation",
                  out="tuned.tl", pretrained="base.tl", epochs=5)
```

`pretrained=` initializes training from an existing `.tl` checkpoint's weights
instead of from scratch. The architecture (`d_model`, `layers`, `heads`, etc.)
and tokenizer are locked to match the pretrained model exactly -- passing a
conflicting override raises a clear error rather than silently ignoring it,
since fine-tuning only works if token ids and embeddings line up with what the
pretrained weights actually learned. You can also switch tasks while
fine-tuning (e.g. a pretrained text-generation backbone into a
text-classification model): matching layers (embeddings, attention, MLP
blocks) transfer, and only the mismatched task head is reinitialized.

## Other tasks

```python
tl.train("reviews/", task="text-classification")
tl.train("housing.csv", task="regression")
```

Tabular preprocessing automatically handles numeric values, ISO dates, and
high-cardinality categories. Missing and rare values are handled using the
fitted training data, and the same preprocessing is stored in the `.tl` file.

## Flexible data formats

`tl.train()` accepts far more than a single `{"text": "..."}` shape.
JSON/JSONL/YAML records are auto-detected and normalized, in order: a
known text field; chat-style turn lists (`"messages"`/`"conversations"`,
OpenAI- or ShareGPT-style); flat conversational pairs under common aliases
(`user`/`bot`, `human`/`gpt`, `instruction`/`input`/`output`, `prompt`/
`completion`, ...); and, as a last resort, any other record is flattened
into readable text rather than rejected. See
[docs/training.md](docs/training.md#supported-data-formats) for the full
list.

```python
tl.train("chats.jsonl", task="text-generation")  # [{"user": "...", "bot": "..."}, ...] just works
```

## Internet browsing at inference time

Off by default; turn it on per-call, per-session, or from the CLI to let
a text-generation model search the web for extra context before
answering:

```python
model = tl.load("model.tl")
model.generate("What's new in PyTorch this week?", internet="connect")
```

```bash
tensorless run model.tl --internet connect
```

See [docs/inference.md](docs/inference.md#internet-browsing-opt-in-off-by-default).

Models are saved as `.tl` files and can be loaded later:

```python
model = tl.load("model.tl")
print(model.info())
```

See the [documentation](docs/quickstart.md) for data formats, configuration,
mixed precision, checkpointing, and the command-line interface.