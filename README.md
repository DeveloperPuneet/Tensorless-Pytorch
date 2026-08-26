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
memory is limited.

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

## Other tasks

```python
tl.train("reviews/", task="text-classification")
tl.train("housing.csv", task="regression")
```

Tabular preprocessing automatically handles numeric values, ISO dates, and
high-cardinality categories. Missing and rare values are handled using the
fitted training data, and the same preprocessing is stored in the `.tl` file.

Models are saved as `.tl` files and can be loaded later:

```python
model = tl.load("model.tl")
print(model.info())
```

See the [documentation](docs/quickstart.md) for data formats, configuration,
mixed precision, checkpointing, and the command-line interface.