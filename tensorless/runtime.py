"""Runtime inference wrapper.

`tl.load("model.tl")` returns a `LoadedModel`, which knows how to rebuild
the exact architecture used at training time, load the weights, and
expose a task-appropriate prediction API:

  - text-generation      -> `.generate(prompt)` and `.chat()`
  - text-classification  -> `.predict(text)`
  - classification/regression (tabular) -> `.predict(record_or_records)`
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F

from .auto.detector import target_column
from .data.loader import load_dataset
from .data.tabular import TabularPreprocessor
from .devices.device import get_torch_device
from .errors import DataError, ModelError
from .metrics import ClassificationAccumulator, RegressionAccumulator, perplexity_from_loss
from .models.registry import build_model
from .serialization.tl_format import load_tl
from .tokenization import tokenizer_from_state_dict
from .web.browser import browse_for_context, web_search

_INTERNET_ON_VALUES = {"connect", "on", "true", "1", "yes"}


def _internet_enabled(value: Union[str, bool, None]) -> bool:
    """Normalize the many spellings of "turn internet browsing on" a
    caller might pass (bool, or a string like "connect"/"on"/"off").
    Internet browsing is opt-in: anything falsy, None, or unrecognized
    is treated as off.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _INTERNET_ON_VALUES


class LoadedModel:
    def __init__(self, payload: Dict[str, Any], device: str = None, internet: Union[str, bool, None] = "off"):
        self.payload = payload
        # Internet browsing is OFF by default -- a model only searches the
        # web for a response when this is explicitly turned on, either
        # here, via `set_internet("connect")`, or per-call with
        # `generate(..., internet="connect")` / `predict(..., internet="connect")`.
        self._internet_default = _internet_enabled(internet)
        self.last_web_sources: List[Any] = []
        self.task: str = payload["task"]
        self.model_type: str = payload["model_type"]
        self.config: Dict[str, Any] = payload["config"]
        self.meta: Dict[str, Any] = payload["meta"]
        self.metrics: Dict[str, Any] = payload.get("metrics", {})
        self.dataset_fingerprint: str = payload.get("dataset_fingerprint")

        self.tokenizer = None
        if payload.get("tokenizer_state") is not None:
            self.tokenizer = tokenizer_from_state_dict(payload["tokenizer_state"])

        self.preprocessor = None
        if payload.get("preprocessor_state") is not None:
            self.preprocessor = TabularPreprocessor.from_state_dict(payload["preprocessor_state"])

        device_name = device or self.config.get("device", "cpu")
        self.device = get_torch_device(device_name)

        self.model = build_model(self.task, self.model_type, self.config, self.meta)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Internet browsing (opt-in, off by default)
    # ------------------------------------------------------------------
    def set_internet(self, internet: Union[str, bool, None]) -> None:
        """Set the default internet-browsing mode for this loaded model,
        e.g. `model.set_internet("connect")` / `model.set_internet("off")`.
        Individual `generate()`/`predict()` calls can still override this
        per-call via their own `internet=` argument.
        """
        self._internet_default = _internet_enabled(internet)

    def _browse(self, query: str, max_results: int = 3, verbose: bool = True) -> str:
        """Search the web for `query` and return a context block to
        prepend to the prompt, or "" if nothing could be found (network
        unreachable, no results, etc.) -- browsing failures never raise,
        they just silently fall back to answering from the model alone.
        """
        self.last_web_sources = []
        context = browse_for_context(query, max_results=max_results)
        if context:
            self.last_web_sources = web_search(query, max_results=max_results)
            if verbose:
                print(f"[tensorless] internet=connect -- searched the web for: {query!r}")
        elif verbose:
            print(f"[tensorless] internet=connect -- no web results found for: {query!r}; answering from the model alone")
        return context

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        internet: Union[str, bool, None] = None,
        internet_max_results: int = 3,
    ) -> str:
        """Generate a continuation of `prompt`.

        `internet` controls web browsing for this call: "connect" (or
        `True`) searches the web for `prompt` and folds the results in as
        extra context before generating; "off" (or `False`) never
        browses. Defaults to whatever `set_internet()` last configured
        (itself off unless explicitly turned on) -- browsing is always
        opt-in, never automatic.
        """
        if self.task != "text-generation":
            raise ModelError(f"generate() is only available for text-generation models, not '{self.task}'.")

        use_internet = self._internet_default if internet is None else _internet_enabled(internet)
        effective_prompt = prompt
        if use_internet:
            context = self._browse(prompt, max_results=internet_max_results, verbose=self.config.get("verbose", True))
            if context:
                effective_prompt = f"{context}\n\nQuestion: {prompt}\nAnswer:"

        ids = self.tokenizer.encode(effective_prompt, add_special_tokens=True)[:-1]  # drop trailing eos
        if not ids:
            ids = [self.tokenizer.bos_id]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=self.tokenizer.eos_id,
        )
        # top_p / repetition_penalty and KV-cached decoding are only
        # supported by the v2 architecture; v1 checkpoints silently fall
        # back to the arguments they understand.
        if self.config.get("architecture", "v1") == "v2":
            gen_kwargs["top_p"] = top_p
            gen_kwargs["repetition_penalty"] = repetition_penalty
        out = self.model.generate(input_ids, **gen_kwargs)
        return self.tokenizer.decode(out[0].tolist())

    def chat(self, internet: Union[str, bool, None] = None) -> None:
        """Interactive terminal chat loop for text-generation models.

        Internet browsing defaults to off; pass `internet="connect"` to
        start with it on, or type `internet on` / `internet off` at the
        prompt to toggle it mid-conversation.
        """
        if self.task != "text-generation":
            raise ModelError(f"chat() is only available for text-generation models, not '{self.task}'.")
        use_internet = self._internet_default if internet is None else _internet_enabled(internet)
        print("Tensorless PyTorch interactive chat. Type 'exit' or Ctrl+C to quit.")
        print(f"[tensorless] internet browsing: {'on' if use_internet else 'off'} "
              f"(type 'internet on' / 'internet off' to toggle)")
        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            stripped = prompt.strip().lower()
            if stripped in ("exit", "quit"):
                break
            if stripped in ("internet on", "internet connect"):
                use_internet = True
                print("[tensorless] internet browsing: on")
                continue
            if stripped == "internet off":
                use_internet = False
                print("[tensorless] internet browsing: off")
                continue
            reply = self.generate(prompt, max_new_tokens=200, internet=use_internet)
            print(reply)

    # ------------------------------------------------------------------
    # Text classification
    # ------------------------------------------------------------------
    def _predict_text_classification(self, texts: List[str]) -> List[str]:
        block_size = self.config["max_seq_len"]
        batch_ids, batch_mask = [], []
        for t in texts:
            ids = self.tokenizer.encode(t, add_special_tokens=True)[:block_size]
            mask = [1] * len(ids)
            if len(ids) < block_size:
                pad_len = block_size - len(ids)
                ids = ids + [self.tokenizer.pad_id] * pad_len
                mask = mask + [0] * pad_len
            batch_ids.append(ids)
            batch_mask.append(mask)
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        attn_mask = torch.tensor(batch_mask, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attn_mask)
        pred_idx = logits.argmax(dim=-1).tolist()
        classes = self.meta["classes"]
        return [classes[i] for i in pred_idx]

    # ------------------------------------------------------------------
    # Tabular classification / regression
    # ------------------------------------------------------------------
    def _predict_tabular(self, records: List[Dict[str, Any]]) -> List[Any]:
        transformed = self.preprocessor.transform(records, with_target=False)
        numeric = transformed["numeric"].to(self.device)
        categorical = transformed["categorical"].to(self.device)
        with torch.no_grad():
            out = self.model(numeric, categorical)
        if self.task == "classification":
            pred_idx = out.argmax(dim=-1)
            return self.preprocessor.inverse_target(pred_idx)
        else:
            return self.preprocessor.inverse_target(out)

    # ------------------------------------------------------------------
    # Unified predict()
    # ------------------------------------------------------------------
    def predict(self, x: Union[str, Dict[str, Any], List[Any]], internet: Union[str, bool, None] = None) -> Any:
        single = not isinstance(x, list)
        items = [x] if single else x

        if self.task == "text-classification":
            preds = self._predict_text_classification(items)
        elif self.task in ("classification", "regression"):
            preds = self._predict_tabular(items)
        elif self.task == "text-generation":
            preds = [self.generate(prompt=str(i), internet=internet) for i in items]
        else:
            raise ModelError(f"predict() not supported for task '{self.task}'.")

        return preds[0] if single else preds

    # ------------------------------------------------------------------
    # Post-hoc evaluation on new data
    # ------------------------------------------------------------------
    def evaluate(self, path: str, batch_size: int = 32) -> Dict[str, Any]:
        """Evaluate this model against new data at `path` -- e.g. a held-out
        test set that was never used for training or validation. Uses the
        model's already-fitted tokenizer/preprocessor as-is (nothing is
        re-fit on this data), and returns a metrics dict with `loss` plus
        the task-appropriate metric(s): `perplexity` for text-generation,
        `accuracy` for (text-)classification, `mae`/`rmse`/`r2` for
        regression -- computed the same way as the live validation metrics
        reported during training (see `tensorless.metrics`).

        This is independent of the train/validation split used *during*
        `tl.train()`: it accepts any dataset in the same format `tl.train()`
        does and can be called any time after loading a model, e.g. weeks
        later on data that didn't exist yet at training time.
        """
        ds = load_dataset(path)

        if self.task == "text-generation":
            if ds.kind != "text" or not ds.texts:
                raise ModelError(
                    "evaluate() for a text-generation model expects a plain "
                    "text dataset (the same format tl.train() accepts)."
                )
            block = self.config["max_seq_len"]
            all_ids: List[int] = []
            for t in ds.texts:
                all_ids.extend(self.tokenizer.encode(t, add_special_tokens=True))
            if len(all_ids) < 2:
                raise DataError("Not enough tokens in the evaluation data to compute a loss.")
            losses = []
            with torch.no_grad():
                for start in range(0, len(all_ids) - 1, block):
                    chunk = all_ids[start:start + block + 1]
                    if len(chunk) < 2:
                        continue
                    x = torch.tensor([chunk[:-1]], dtype=torch.long, device=self.device)
                    y = torch.tensor([chunk[1:]], dtype=torch.long, device=self.device)
                    logits = self.model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                        ignore_index=self.tokenizer.pad_id,
                    )
                    losses.append(loss.item())
            mean_loss = sum(losses) / max(1, len(losses))
            return {"loss": mean_loss, "perplexity": perplexity_from_loss(mean_loss), "n_chunks": len(losses)}

        elif self.task == "text-classification":
            if ds.kind != "text_labeled":
                raise ModelError(
                    "evaluate() for a text-classification model expects a directory "
                    "of class subfolders (the same format tl.train() accepts)."
                )
            classes = self.meta["classes"]
            block_size = self.config["max_seq_len"]
            losses, acc = [], ClassificationAccumulator()
            with torch.no_grad():
                for start in range(0, len(ds.texts), batch_size):
                    texts = ds.texts[start:start + batch_size]
                    labels_batch = ds.labels[start:start + batch_size]
                    batch_ids, batch_mask = [], []
                    for t in texts:
                        ids = self.tokenizer.encode(t, add_special_tokens=True)[:block_size]
                        mask = [1] * len(ids)
                        if len(ids) < block_size:
                            pad_len = block_size - len(ids)
                            ids = ids + [self.tokenizer.pad_id] * pad_len
                            mask = mask + [0] * pad_len
                        batch_ids.append(ids)
                        batch_mask.append(mask)
                    input_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
                    attn_mask = torch.tensor(batch_mask, dtype=torch.long, device=self.device)
                    # Labels this model never saw during training map to -1
                    # (never counted "correct") rather than crashing --
                    # evaluation data isn't guaranteed to only contain
                    # already-known classes.
                    target_idx = torch.tensor(
                        [classes.index(label) if label in classes else -1 for label in labels_batch],
                        dtype=torch.long,
                    )
                    target_dev = target_idx.clamp(min=0).to(self.device)
                    logits = self.model(input_ids, attention_mask=attn_mask)
                    loss = F.cross_entropy(logits, target_dev)
                    losses.append(loss.item())
                    pred = logits.argmax(dim=-1).cpu()
                    correct = ((pred == target_idx) & (target_idx != -1)).sum().item()
                    acc.update(correct, len(labels_batch))
            mean_loss = sum(losses) / max(1, len(losses))
            return {"loss": mean_loss, **acc.compute(), "n_examples": acc.total}

        elif self.task in ("classification", "regression"):
            target_col = target_column(ds)
            if target_col is None:
                raise DataError(
                    f"Could not find a target column in the evaluation data at '{path}'."
                )
            transformed = self.preprocessor.transform(ds.records, with_target=True)
            n = transformed["numeric"].shape[0]
            losses = []
            cls_acc, reg_acc = ClassificationAccumulator(), RegressionAccumulator()
            with torch.no_grad():
                for start in range(0, n, batch_size):
                    sl = slice(start, start + batch_size)
                    numeric = transformed["numeric"][sl].to(self.device)
                    categorical = transformed["categorical"][sl].to(self.device)
                    target = transformed["target"][sl].to(self.device)
                    out = self.model(numeric, categorical)
                    if self.task == "classification":
                        loss = F.cross_entropy(out, target)
                        correct = (out.argmax(dim=-1) == target).sum().item()
                        cls_acc.update(correct, target.numel())
                    else:
                        loss = F.mse_loss(out, target)
                        # Metrics are computed in the *original* target scale
                        # (not the standardized training scale), since MAE
                        # or R2 in z-score units isn't meaningful to a user.
                        pred_orig = out.double() * self.preprocessor.target_std + self.preprocessor.target_mean
                        target_orig = target.double() * self.preprocessor.target_std + self.preprocessor.target_mean
                        err = pred_orig - target_orig
                        reg_acc.update(
                            (err * err).sum().item(), err.abs().sum().item(),
                            target_orig.sum().item(), (target_orig * target_orig).sum().item(),
                            target.numel(),
                        )
                    losses.append(loss.item())
            mean_loss = sum(losses) / max(1, len(losses))
            extra = cls_acc.compute() if self.task == "classification" else reg_acc.compute()
            return {"loss": mean_loss, **extra, "n_examples": n}

        else:
            raise ModelError(f"evaluate() not supported for task '{self.task}'.")

    def info(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "model_type": self.model_type,
            "tensorless_version": self.payload.get("tensorless_version"),
            "tl_format_version": self.payload.get("tl_format_version"),
            "config": self.config,
            "metrics": self.metrics,
            "training_complete": self.payload.get("training_complete"),
            "n_parameters": sum(p.numel() for p in self.model.parameters()),
            "internet": "on" if self._internet_default else "off",
        }


def load_model(path: str, device: str = None, internet: Union[str, bool, None] = "off") -> LoadedModel:
    payload = load_tl(path)
    return LoadedModel(payload, device=device, internet=internet)
