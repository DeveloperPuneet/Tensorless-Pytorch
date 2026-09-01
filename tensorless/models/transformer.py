"""A small, dependency-free (beyond PyTorch) GPT-style decoder transformer.

Used for:
  - "text-generation": next-token prediction over the char vocabulary
  - "text-classification": same backbone, with a classification head on
    the final token's hidden state instead of a language-modeling head

Kept intentionally compact -- this is not meant to compete with
production LLM training frameworks, it's meant to give Tensorless PyTorch a real,
working, from-scratch model that trains fast enough on CPU for the
"zero setup" experience to actually be pleasant.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..errors import ModelError


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=attn_mask is None
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * ff_mult)
        self.fc2 = nn.Linear(d_model * ff_mult, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, ff_mult, dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


def _check_finite_logits(logits: torch.Tensor) -> None:
    """A NaN/Inf logit means the model's weights are themselves corrupted
    (most commonly: fp16 training without effective gradient clipping,
    letting a bad update push some weight to inf, which then poisons
    every subsequent forward pass). Raising here turns an opaque CUDA
    device-side assert crash inside `torch.multinomial` into a clear,
    actionable message.
    """
    if not torch.isfinite(logits).all():
        raise ModelError(
            "This model produced non-finite (NaN/Inf) logits during generation, "
            "which means its saved weights are corrupted -- not something that can "
            "be worked around at generation time. This is almost always caused by "
            "instability during training (e.g. fp16 training diverging). Retrain "
            "the model to get a usable checkpoint; if training is under your "
            "control, try a lower learning_rate= and/or precision='bf16' if your "
            "GPU supports it natively (Ampere or newer)."
        )


class TinyTransformerV1(nn.Module):
    """Decoder-only transformer usable for LM or sequence classification.

    This is the original ("v1") architecture, kept byte-for-byte as-is so
    that any `.tl` checkpoint trained before the v2 upgrade still loads
    and runs correctly (`registry.build_model` picks this class whenever
    a checkpoint's config has no `architecture` key, or has
    `architecture == "v1"`).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        max_seq_len: int,
        task: str = "text-generation",
        n_classes: int = 0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.task = task
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, heads, ff_mult, dropout) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        if task == "text-generation":
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight  # weight tying
        elif task == "text-classification":
            assert n_classes > 0, "n_classes must be set for text-classification"
            self.head = nn.Linear(d_model, n_classes)
        else:
            raise ValueError(f"Unsupported task for TinyTransformer: {task}")

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        )
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        attn_mask = None
        if attention_mask is not None:
            # Combine causal mask with padding mask.
            causal = torch.tril(torch.ones(T, T, device=input_ids.device, dtype=torch.bool))
            pad = attention_mask.bool().unsqueeze(1).unsqueeze(1)  # B,1,1,T
            attn_mask = (causal.unsqueeze(0).unsqueeze(0) & pad)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.ln_f(x)

        if self.task == "text-generation":
            return self.head(x)  # B, T, vocab_size
        else:
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            else:
                lengths = torch.full((B,), T - 1, device=input_ids.device)
            pooled = x[torch.arange(B, device=input_ids.device), lengths]
            return self.head(pooled)  # B, n_classes

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            cond = input_ids[:, -self.max_seq_len:]
            logits = self(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            _check_finite_logits(logits)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return input_ids


# ---------------------------------------------------------------------------
# V2 architecture: RMSNorm + Rotary Position Embeddings + SwiGLU feed-forward
# + KV-cached generation + optional gradient checkpointing. This is what
# `tl.train(...)` uses by default for new runs (see `models/registry.py`);
# it scales meaningfully better than v1 at the "upper-mid" sizes Tensorless
# now auto-configures for large datasets, while staying a drop-in
# same-interface swap (`forward(...)` / `generate(...)`).
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean-centering, no bias) -- cheaper
    than LayerNorm and what most modern decoder-only LMs use."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def _rotary_cos_sin(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, half)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, heads, T, head_dim); cos/sin: (T, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class RotaryCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float, max_seq_len: int, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        assert (d_model // heads) % 2 == 0, "head_dim must be even for rotary embeddings"
        self.heads = heads
        self.head_dim = d_model // heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        total_len = position_offset + T
        cos, sin = _rotary_cos_sin(total_len, self.head_dim, x.device, q.dtype, self.rope_base)
        cos, sin = cos[position_offset:total_len], sin[position_offset:total_len]
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v

        is_causal = attn_mask is None and (kv_cache is None or position_offset == 0) and T > 1
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class SwiGLU(nn.Module):
    """Gated feed-forward network (SwiGLU), as used in LLaMA/PaLM-style
    models -- consistently outperforms a plain ReLU/GELU MLP at equal
    parameter budgets."""

    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        hidden = int(d_model * ff_mult * (2 / 3))  # keep param count comparable to a 2x MLP
        hidden = max(hidden, d_model)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class BlockV2(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_mult: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RotaryCausalSelfAttention(d_model, heads, dropout, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, ff_mult, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, kv_cache=kv_cache, position_offset=position_offset)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformerV2(nn.Module):
    """Decoder-only transformer with RoPE + RMSNorm + SwiGLU + KV-cache
    generation. Same public interface as `TinyTransformerV1`
    (`forward(...)` / `generate(...)`), scaled to comfortably reach
    "upper-mid" model sizes (~100M-350M parameters) when auto-configured
    for large datasets.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        max_seq_len: int,
        task: str = "text-generation",
        n_classes: int = 0,
        pad_id: int = 0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.task = task
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.gradient_checkpointing = gradient_checkpointing
        self.d_model = d_model
        self.layers = layers
        self.heads = heads

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [BlockV2(d_model, heads, ff_mult, dropout, max_seq_len) for _ in range(layers)]
        )
        self.norm_f = RMSNorm(d_model)

        if task == "text-generation":
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight  # weight tying
        elif task == "text-classification":
            assert n_classes > 0, "n_classes must be set for text-classification"
            self.head = nn.Linear(d_model, n_classes)
        else:
            raise ValueError(f"Unsupported task for TinyTransformerV2: {task}")

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        if kv_caches is None:
            assert T <= self.max_seq_len, (
                f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
            )
        x = self.tok_emb(input_ids)
        x = self.drop(x)

        attn_mask = None
        if attention_mask is not None:
            causal = torch.tril(torch.ones(T, T, device=input_ids.device, dtype=torch.bool))
            pad = attention_mask.bool().unsqueeze(1).unsqueeze(1)  # B,1,1,T
            attn_mask = (causal.unsqueeze(0).unsqueeze(0) & pad)

        use_checkpoint = self.gradient_checkpointing and self.training and kv_caches is None
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    lambda x, block=block: block(x, attn_mask=attn_mask), x, use_reentrant=False
                )
            else:
                x = block(x, attn_mask=attn_mask, kv_cache=cache, position_offset=position_offset)
        x = self.norm_f(x)

        if self.task == "text-generation":
            return self.head(x)
        else:
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            else:
                lengths = torch.full((B,), T - 1, device=input_ids.device)
            pooled = x[torch.arange(B, device=input_ids.device), lengths]
            return self.head(pooled)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive sampling with an incremental KV cache -- each new
        token only re-processes itself instead of the whole prefix, which
        is what makes generation from upper-mid-sized models practical.
        """
        self.eval()
        prompt = input_ids[:, -self.max_seq_len:]
        kv_caches = [dict(k=None, v=None) for _ in range(self.layers)]

        logits = self(prompt, kv_caches=kv_caches, position_offset=0)
        next_logits = logits[:, -1, :]
        _check_finite_logits(next_logits)
        generated = input_ids

        for step in range(max_new_tokens):
            next_logits = next_logits / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                for b in range(generated.size(0)):
                    seen = torch.unique(generated[b])
                    next_logits[b, seen] = next_logits[b, seen] / torch.where(
                        next_logits[b, seen] > 0, torch.tensor(repetition_penalty, device=next_logits.device),
                        torch.tensor(1.0 / repetition_penalty, device=next_logits.device),
                    )

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], -float("inf"))

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                remove = cumprobs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                next_logits = torch.full_like(next_logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)

            if eos_id is not None and (next_id == eos_id).all():
                break
            if step == max_new_tokens - 1:
                break

            position_offset = kv_caches[0]["k"].shape[2]
            if position_offset >= self.max_seq_len:
                # Cache is full: fall back to a fresh (uncached) forward
                # pass over the trailing window rather than growing forever.
                kv_caches = [dict(k=None, v=None) for _ in range(self.layers)]
                window = generated[:, -self.max_seq_len:]
                logits = self(window, kv_caches=kv_caches, position_offset=0)
            else:
                logits = self(next_id, kv_caches=kv_caches, position_offset=position_offset)
            next_logits = logits[:, -1, :]
            _check_finite_logits(next_logits)

        return generated


# Backward/forward-compatible aliases: existing call sites (and any user
# code doing `from tensorless.models.transformer import TinyTransformer`)
# keep working, defaulting to the new v2 architecture.
TinyTransformer = TinyTransformerV2
