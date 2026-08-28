"""Startup reporting: show every resolved config value and whether it was
set manually (by the caller / CLI flag) or chosen automatically, before
training begins. Makes `tl.train("./data")`'s "fully automatic" behavior
inspectable instead of a black box.
"""

from __future__ import annotations

from typing import Any, Dict, Set

# Printed in a stable, grouped order rather than dict/dataclass field
# order, so related knobs (architecture, then optimization, then
# hardware, ...) read together.
_FIELD_GROUPS = (
    ("task / architecture", (
        "task", "model_type", "architecture", "d_model", "layers", "heads",
        "ff_mult", "dropout", "max_seq_len", "tokenizer", "bpe_vocab_size",
        "gradient_checkpointing", "compile", "num_workers", "multi_gpu",
    )),
    ("optimization", (
        "optimizer", "learning_rate", "weight_decay", "batch_size", "epochs",
        "max_steps", "gradient_accumulation_steps", "grad_clip", "warmup_steps",
    )),
    ("validation / early stopping", ("val_split", "patience", "min_delta")),
    ("hardware", ("device", "precision")),
    ("checkpointing", ("checkpoint_every", "checkpoint_dir")),
    ("misc", ("out", "seed", "verbose")),
)


def print_config_report(cfg: Dict[str, Any], manual_fields: Set[str], locked_fields: Set[str] = frozenset()) -> None:
    """Print every resolved training parameter, tagged with how its value
    was decided:

      (manual)  -- explicitly passed by the caller (tl.train(...) kwarg
                   or a CLI flag)
      (locked)  -- forced to match a --pretrained checkpoint's
                   architecture/tokenizer (fine-tuning)
      (auto)    -- chosen automatically from dataset size + hardware
    """
    seen = set()
    print("[tensorless] ===== training configuration =====")
    manual_count = locked_count = auto_count = 0
    for group_name, fields in _FIELD_GROUPS:
        rows = [f for f in fields if f in cfg]
        if not rows:
            continue
        print(f"[tensorless] -- {group_name} --")
        for field in rows:
            seen.add(field)
            value = cfg[field]
            if field in locked_fields:
                source, manual_count_delta, locked_delta, auto_delta = "locked", 0, 1, 0
            elif field in manual_fields:
                source, manual_count_delta, locked_delta, auto_delta = "manual", 1, 0, 0
            else:
                source, manual_count_delta, locked_delta, auto_delta = "auto", 0, 0, 1
            manual_count += manual_count_delta
            locked_count += locked_delta
            auto_count += auto_delta
            print(f"[tensorless]   {field:28s} = {value!r:<12} ({source})")

    # Any resolved fields not covered by the grouping above (forward
    # compatibility if ResolvedConfig grows a field this module doesn't
    # know about yet) -- still report them rather than silently hiding them.
    leftover = [f for f in cfg if f not in seen and f != "extra"]
    if leftover:
        print("[tensorless] -- other --")
        for field in sorted(leftover):
            value = cfg[field]
            if field in locked_fields:
                source = "locked"
                locked_count += 1
            elif field in manual_fields:
                source = "manual"
                manual_count += 1
            else:
                source = "auto"
                auto_count += 1
            print(f"[tensorless]   {field:28s} = {value!r:<12} ({source})")

    summary = f"[tensorless] {manual_count} manual, {auto_count} auto"
    if locked_count:
        summary += f", {locked_count} locked (pretrained)"
    print(summary + f" -- {manual_count + auto_count + locked_count} parameters total")
    print("[tensorless] ===================================")


def print_resumed_config_report(cfg: Dict[str, Any]) -> None:
    """Shorter report for resumed runs: the config is whatever was used
    when the checkpoint was first created, so a fresh manual/auto
    breakdown wouldn't reflect this invocation's arguments.
    """
    print("[tensorless] ===== resuming with checkpoint's original configuration =====")
    print(
        f"[tensorless]   task={cfg.get('task')} model_type={cfg.get('model_type')} "
        f"architecture={cfg.get('architecture')} d_model={cfg.get('d_model')} "
        f"layers={cfg.get('layers')} batch_size={cfg.get('batch_size')} "
        f"epochs={cfg.get('epochs')}"
    )
    print("[tensorless] (pass force=True to discard the checkpoint and re-resolve config from scratch)")
    print("[tensorless] ================================================================")
