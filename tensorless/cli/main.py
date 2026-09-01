"""Command-line interface.

    tensorless train ./data
    tensorless run model.tl
    tensorless inspect ./data
    tensorless info model.tl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import typing

from .. import api
from ..config import TrainConfig
from ..errors import TensorlessError
from ..serialization.tl_format import load_tl

# Fields handled by their own hand-written flags below (either because they
# need special argparse behavior, like `path`/`--force`/`--quiet`, or
# because exposing them on the CLI wouldn't make sense, like `extra`).
_HANDPICKED_FIELDS = {"out", "force", "verbose", "extra"}


def _add_train_parser(subparsers) -> None:
    p = subparsers.add_parser("train", help="Train a model on a dataset (fully automatic by default)")
    p.add_argument("path", help="Path to a dataset file or directory")
    p.add_argument("--out", default=None, help="Output .tl file path (default: model.tl)")
    p.add_argument("--force", action="store_true", help="Force retraining even if a matching model exists")
    p.add_argument("--quiet", action="store_true", help="Suppress training logs")

    # Every other `TrainConfig` field gets a matching CLI flag, generated
    # from the dataclass itself. This is what gives the CLI *parity* with
    # `TrainConfig` -- previously only ~12 of its ~30 fields were exposed
    # here, so anything not on that hand-picked list (optimizer,
    # weight_decay, val_split, patience, precision, checkpoint_every,
    # seed, resume, ...) was simply unreachable from the command line.
    # Generating the flags instead of listing them by hand means a new
    # `TrainConfig` field automatically gets a CLI flag too, so the two
    # can't drift out of sync again.
    hints = typing.get_type_hints(TrainConfig)
    for f in dataclasses.fields(TrainConfig):
        if f.name in _HANDPICKED_FIELDS:
            continue
        flag = "--" + f.name.replace("_", "-")
        field_type = hints[f.name]
        origin = typing.get_origin(field_type)
        if origin is typing.Union:
            # Optional[X] == Union[X, None] -- unwrap to X.
            args = [a for a in typing.get_args(field_type) if a is not type(None)]
            field_type = args[0] if args else str

        if field_type is bool:
            # BooleanOptionalAction gives both --flag/--no-flag and
            # defaults to None, so we can tell "not passed" (use the
            # auto-config default) apart from an explicit False.
            p.add_argument(flag, dest=f.name, default=None, action=argparse.BooleanOptionalAction,
                            help=f"Override TrainConfig.{f.name}")
        elif field_type in (int, float, str):
            p.add_argument(flag, dest=f.name, type=field_type, default=None,
                            help=f"Override TrainConfig.{f.name}")
        # Any other type (e.g. `extra`'s Dict) is skipped -- not
        # meaningfully expressible as a single CLI flag.


def _add_run_parser(subparsers) -> None:
    p = subparsers.add_parser("run", help="Run a trained .tl model")
    p.add_argument("path", help="Path to a .tl model file")
    p.add_argument("--prompt", default=None, help="Input text/prompt (skips interactive chat)")
    p.add_argument(
        "--internet", default="off", choices=["off", "connect"],
        help="Let the model browse the web for extra context before answering (default: off)",
    )


def _add_inspect_parser(subparsers) -> None:
    p = subparsers.add_parser("inspect", help="Inspect a dataset without training")
    p.add_argument("path", help="Path to a dataset file or directory")


def _add_info_parser(subparsers) -> None:
    p = subparsers.add_parser("info", help="Show information about a trained .tl model")
    p.add_argument("path", help="Path to a .tl model file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensorless", description="Tensorless PyTorch: ML with maximum automation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers)
    _add_run_parser(subparsers)
    _add_inspect_parser(subparsers)
    _add_info_parser(subparsers)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            overrides = {}
            if args.out:
                overrides["out"] = args.out
            if args.force:
                overrides["force"] = True
            if args.quiet:
                overrides["verbose"] = False
            # Every generically-generated flag (see _add_train_parser)
            # shares its dest name with the TrainConfig field it maps
            # to, and defaults to None when not passed -- so we can just
            # sweep them all up here instead of listing each by hand.
            for f in dataclasses.fields(TrainConfig):
                if f.name in _HANDPICKED_FIELDS:
                    continue
                value = getattr(args, f.name, None)
                if value is not None:
                    overrides[f.name] = value
            api.train(args.path, **overrides)
            return 0

        elif args.command == "run":
            api.run(args.path, prompt=args.prompt, internet=args.internet)
            return 0

        elif args.command == "inspect":
            api.inspect(args.path)
            return 0

        elif args.command == "info":
            payload = load_tl(args.path)
            info = {
                "task": payload["task"],
                "model_type": payload["model_type"],
                "tensorless_version": payload.get("tensorless_version"),
                "tl_format_version": payload.get("tl_format_version"),
                "training_complete": payload.get("training_complete"),
                "metrics": payload.get("metrics"),
                "dataset_fingerprint": (payload.get("dataset_fingerprint") or "")[:16],
            }
            print(json.dumps(info, indent=2, default=str))
            return 0

    except TensorlessError as e:
        print(f"tensorless: error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
