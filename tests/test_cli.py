import os

from tensorless.cli.main import main


def test_cli_train(text_corpus, workdir):
    rc = main(["train", text_corpus, "--out", "model.tl", "--epochs", "1", "--d-model", "16", "--layers", "1", "--heads", "2", "--batch-size", "16", "--quiet"])
    assert rc == 0
    assert os.path.isfile("model.tl")


def test_cli_inspect(text_corpus, workdir, capsys):
    rc = main(["inspect", text_corpus])
    assert rc == 0
    captured = capsys.readouterr()
    assert "text-generation" in captured.out


def test_cli_info(text_corpus, workdir, capsys):
    main(["train", text_corpus, "--out", "model.tl", "--epochs", "1", "--d-model", "16", "--layers", "1", "--heads", "2", "--batch-size", "16", "--quiet"])
    rc = main(["info", "model.tl"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "text-generation" in captured.out


def test_cli_run_with_prompt(text_corpus, workdir, capsys):
    main(["train", text_corpus, "--out", "model.tl", "--epochs", "1", "--d-model", "16", "--layers", "1", "--heads", "2", "--batch-size", "16", "--quiet"])
    rc = main(["run", "model.tl", "--prompt", "the quick"])
    assert rc == 0
    captured = capsys.readouterr()
    assert len(captured.out.strip()) > 0


def test_cli_error_returns_nonzero(workdir):
    rc = main(["inspect", "does_not_exist_path"])
    assert rc == 1


def test_cli_exposes_all_train_config_fields():
    """The `train` subcommand's flags must have parity with TrainConfig --
    previously only a hand-picked subset of fields (~12 of ~30) had a
    corresponding CLI flag at all."""
    import dataclasses

    from tensorless.cli.main import build_parser
    from tensorless.config import TrainConfig

    parser = build_parser()
    train_parser = next(
        a.choices["train"] for a in parser._subparsers._group_actions if "train" in a.choices
    )
    known_dests = {a.dest for a in train_parser._actions}

    for f in dataclasses.fields(TrainConfig):
        if f.name in ("extra", "verbose"):
            # "extra" is a dict of raw overrides, not a single-flag field.
            # "verbose" is deliberately exposed as its inverse, --quiet,
            # rather than duplicated as its own flag.
            continue
        assert f.name in known_dests, f"TrainConfig.{f.name} has no matching CLI flag"


def test_cli_previously_unreachable_flags_work(text_corpus, workdir):
    """`--optimizer`, `--val-split`, and `--seed` were unreachable from the
    CLI before CLI/TrainConfig parity was fixed."""
    rc = main([
        "train", text_corpus, "--out", "model.tl", "--epochs", "1",
        "--d-model", "16", "--layers", "1", "--heads", "2", "--batch-size", "16",
        "--optimizer", "sgd", "--val-split", "0.2", "--seed", "7", "--quiet",
    ])
    assert rc == 0
    from tensorless.serialization.tl_format import load_tl

    payload = load_tl("model.tl")
    assert payload["config"]["optimizer"] == "sgd"
    assert payload["config"]["val_split"] == 0.2
    assert payload["config"]["seed"] == 7
