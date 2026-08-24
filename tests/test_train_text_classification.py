import tensorless as tl
from tensorless.data.loader import load_dataset
from tensorless.auto.config import resolve_config
from tensorless.config import TrainConfig
from tensorless.training.data_prep import prepare_text_classification


def test_train_text_classification(text_classification_dir, workdir):
    model = tl.train(
        text_classification_dir,
        out="sentiment.tl",
        epochs=10,
        d_model=32,
        layers=2,
        heads=2,
        batch_size=16,
        max_seq_len=64,
    )
    assert model.task == "text-classification"
    assert set(model.meta["classes"]) == {"positive", "negative"}


def test_text_classification_accuracy_on_train_set(text_classification_dir, workdir):
    model = tl.train(
        text_classification_dir,
        out="sentiment.tl",
        epochs=15,
        d_model=32,
        layers=2,
        heads=2,
        batch_size=16,
        max_seq_len=64,
    )
    ds = load_dataset(text_classification_dir)
    correct = sum(1 for t, l in zip(ds.texts, ds.labels) if model.predict(t) == l)
    accuracy = correct / len(ds.texts)
    assert accuracy > 0.9


def test_text_classification_uses_dynamic_padding(text_classification_dir):
    ds = load_dataset(text_classification_dir)
    cfg = resolve_config(
        ds, TrainConfig(task="text-classification", max_seq_len=128, val_split=0)
    ).to_dict()
    prepared = prepare_text_classification(ds, cfg)
    input_ids, attention_mask, _ = next(iter(prepared.train_loader))
    assert input_ids.shape[1] < cfg["max_seq_len"]
    assert input_ids.shape == attention_mask.shape
