import tensorless as tl

from .conftest import TINY_TABULAR_KWARGS, TINY_TEXT_KWARGS


def test_text_generation_reports_perplexity(text_corpus, workdir):
    kwargs = dict(TINY_TEXT_KWARGS)
    kwargs["val_split"] = 0.2
    model = tl.train(text_corpus, out="model.tl", **kwargs)
    assert "final_val_perplexity" in model.metrics
    assert model.metrics["final_val_perplexity"] > 0


def test_classification_reports_accuracy(tabular_classification_csv, workdir):
    kwargs = dict(TINY_TABULAR_KWARGS)
    kwargs["val_split"] = 0.2
    model = tl.train(tabular_classification_csv, out="model.tl", task="classification", **kwargs)
    assert "final_val_accuracy" in model.metrics
    assert 0.0 <= model.metrics["final_val_accuracy"] <= 1.0


def test_regression_reports_mae_rmse_r2(tabular_regression_csv, workdir):
    kwargs = dict(TINY_TABULAR_KWARGS)
    kwargs["val_split"] = 0.2
    model = tl.train(tabular_regression_csv, out="model.tl", task="regression", **kwargs)
    for key in ("final_val_mae", "final_val_rmse", "final_val_r2"):
        assert key in model.metrics
        assert model.metrics[key] is not None


def test_evaluate_on_held_out_classification_data(tabular_classification_csv, workdir):
    model = tl.train(
        tabular_classification_csv, out="model.tl", task="classification", val_split=0.2,
        **TINY_TABULAR_KWARGS,
    )
    # Evaluate against a completely separate file the model never trained
    # or validated on.
    import csv
    import random

    random.seed(99)
    test_path = workdir / "held_out.csv"
    with open(test_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["age", "income", "city", "label"])
        for _ in range(40):
            age = random.randint(18, 70)
            income = random.randint(20000, 150000)
            city = random.choice(["NYC", "LA", "Chicago"])
            label = 1 if income > 70000 else 0
            w.writerow([age, income, city, label])

    result = model.evaluate(str(test_path))
    assert "loss" in result
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["n_examples"] == 40


def test_evaluate_on_held_out_text_generation_data(text_corpus, workdir):
    model = tl.train(text_corpus, out="model.tl", val_split=0.2, **TINY_TEXT_KWARGS)

    test_path = workdir / "held_out.txt"
    test_path.write_text("a totally different sentence about cats and dogs playing outside")

    result = model.evaluate(str(test_path))
    assert "loss" in result
    assert "perplexity" in result
    assert result["perplexity"] > 0


def test_evaluate_on_held_out_regression_data(tabular_regression_csv, workdir):
    model = tl.train(
        tabular_regression_csv, out="model.tl", task="regression", val_split=0.2,
        **TINY_TABULAR_KWARGS,
    )
    import csv
    import random

    random.seed(123)
    test_path = workdir / "held_out.csv"
    with open(test_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sqft", "bedrooms", "city", "price"])
        for _ in range(40):
            sqft = random.randint(500, 3500)
            bedrooms = random.randint(1, 5)
            city = random.choice(["NYC", "LA", "Chicago"])
            price = sqft * 150 + bedrooms * 10000 + random.randint(-5000, 5000)
            w.writerow([sqft, bedrooms, city, price])

    result = model.evaluate(str(test_path))
    for key in ("loss", "mae", "rmse", "r2"):
        assert key in result
