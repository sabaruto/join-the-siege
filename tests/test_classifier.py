import pytest
from src.classifier import (
    chooseClassifier,
)
import pandas as pd


class_data = ("TokenFrequency", "RandomForest")
class_model_data = (
    ["TokenFrequency", "src/models/TokenFrequency/test_model.jsonl"],
    ["RandomForest", ""],
)
class_training_data = (["RandomForest", pd.DataFrame([])],)
class_model_input_output_data = (
    [
        "TokenFrequency",
        "src/models/TokenFrequency/test_model.jsonl",
        "DRIVER LICENSE License No. P99999999 Expires 00-00-00",
        "drivers_license",
    ],
    [
        "RandomForest",
        "",
        "BANK 2 of Testing Account Holder: Jonh Doe 2 Wire Transfer",
        "bank_statement",
    ],
)


@pytest.mark.parametrize("classifier_name", class_data)
def test_create_empty_classifier(classifier_name: str):
    classifier = chooseClassifier(classifier_name)
    input_data = "test data"

    with pytest.raises(ValueError):
        classifier.classify(input_data)


@pytest.mark.parametrize("classifier_name", class_data)
def test_load_improper_model(classifier_name: str):
    classifier = chooseClassifier(classifier_name)
    empty_model_dir = ""

    with pytest.raises(ValueError):
        classifier.load(empty_model_dir)


def test_improper_training_data():
    classifier = chooseClassifier("RandomForest")
    empty_training_data = pd.DataFrame([])

    with pytest.raises(ValueError):
        classifier.train(empty_training_data)


@pytest.mark.parametrize("classifier_name,training_data", class_training_data)
def test_simple_training_data(classifier_name: str, training_data: pd.DataFrame):
    classifier = chooseClassifier(classifier_name)
    classifier.train(training_data)


@pytest.mark.parametrize("classifier_name,model_dir", class_model_data)
def test_empty_input(classifier_name: str, model_dir: str):
    classifier = chooseClassifier(classifier_name)
    classifier.load(model_dir)

    with pytest.raises(ValueError):
        classifier.classify("")


@pytest.mark.parametrize(
    "classifier_name,model_dir,input,expected_output", class_model_input_output_data
)
def test_classify(
    classifier_name: str, model_dir: str, input: str, expected_output: str
):
    classifier = chooseClassifier(classifier_name)
    classifier.load(model_dir)
    model_output = classifier.classify(input)

    assert model_output == expected_output
