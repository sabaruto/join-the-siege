import pytest
from src.classifier import (
    retrieve_classifier,
)
import pandas as pd


class_data = ("TokenFrequency", "RandomForest")
class_model_data = (
    ["TokenFrequency", "src/models/TokenFrequency/latest"],
    ["RandomForest", "src/models/RandomForest/latest"],
)
class_training_data = (
    [
        "RandomForest",
        "media/datasets/dataset.csv",
        "src/training_data/RandomForest/simple_params.json",
    ],
)
class_model_input_output_data = (
    [
        "TokenFrequency",
        "src/models/TokenFrequency/latest",
        "DRIVER LICENSE License No. P99999999 Expires 00-00-00",
        "drivers_license",
    ],
    [
        "RandomForest",
        "src/models/RandomForest/latest",
        "BANK 2 of Testing Account Holder: Jonh Doe 2 Wire Transfer",
        "bank_statement",
    ],
)


@pytest.mark.parametrize("classifier_name", class_data)
def test_load_improper_model(classifier_name: str):
    classifier = retrieve_classifier(classifier_name)
    empty_model_dir = ""

    with pytest.raises(ValueError):
        classifier.load(empty_model_dir)


def test_improper_training_data():
    classifier = retrieve_classifier("RandomForest")
    empty_training_data = pd.DataFrame([[]])

    with pytest.raises(ValueError):
        classifier.train(empty_training_data, "")


@pytest.mark.parametrize(
    "classifier_name,training_data_dir,training_parameters_dir", class_training_data
)
def test_simple_training_data(
    classifier_name: str, training_data_dir: str, training_parameters_dir: str
):
    classifier = retrieve_classifier(classifier_name)
    classifier.train(training_data_dir, training_parameters_dir)


@pytest.mark.parametrize("classifier_name,model_dir", class_model_data)
def test_empty_input(classifier_name: str, model_dir: str):
    classifier = retrieve_classifier(classifier_name)
    with pytest.raises(ValueError):
        classifier.classify("")


@pytest.mark.parametrize(
    "classifier_name,model_dir,input,expected_output", class_model_input_output_data
)
def test_classify(
    classifier_name: str, model_dir: str, input: str, expected_output: str
):
    classifier = retrieve_classifier(classifier_name)
    classifier.load(model_dir)
    model_output = classifier.classify(input)

    assert model_output == expected_output
