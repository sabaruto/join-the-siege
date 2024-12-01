from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from pandas import DataFrame

import jsonlines
import jsonschema
import pandas as pd
import xgboost

from src.preprocessing import (
    create_token_vector,
    vectoriseDataset,
    vectoriseSeries,
)


class Classifier(ABC):

    @abstractmethod
    def classify(self, input: str) -> str:
        pass

    def train(self, training_data: DataFrame, training_parameters_dir: str):
        pass

    def from_saved_model(cls, model_dir: str) -> "Classifier":
        classifier = cls()
        classifier.load(model_dir)

        return classifier

    @abstractmethod
    def load(self, dir: str):
        pass

    def export(self, dir: str):
        pass


class TokenFrequencyClassifier(Classifier):
    model_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "",
        "type": "object",
        "properties": {
            "category": {"type": "string", "minLength": 1},
            "tokens": {"type": "array"},
        },
        "required": ["category", "tokens"],
    }

    def __init__(self):
        self.category_tokens: dict[str, list[str]] = {}

    def load(self, trained_model_dir: str):
        self.category_tokens = {}
        try:
            with jsonlines.open(
                f"{trained_model_dir}/model.jsonl", "r"
            ) as model_json_reader:
                for json_obj in model_json_reader:
                    jsonschema.validate(
                        json_obj, schema=TokenFrequencyClassifier.model_schema
                    )

                    category = json_obj["category"]
                    token_list = json_obj["tokens"]
                    self.category_tokens[category] = token_list
        except Exception as e:
            raise ValueError(f"error loading model data: {e}")

    def get_token_frequency(self, input: str):
        if not self.category_tokens:
            raise ValueError("Classifier data has not been loaded")

        categories = list(self.category_tokens.keys())
        frequency: dict[str, int] = {}

        for category in categories:
            tokens = self.category_tokens[category]
            frequency[category] = 0
            for word in tokens:
                frequency[category] += input.count(word)

        return frequency

    def classify(self, input: str):
        if not self.category_tokens:
            raise ValueError("Classifier data has not been loaded")

        if len(input) < 1:
            raise ValueError("Recieved empty input")

        categories = list(self.category_tokens.keys())

        frequency = self.get_token_frequency(input)
        print(frequency)

        # Sort the categories based
        categories.sort(key=lambda category: frequency[category], reverse=True)
        print(categories)
        return categories[0]


class RandomForestClassifier(Classifier):
    training_params_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "",
        "type": "object",
        "properties": {
            "xgboost_params": {
                "type": "object",
            },
            "num_boost_rounds": {"type": "number"},
        },
        "required": ["xgboost_params", "num_boost_rounds"],
    }

    def __init__(self):
        self.vocab_dict: dict[str, int]
        self.label_mapping: dict[int, str]
        self.model: xgboost.Booster

    @classmethod
    def from_dataset(cls, dataset: pd.DataFrame) -> "RandomForestClassifier":
        classifier = cls()
        classifier.initialise_mappings(dataset)
        return classifier

    def initialise_mappings(self, dataset: pd.DataFrame):
        _, feature_column, label_column = dataset.columns

        label_tokens = dataset[label_column]
        feature_tokens = dataset[feature_column]
        feature_tokens = feature_tokens.aggregate("sum")
        feature_tokens = feature_tokens.split()

        label_dict = create_token_vector(label_tokens)
        self.label_mapping = {index: token for token, index in label_dict.items()}
        self.vocab_dict = create_token_vector(feature_tokens, shuffle_input=True)

    def train(self, training_data_path: str, training_parameters_path: str):
        try:
            training_data = pd.read_csv(training_data_path)
        except Exception as e:
            raise ValueError(f"Unable to read csv: {e}")

        with open(training_parameters_path, "rb") as params_file:
            training_params_json = json.load(params_file)
        jsonschema.validate(training_params_json, schema=self.training_params_schema)

        if not self.vocab_dict or not self.label_mapping:
            self.initialise_mappings(training_data)

        xgboost_params = training_params_json["xgboost_params"]
        num_boost_rounds = training_params_json["num_boost_rounds"]

        features_dataset, labels_dataset = vectoriseDataset(
            training_data, self.vocab_dict, self.label_mapping
        )

        xgb_training_matrix = xgboost.DMatrix(features_dataset, labels_dataset)

        xgboost_params["num_class"] = len(self.label_mapping)
        self.model = xgboost.train(
            xgboost_params, xgb_training_matrix, num_boost_rounds
        )

    def classify(self, input: str):
        if not self.model:
            raise AttributeError("Classifier not initialised yet")

        if len(input) < 1:
            raise ValueError("Recieved empty input")

        labels = [f"feature_{i}" for i in range(len(self.vocab_dict))]
        vectored_input = vectoriseSeries(pd.Series([input]), self.vocab_dict)
        vectored_input = pd.DataFrame(vectored_input, columns=labels)
        vectored_input = xgboost.DMatrix(vectored_input)

        prediction_probs = self.model.predict(vectored_input)
        prediction_index = prediction_probs.argmax(axis=1)[0]
        prediction = self.label_mapping[prediction_index]

        return prediction

    def load(self, dir: str):
        if not os.path.exists(f"{dir}/model.json"):
            raise ValueError(f"model,json file not found in {dir}")
        if not os.path.exists(f"{dir}/labels.json"):
            raise ValueError(f"labels.json file not found in {dir}")
        if not os.path.exists(f"{dir}/vocab.json"):
            raise ValueError(f"vocab.json file not found in {dir}")

        self.model = xgboost.Booster()
        self.model.load_model(f"{dir}/model.json")

        with open(f"{dir}/vocab.json") as vocab_file:
            imported_object = json.load(vocab_file)
            assert isinstance(imported_object, dict)
            self.vocab_dict = imported_object

        with open(f"{dir}/labels.json") as label_file:
            imported_object = json.load(label_file)
            assert isinstance(imported_object, dict)
            self.label_mapping = {
                index: label for label, index in imported_object.items()
            }

    def export(self, dir: str):
        if not self.model:
            raise AttributeError("Classifier not initialised yet")

        Path(dir).mkdir(parents=True, exist_ok=True)
        self.model.save_model(f"{dir}/model.json")

        with open(f"{dir}/labels.json", "w") as label_file:
            label_dict = {label: index for index, label in self.label_mapping.items()}
            json.dump(label_dict, label_file)

        with open(f"{dir}/vocab.json", "w") as vocab_file:
            json.dump(self.vocab_dict, vocab_file)


def retrieve_classifier(name: str) -> Classifier:
    match name:
        case "TokenFrequency":
            classifier_class = TokenFrequencyClassifier
            model_dir = "src/models/TokenFrequency/latest"
        case "RandomForest":
            classifier_class = RandomForestClassifier
            model_dir = "src/models/RandomForest/latest"
        case _:
            raise ValueError(f"there's no classifier of type {name}")

    classifier = Classifier.from_saved_model(classifier_class, model_dir)
    return classifier
