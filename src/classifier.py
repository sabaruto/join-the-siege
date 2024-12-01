from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from pandas import DataFrame

import jsonlines
import jsonschema
import pandas as pd
import xgboost

from src.preprocessing import vectoriseDataset, vectoriseList, vectoriseSeries


class Classifier(ABC):

    @abstractmethod
    def classify(self, input: str) -> str:
        pass

    def train(self, training_data: DataFrame, training_parameters_dir: str):
        pass

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
            with jsonlines.open(trained_model_dir, "r") as model_json_reader:
                for json_obj in model_json_reader:
                    jsonschema.validate(
                        json_obj, schema=TokenFrequencyClassifier.model_schema
                    )

                    category = json_obj["category"]
                    token_list = json_obj["tokens"]
                    self.category_tokens[category] = token_list
        except Exception as e:
            raise ValueError(f"error loading model data: {e}")

    def classify(self, input: str):
        if not self.category_tokens:
            raise ValueError("Classifier data has not been loaded")

        if len(input) < 1:
            raise ValueError("Recieved empty input")

        categories = list(self.category_tokens.keys())
        frequency: dict[str, int] = {}

        for category in categories:
            tokens = self.category_tokens[category]
            for word in tokens:
                frequency[category] = input.count(word)

        print(frequency)

        # Sort the categories based
        categories.sort(key=lambda category: frequency[category], reverse=True)

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

    def train(self, training_data_dir: str, training_parameters_dir: str):
        training_data = pd.read_csv(training_data_dir)
        with open(training_parameters_dir, "rb") as params_file:
            training_params_json = json.load(params_file)
        jsonschema.validate(training_params_json, schema=self.training_params_schema)

        xgboost_params = training_params_json["xgboost_params"]
        num_boost_rounds = training_params_json["num_boost_rounds"]

        self.vocab_dict, self.label_mapping, features_dataset, labels_dataset = (
            vectoriseDataset(training_data)
        )

        xgb_training_matrix = xgboost.DMatrix(features_dataset, labels_dataset)

        xgboost_params["num_class"] = len(self.label_mapping)
        self.model = xgboost.train(
            xgboost_params, xgb_training_matrix, num_boost_rounds
        )

    def classify(self, input: list[str]):
        if not self.model:
            raise AttributeError("Classifier not initialised yet")

        labels = [f"feature_{i}" for i in range(len(self.vocab_dict))]
        vectored_input = vectoriseSeries(pd.Series(input), self.vocab_dict)
        vectored_input = pd.DataFrame(vectored_input, columns=labels)
        vectored_input = xgboost.DMatrix(vectored_input)

        prediction_index = self.model.predict(vectored_input)

        prediction = {
            self.label_mapping[index]: value
            for index, value in enumerate(prediction_index[0])
        }
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
            self.label_mapping = imported_object

    def export(self, dir: str):
        if not self.model:
            raise AttributeError("Classifier not initialised yet")

        Path(dir).mkdir(parents=True, exist_ok=True)
        self.model.save_model(f"{dir}/model.json")

        with open(f"{dir}/labels.json", "w") as label_file:
            json.dump(self.label_mapping, label_file)

        with open(f"{dir}/vocab.json", "w") as vocab_file:
            json.dump(self.vocab_dict, vocab_file)


def chooseClassifier(name: str = "TokenFrequency") -> Classifier:
    match name:
        case "TokenFrequency":
            return TokenFrequencyClassifier()
        case "RandomForest":
            return RandomForestClassifier()
        case _:
            raise ValueError(f"there's no classifier of type {name}")
