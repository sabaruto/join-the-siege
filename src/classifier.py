from abc import ABC, abstractmethod
from pandas import DataFrame

import jsonlines
import jsonschema
import pandas as pd
import xgboost


class Classifier(ABC):

    @abstractmethod
    def classify(self, input: str) -> str:
        pass

    def _preprocessData(self, input: str) -> any:
        pass

    def train(self, training_data: DataFrame):
        pass

    @abstractmethod
    def load(self, trained_model_dir: str):
        pass

    def export(self, dir: str):
        pass


class KeyWordsClassifier(Classifier):
    model_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": "",
        "type": "object",
        "properties": {
            "category": {"type": "string", "minLength": 1},
            "key_words": {"type": "array"},
        },
        "required": ["category", "key_words"],
    }

    def __init__(self):
        self.keywords: dict[str, list[str]] = {}

    def load(self, trained_model_dir: str):
        self.keywords = {}
        try:
            with jsonlines.open(trained_model_dir, "r") as model_json_reader:
                for json_obj in model_json_reader:
                    jsonschema.validate(
                        json_obj, schema=KeyWordsClassifier.model_schema
                    )

                    category = json_obj["category"]
                    key_words = json_obj["key_words"]
                    self.keywords[category] = key_words
        except Exception as e:
            raise ValueError(f"error loading model data: {e}")

    def classify(self, input: str):
        if not self.keywords:
            raise ValueError("Classifier data has not been loaded")

        if len(input) < 1:
            raise ValueError("Recieved empty input")

        categories = list(self.keywords.keys())
        frequency: dict[str, int] = {}

        for category in categories:
            key_words = self.keywords[category]
            for word in key_words:
                frequency[category] = input.count(word)

        print(frequency)

        # Sort the categories based
        categories.sort(key=lambda category: frequency[category], reverse=True)

        return categories[0]


class RandomForestClassifier(Classifier):
    def __init__(self):
        self.word_to_vector: dict[int, str]
        xgboost.DMatrix()

    def _preprocessData(self, input: str):
        # Convert the data into a vector

        if not self.word_to_vector:
            raise ValueError("Classifier data has not been loaded")

        return [self.word_to_vector[word] for word in input]

    def createWordVector(self, training_words: list[str]):
        vocab: dict[str, int] = {}
        index = 1

        vocab["<pad>"] = 0
        for word in training_words:
            if word not in vocab:
                vocab[word] = index
                index += 1

        self.word_to_vector = {index: word for word, index in vocab.items()}

    # TODO Look into fixing the training algorithm
    def train(self, training_data_dir: str):
        training_data = pd.read_csv(training_data_dir)
        word_groups = training_data["filecontents"]
        word_str = " ".join(word_groups)
        words = word_str.split()

        self.createWordVector(words)
        training_data["filecontents"] = training_data["filecontents"].apply(
            self._preprocessData
        )

        xgboost.train()

    pass


def chooseClassifier(name: str = "KeyWords") -> Classifier:
    match name:
        case "KeyWords":
            return KeyWordsClassifier()
        case "RandomForest":
            return RandomForestClassifier()
        case _:
            raise ValueError(f"there's no classifier of type {name}")
