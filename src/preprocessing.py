import pandas as pd
import random


def create_token_vector(
    training_tokens: list[str], max_size: int = 5000, shuffle_input: bool = False
):
    if shuffle_input:
        random.shuffle(training_tokens)
    vocab: dict[str, int] = {}
    index = 0

    for token in training_tokens:
        token = token.lower()

        if token not in vocab:
            vocab[token] = index
            index += 1

        # if index >= max_size:
        #     break

    return vocab


def vectorise_list(input_tokens: list[str], vocab_dict: dict[str, int]) -> list[int]:
    vector = [0 for _ in range(len(vocab_dict))]

    for token in input_tokens:
        token = token.lower()
        if token in vocab_dict:
            token_index = vocab_dict[token]
            vector[token_index] += 1
    return vector


def vectoriseSeries(feature_series: pd.Series, vocab_dict: dict[str, int]) -> pd.Series:
    feature_series = feature_series.str.split(" ", expand=False)
    feature_series = feature_series.map(lambda x: vectorise_list(x, vocab_dict))
    feature_series = feature_series.to_list()

    feature_len = len(feature_series[0])
    feature_series = pd.DataFrame(
        feature_series,
        columns=[f"feature_{i}" for i in range(feature_len)],
    )

    return feature_series


def vectoriseDataset(
    dataset: pd.DataFrame, vocab_dict: dict[str, int], label_mapping: dict[int, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes a database of form (X: str, y) and converts
    it to (X_0: int, X_1: int ..., X_n: int) (y), decomposing the input string to
    token vectors defined in token_vector
    """

    _, feature_column, label_column = dataset.columns

    features_dataest = vectoriseSeries(dataset[feature_column], vocab_dict)

    labels_dataset = dataset[label_column]
    label_dict = {token: index for index, token in label_mapping.items()}
    labels_dataset = labels_dataset.map(label_dict)

    return features_dataest, labels_dataset
