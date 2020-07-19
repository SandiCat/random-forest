from typing import *

from dataset import DataSet, Label, Row
from collections import Counter


def predictions_and_ground(predict: Callable[[Row], Label], test_set: DataSet) -> List[Tuple[Label, Label]]:
    return [(predict(row), row.label) for row in test_set.rows]


def print_confusion_matrix(labels: List[Tuple[Label, Label]], dataset: DataSet):
    counter = Counter(labels)
    sorted_labels = sorted(dataset.labels)
    for true_label in sorted_labels:
        print(" ".join(str(counter[(predicted_label, true_label)]) for predicted_label in sorted_labels))


def accuracy(labels: List[Tuple[Label, Label]]) -> float:
    return len(list(filter(lambda x: x[0] == x[1], labels))) / len(labels)


def test_set_predictions(predict: Callable[[Row], Label], test_set: DataSet) -> List[Label]:
    return [predict(row) for row in test_set.rows]


M = TypeVar('M')


def make_predictor(model: M, predict: Callable[[M, Row], Label]) -> Callable[[Row], Label]:
    def predictor(row: Row) -> Label:
        return predict(model, row)
    return predictor
