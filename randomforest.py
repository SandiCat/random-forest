from dataclasses import dataclass
from typing import List

from dataset import DataSet, Row, Label
from decisiontree import DecisionTree
import decisiontree
import util


@dataclass
class TreeWithDataset:
    tree: DecisionTree
    dataset: DataSet


RandomForest = List[TreeWithDataset]


def train(
        train_set: DataSet,
        num_trees: int, max_depth: int, example_ratio: float, feature_ratio: float) -> RandomForest:
    import random
    forest: RandomForest = []
    for i in range(num_trees):
        feature_subset = set(random.sample(
            train_set.features,
            round(len(train_set.features) * feature_ratio)
        ))
        subset: DataSet = DataSet(
            rows=random.sample(train_set.rows, round(train_set.size * example_ratio))
        ).feature_subset(feature_subset)
        forest.append(TreeWithDataset(
            tree=decisiontree.id3(subset, max_depth),
            dataset=subset
        ))
    return forest


def predict(forest: RandomForest, row: Row) -> Label:
    predictions = [decisiontree.predict(tree.tree, row) for tree in forest]
    return util.most_frequent_alphabetically(predictions)


def print_forest(forest: RandomForest):
    for tree in forest:
        print(" ".join(tree.dataset.features))
        print(" ".join(str(row.order) for row in tree.dataset.rows))

