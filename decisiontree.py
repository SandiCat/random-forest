from dataclasses import dataclass
from typing import Dict, Union, NewType, Set, List, Tuple

from dataset import Feature, FeatureValue, Label, DataSet, Row


@dataclass
class Node:
    feature_name: Feature
    children: Dict[FeatureValue, "DecisionTree"]
    default_label: Label


@dataclass
class Leaf:
    label: Label


DecisionTree = Union[Node, Leaf]
Entropy = NewType("Entropy", float)


def dataset_entropy(dataset: DataSet) -> Entropy:
    import math

    def p_log_p(p):
        return p * math.log2(p)

    return Entropy(
        - sum(
            p_log_p(dataset.label_occurrences[label] / dataset.size)
            for label in dataset.labels
        )
    )


def information_gain(dataset: DataSet, x: Feature) -> Entropy:
    expected_entropy_post = 0
    for v in dataset.values(x):
        filtered = dataset.filter_by_feature(x, v)
        expected_entropy_post += filtered.size / dataset.size * dataset_entropy(filtered)
    return Entropy(dataset_entropy(dataset) - expected_entropy_post)


def id3(original_dataset: DataSet, max_depth=None) -> DecisionTree:
    def go(parent_dataset: DataSet, dataset: DataSet, features_left: Set[Feature], depth) -> DecisionTree:
        if not dataset.size:
            return Leaf(parent_dataset.most_frequent_label)
        if not features_left or len(dataset.label_occurrences) == 1:
            return Leaf(dataset.most_frequent_label)
        if depth is not None and depth == 0:
            return Leaf(dataset.most_frequent_label)

        most_discriminative_feature = max(
            sorted(features_left),
            key=lambda feature: information_gain(dataset, feature)
        )
        node = Node(
            feature_name=most_discriminative_feature,
            children={
                val: go(
                    parent_dataset=dataset,
                    dataset=dataset.filter_by_feature(most_discriminative_feature, val),
                    features_left=features_left - {most_discriminative_feature},
                    depth=depth - 1 if depth else None
                )
                for val in original_dataset.values(most_discriminative_feature)
            },
            default_label=dataset.most_frequent_label
        )
        return node

    return go(original_dataset, original_dataset, original_dataset.features, max_depth)


def show_decision_tree(tree: DecisionTree) -> str:
    def go(depth: int, tree: DecisionTree) -> List[Tuple[int, Feature]]:
        if isinstance(tree, Leaf):
            return []
        elif isinstance(tree, Node):
            ret = [(depth, tree.feature_name)]
            for _, child in tree.children.items():
                ret.extend(go(depth + 1, child))
            return ret
        else:
            print(tree)

    return ", ".join(
        map(
            lambda x: f"{x[0]}:{x[1]}",
            sorted(go(0, tree), key=lambda x: x[0])
        )
    )


def predict(tree: DecisionTree, row: Row) -> Label:
    if isinstance(tree, Leaf):
        return tree.label
    elif isinstance(tree, Node):
        if row.features[tree.feature_name] not in tree.children:
            return tree.default_label
        else:
            return predict(tree.children[row.features[tree.feature_name]], row)
