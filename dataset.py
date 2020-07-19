from dataclasses import dataclass
from typing import NewType, Dict, Set, List
from collections import Counter

import util

Feature = NewType("Feature", str)
FeatureValue = NewType("FeatureValue", str)
Label = NewType("Label", str)


@dataclass
class Row:
    features: Dict[Feature, FeatureValue]
    label: Label
    order: int

    def feature_subset(self, features_to_keep: Set[Feature]) -> "Row":
        return Row(
            features={f: val for f, val in self.features.items() if f in features_to_keep},
            label=self.label,
            order=self.order
        )


@dataclass
class DataSet:
    rows: List[Row]

    def __post_init__(self):
        self.label_column = list(map(lambda row: row.label, self.rows))
        self.label_occurrences = Counter(self.label_column)
        self.labels: Set[Label] = set(self.label_occurrences.keys())
        self.most_frequent_label = util.most_frequent_alphabetically(self.label_column)
        self.features: Set[Feature] = set() if not self.rows else set(self.rows[0].features.keys())
        self.feature_values: Dict[Feature, Set[FeatureValue]] = {
            feature: set(row.features[feature] for row in self.rows)
            for feature in self.features
        }
        self.columns: Dict[Feature, List[FeatureValue]] = {
            feature: [row.features[feature] for row in self.rows]
            for feature in self.features
        }

    def filter(self, predicate) -> "DataSet":
        return DataSet(list(filter(predicate, self.rows)))

    def filter_by_feature(self, feature: Feature, value: FeatureValue) -> "DataSet":
        return self.filter(lambda row: row.features[feature] == value)

    def feature_subset(self, features_to_keep: Set[Feature]) -> "DataSet":
        return DataSet(
            rows=[row.feature_subset(features_to_keep) for row in self.rows]
        )

    def column(self, feature: Feature) -> List[FeatureValue]:
        return self.columns[feature]

    def values(self, feature: Feature) -> Set[FeatureValue]:
        return self.feature_values[feature]

    @property
    def size(self):
        return len(self.rows)


def load_dataset(filename: str) -> DataSet:
    import csv
    with open(filename) as f:
        reader = csv.reader(f)
        column_names = next(reader)
        features: List[Feature] = list(map(Feature, column_names[:-1]))
        return DataSet(
            rows=[
                Row(
                    features=dict(zip(features, map(FeatureValue, row[:-1]))),
                    label=Label(row[-1]),
                    order=i
                )
                for i, row in enumerate(reader)
            ]
        )
