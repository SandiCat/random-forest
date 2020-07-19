from typing import *
from collections import Counter

T = TypeVar("T")


def most_frequent_alphabetically(ls: List[T]) -> Optional[T]:
    top_labels = []
    top_freq = None
    for (label, freq) in Counter(ls).most_common():
        if top_freq is None:
            top_freq = freq
        if freq != top_freq:
            break
        top_labels.append(label)
    return None if not top_labels else sorted(top_labels)[0]
