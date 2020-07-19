import assessment
import randomforest
from dataset import load_dataset
import decisiontree

from config import *


def print_whole_assessment(predictor, test_set):
    print(" ".join(assessment.test_set_predictions(predictor, test_set)))
    labels = assessment.predictions_and_ground(predictor, test_set)
    print(assessment.accuracy(labels))
    assessment.print_confusion_matrix(labels, test_set)


def main(train_set_filename, test_set_filename, config_filename):
    train_set = load_dataset(train_set_filename)
    test_set = load_dataset(test_set_filename)
    config = load_config(config_filename)
    if config.model == Model.ID3:
        tree = decisiontree.id3(train_set, config.max_depth)
        print(decisiontree.show_decision_tree(tree))
        predictor = assessment.make_predictor(tree, decisiontree.predict)
        print_whole_assessment(predictor, test_set)
    elif config.model == Model.RF:
        forest = randomforest.train(
            train_set, config.num_trees, config.max_depth, config.example_ratio, config.feature_ratio)
        randomforest.print_forest(forest)
        predictor = assessment.make_predictor(forest, randomforest.predict)
        print_whole_assessment(predictor, test_set)


if __name__ == "__main__":
    # main("datasets/volleyball.csv", "datasets/volleyball_test.csv", "config/id3.cfg")
    # main("datasets/volleyball.csv", "datasets/volleyball_test.csv", "config/rf_n5_sub05.cfg")
    import sys
    main(*sys.argv[1:])
