"""
Single Programmer Affidavit
I the undersigned promise that the attached assignment is my own work. While I was free to discuss ideas with others,
the work contained is my own. I recognize that should this not be the case, I will be subject to penalties as outlined
in the course syllabus.
Programmer (Opal Issan. Nov 24th, 2020)
"""
import time
from ml_lib.ml_util import DataSet
from decision_tree import DecisionTreeLearner
from ml_lib.crossval import cross_validation
from statistics import mean, stdev


def main():
    """
    Machine learning with decision trees.
    Runs cross validation on data sets and reports results/trees
    """
    # if true will run a decision tree on the mushroom dataset.
    run_mushroom = True
    # if true will run a decision tree on the zoo dataset.
    run_zoo = True
    # if true will run a decision tree on the tiny_animal dataset.
    run_tiny_animal = False
    # if true will run a decision tree on the restaurant dataset.
    run_restaurant = False

    if run_mushroom:
        # the mushroom label is the first index of the mushroom dataset.
        # target=0 will exclude the label from mushroom.inputs list of attributes.
        data = DataSet(name="mushrooms", attr_names=True, target=0, exclude=[0])

    if run_zoo:
        # the label is the last index of the zoo dataset.
        # target=-1 will exclude the label from zoo.inputs list of attributes.
        data = DataSet(name="zoo", attr_names=True, target=-1, exclude=[0])

    if run_tiny_animal:
        # the label is the last index of the tiny_animal dataset.
        # target=-1 will exclude the label from tiny_animals_set.inputs list of attributes.
        data = DataSet(name="tiny_animal_set", attr_names=True, target=-1)

    if run_restaurant:
        # the label is the last index of the restaurant dataset.
        # target=-1 will exclude the label from tiny_animals_set.inputs list of attributes.
        data = DataSet(name="restaurant", attr_names=True, target=-1)

    tree = DecisionTreeLearner(dataset=data, debug=True, p_value=0.05)
    tree.chi_annotate(p_value=0.05)
    print(tree)

    results = cross_validation(learner=DecisionTreeLearner, dataset=data, p_value=0.05)

    print("Mean Error = ", mean(results[0]))
    print("Standard deviation = ", stdev(results[0]))


if __name__ == '__main__':
    """
    # TODO:
    2. One pruned with p-value of 0.05 and one without pruning. 

    """
    main()
