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
    raise NotImplementedError


if __name__ == '__main__':
    """
    # TODO:
    1. Run a decision tree on the Zoo and Mushroom dataset. 
    2. One pruned with p-value of 0.05 and one without pruning. 
    3. provide a cross-validation class to conduct two 10-fold-cross-validation. 
    4. The driver needs to print out the mean error and standard deviation. 
    5. Print out the decision tree + call chi_annotate on the tree before you print it so that you can see the chi2 
    statistic for each decision node. 
    """
    main()
