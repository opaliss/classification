from ml_lib.ml_util import argmax_random_tie, best_index

class DecisionFork:
    """
    A fork of a decision tree holds an attribute to test, and a dict
    of branches, one for each of the attribute's values.
    """

    indent = 4  # Tab size for printing

    def __init__(self, attr, distribution, attr_name=None, default_child=None,
                 branches=None, parent=None):
        """DecisionFork constructor
        A branch in the decision tree.
        attr - Attribute index that we are splitting on
        distribution - Tuple or list indicating the distribution of class instances
           for this branch, e.g. (8, 2, 3) would indicate 8 instances of
           target 0, 2 instances of target 1, and 3 instances of target 2
           appeared in the training data for this decision fork.  These values
           should appear in the same order as DataSet.values[DataSet.target].
           If no example of a specific target appears in the set, then the
           count for that class should be zero.  e.g. If target 1 had no
           instances in the example above, the distribution would be (8, 0, 3).
           This information is not required for training, but can be used
           to support chi^2 pruning of trees.
        attr_name - Human readable attribute name
        default_child - Function used to predict class when the example
           contains an attribute value that was not seen in training.
           (e.g. when there is no branch)  This can be important for real world
           data sets.  If it is omitted (None), we plurality class of the
           training data is used.
        branches - Dictionary of branches, defaults to None, use .add to add new
                   branches incrementally
        """

        # Store constructor arguments
        self.attr = attr
        self.distribution = distribution
        self.attr_name = attr_name or attr
        self.default_child = default_child
        self.branches = branches or {}

        self.parent = parent
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def __call__(self, example):
        """Given an example,  classify it using the attribute and the branches."""

        attr_val = example[self.attr]
        # If this attribute value was seen in training and produced a branch,
        # classify according to the branch.  Otherwise use the default_child.
        if attr_val in self.branches:
            # Continue down the tree
            prediction = self.branches[attr_val](example)
        else:
            # We did not see this attribute value in training
            # Predict at this node.
            if self.default_child is None:
                # Nothing specified, use pluarality value
                selectged_class =  best_index(self.distribution)
                prediction = selected_class
            else:
                # User specified their own method for deciding the class
                # in this case.
                prediction = self.default_child(example)

        return prediction

    def predict(self, example):
        """Predict class of example"""
        return self(example)

    def add(self, val, subtree):
        """Add a branch. If self.attr = val, go to the given subtree."""
        self.branches[val] = subtree

    def __str__(self):
        "String representation of this fork of the decision tree"

        tab = " " * self.indent * self.depth
        # Add chi-squared value if it has been computed
        chi2 = f' Chi2={self.chi2.value:.3f}' if hasattr(self, 'chi2') else ""
        dist_str = ",".join([str(d) for d in self.distribution])
        result = [f"{self.attr_name}{chi2} split ({dist_str})"]
        for (val, subtree) in self.branches.items():
            string = str(subtree)
            #tab2 = tab + " " * (self.indent // 2)
            result.append(f"{tab}{val} -> " + str(subtree))
            #result.append(str(subtree))

        return "\n".join(result)


    def __repr__(self):
        return 'DecisionFork({0!r}, {1!r}, {2!r})'.format(
            self.attr, self.attr_name, self.branches)


class DecisionLeaf:
    """A leaf of a decision tree holds just a result."""

    def __init__(self, result, distribution, parent):
        """
        Construct a leaf with result class associated with this leaf.
        result - Class assigned to leaf
        distribution - Count of training examples of each class that were
            assigned to the leaf.  See distribution in the constructor of
            class DecisionFork for details.
        parent DecisionFork
        """
        self.result = result
        self.distribution = distribution
        self.parent = parent
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def __call__(self, example):
        "Any example arriving at this leaf will be classified as result"
        return self.result

    def __str__(self):
        return f'{self.result} is predicted'

    def __repr__(self):
        return repr(self.result)

