"""
machine learning utility functions
"""

# Standard modules
import random
import collections
import numpy as np
from statistics import mean

# Contributed
from ml_lib.utils import open_data, num_or_str


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


# Various distance and error metrics
def euclidean_distance(x, y):
    "euclidean_distance(x,y) - straight line distance between vectors x and y"
    return np.sqrt(sum((_x - _y) ** 2 for _x, _y in zip(x, y)))


def manhattan_distance(x, y):
    "manhattan_distance(x, y) - City block distance between vectors x and y"
    return sum(abs(_x - _y) for _x, _y in zip(x, y))


def hamming_distance(x, y):
    "hamming_distance(x, y) - Number of nonmatching elements in vectors x, y"
    return sum(_x != _y for _x, _y in zip(x, y))


def rms_error(x, y):
    "rms_error(x, y) - Root mean square distance between vectors x, y"
    return np.sqrt(ms_error(x, y))


def ms_error(x, y):
    "ms_error(x, y) - Mean square distance between vectors x, y"
    return mean((x - y) ** 2 for x, y in zip(x, y))


def mean_error(x, y):
    "mean_error(x, y) mean of difference between x, y"
    return mean(abs(x - y) for x, y in zip(x, y))


def mean_boolean_error(x, y):
    "mean_boolean_error(x, y) - Similar to hamming distance; mean instead of sum"
    return mean(_x != _y for _x, _y in zip(x, y))


def best_index(seq):
    """best_index(seq)
    Given a sequence, find the postion of the largest value.
    Ties are broken randomly.
    """
    largest = max(seq)
    indices = []
    for idx, val in enumerate(seq):
        if val == largest:
            indices.append(idx)

    # Randomize if necessary
    if len(indices) > 1:
        random.shuffle(indices)

    return indices[0]



# ______________________________________________________________________________
# argmin and argmax - These are not the classic argmin/argmax, but rather ones
# that handle functions of their values.  See argmax_random_tie for an
# example

# identity function
identity = lambda x: x


def argmin_random_tie(seq, key=identity):
    """Return a minimum element of seq; break ties at random.
    Each element is evaluated with function key to determine its value

    See argmax_random_tie for details on how to use this function.  It operates
    in an analagous manner.
    """
    return min(shuffled(seq), key=key)


def argmax_random_tie(seq, key=identity):
    """Return an element with highest fn(seq[i]) score; break ties at random.
    Each element is evaluated with function key to determine its value

    This function is different than a typical argmax.  It relies on
    key containing a function that ties some set of values to a function that
    maps them.

    In the context of a decision tree, this could be used as follows:

    attributes = Some sequence of attribute indices
    identity = A function such as:
       key=lambda a: some_fn(a, examples)

    where some_fn determines information about a set of examples, such as
    the information present for a given attribute a.
    """
    return max(shuffled(seq), key=key)


def normalize(dist):
    """Scale values in a dictionary or list such that they represent
    a probability distribution.  Each value lies in 0 <= value <= 1
    and the sum of all values is 1.

    :param dist:  The distribution.  May be numeric, or a dictionary of numeric
                  values.  Note that dictionaries are modified, other iterables
                  have copies returned.

    :return val: Returns a dictionary or list.
    """
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
            assert 0 <= dist[key] <= 1  # probabilities must be between 0 and 1
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]


class DataSet:
    """
    A data set for a machine learning problem. It has the following fields:

    d.examples   A list of examples. Each one is a list of attribute values.
    d.attrs      A list of integers to index into an example, so example[attr]
                 gives a value. Normally the same as range(len(d.examples[0])).
    d.attr_names Optional list of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.inputs     The list of attrs without the target.
    d.values     A list of lists: each sublist is the set of possible
                 values for the corresponding attribute. If initially None,
                 it is computed from the known examples by self.set_problem.
                 If not None, an erroneous value raises ValueError.
    d.distance   A function from a pair of examples to a non-negative number.
                 Should be symmetric, etc. Defaults to mean_boolean_error
                 since that can handle any field types.
    d.name       Name of the data set (for output display only).
    d.source     URL or other source where the data came from.
    d.exclude    A list of attribute indexes to exclude from d.inputs. Elements
                 of this list can either be integers (attrs) or attr_names.

    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs.
    """

    def __init__(self, examples=None, attrs=None, attr_names=None, target=-1, inputs=None,
                 values=None, distance=mean_boolean_error, name='', source='', exclude=()):
        """
        Accepts any of DataSet's fields. Examples can also be a
        string or file from which to parse examples using parse_csv.
        Optional parameter: exclude, as documented in .set_problem().
        >>> DataSet(examples='1, 2, 3')
        <DataSet(): 1 examples, 3 attributes>

        examples - Filename or list of examples.  File is read from the
           aima-data subdirectory of this module.
        attrs - Indices of example attirbutes (will be determined from data
           if omitted
        attr_names - Human readable names for attributes.  If None, set to
           indices.  If a filename is passed and attr_names is set to True,
           it is assumed that the first line of the file contains the attribute
           names.
        target - Index of target attribute, self.examples[k,target] is the
           class value or regression target of example k.
        inputs - A method of restricting the data to use.  If specified,
           inputs is a list that makes only attribute indices in inputs
           available for learning.
        values - A list of the values that appear across examples for each
           attribute.  e.g.  self.values[self.target] shows a unique list
           of values associated with the class that we are trying to predict.
           In general, self.values[a] shows the range of values across attribute
           a.  If set to None, values will be calculated from the examples
        distance - Function for measuring the dissimilarity between two examples.
           Default function works regardless of data type.
        name - data set name
        source - Reserved for future use.
        exclude - Opposite of inputs, excludes certain attributes

        """
        self.name = name
        self.source = source
        self.values = values
        self.distance = distance
        self.got_values_flag = bool(values)

        # initialize .examples from string or list or data directory
        if isinstance(examples, str):
            self.examples = parse_csv(examples)
        elif examples is None:
            self.examples = parse_csv(open_data(name + '.csv').read())
            if attr_names == True:
                # First line is attribute names
                attr_names = self.examples.pop(0)
        else:
            self.examples = examples

        # attrs are the indices of examples, unless otherwise stated.
        if self.examples is not None and attrs is None:
            attrs = list(range(len(self.examples[0])))

        self.attrs = attrs

        # initialize .attr_names from string, list, or by default
        if isinstance(attr_names, str):
            self.attr_names = attr_names.split()
        else:
            self.attr_names = attr_names or attrs
        self.set_problem(target, inputs=inputs, exclude=exclude)

    def set_problem(self, target, inputs=None, exclude=()):
        """
        Set (or change) the target and/or inputs.
        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not use in inputs. Attributes can be -n .. n, or an attr_name.
        Also computes the list of possible values, if that wasn't done yet.
        """
        self.target = self.attr_num(target)
        exclude = list(map(self.attr_num, exclude))
        if inputs:
            self.inputs = remove_all(self.target, inputs)
        else:
            self.inputs = [a for a in self.attrs if a != self.target and a not in exclude]
        if not self.values:
            self.update_values()
        self.check_me()

    def check_me(self):
        """Check that my fields make sense."""
        assert len(self.attr_names) == len(self.attrs)
        assert self.target in self.attrs
        assert self.target not in self.inputs
        assert set(self.inputs).issubset(set(self.attrs))
        if self.got_values_flag:
            # only check if values are provided while initializing DataSet
            list(map(self.check_example, self.examples))

    def add_example(self, example):
        """Add an example to the list of examples, checking it first."""
        self.check_example(example)
        self.examples.append(example)

    def check_example(self, example):
        """Raise ValueError if example has any invalid values."""
        if self.values:
            for a in self.attrs:
                if example[a] not in self.values[a]:
                    raise ValueError('Bad value {} for attribute {} in {}'
                                     .format(example[a], self.attr_names[a], example))

    def attr_num(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attr_names.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr

    def update_values(self):
        self.values = list(map(unique, zip(*self.examples)))

    def sanitize(self, example):
        """Return a copy of example, with non-input attributes replaced by None."""
        return [attr_i if i in self.inputs else None for i, attr_i in enumerate(example)][:-1]

    def classes_to_numbers(self, classes=None):
        """Converts class names to numbers."""
        if not classes:
            # if classes were not given, extract them from values
            classes = sorted(self.values[self.target])
        for item in self.examples:
            item[self.target] = classes.index(item[self.target])

    def remove_examples(self, value=''):
        """Remove examples that contain given value."""
        self.examples = [x for x in self.examples if value not in x]
        self.update_values()

    def split_values_by_classes(self):
        """Split values into buckets according to their class."""
        buckets = defaultdict(lambda: [])
        target_names = self.values[self.target]

        for v in self.examples:
            item = [a for a in v if a not in target_names]  # remove target from item
            buckets[v[self.target]].append(item)  # add item to bucket of its class

        return buckets

    def find_means_and_deviations(self):
        """
        Finds the means and standard deviations of self.dataset.
        means     : a dictionary for each class/target. Holds a list of the means
                    of the features for the class.
        deviations: a dictionary for each class/target. Holds a list of the sample
                    standard deviations of the features for the class.
        """
        target_names = self.values[self.target]
        feature_numbers = len(self.inputs)

        item_buckets = self.split_values_by_classes()

        means = defaultdict(lambda: [0] * feature_numbers)
        deviations = defaultdict(lambda: [0] * feature_numbers)

        for t in target_names:
            # find all the item feature values for item in class t
            features = [[] for _ in range(feature_numbers)]
            for item in item_buckets[t]:
                for i in range(feature_numbers):
                    features[i].append(item[i])

            # calculate means and deviations fo the class
            for i in range(feature_numbers):
                means[t][i] = mean(features[i])
                deviations[t][i] = stdev(features[i])

        return means, deviations

    def __repr__(self):
        return '<DataSet({}): {:d} examples, {:d} attributes>'.format(self.name, len(self.examples), len(self.attrs))


def parse_csv(input, delim=','):
    r"""
    Input is a string consisting of lines, each line has comma-delimited
    fields. Convert this into a list of lists. Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.
    >>> parse_csv('1, 2, 3 \n 0, 2, na')
    [[1, 2, 3], [0, 2, 'na']]
    """
    lines = [line for line in input.splitlines() if line.strip()]
    return [list(map(num_or_str, line.split(delim))) for line in lines]


def err_ratio(learner, dataset, examples=None):
    """
    Return the proportion of the examples that are NOT correctly predicted.
    verbose - 0: No output; 1: Output wrong; 2 (or greater): Output correct
    """
    examples = examples or dataset.examples
    if len(examples) == 0:
        return 0.0
    right = 0
    for example in examples:
        desired = example[dataset.target]
        output = learner(example)

        # Check if prediction is correct
        if isinstance(desired, str):
            correct = desired == output
        else:
            correct = np.allclose(output, desired)
        if correct:
            right += 1
    return 1 - (right / len(examples))


def grade_learner(learner, tests):
    """
    Grades the given learner based on how many tests it passes.
    tests is a list with each element in the form: (values, output).
    """
    return mean(int(learner.predict(X) == y) for X, y in tests)



# ______________________________________________________________________________
# Functions on Sequences and Iterables


def sequence(iterable):
    """Converts iterable to sequence, if it is not already one."""
    return (iterable if isinstance(iterable, collections.abc.Sequence)
            else tuple([iterable]))


def remove_all(item, seq):
    """Return a copy of seq (or string) with all occurrences of item removed."""
    if isinstance(seq, str):
        return seq.replace(item, '')
    elif isinstance(seq, set):
        rest = seq.copy()
        rest.remove(item)
        return rest
    else:
        return [x for x in seq if x != item]


def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements."""
    return list(set(seq))

