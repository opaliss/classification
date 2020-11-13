from copy import copy
import random

from ml_lib.ml_util import err_ratio


def cross_validation(learner, dataset, *learner_posn_args, k=10, trials=1,
                     **learner_kw_args):
    """
    Perform k-fold cross_validation

    :param learner:  Class of machine learning algorithm.  Constructor
       must accept an instance of ml_lib.ml_util.DataSet.  Any optional
       positional or keyword arguments may be passed in to this function,
       e.g. cross_validation(NeuralNetLearner, my_data, 50, 3, k=10, l2=.01)
            would train the models as follows:
            NeuralNetLearner(fold_k_dataset, 50, 3, l2=.01)
            where the arguments are specific to any given learning algorithm.
        fold_k_dataset is handled by this function which will create a version
        of the dataset with appropriate training data for each test fold.
    :param dataset: cross validation data, instance of ml_lib.ml_util.DataSet
    :param k:  The data are split into k folds, and each fold is tested with
        a model trained from the remaining k-1 folds.
    :param trials: Number of times to repeat the k-fold experiment.
    :param *learner_posn_args:  List of positional arguments for the learner
       that are passed after the dataset argument.
    :param **learner_kw_args:  List of keyward arguments for the learner
       that are passed after the dataset argument.


    Data are shuffled before each of the trials.

    If trials = 1:
       Returns a tuple containing:
          list of cross validation errors
          list of models used to test each fold
    else:
       Returns a list of tuples where each entry is as for the above case where
       trials = 1.

    """

    # If k not specified, split such that each trial consists of a single
    # example.  This is usually called leave-one-out or jackknife training,
    # and is usually only done for very small samples.
    k = k or len(dataset.examples)

    if trials > 1:
        # Run K-fold cross validation multiple times.
        trial_errs = 0
        results = []
        for t in range(trials):
            results.append(cross_validation(learner, dataset, 1, k))

    else:
        fold_errs = []  # error per k folds
        models = []  # models trained with k-1 folds

        n = len(dataset.examples)  # Number of examples
        foldN = n // k  # Number of items per fold

        examples = dataset.examples
        random.shuffle(dataset.examples)
        fold_data = copy(dataset)  # Shallow copy of data set
        for fold in range(k):
            # Specify which examples are to be used in this fold
            train_data, val_data = \
                train_test_split(dataset, fold*foldN, (fold+1)*foldN)

            fold_data.examples = train_data
            # Learn the classification/regression function
            h = learner(fold_data, *learner_posn_args, **learner_kw_args)

            # See how we did on validation data.
            fold_err = err_ratio(h.predict, fold_data, val_data)

            # Prune the model, then see how well we do on validation data
            p_value = .05
            h.chi_annotate(p_value)
            h.prune(p_value)
            pruned_err = err_ratio(h.predict, fold_data, val_data)

            # track fold errors and models
            fold_errs.append(pruned_err)
            models.append(h)

        results = (fold_errs, models)

    return results


def train_test_split(dataset, start=None, end=None, test_split=None):
    """
    If you are giving 'start' and 'end' as parameters,
    then it will return the testing set from index 'start' to 'end'
    and the rest for training.
    If you give 'test_split' as a parameter then it will return
    test_split * 100% as the testing set and the rest as
    training set.
    """
    examples = dataset.examples
    if test_split is None:
        # User requested specific test examples
        val = examples[start:end]  # The ones they wanted
        train = examples[:start] + examples[end:]  # all the rest
    else:
        # Compute test size as a rate of test_split, e.g.
        # .25 means 75% train, 25% test
        total_size = len(examples)

        val_size = int(total_size * test_split)
        train_size = total_size - val_size

        train = examples[:train_size]
        val = examples[train_size:total_size]

    return train, val

