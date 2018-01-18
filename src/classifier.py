from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline


# List of estimators to evaluate. Add new estimators to this dictionary
estimators = {
    'l1': LogReg(random_state=0, penalty='l1'),
    'l2': LogReg(random_state=0),
    'svc': SVC(random_state=0),
    'rf': RandomForestClassifier(random_state=0, n_estimators=75),
    'dt': DecisionTreeClassifier(random_state=0),
    'gbm': GradientBoostingClassifier(random_state=0)
}

scoring = ['accuracy', 'precision', 'recall', 'roc_auc']


def runclassifiers(X, y, cv=10, test_size=0.15, sampling_type=None, classifiers='all',
                   metrics='all', print_coeff=False, col_names=None):
    """
    Evaluate diffferent classifiers on dataset
    :param X: array-like, The data to fit (X).
    :param y: array-like, Target values (y).
    :param cv: int, default=10. Number of cross-validation folds.
    :param test_size: float, default=0.15. Test set size
    :param sampling_type: str, default=None. `oversample` or `even`.
    :param classifiers: list, default='all'. List of classifiers to evaluate
    :param metrics: list, default='all'. List of scoring metrics to use. Ref scikit-learn default
    :param print_coeff: bool, default=False. Print feature coefficients for LogReg
    :param col_names: list, default=None. Feature names used to print coefs
    :return:
    """

    global estimators
    global scoring

    def data_sampler():
        split = 0
        while split < cv:
            if sampling_type == 'over':
                sampler = RandomOverSampler()
            else:
                sampler = RandomUnderSampler()
            X_resampled, y_resampled = sampler.fit_sample(X, y)
            yield (X_resampled, y_resampled)
            split += 1

    # Only use classifiers specified by param
    if classifiers != 'all':
        estimators = {k: v for k, v in estimators.items() if k in classifiers}

    # Only use scoring metric specified by param
    if metrics != 'all':
        scoring = [s for s in scoring if s in metrics]

    if not sampling_type:
        split_generator = cv  # This will do a regulr train test split in the cross_validate function
    else:
        split_generator = data_sampler()

    # TODO Use GridSearchCV in place of naive for loop
    # Train and eval classifiers and print results
    for name, estimator in estimators.items():
        scores = cross_validate(estimator, X, y, scoring=scoring,
                                cv=split_generator, return_train_score=True)

        for metric in scoring:
            print('{}-{}:'.format(name.upper(), metric.capitalize()), str(np.mean(scores['test_{}'.format(metric)])),
                  'std:', str(np.std(scores['test_{}'.format(metric)])),
                  'model-fit:', str(np.mean(scores['train_{}'.format(metric)]) -
                                    np.mean(scores['test_{}'.format(metric)])),
                  'score-time:', str(np.std(scores['score_time'])),
                  'fit-time:', str(np.std(scores['fit_time'])),
                  )
        print('')

    # TODO Find a better way to do this
    # Display coefficients
    if print_coeff and col_names is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        l1_coefficients = LogReg().fit(X_train, y_train).coef_
        print('L1 Coefficients')
        for i, v in enumerate(np.mean(np.asarray(l1_coefficients), axis=0)[0]):
            print(col_names[i], v)


def tuneclassifiers(X, y, classifiers=['rf'], cv=5, metric='roc_auc',
                    hyper_params={'rf': ['n_estimators', 'max_depth']},
                    param_values={'rf': [[75, 150], [1, 3, None]]}):
    """
    Tune the parameters for the selected classifiers
    :param classifiers: list, default=['rf']. List of classifiers to tune
    :param hyper_params: dict, default={'rf': ['n_estimators', 'max_depth']}. Order with param_values
    :param param_values: dict, default={'rf': [[75, 150], [1, 3, None]]}. Param values
    :param cv: int, default=5. Num of cv folds in gridsearch
    :param metric, default='roc_auc'. Metric to optimize gridsearch for.
    :return: list. Best estimator for each classifier.
    """
    best_estimators = []
    for classifier in classifiers:
        pipe = Pipeline([('imputer', Imputer(strategy='most_frequent')),
                         (classifier, estimators[classifier])])

        param_grid = {}
        for i in range(len(hyper_params)):
            param_grid['{}__{}'.format(classifier, hyper_params[classifier][i])] = param_values[classifier][i]

        grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring=metric, return_train_score=True)
        grid.fit(X, y)
        print(pd.DataFrame(grid.cv_results_))

        best_estimators.append(grid.best_estimator_)

        return best_estimators
