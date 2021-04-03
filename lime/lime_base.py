"""
Contains abstract functionality for learning locally linear sparse model.
"""
from __future__ import print_function

import os
import pickle
import itertools
from collections import Counter, defaultdict

import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state

from sigdirect import SigDirect

def get_all_rules(neighborhood_data, labels_column, clf):

    clf.fit(neighborhood_data, labels_column)
    local_pred =  clf.predict(neighborhood_data[0].reshape((1,-1)), 2).astype(int)[0]

    all_rules = clf.get_all_rules()
    return all_rules, local_pred

def get_features_sigdirect(all_rules, true_label):
    """ use applied rules first, and then the rest of the applicable rules, 
        and then all rules (other labels, rest of them match)
    """

    # applied rules,
    applied_sorted_rules = sorted(all_rules[true_label], 
                          key=lambda x:(
                                        len(x[0].get_items()),  
                                        - x[0].get_confidence() * x[0].get_support(), 
                                        x[0].get_log_p(),
                                        - x[0].get_support(),
                                        -x[0].get_confidence(), 
                                       ), 
                          reverse=False)

    # applicable rules, except the ones in applied rules.
    applicable_sorted_rules = sorted(itertools.chain(*[all_rules[x] for x in all_rules if x!=true_label]), 
                          key=lambda x:(
                                        len(x[0].get_items()),  
                                        - x[0].get_confidence() * x[0].get_support(), 
                                        x[0].get_log_p(),
                                        - x[0].get_support(),
                                        -x[0].get_confidence(), 
                                       ), 
                          reverse=False)

    # all rules, except the ones in applied rules.
    other_sorted_rules = sorted(itertools.chain(*[all_rules[x] for x in all_rules if x!=true_label]), 
                          key=lambda x:(
                                        len(x[0].get_items()),  
                                        - x[0].get_confidence() * x[0].get_support(), 
                                        x[0].get_log_p(),
                                        - x[0].get_support(),
                                        -x[0].get_confidence(), 
                                       ), 
                          reverse=False)

    counter = len(all_rules)
    bb_features = defaultdict(int)

    # First add applied rules
    applied_rules = []
    for rule,ohe,original_point_sd in applied_sorted_rules:
        temp = np.zeros(original_point_sd.shape[0]).astype(int)
        temp[rule.get_items()] = 1
        if np.sum(temp & original_point_sd.astype(int))!=temp.sum():
            continue
        else:
            applied_rules.append(rule)
        rule_items = ohe.inverse_transform(temp.reshape((1,-1)))[0] ## TEXT (comment for TEXT)
#         rule_items = temp ## TEXT (uncomment for TEXT)
        for item, val in enumerate(rule_items):
            if val is None:
                continue
#                 if val==0: ## TEXT (uncomment for TEXT)
#                     continue ## TEXT (uncomment for TEXT)
#                 if item not in bb_features:
            bb_features[item] += rule.get_support()
#                     bb_features[item] += counter
#                 bb_features[item] = max(bb_features[item],  rule.get_confidence()/len(rule.get_items()))
        counter -= 1
    set_size_1 = len(bb_features)

    # Second, add applicable rules
    applicable_rules = []
    for rule,ohe,original_point_sd in applicable_sorted_rules:
        temp = np.zeros(original_point_sd.shape[0]).astype(int)
        temp[rule.get_items()] = 1
        if np.sum(temp & original_point_sd.astype(int))!=temp.sum():
            continue
        else:
            applicable_rules.append(rule)
        rule_items = ohe.inverse_transform(temp.reshape((1,-1)))[0] ## TEXT (comment for TEXT)
#         rule_items = temp ## TEXT (uncomment for TEXT)
        for item, val in enumerate(rule_items):
            if val is None:
                continue
            if item not in bb_features:
#                 bb_features[item] += rule.get_support()
                bb_features[item] += counter
        counter -= 1

    # Third, add other rules.
    other_rules = []
    for rule,ohe,original_point_sd in other_sorted_rules:
        temp = np.zeros(original_point_sd.shape[0]).astype(int)
        temp[rule.get_items()] = 1
        # avoid applicable rules
        if np.array_equal(temp, temp & original_point_sd.astype(int)): # error??? it was orig...[0].astype
            continue
#             elif temp.sum()==1:
#                 continue
        elif temp.sum() - np.sum(temp & original_point_sd.astype(int)) >1: # error??? 
            continue
#             else:
        rule_items = ohe.inverse_transform(temp.reshape((1,-1)))[0] ## TEXT (comment for TEXT)
#         rule_items = temp ## TEXT (uncomment for TEXT)
        seen_set = 0
        for item, val in enumerate(rule_items):
            if val is None:
                continue
            if item not in bb_features:
#                 bb_features[item] += rule.get_support()
#                     bb_features[item] += counter
                candid_feature = item
                pass
            else:
                seen_set += 1
        if seen_set==temp.sum()-1: # and (item not in bb_features):
            bb_features[candid_feature] += counter
            other_rules.append(rule)
        counter -= 1

    feature_value_pairs = sorted(bb_features.items(), key=lambda x:x[1], reverse=True)

    return feature_value_pairs, None


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None,
                                   neighborhood_data_sd=None,
                                   ohe=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)

        if isinstance(model_regressor, SigDirect):
            all_rules = defaultdict(list)
            true_label = neighborhood_labels[0].argmax()
            labels_column = np.argmax(neighborhood_labels, axis=1)
            all_raw_rules, predicted_label = get_all_rules(neighborhood_data_sd, labels_column, model_regressor)

            # convert raw rules to rules (one-hot-decoding them)
            if predicted_label==true_label:
                for x,y in all_raw_rules.items():
                    all_rules[x] = [(t,ohe,neighborhood_data_sd[0]) for t in y]
            else:
                predicted_label = -1 # to show we couldn't predict it correctly

            feature_value_pairs, prediction_score = get_features_sigdirect(all_rules, true_label)
            return (0, feature_value_pairs, prediction_score, predicted_label)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)
