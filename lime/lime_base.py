"""
Contains abstract functionality for learning locally linear sparse model.
"""
# from __future__ import print_function

import sys
import math
import gc
import copy
import os.path
import pickle
import itertools
from collections import Counter, defaultdict
from importlib import reload  

import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
import sklearn.tree
from pprint import pprint
from sklearn.metrics import accuracy_score

from sigdirect import SigDirect

def get_all_rules(input_):
    neighborhood_data, labels_column, ohe, true_label = input_

    early_stopping = True # TODO
    do_pruning = True
    is_cpp_code=True # TODO: to change to python, update the path in experiments_config.py
    threshold = 0.00000005
    

    hash_value = hash(neighborhood_data.tostring() 
                    + labels_column.tostring()
                    + str.encode(str(do_pruning))
                    + str.encode(str(early_stopping))
#                         + str.encode(str(config.ALPHA)) # or threshold
                     )
    clf_path = "{}.pickle".format(hash_value)
#         if False:
    if os.path.exists(clf_path):
        clf = pickle.load(open(clf_path, 'rb'), encoding='latin1')
    else:
        if is_cpp_code: # C++ / cython implementation
            clf = SigDirect(clf_version=1, alpha=threshold,  # TODO
                            early_stopping=early_stopping, confidence_threshold=0.5, is_binary=True) # TODO
        else: # python implementation
            clf = SigDirect(early_stopping=early_stopping)
        if do_pruning:
            clf.fit(neighborhood_data, labels_column)
        else: # not working with cpp
            clf.fit(neighborhood_data, labels_column, prune=False)
        pickle.dump(clf, open(clf_path, 'wb'))
    local_pred =  clf.predict(neighborhood_data[0].reshape((1,-1)), 2).astype(int)[0]

    all_rules = clf.get_all_rules()
    return all_rules, local_pred

def get_features_sd_4(all_rules, true_label):
    
    approach = 14
    
    # use applied rules first, and then the rest of the applicable rules, and then all rules (other labels, rest of them match)
    if approach==14:
#         predicted_label = clf.predict(neighborhood_data[0].reshape((1,-1)), heuristic=2).astype(int)[0]
        
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
        applied_sorted_rules = [x for x in applied_sorted_rules if x[0].get_confidence() * x[0].get_support()>0.00]

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
        applicable_sorted_rules = [x for x in applicable_sorted_rules if (x[0].get_confidence()* x[0].get_support() )>1.00]

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
        other_sorted_rules = [x for x in other_sorted_rules if (x[0].get_confidence() * x[0].get_support())>1.000]
        
        
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
#             rule_items = ohe.inverse_transform(temp.reshape((1,-1)))[0] ## TEXT
            rule_items = temp
            print(rule)
            for item, val in enumerate(rule_items):
                if val is None:
                    continue
                if val==0: ## TEXT
                    continue ## TEXT
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
#             rule_items = ohe.inverse_transform(temp.reshape((1,-1)))[0] ## TEXT
            rule_items = temp
            for item, val in enumerate(rule_items):
                if val is None:
                    continue
                if item not in bb_features:
#                 bb_features[item] += rule.get_support()
                    bb_features[item] += counter
            counter -= 1
        set_size_2 = len(bb_features)

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
#             rule_items = ohe.inverse_transform(temp.reshape((1,-1)))[0] ## TEXT
            rule_items = temp
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
        set_size_3 = len(bb_features)

        feature_value_pairs = sorted(bb_features.items(), key=lambda x:x[1], reverse=True)        


    return (0, feature_value_pairs, None, 0)


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
                # print(data[0])
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
    # @profile(stream=fp)
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
        feature_selection = 'none'
        weights = self.kernel_fn(distances)

        if isinstance(model_regressor, sklearn.tree.DecisionTreeRegressor):
            def get_features_dt(clf, row):
                feature = clf.tree_.feature
                leave_id = clf.apply(row.reshape(1, -1))
                node_indicator = clf.decision_path([row])
                features = []
                node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

                for node_id in node_index:
                    if leave_id[0] == node_id:  # <-- changed != to ==
                        continue # <-- comment out
                    else: # < -- added else to iterate through decision nodes
                        features.append(feature[node_id])
                return features
            
            easy_model = model_regressor
            # print(labels_column)
            easy_model.fit(neighborhood_data[:, :], 
                            labels_column)
            prediction_score = easy_model.score(
                neighborhood_data[:, :],
                labels_column, sample_weight=weights)
            local_pred = easy_model.predict(neighborhood_data[0, :].reshape(1, -1))
            important_features = get_features_dt(easy_model, neighborhood_data[0,:])
            feature_value_pairs = [(x,1./len(important_features)) for x in important_features]
            return (0, feature_value_pairs, prediction_score, local_pred)

        elif isinstance(model_regressor, str):
            try:
                if type(neighborhood_data_sd)==list:
                    fidelity = 0.0
                    all_rules = defaultdict(list)
                    for x,y,z in zip(neighborhood_data_sd, neighborhood_labels, ohe):
                        ##### multi-class vs binary dataset for neighbourhood generation #####
                        true_label = y[0].argmax()
                        labels_column = np.argmax(y, axis=1)
                        r_, predicted_label = get_all_rules((x, labels_column, z, true_label))
                        if predicted_label!=true_label:
                            continue
                        else:
                            fidelity = 1.0
                        for i,j in r_.items():
#                             all_rules[i].extend([(x_,z,x[0]) for x_ in j])
                            all_rules[i] = [(t,z,x[0]) for t in j]
                        break
                    _, feature_value_pairs, prediction_score, local_pred = get_features_sd_4(
                                                                                    all_rules, 
                                                                                    true_label, 
                                                                                    )
#                     if len(all_rules)==0:
#                         fidelity = 0.0
                else:
                    ##### multi-class vs binary dataset for neighbourhood generation #####
#                     true_label = 1
#                     labels_column = np.argmax(neighborhood_labels, axis=1)==neighborhood_labels[0].argmax()
                    true_label = neighborhood_labels[0].argmax()
                    labels_column = np.argmax(neighborhood_labels, axis=1)
                    fidelity = 1.0
                    all_rules = defaultdict(list)
                    
                    r_, predicted_label = get_all_rules((neighborhood_data_sd, labels_column, ohe, true_label))
                    no_retry = True # 
                    if predicted_label==true_label:                 
                        for x,y in r_.items():
                            all_rules[x] = [(t,ohe,neighborhood_data_sd[0]) for t in y]
                        _, feature_value_pairs, prediction_score, local_pred = get_features_sd_4(all_rules, true_label)
                        return (0, feature_value_pairs, prediction_score, fidelity)
                    elif no_retry: # 
                        fidelity = 0.0
                        _, feature_value_pairs, prediction_score, local_pred = get_features_sd_4(all_rules, true_label)
                        return (0, feature_value_pairs, prediction_score, fidelity)


                    labels_column_2 = np.zeros_like(labels_column)
                    neighborhood_data_sd_2 = np.zeros_like(neighborhood_data_sd)
                    
                    distances = - np.abs(np.subtract(neighborhood_data_sd[0], neighborhood_data_sd)).sum()
                    ps_p = sp.special.softmax(1.0000005 * distances / np.equal(labels_column,labels_column[0]).astype(int)) 
                    ps_n = sp.special.softmax(0.500001 * distances / np.not_equal(labels_column,labels_column[0]).astype(int)) 

#                     p_size = int((10/math.sqrt(neighborhood_labels.shape[1])) * labels_column.shape[0]/10)
                    p_size = int(5 * labels_column.shape[0]/10)
                    n_size = labels_column.shape[0] - p_size

                    i_p = np.random.choice(labels_column.shape[0], 
                                    size=p_size, 
                                    replace=True, 
                                    p=ps_p)
                    i_n = np.random.choice(labels_column.shape[0], 
                                    size=n_size, 
                                    replace=True, 
                                    p=ps_n)
#                     print(i_p.shape, i_n.shape, p_size)
                    labels_column_2[:p_size] =  labels_column[i_p]
                    labels_column_2[p_size:] =  labels_column[i_n]
                    neighborhood_data_sd_2[:p_size] = neighborhood_data_sd[i_p]
                    neighborhood_data_sd_2[p_size:] = neighborhood_data_sd[i_n]

                    neighborhood_data_sd_2[:10] = neighborhood_data_sd[0]
                    labels_column_2[:10] = labels_column[0]
                    neighborhood_data_sd = neighborhood_data_sd_2
                    labels_column = labels_column_2


                    # more than half belong to target class:
                    if Counter(labels_column)[labels_column[0]]>labels_column.shape[0]/2:
                        pass
                    # less than 1/n belong to target class (n classes):
                    elif Counter(labels_column)[labels_column[0]]<labels_column.shape[0]/len(Counter(labels_column)):
                        pass

                    #########################
                    all_rules = defaultdict(list)
                    r_, predicted_label = get_all_rules((neighborhood_data_sd, labels_column, ohe, true_label))
                    if predicted_label==true_label:                 
                        for x,y in r_.items():
                            all_rules[x] = [(t,ohe,neighborhood_data_sd[0]) for t in y]
                    else:
                        fidelity = 0.0
                _, feature_value_pairs, prediction_score, local_pred = get_features_sd_4(all_rules, true_label)
                return (0, feature_value_pairs, prediction_score, fidelity)
            except Exception as e:
                print(repr(e))
#                 raise e
                return (0, [], 0.0, 0.0)
        else:
            if model_regressor is None:
                model_regressor = Ridge(alpha=1, fit_intercept=True,
                                        random_state=self.random_state)
            easy_model = model_regressor
            easy_model.fit(neighborhood_data[:, used_features],
                           labels_column, sample_weight=weights)
            prediction_score = easy_model.score(
                neighborhood_data[:, used_features],
                labels_column, sample_weight=weights)
            print('model type:', type(easy_model))
            local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

            if self.verbose:
                print('Intercept', easy_model.intercept_)
                print('Prediction_local', local_pred,)
                print('Right:', neighborhood_labels[0, label])
            return (easy_model.intercept_,
                    sorted(zip(used_features, easy_model.coef_),
                           key=lambda x: np.abs(x[1]), reverse=True),
                    prediction_score, local_pred)
