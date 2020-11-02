########################################################################################################################
# Class to find best SVM features for single output
########################################################################################################################
import itertools
import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

__author__ = "c.magg"


class SVMSearch:
    """
    Class to search for the best SVM for single- and multi-output
    """

    def __init__(self, files_indep_features_train, files_dep_features_train,
                 files_indep_features_test=None, files_dep_features_test=None):
        """
        Constructor
        :param files_indep_features_train: csv fn for independent features (segmentation features)
        :param files_dep_features_train: csv fn for dependent features (error metrics)
        :param files_indep_features_test: csv fn for independent features test set (segmentation features) [optional]
        :param files_dep_features_test: csv fn for dependent features test set(error metrics) [optional]
        """
        self.X_test = []
        self.X = []
        self.y_test = []
        self.y = []
        self.fn_names_train = []
        self.fn_names_test = []
        # train data for k-fold cross validation
        for idx, fn_features, fn_errors in zip(range(len(files_indep_features_train)), files_indep_features_train,
                                               files_dep_features_train):
            features = pd.read_csv(fn_features, sep=';')
            errors = pd.read_csv(fn_errors, sep=';')
            self.X.append(np.array(features))  # independent features
            self.y.append(np.array(errors))  # dependent features
            self.fn_names_train.append(fn_features.split('/')[-1].split('.')[0].replace('features_', ''))
        self.error_names = errors.keys()

        # test data for final validation
        if files_indep_features_test is not None:
            for idx, fn_features, fn_errors in zip(range(len(files_indep_features_test)), files_indep_features_test,
                                                   files_dep_features_test):
                features = pd.read_csv(fn_features, sep=';')
                errors = pd.read_csv(fn_errors, sep=';')
                self.X_test.append(np.array(features))
                self.y_test.append(np.array(errors))
                self.fn_names_test.append(fn_features.split / '/'[-1].split('.')[0].replace('features_', ''))

    def single_output_svm_kfold(self, n_splits=5, kernel='rbf', nc=10, seed=13771, path=''):
        """
        Test for single-output SVR with k-fold cross validation using the following metrics:
        * train/test regression score
        * train/test mean absolute deviation
        & store results to csv with init path
        :param n_splits: number of splits for K-fold cross validation
        :param kernel: kernel type for SVR
        :param nc: maximal number of regularization parameter C
        :param seed: random seed
        :param path: path to store csv files
        :return: dataframe with results
        """
        logging.info(" K-fold cross validation of single SVM with %d splits, %s kernel and maximal C %d", n_splits,
                     kernel, nc)
        np.random.seed(seed)
        C = range(1, nc)

        for idx in range(len(self.X)):
            df = pd.DataFrame(
                columns=['error_metrics', 'C', 'score_train', 'score_test', 'mean_error_train', 'mean_error_test'])
            logging.info(' Dataset %s', self.fn_names_train[idx])

            # go through all error metrics
            for i in range(np.shape(self.y[idx])[1]):
                logging.info('  Process error metrics: %d %s', i, self.error_names[i])

                #  go through all C values
                for c in C:
                    row = self.cross_validation_single_svm_results(self.X[idx], self.y[idx], i, n_splits, kernel, c,
                                                                   self.error_names[i])
                    df = df.append(row, ignore_index=True)
            fn = os.path.join(path, 'singleoutput_result_' + self.fn_names_train[idx] + '_kfold.csv')
            df.to_csv(fn, index=False, sep=';')
        return df

    def single_output_svm_test(self, kernel='rbf', nc=10, seed=13771, path=''):
        """
        Test for single-output SVR with test set using the following metrics:
        * train/test regression score
        * train/test mean absolute deviation
        & store results to csv with init path
        :param kernel: kernel type for SVR
        :param nc: maximal number of regularization parameter C
        :param seed: random seed
        :param path: path to store csv files
        :return: dataframe with results
        """
        logging.info(" Validation with testset of single SVM with %s kernel and maximal C %d", kernel, nc)
        np.random.seed(seed)
        C = range(1, nc)

        if len(self.X_test) == 0:
            raise ValueError("Test set is not given.")

        for idx in range(len(self.X)):
            df = pd.DataFrame(
                columns=['error_metrics', 'C', 'score_train', 'score_test', 'mean_error_train', 'mean_error_test'])
            logging.info(' Dataset %s', self.fn_names_train[idx])

            # go through all error metrics
            for i in range(np.shape(self.y[idx])[1]):
                logging.info('  Process error metrics: %d %s', i, self.error_names[i])

                #  go through all C values
                for c in C:
                    row = self.testset_single_svm_results(self.X[idx], self.y[idx],
                                                          self.X_test[idx], self.y_test[idx],
                                                          i, kernel, c, self.error_names[i])
                    df = df.append(row, ignore_index=True)
            fn = os.path.join(path, 'singleoutput_result_' + self.fn_names_train[idx] + '_test.csv')
            df.to_csv(fn, index=False, sep=';')
        return df

    @staticmethod
    def cross_validation_single_svm_results(X, y, idx, n_splits, kernel, c, name=''):
        """
        K-fold cross validation for single SVR
        :param X: independent features
        :param y: dependent features (all dependent features)
        :param idx: index for dependent feature to be tested
        :param n_splits: number of splits for K-fold cross validation
        :param kernel: kernel type for SVR
        :param c: regularization parameter C
        :param name: name of dependent feature to be tested
        :return: dict with results
        """
        kf = KFold(n_splits=n_splits)
        score_train_fold = []
        score_test_fold = []
        mean_train_fold = []
        mean_test_fold = []
        # k-fold cross validation
        for train_index, test_index in kf.split(X):
            X_ = X[train_index, :]
            y_ = y[train_index, :][:, idx]
            X_test_ = X[test_index, :]
            y_test_ = y[test_index, :][:, idx]
            # regressor
            regr = svm.SVR(kernel=kernel, C=c, epsilon=0.1)
            regr.fit(X_, y_)
            score_train_fold.append(regr.score(X_, y_))
            mean_train_fold.append(np.mean(np.abs(regr.predict(X_) - y_)))
            score_test_fold.append(regr.score(X_test_, y_test_))
            mean_test_fold.append(np.mean(np.abs(regr.predict(X_test_) - y_test_)))
            # print('     train set', score_train[-1], mean_train[-1])
            # print('     test set', score_test[-1], mean_test[-1])
        row = {'error_metrics': name,
               'C': c,
               'score_train': np.mean(score_train_fold),
               'score_test': np.mean(score_test_fold),
               'mean_error_train': np.mean(mean_train_fold),
               'mean_error_test': np.mean(mean_test_fold)}
        return row

    @staticmethod
    def testset_single_svm_results(X, y, X_test, y_test, idx, kernel, c, name=''):
        """
        Validation for single SVR with test set
        :param X_test: independent features test set
        :param y_test: dependent features test set (all dependent features)
        :param X: independent features
        :param y: dependent features (all dependent features)
        :param idx: index for dependent feature to be tested
        :param kernel: kernel type for SVR
        :param c: regularization parameter C
        :param name: name of dependent feature to be tested
        :return: dict with results
        """
        # get data
        X_ = X
        y_ = y[:, idx]
        X_test_ = X_test
        y_test_ = y_test[:, idx]
        # regressor
        regr = svm.SVR(kernel=kernel, C=c, epsilon=0.1)
        regr.fit(X_, y_)
        # results
        row = {'error_metrics': name,
               'C': c,
               'score_train': regr.score(X_, y_),
               'score_test': regr.score(X_test_, y_test_),
               'mean_error_train': np.mean(np.abs(regr.predict(X_) - y_)),
               'mean_error_test': np.mean(np.abs(regr.predict(X_test_) - y_test_))}
        return row

    def multi_output_svm_kfold(self, n_splits=5, kernel='rbf', nc=10, seed=13771, path=''):
        """
        Test for multi-output SVR with k-fold cross validation using the following metrics:
        * train/test regression score
        * train/test mean absolute deviation
        & store results to csv with init path
        :param n_splits: number of splits for K-fold cross validation
        :param kernel: kernel type for SVR
        :param nc: maximal number of regularization parameter C
        :param seed: random seed
        :param path: path to store csv files
        :return: dataframe with results of last file
        """
        logging.info(" K-fold cross validation of multi-output SVM with %d splits, %s kernel and maximal C %d",
                     n_splits, kernel, nc)
        np.random.seed(seed)
        C = range(1, nc)

        for idx in range(len(self.X)):
            df = pd.DataFrame(
                columns=['error_metrics', 'C', 'score_train', 'score_test', 'mean_error_train', 'mean_error_test'])
            logging.info(' Dataset %s', self.fn_names_train[idx])

            # go through all error metrics combinations
            all_idx = list(range(np.shape(self.y[idx])[1]))
            for ind in range(2, np.shape(self.y[idx])[1] + 1):
                logging.info(" Take all %d-combinations", ind)
                combinations = list(itertools.combinations(all_idx, ind))
                for combi in combinations:
                    logging.info('  Process error metrics: %s %s', str(combi), list(self.error_names[list(combi)]))

                    #  go through all C values
                    for c in C:
                        row = self.cross_validation_multiple_svm_output(self.X[idx], self.y[idx], combi, n_splits,
                                                                        kernel, c,
                                                                        list(self.error_names[list(combi)]))
                        df = df.append(row, ignore_index=True)
            fn = os.path.join(path, 'multioutput_result_' + self.fn_names_train[idx] + '_kfold.csv')
            df.to_csv(fn, index=False, sep=';')
        return df

    def multi_output_svm_test(self, kernel='rbf', nc=10, seed=13771, path=''):
        """
        Test for multi-output SVR with validation with test set using the following metrics:
        * train/test regression score
        * train/test mean absolute deviation
        & store results to csv with init path
        :param kernel: kernel type for SVR
        :param nc: maximal number of regularization parameter C
        :param seed: random seed
        :param path: path to store csv files
        :return: dataframe with results of last file
        """
        logging.info(" Validation with dataset of multi-output SVM with %s kernel and maximal C %d",
                     kernel, nc)
        np.random.seed(seed)
        C = range(1, nc)

        if len(self.X_test) == 0:
            raise ValueError("Test set is not given.")

        for idx in range(len(self.X)):
            df = pd.DataFrame(
                columns=['error_metrics', 'C', 'score_train', 'score_test', 'mean_error_train', 'mean_error_test'])
            logging.info(' Dataset %s', self.fn_names_train[idx])

            # go through all error metrics combinations
            all_idx = list(range(np.shape(self.y[idx])[1]))
            for ind in range(2, np.shape(self.y[idx])[1] + 1):
                logging.info(" Take all %d-combinations", ind)
                combinations = list(itertools.combinations(all_idx, ind))
                for combi in combinations:
                    logging.info('  Process error metrics: %s %s', str(combi), list(self.error_names[list(combi)]))

                    #  go through all C values
                    for c in C:
                        row = self.testset_multiple_svm_results(self.X[idx], self.y[idx],
                                                                self.X_test[idx], self.y_test[idx],
                                                                combi, kernel, c, list(self.error_names[list(combi)]))
                        df = df.append(row, ignore_index=True)
            fn = os.path.join(path, 'multioutput_result_' + self.fn_names_train[idx] + '_test.csv')
            df.to_csv(fn, index=False, sep=';')
        return df

    @staticmethod
    def cross_validation_multiple_svm_output(X, y, idx, n_splits, kernel, c, name=''):
        """
        K-fold cross validation for multioutput SVR
        :param X: independent features
        :param y: dependent features (all dependent features)
        :param idx: index for dependent features to be tested
        :param n_splits: number of splits for K-fold cross validation
        :param kernel: kernel type for SVR
        :param c: regularization parameter C
        :param name: name of dependent features to be tested
        :return: dict with results
        """
        kf = KFold(n_splits=n_splits)
        score_train_fold = []
        score_test_fold = []
        mean_train_fold = []
        mean_test_fold = []
        # k-fold cross validation
        for train_index, test_index in kf.split(X):
            X_ = X[train_index, :]
            y_ = y[train_index, :][:, idx]
            X_test_ = X[test_index, :]
            y_test_ = y[test_index, :][:, idx]
            # regressor
            regr = svm.SVR(kernel=kernel, C=c, epsilon=0.1)
            wrapper = MultiOutputRegressor(regr)
            wrapper.fit(X_, y_)
            score_train_fold.append(wrapper.score(X_, y_))
            mean_train_fold.append(np.mean(np.abs(wrapper.predict(X_) - y_)))
            score_test_fold.append(wrapper.score(X_test_, y_test_))
            mean_test_fold.append(np.mean(np.abs(wrapper.predict(X_test_) - y_test_)))
        row = {'error_metrics': name,
               'C': c,
               'score_train': np.mean(score_train_fold),
               'score_test': np.mean(score_test_fold),
               'mean_error_train': np.mean(mean_train_fold),
               'mean_error_test': np.mean(mean_test_fold)}
        return row

    @staticmethod
    def testset_multiple_svm_results(X, y, X_test, y_test, idx, kernel, c, name=''):
        """
        Validation for multioutput SVR with test set
        :param X_test: independent features test set
        :param y_test: dependent features test set (all dependent features)
        :param X: independent features
        :param y: dependent features (all dependent features)
        :param idx: index for dependent feature to be tested
        :param kernel: kernel type for SVR
        :param c: regularization parameter C
        :param name: name of dependent feature to be tested
        :return: dict with results
        """
        # get data
        X_ = X
        y_ = y[:, idx]
        X_test_ = X_test
        y_test_ = y_test[:, idx]
        # regressor
        regr = svm.SVR(kernel=kernel, C=c, epsilon=0.1)
        wrapper = MultiOutputRegressor(regr)
        wrapper.fit(X_, y_)
        # results
        row = {'error_metrics': name,
               'C': c,
               'score_train': wrapper.score(X_, y_),
               'score_test': wrapper.score(X_test_, y_test_),
               'mean_error_train': np.mean(np.abs(wrapper.predict(X_) - y_)),
               'mean_error_test': np.mean(np.abs(wrapper.predict(X_test_) - y_test_))}
        return row
