########################################################################################################################
# Class to create segmentation mask ready to be rendered by VTK
# This includes segmentation mask modification in numpy
########################################################################################################################
import sys
import os
import pandas as pd
import numpy as np
from sklearn import svm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from SVMFeatures import SVMFeatures

__author__ = "c.magg"


class SVRPrediction:
    """
    Class to generate the SVM predictions for given mask
    Loads and trains internally a single-output SVM for jaccard distance
    """

    def __init__(self, standardized=True):
        path_data = "../../Data/features_standardized/"
        if standardized is False:
            path_data = "../../Data/features/"
        all_files = [filename for filename in os.listdir(path_data) if filename.endswith("csv")]
        all_files_error = [os.path.join(path_data, fn) for fn in all_files if 'error_metrics' in fn]
        all_files_features = [os.path.join(path_data, fn) for fn in all_files if 'features' in fn]
        self.X = []
        self.y = []
        for fn_features, fn_errors in zip(all_files_features, all_files_error):
            features = pd.read_csv(fn_features, sep=';')
            errors = pd.read_csv(fn_errors, sep=';')
            self.X.append(np.array(features))  # independent features
            self.y.append(np.array(errors))  # dependent features
        self.structures = ['Brain', 'CerebellPOSTYL', 'Cingulumleft', 'Cingulumright', 'Corpuscallosum', 'CTV', 'CTV1',
                      'CTV2', 'Fornix', 'GTV', 'Hypothalamus', 'PapezCircle', 'PTV1', 'PTV2', 'Scalp',
                      'TemporalLobeLt', 'TemporalLobeRt', 'ThalamusantL', 'ThalamusantR', 'Thalamusleft',
                      'Thalamusright']
        self.C = [85, 99, 9, 1, 5, 13, 14, 13, 99, 6, 1, 99, 3, 21, 64, 74, 89, 1, 99, 1, 1]
        self.C_per_structure = dict(zip(self.structures, self.C))
        self.regressions = dict().fromkeys(self.structures)

        self.train_svm()

    def train_svm(self):
        for idx in range(len(self.X)):
            X = self.X[idx]
            y = self.y[idx][:, 1]  # jaccard distance
            regr = svm.SVR(kernel='rbf', C=self.C_per_structure[self.structures[idx]], epsilon=0.1)
            regr.fit(X, y)
            self.regressions[self.structures[idx]] = regr

    def make_prediction(self, img, mask, structure, standardized=True):
        features, tmp = SVMFeatures([img], preds=[mask], k=255).calculate(standardize=standardized)
        features = features.values[0].reshape(1, -1)
        return self.regressions[structure].predict(features)[0]
