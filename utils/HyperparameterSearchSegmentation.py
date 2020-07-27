########################################################################################################################
# Class to implement hyperparameter search for segmentation
########################################################################################################################
import pandas as pd
import numpy as np
import os.path
import sys
import logging as log
import itertools

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

__author__ = "c.magg"


class HyperparameterSearchSegmentation:

    def __init__(self, segmentor, struct=None, first=None, last=None):
        self.segm = segmentor
        self.result = pd.DataFrame(columns=['Struct', 'Values', 'Dice'])
        if struct is None or first is None or last is None:
            # TODO: add error handling
            self.found_struct = self.segm.patient.get_filtered_contour_names().values
            self.index_first = self.segm.patient.contour_list_names_filtered['first'].values
            self.index_last = self.segm.patient.contour_list_names_filtered['last'].values
        else:
            self.found_struct = struct
            self.index_first = first
            self.index_last = last

    def eval(self, params, path):
        """
        Method to evaluate the best hyperparameter setting with the provided settings
        :return:
        """
        liste = []
        combinations = []
        if type(params) == dict:
            for k in params.keys():
                liste.append(params[k])
            combinations = list(itertools.product(liste[0], liste[1], liste[2]))
            log.info(" Start hyperparameter search with # combinations %s", len(combinations))

        for struct, first, last in zip(self.found_struct, self.index_first, self.index_last):
            res = []
            if type(params) == list:
                liste = []
                i = [idx for idx, p in enumerate(params) for k, v in p.items() if v == struct][0]
                for k in params[i].keys():
                    if k == 'struct':
                        continue
                    liste.append(params[i][k])
                combinations = list(itertools.product(liste[0], liste[1], liste[2]))
                log.info(" Start hyperparameter search with # combinations %s", len(combinations))
            for idx, comb in enumerate(combinations):
                log.info(" Combination %s", idx)
                self.segm.active_contour(struct, postprocess=-1,
                                         first=first, last=last,
                                         kernel=(comb[0], comb[0]),
                                         beta=comb[1], w_edge=0.1, max_iterations=comb[2])
                dice, vol_error, hausdorff = self.segm.evaluate_segmentation()
                tmp = {'Struct': struct, 'Combi': comb,
                       'Dice': np.mean(dice), 'Vol_error': np.mean(vol_error), 'Hausdorff': np.mean(hausdorff)}
                res.append(tmp)
            dice = []
            for r in res:
                dice.append(r['Dice'])
            idx_best = np.argmax(dice)
            self.result = self.result.append({'Struct': struct,
                                              'Values': res[idx_best]['Combi'],
                                              'Dice': np.max(dice)},
                                             ignore_index=True)
            self.result.to_csv(path, index=False)
        return self.result
