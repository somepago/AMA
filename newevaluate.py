""" Evaluate ROC
Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
import numpy as np
from sklearn.metrics import roc_auc_score

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")


def pr(labels,scores,percen):
    thresh = np.percentile(scores, 100 - percen)
    print("Threshold :", thresh)
    pred = (scores>=thresh).astype(int)
    labels = np.array(labels)
    gt = labels.astype(int)
    precision, recall, f_score, support = prf(gt, pred, average='binary')
    print('precision %f , recall %f , f1: %f' % (precision, recall, f_score))


##
def roc(labels, scores, saveto=None):	
    return roc_auc_score(labels, scores)
#     """Compute ROC curve and ROC area for each class"""
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     labels = labels
#     scores = scores

#     # True/False Positive Rates.
#     fpr, tpr, _ = roc_curve(labels, scores)
#     roc_auc = auc(fpr, tpr)

#     # Equal Error Rate
#     eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

#     if saveto:
#         plt.figure()
#         lw = 2
#         plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
#         plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
#         plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver operating characteristic')
#         plt.legend(loc="lower right")
#         plt.savefig(os.path.join(saveto, "ROC.pdf"))
#         plt.close()

#     return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap



recall_level_default = 0.8

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  