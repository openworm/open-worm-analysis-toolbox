# -*- coding: utf-8 -*-
"""
Insert description here
"""

import numpy as np
from . import utils


def fp_isequal(x, y, feature_name, tol=1e-6):
    if np.abs(x - y) <= tol:
        return True
    else:
        print('Values not equal: %s' % feature_name)


def corr_value_high(x, y, feature_name, high_corr_value=0.999, merge_nans=False):

    # NOTE: For now I am printing everything, eventually it would be nice
    # to optionally print things ...

    #  try:
    if x.shape != y.shape:
        print('Shape mismatch: %s' % feature_name)
        return False
    else:
        np.reshape(x, x.size)
        np.reshape(y, y.size)

        if merge_nans:
            keep_mask = ~np.logical_or(np.isnan(x), np.isnan(y))
            xn = x[keep_mask]
            yn = y[keep_mask]
        else:
            xn = x[~np.isnan(x)]  # xn -> x without NaNs or x no NaN -> xn
            yn = y[~np.isnan(y)]

        if xn.shape != yn.shape:
            print('Shape mismatch after NaN filter: %s' % feature_name)
            return False
        else:
            c = np.corrcoef(xn, yn)
            is_good = c[1, 0] > high_corr_value
            if not is_good:

                #import pdb
                # pdb.set_trace()
                print('Corr value too low for %s: %0.3f' %
                      (feature_name, c[1, 0]))
            return is_good
#  except:
#    import pdb
#    pdb.set_trace()
