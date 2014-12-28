# -*- coding: utf-8 -*-
"""
This code is meant to implement the functions that actually compare feature 
data between two different instances without knowing anything about the 
features they are comparing (i.e. just looking at the numbers)

"""

import numpy as np

from .. import utils


def fp_isequal(x, y, feature_name, tol=1e-6):
    if np.isnan(x) and np.isnan(y):
        return True
    elif np.logical_or(np.isnan(x),np.isnan(y)): 
        print('Values not equal: %s' % feature_name)
        #import pdb
        #pdb.set_trace()
        return False
    elif np.abs(x - y) <= tol:
        return True
    else:
        print('Values not equal: %s' % feature_name)
        #import pdb
        #pdb.set_trace()
        return False


def corr_value_high(x, y, feature_name, high_corr_value=0.999, merge_nans=False):

    # NOTE: For now I am printing everything, eventually it would be nice
    # to optionally print things ...

    return_value = False

    try:
        if type(x) != type(y):
            print('Type mismatch %s vs %s: %s' % (type(x),type(y),feature_name))
        elif x.shape != y.shape:
            #TODO: Include shapes of both
            print('Shape mismatch %s vs %s: %s' % (str(x.shape),str(y.shape),feature_name))
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
    
            if (xn.size == 0) and (yn.size == 0):
                return_value = True
            elif (xn.size == 1) and (yn.size == 1):
                #Can't take correlation coefficient with single values
                return_value = True
            elif xn.shape != yn.shape:
                print('Shape mismatch after NaN filter: %s' % feature_name)
            else:
                c = np.corrcoef(xn, yn)
                is_good = c[1, 0] > high_corr_value
                if not is_good:
    
                    #import pdb
                    # pdb.set_trace()
                    print('Corr value too low for %s: %0.3f' %
                          (feature_name, c[1, 0]))
                return_value = is_good
                
#        if not return_value:
#            import pdb
#            pdb.set_trace()
            
        return return_value
    except:
        print('Doh, something went wrong, this should never run')
        import pdb
        pdb.set_trace()
