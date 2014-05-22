# -*- coding: utf-8 -*-
"""
Insert description here
"""

import numpy as np

def fp_isequal(x,y,feature_name,tol=1e-6):
  if np.abs(x-y) <= tol:
    return True
  else:
    print 'Values not equal: %s' % feature_name

def corr_value_high(x,y,feature_name,high_corr_value=0.999):
  
  #NOTE: For now I am printing everything, eventually it would be nice
  #to optionally print things ...  
  
  if x.shape != y.shape:
    print 'Shape mismatch: %s' % feature_name
    return False
  else:
    np.reshape(x,x.size)
    np.reshape(y,y.size)
    
    xn = x[~np.isnan(x)] #xn -> x without NaNs or x no NaN -> xn
    yn = y[~np.isnan(y)]
    
    if xn.shape != yn.shape:
      print 'Shape mismatch after NaN filter: %s' % feature_name
      return False
    else:
      c = np.corrcoef(xn,yn)
      is_good = c[1,0] > high_corr_value
      if not is_good:
        print 'Corr value too low for %s: %0.3f' % (feature_name,c[1,0])
      return is_good