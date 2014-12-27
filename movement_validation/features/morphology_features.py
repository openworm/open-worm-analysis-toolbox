# -*- coding: utf-8 -*-
"""
morphology_features.py

"""

import numpy as np

from .. import utils

from . import feature_comparisons as fc

class Widths(object):
    """
    Attributes
    ----------    
    head :
    midbody :
    tail :
    """
    
    fields = ('head', 'midbody', 'tail')
    
    def __init__(self,features_ref):
        """
        Parameters
        ----------
        features_ref : WormFeatures
        
        Note the current approach just computes the mean of the different 
        body section widths. Eventually this should be computed in this class.
        """
        nw = features_ref.nw
    
        for partition in self.fields:
            setattr(self,partition, np.mean(nw.get_partition(partition, 'widths'),0))
    
    @classmethod
    def from_disk(cls,width_ref):

        self = cls.__new__(cls)

        for partition in self.fields:
            setattr(self,partition, utils._extract_time_from_disk(width_ref,partition))
    
        return self
        
    def __eq__(self, other):
        return fc.corr_value_high(self.head, other.head, 'morph.width.head') and \
            fc.corr_value_high(self.midbody, other.midbody, 'morph.width.midbody') and \
            fc.corr_value_high(self.tail, other.tail, 'morph.width.tail')

    def __repr__(self):
        return utils.print_object(self)    