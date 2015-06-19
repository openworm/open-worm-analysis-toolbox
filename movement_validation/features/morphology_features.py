# -*- coding: utf-8 -*-
"""
morphology_features.py

"""
import numpy as np

from .. import utils

class Widths(object):
    """
    Attributes
    ----------    
    head :
    midbody :
    tail :
    """
    
    fields = ('head', 'midbody', 'tail')
    
    def __init__(self, features_ref):
        """
        Parameters
        ----------
        features_ref : WormFeatures instance
        
        Note the current approach just computes the mean of the 
        different body section widths. Eventually this should be 
        computed in this class.

        """
        nw = features_ref.nw
    
        for partition in self.fields:
            widths_in_partition = nw.get_partition(partition, 'widths')
            setattr(self, partition, np.mean(widths_in_partition, 0))
    
    @classmethod
    def from_disk(cls,width_ref):

        self = cls.__new__(cls)

        for partition in self.fields:
            widths_in_partition = utils._extract_time_from_disk(width_ref, 
                                                                partition)
            setattr(self, partition, widths_in_partition)
    
        return self
        
    def __eq__(self, other):
        return (
                utils.correlation(self.head, other.head, 
                                   'morph.width.head') and
                utils.correlation(self.midbody, other.midbody,
                                   'morph.width.midbody') and
                utils.correlation(self.tail, other.tail, 
                                   'morph.width.tail'))

    def __repr__(self):
        return utils.print_object(self)  
