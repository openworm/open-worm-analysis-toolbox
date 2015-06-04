# -*- coding: utf-8 -*-
"""
morphology_features.py

Any more complex morphology features requiring classes are defined here.

"""

import numpy as np

from .. import utils
from . import feature_comparisons as fc

class Widths(object):
    """
    Stores the mean width of various partitions of the worm, for each frame.
    
    Attributes
    ----------    
    head:    The mean width of the worm along the 'head' partition
    midbody: The mean width of the worm along the 'midbody' partition
    tail:    The mean width of the worm along the 'tail' partition

    """
    
    def __init__(self, nw=None):
        """
        Parameters
        ----------
        nw: An instance of NormalizedWorm

        """
        self.fields = ('head', 'midbody', 'tail')

        if nw is not None:
            for partition in self.fields:
                setattr(self, partition, 
                        np.mean(nw.get_partition(partition, 'widths'), 0))
    
    @classmethod
    def from_disk(cls, width_ref):
        widths_instance = cls()

        for partition in widths_instance.fields:
            setattr(widths_instance, partition, 
                    utils._extract_time_from_disk(width_ref, partition))
    
        return widths_instance
        
    def __eq__(self, other):
        return \
            fc.corr_value_high(self.head, other.head, 'morph.width.head') and \
            fc.corr_value_high(self.midbody, other.midbody, 
                               'morph.width.midbody') and \
            fc.corr_value_high(self.tail, other.tail, 'morph.width.tail')

    def __repr__(self):
        return utils.print_object(self)    