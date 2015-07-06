# -*- coding: utf-8 -*-
"""
morphology_features.py

"""
import numpy as np

from .generic_features import Feature
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
        
#======================================================================
#                       NEW CODE 
#======================================================================
#
#
#   Still need to handle equality comparison and loading from disk
#   
class Length(Feature):
    
    def __init__(self,wf):
        self.name = 'morphology.length' 
        self.value = wf.nw.length
                
class Width(Feature):
    """
    This should only be called by the subclasses
    
    Attributes
    ----------
    partition_name
    """
    
    def __init__(self,wf,partition_name):

        """
        Parameters
        ----------
        wf : WormFeatures instance
        """
        
        self.name = 'morphology.width.' + partition_name
        self.partition_name = partition_name
        #I'm not thrilled with the name of this method
        self.value = wf.nw.get_partition(partition_name, 'widths')
        
class HeadWidth(Width):
    
    def __init__(self,wf):
        Width.__init__(self,wf,'head')

class MidbodyWidth(Width):
    
    def __init__(self,wf):
        Width.__init__(self,wf,'midbody')
        
class TailWidth(Width):
    
    def __init__(self,wf):
        Width.__init__(self,wf,'tail')        
    
class Area(Feature):

    def __init__(self,wf):
        self.name = 'morphology.area'
        self.value = wf.nw.area
        
class AreaPerLength(Feature):
    
    def __init__(self,wf):
        self.name = 'morphology.area_per_length'
        area = wf['morphology.area'].value
        length = wf['morphology.length'].value
        self.value = area/length

class WidthPerLength(Feature):
    
    def __init__(self,wf):
        self.name = 'morphology.width_per_length'
        width = wf['morphology.width.midbody'].value
        length = wf['morphology.length'].value
        self.value = width/length
    
    
    
    
    
    
    
    
    
    
    