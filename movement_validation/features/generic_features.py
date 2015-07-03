# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:26:27 2015

@author: RNEL
"""

from .. import utils

#This needs to move elsewhere so that the feature files can import it
class Feature(object):

    """
    Attributes
    ----------
    name
    value
    """
    def __repr__(self):
        return utils.print_object(self)

    #This should be part of the features
    def __eq__(self,other):
        #TODO: Figure out how to pass parameters into this
        #We should probably just overload it ...
        return utils.correlation(self.value,other.value,self.name)

#How are we going to do from disk?
#
#1) Need some reference to a saved file
#2) 
        
#    @classmethod
#    def from_disk(cls,width_ref):
#
#        self = cls.__new__(cls)
#
#        for partition in self.fields:
#            widths_in_partition = utils._extract_time_from_disk(width_ref, 
#                                                                partition)
#            setattr(self, partition, widths_in_partition)