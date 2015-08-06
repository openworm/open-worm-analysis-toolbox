# -*- coding: utf-8 -*-
"""
generic_features

@author: RNEL
"""

from .. import utils

import re

#Get everything until a period that is followed by no periods
_parent_feature_pattern = re.compile('(.*)\.([^\.]+)')

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

    def get_feature(self,wf,feature_name):
        
        """
        This was put in to do anything we need when getting a feature
        rather than calling the feature directly
        """

        #1) Do logging - NYI
        #What is the easiest way to initialize without forcing a init super call?
    
        #2) Make the call to 

        return wf.get_feature(feature_name)
        

def get_parent_feature_name(feature_name):
    
    """
    Go from something like:
    locomotion.crawling_bends.head.amplitude
    TO
    locomotion.crawling_bends.head
    """    
        
    return get_feature_name_info(feature_name)[0]


def get_feature_name_info(feature_name):

    """
    Outputs
    -------
    (parent_name,specific_feature_name)
    """
    #We could make this more obvious by using split ...
    #I might want to also remove the parens and only get back the 1st string somehow
    #0 - the entire match
    #1 - the first parenthesized subgroup

    
    result = _parent_feature_pattern.match(feature_name)
    return result.group(1),result.group(2)

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
        
#event_durations
#distance_during_events
#time_between_events
#distance_between_events
#frequency
#time_ratio
#data_ratio

def get_event_attribute(event_object,attribute_name):
    
    #We might want to place some logic in here

    if event_object.is_null:
        return None
    else:
        return getattr(event_object,attribute_name)

class EventFeature(Feature):
    
    def __init__(self,wf,feature_name):
        self.name = feature_name
        event_name, feature_type = get_feature_name_info(feature_name)
        event_name = get_parent_feature_name(feature_name)
        event_value = self.get_feature(wf,event_name).value              
        self.value = get_event_attribute(event_value,feature_type)        

    @classmethod    
    def from_schafer_file(cls,wf,feature_name):
        return cls(wf,feature_name)

#We might want to make things specific again but for now we'll use
#a single class

#class EventDuration(Feature):
#    
#    def __init__(self, wf, feature_name):
#        parent_name = get_parent_feature_name(feature_name)
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'event_durations')
#        self.name = event_name + '.event_durations'
#    
#class DistanceDuringEvents(Feature):
#    
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'distance_during_events')
#        self.name = event_name + '.distance_during_events'
#
#class TimeBetweenEvents(Feature):
#
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'time_between_events')
#        self.name = event_name + '.time_between_events'    
#
#class DistanceBetweenEvents(Feature):
#
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'distance_between_events')
#        self.name = event_name + '.distance_between_events'   
#    
#class Frequency(Feature):
#
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'frequency')
#        self.name = event_name + '.frequency'   
#
#class EventTimeRatio(Feature):
#
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'time_ratio')
#        self.name = event_name + '.time_ratio'   
#    
#class EventDataRatio(Feature):
#
#    def __init__(self,wf,event_name):
#        temp = wf[event_name]
#        self.value = get_event_attribute(temp.value,'data_ratio')
#        self.name = event_name + '.data_ratio'   
    
    