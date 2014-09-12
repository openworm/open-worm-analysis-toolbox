# -*- coding: utf-8 -*-
"""
This module will hold a class that will be referenced when processing features.

I'd like to move things from "config" into here ...
- @JimHokanson

"""

class FeatureProcessingOptions(object):
    
    def __init__(self):    
    
        #The idea with this attribute is that functions will check if they are
        #in this list. If they are then they can 
        #
        #NOTE: No functions use this yet ...
        self.functions_to_explain = []
        
        #This indicates that, where possible, code should attempt to replicate
        #the errors and inconsistencies present in the way that the Schafer lab
        #computed features. This can be useful for ensuring that we are able to
        #compute features in the same way that they did.
        #
        #NOTE: There are a few instances where this is not supported such that
        #the behavior will not match
        self.mimic_old_behaviour = True
    
        self.locomotion = LocomotionOptions()

class LocomotionOptions(object):
    
    def __init__(self):
        #Used in locomotion_features.get_worm_velocity
        #
        #Units: seconds
        #NOTE: We could get the defaults from the class ...
        self.velocity_tip_diff = 0.25
        self.velocity_body_diff = 0.5