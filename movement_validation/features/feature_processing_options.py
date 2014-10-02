# -*- coding: utf-8 -*-
"""
This module will hold a class that will be referenced when processing features.

I'd like to move things from "config" into here ...
- @JimHokanson

"""

class FeatureProcessingOptions(object):
    
    def __init__(self):    
    
        #The idea with this attribute is that functions will check if they are
        #in this list. If they are then they can display some sort of popup that
        #clarifies how they are working.
        #
        #NOTE: No functions actually use this yet. It is just a place holder.
        self.functions_to_explain = []
        
        #This indicates that, where possible, code should attempt to replicate
        #the errors and inconsistencies present in the way that the Schafer lab
        #computed features. This can be useful for ensuring that we are able to
        #compute features in the same way that they did.
        #
        #NOTE: There are a few instances where this is not supported such that
        #the behavior will not match even if this value is set to True.
        self.mimic_old_behaviour = True
    
        self.locomotion = LocomotionOptions()
        
        #This is not yet implemented. The idea is to support not 
        #computing certain features. We might also allow disabling certain 
        #groups of feature.
        self.features_to_ignore = []
        
    def disable_contour_features(self):
         #see self.features_to_ignore
         pass
         

class LocomotionOptions(object):
    
    def __init__(self):
        #locomotion_features.LocomotionVelocity
        #-------------------------------------
        #Units: seconds
        #NOTE: We could get the defaults from the class ...
        self.velocity_tip_diff = 0.25
        self.velocity_body_diff = 0.5
        
        #locomotion_features.MotionEvents
        #--------------------------------------
        # Interpolate only this length of NaN run; anything longer is
        # probably an omega turn.
        # If set to "None", interpolate all lengths (i.e. infinity)
        #TODO - Inf would be a better specification
        self.motion_codes_longest_nan_run_to_interpolate = None
        # These are a percentage of the worm's length
        self.motion_codes_speed_threshold_pct = 0.05
        self.motion_codes_distance_threshold_pct = 0.05
        self.motion_codes_pause_threshold_pct = 0.025
    
        #   These are times (s)
        self.motion_codes_min_frames_threshold = 0.5
        self.motion_codes_max_interframes_threshold = 0.25  

        
        
        