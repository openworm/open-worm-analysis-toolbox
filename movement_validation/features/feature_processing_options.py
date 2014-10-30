# -*- coding: utf-8 -*-
"""
This module will hold a class that will be referenced when processing features.

I'd like to move things from "config" into here ...
- @JimHokanson

"""

from .. import utils

class FeatureProcessingOptions(object):
    
    def __init__(self,fps):    
    
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
    
        self.locomotion = LocomotionOptions(fps)
        
        #This is not yet implemented. The idea is to support not 
        #computing certain features. We might also allow disabling certain 
        #groups of feature.
        self.features_to_ignore = []
        
    def disable_contour_features(self):
         #see self.features_to_ignore
         pass

    def __repr__(self):
        return utils.print_object(self)
         

class LocomotionOptions(object):
    
    def __init__(self,fps):
        #locomotion_features.LocomotionVelocity
        #-------------------------------------
        #Units: seconds
        #NOTE: We could get the defaults from the class ...
        self.velocity_tip_diff  = 0.25
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

        #locomotion_bends.LocomotionCrawlingBends
        self.crawling_bends = LocomotionCrawlingBends(fps)
        self.foraging_bends = LocomotionForagingBends(fps)
        self.locomotion_turns = LocomotionTurns(fps)

    def __repr__(self):
        return utils.print_object(self)

class LocomotionTurns(object):
    
    def __init__(self,fps):
        self.max_interpolation_gap_allowed = 9 #frames
        self.min_omega_event_length = round(fps/4)

class LocomotionForagingBends(object):
    
    def __init__(self,fps):
        #NOTE: The nose & neck can also be thought of as the head tip 
        #and head neck
        self.min_nose_window_samples = round(0.1 * fps)
        self.max_samples_interp_nose = 2*self.min_nose_window_samples - 1

class LocomotionCrawlingBends(object):

    def __init__(self,fps):
        self.fft_n_samples = 2 ** 14
        
        self.bends_partitions = \
                            {'head':     (5, 10),
                            'midbody':  (22, 27),
                            'tail':     (39, 44)}          
        

        self.peak_energy_threshold = 0.5
        
        # max_amplitude_pct_bandwidth - when determining the bandwidth,
        # the minimums that are found can't exceed this percentage of the maximum.
        # Doing so invalidates the result.        
        self.max_amplitude_pct_bandwidth = 0.5        
        
        self.min_time_for_bend = 0.5
        self.max_time_for_bend = 15
        
        #TODO: What are the units on these things ????
        #This is a spatial frequency
        self.min_frequency = 0.25 * self.max_time_for_bend
        
        #What is the technical max???? 0.5 fps????
        self.max_frequency = 0.25 * fps
        
        #This is a processing optimization.
        #How far into the maximum peaks should we look ...
        #If this value is low, an expensive computation could go faster. If it
        #is too low, then we end up rerunning the calculation the whole dataset
        #and we end up losing time
        self.initial_max_I_pct = 0.5
        
    def __repr__(self):
        return utils.print_object(self)        
        