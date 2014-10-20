# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:01:04 2014

@author: mcurrie

Notes
-------------------------
Formerly SegwormMatlabClasses / +seg_worm / +stats / @hist / hist.m
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/%40hist/hist.m

"""
import numpy as np


class Histogram:
    """    
    
    %  TODO  Missing Features:
    %   -----------------------------------------
    %   - saving to disk
    %   - version comparison
    """    
    
    
    pass

    def __init__(self, original_instance=None):
        """
        If original is not None, then this becomes a copy constructor.
        
        """

        # Identification
        #####################

        if not original_instance:
            self.field = None
            self.name = None
            self.short_name = None
            self.units = None            # units associated with the bin values
            self.feature_category = None # posture, locomotion, morphology, path
            
            self.hist_type = None      # 'motion', 'simple', 'event
            self.motion_type = None # 'all', 'forward', 'paused', 'backward'
            self.data_type = None # 'all', 'absolute', 'postive', 'negative'
                        
            self.resolution  = None # bin resolution
            self.is_zero_bin = None  # this might be useless - @JimHokanson
            self.is_signed = None
    
        else:
            """
            %   This is used to create a merged object without affecting
            %   the originals ...
            %   Formerly:
            %   obj_out = seg_worm.stats.hist.createCopy(obj_in)
            %
            %   NOTE: The stats are not currently copied as this is
            %   primarily for merging objects where the stats will be
            %   overwritten.
            """            
            self.field            = original_instance.field
            self.name             = original_instance.name
            self.short_name       = original_instance.short_name
            self.units            = original_instance.units
            self.feature_category = original_instance.feature_category

            self.hist_type   = original_instance.hist_type
            self.motion_type = original_instance.motion_type
            self.data_type   = original_instance.data_type

            self.resolution       = original_instance.resolution
            self.is_zero_bin      = original_instance.is_zero_bin
            self.is_signed        = original_instance.is_signed
            

        #####################

        # The probability density value for each bin    
        self.pdf = None  # single-dimension numpy array of length num_videos
        # The centre of each bin
        self.bins = None # single-dimension numpy array of length num_videos
        
        self.counts = 0  #%[n_videos x bins] %The # of values in each bin
        #TODO: This needs to be clarified, I also don't like the name:      
        self.num_samples = 0 # %[n_videos x 1] # of samples for each video
        
        
        self.mean_per_video = None # single-dimension numpy array of length num_videos
        self.std_per_video = None  # single-dimension numpy array of length num_videos
        

        """
        TODO: Not included yet ...
        p_normal, only for n_valid_measurements >= 3
        [~,cur_s.p_normal]  = seg_worm.fex.swtest(cur_h_e.mean, 0.05, 0);
    
        q_normal - 
        
        """
        



    @property
    def valid_means(self):
        """
        mean_per_video, excluding NaN
        """
        return self.mean_per_video(~np.isnan(self.mean_per_video))

    @property
    def mean(self):
        return np.nanmean(self.mean_per_video)

    @property
    def std(self):
        """
        Standard deviation of means
        """
        return np.nanstd(self.mean_per_video)

    @property
    def num_valid_measurements(self):
        return sum(~np.isnan(self.mean_per_video))

    @property
    def num_videos(self):
        """
        The number of  of videos that this instance contains.
        """
        return len(self.mean_per_video)

    @property
    def first_bin(self):
        return self.bins[0]

    @property
    def last_bin(self):
        return self.bins[-1]

    @property
    def num_bins(self):
        return len(self.bins)

    @property
    def all_valid(self):
        return all(~np.isnan(self.mean_per_video))

    @property
    def none_valid(self):
        return self.num_valid_measurements == 0 



    @staticmethod
    def merge_objects(hist_cell_array):
        """            
        %The goal of this function is to go from n collections of 708
        %histogram summaries of features each, to one set of 708 histogram
        %summaries, that has n elements, one for each video
        """

        # DEBUG: this is just a placeholder; instead of merging it just returns
        #        the first feature set
        return hist_cell_array[0]
        
        
        
        
        