# -*- coding: utf-8 -*-
"""
Notes
-------------------------
Formerly SegwormMatlabClasses / +seg_worm / +stats / @hist / hist.m
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/%40hist/hist.m

"""
import numpy as np


class Histogram(object):
    """    
    Encapsulates the notion of a histogram for features.
    
    TODO:  Missing Features:
    -----------------------------------------
        - saving to disk
        - version comparison
        - allow loading from a saved file

    """    

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

            # This is what the midbody of the worm is doing while 
            # these values are obtained:
            self.motion_type = None # 'all', 'forward', 'paused', 'backward'
            
            # This is an additional filter on the values of the data. 
            # Either all values are included or:
            #   - the absolute values are used
            #   - only positive values are used
            #   - only negative values are used            
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
        #                   - @JimHokanson
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
        The goal of this function is to go from n collections of 708
        histogram summaries of features each, to one set of 708 histogram
        summaries, that has n elements, one for each video
        
        i.e. from something like:
        {a.b a.b a.b a.b} where .b may have size [1 x m]
        
        to:
        
        a.b, where .b is of size [n x m], in this case [4 x m]
        
        This requires merging histograms that are computed using different
        bins. IMPORTANTLY, because of the way that we do the bin
        definitions, bin edges will always match if they are close, i.e.
        we'll NEVER have:
        
        edges 1: 1,2,3,4,5
        edges 2: 3.5,4.5,5.5,6.5
        
        Instead we might have:
        edges 2: 3,4,5,6,7,8
        
        This simplifies the merging process a bit. This is accomplished by
        always setting bin edges at multiples of the resolution. This was
        not done previously.

        Parameters
        -------------------------
        Formerly objs = seg_worm.stats.hist.mergeObjects(hist_cell_array)

        Parameters
        -------------------------
        hist_cell_array: a list of objects
            Currently each object should only contain a single set of data
            (i.e. single video) prior to merging. This could be changed.


        Returns
        -------------------------
        One object

        """
        # DEBUG: just for fun
        print("In Histogram.merge_objects... # of objects to merge:", len(hist_cell_array))

        #all_objs = [hist_cell_array{:}]

        """
        num_videos_per_object = [all_objs(1,:).n_videos]
        
        if any(n_videos_per_object ~= 1):
            error('Multiple videos per object not yet implemented')
        
        num_videos   = size(all_objs,2)
        num_features = size(all_objs,1)
        
        temp_results = cell(1,n_features)
        
        for iFeature in range(num_features):
            
            cur_feature_array = all_objs(iFeature,:)
            
            # Create an output object with same meta properties
            final_obj   =  cur_feature_array(1).createCopy();
            
            
            # Align all bins
            # ---------------------------------------------------------------
            n_bins     = [cur_feature_array.n_bins]
            start_bins = [cur_feature_array.first_bin]
            min_bin    = min(start_bins)
            max_bin    = max([cur_feature_array.last_bin])
            
            cur_resolution = final_obj.resolution
            new_bins       = min_bin:cur_resolution:max_bin
            
            # Colon operator was giving warnings about non-integer indices :/
            # - @JimHokanson
            start_indices = round((start_bins - min_bin)./cur_resolution + 1);
            end_indices   = start_indices + n_bins - 1;
            
            new_counts = zeros(length(new_bins),n_videos);
            
            for iVideo in range(num_videos):
                cur_start = start_indices(iVideo)
                if ~isnan(cur_start):
                    cur_end   = end_indices(iVideo)
                    new_counts(cur_start:cur_end,iVideo) = cur_feature_array(iVideo).counts
            
            # Update final properties
            # ---------------------------------------------------------------
            final_obj.bins      = new_bins
            final_obj.counts    = new_counts
            final_obj.n_samples       = cat(1,cur_feature_array.n_samples)
            final_obj.mean_per_video  = cat(1,cur_feature_array.mean_per_video)
            final_obj.std_per_video   = cat(1,cur_feature_array.std_per_video)
            final_obj.pdf       = sum(final_obj.counts,2)./sum(final_obj.n_samples)
            
            # Hold onto final object for output
            temp_results{iFeature} = final_obj
        

        objs = [temp_results{:}]
        """
        # DEBUG: this is just a placeholder; instead of merging it just returns
        #        the first feature set
        return hist_cell_array[0]
        


