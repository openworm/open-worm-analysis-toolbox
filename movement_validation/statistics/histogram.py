# -*- coding: utf-8 -*-
"""
Notes
-------------------------
Formerly SegwormMatlabClasses / +seg_worm / +stats / @hist / hist.m
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/%40hist/hist.m

"""
import numpy as np
from .. import config

class Histogram(object):
    """    
    Encapsulates the notion of a histogram for features.
    
    All bins in this histogram have an equal bin width.

   Current Status:
    %   -----------------------------------------------------------------
    %   Working but could be optimized a bit and there are some missing
    %   features. At this point however it will work for computing the
    %   statistics objects. Documentation could also be improved.
    
    TODO:  Missing Features:
    -----------------------------------------
        - saving to disk
        - version comparison
        - allow loading from a saved file

    """    
    def __init__(self, data, specs, hist_type, motion_type, data_type):
        """
        
        Parameters
        ------------------
        data: numpy array
            The data to be counted for the histogram
        specs: instance of Specs class
        hist_type: string
            histogram type  # 'motion', 'simple', 'event
        motion_type: string
             # 'all', 'forward', 'paused', 'backward'
        data_type: string
            # 'all', 'absolute', 'postive', 'negative'
            This is an additional filter on the values of the data. 
            Either all values are included or:
              - the absolute values are used
              - only positive values are used
              - only negative values are used     
        Returns        
        
        """
        self.field            = specs.long_field
        self.name             = specs.name
        self.short_name       = specs.short_name
        self.units            = specs.units
        self.feature_category = specs.feature_category
        self.bin_width        = specs.bin_width
        self.is_zero_bin      = specs.is_zero_bin
        self.is_signed        = specs.is_signed

        self.hist_type   = hist_type
        self.motion_type = motion_type
        self.data_type   = data_type

        self.data             = data
        self.num_samples      = len(self.data)

        # Find a set of bins that will cover the data
        self.compute_covering_bins()
        # Compute the histogram counts using those bins we just found
        # i.e. populate self.counts and self.pdf
        #      (pdf = the probability density value for each bin)
        self.compute_histogram_counts()

        self.compute_summary_statistics()

        # TODO: Not included yet ...
        # p_normal, only for n_valid_measurements >= 3
        # [~,cur_s.p_normal]  = seg_worm.fex.swtest(cur_h_e.mean, 0.05, 0);
        # q_normal - 
        # - @JimHokanson


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

   
    def compute_covering_bins(self):
        """
        Compute histogram bin boundaries that will be enough to cover 
        the given data
        
        Parameters
        ---------------------
        None, but we will use member variables:
        self.data: numpy array
            This is the data for which we must have enough bins to cover
        self.bin_width: float
            The width of the bins

        Returns
        ---------------------
        None
            However, two member variables are populated:
        (bin_boundaries, bin_midpoints): Two numpy arrays
            The bin_boundaries are the boundaries of the bins that will 
            accomodate the data given.
            All bins are right half-open except the last, which is closed.
            i.e. if the array edges = (a1, a2, ..., a(n+1) was returned, 
                 there are n bins and  
                 bin #1 = [a1, a2)
                 bin #2 = [a2, a3)
                 ...
                 bin #n = [an, an+1]
            Note that the "bin_midpoints" array that is returned is an array 
            of the midpoints of all the bins, strangely.  I'm not sure why 
            we'd need that.

        Notes
        ---------------------
        This version may have an extra bin than the previous version but
        this one is MUCH simpler and merging should be much simpler as edges
        should always align ...
        %   min -65.4
        %   max 20.01
        %   bin_width 1
        %   Old:
        %   boundaries -65.5 to 20.5
        %   New:
        %   boundaries -70 to 21

        Formerly: 
        function [bins,edges] = h__computeBinInfo(data,bin_width)

        """
        # Compute the data range
        min_data = min(self.data)
        max_data = max(self.data)
        
        # Let's "snap the bins to a grid" if you will, so that they will
        # line up when we try to merge multiple histograms later.
        # so if the bin_width = 2 and the min_data = 11, we will
        # start the first bin at 10, since that is a multiple of the 
        # bin width.
        min_boundary = np.floor(min_data/self.bin_width) * self.bin_width
        max_boundary = np.ceil(max_data/self.bin_width) * self.bin_width
        
        # If we have a singular value, then we will get a singular edge, 
        # which isn't good for binning. We always need to make sure that 
        # our data is bounded by a high and low end. Given how hist works 
        # (it is inclusive on the low end, when we only have one edge we 
        # add a second edge by increasing the high end, NOT by decreasing 
        # the low end.
        #
        # i.e. In Matlab you can't bound 3 by having edges at 2 & 3, the 
        #      edges would need to be at 3 & 4
        if min_boundary == max_boundary:
            max_boundary = min_boundary + self.bin_width
        
        num_bins = (max_boundary - min_boundary) / self.bin_width
        
        if num_bins > config.MAX_NUMBER_BINS:
            raise Exception("Given the specified resolution of " + \
                            str(self.bin_width) + ", the number of data " + \
                            "bins exceeds the maximum, which has been " + \
                            "set to MAX_NUMBER_BINS = " + \
                            str(config.MAX_NUMBER_BINS))
        
        self.bin_boundaries = np.arange(min_boundary, 
                                        max_boundary + self.bin_width, 
                                        step=self.bin_width)
        self.bin_midpoints  = self.bin_boundaries + self.bin_width/2

        # Because of the nature of floating point figures we can't guarantee
        # that these asserts work without the extra buffer of + self.bin_width
        # (though this bound could probably be greatly improved)
        assert(min_data >= self.bin_boundaries[0] - self.bin_width)
        assert(max_data <= self.bin_boundaries[-1] + self.bin_width)

        return None
    
    
    def compute_histogram_counts(self):
        """
        Compute the actual counts for the bins given the data
       
        """
        self.counts = np.histogram(self.data, bins = self.bin_boundaries)[0]

        if sum(self.counts) == 0:
            # Handle the divide-by-zero case
            self.pdf = None
        else:
            self.pdf = self.counts / sum(self.counts)

    
    def compute_summary_statistics(self):
        """
        Compute the mean and standard deviation on the data array
        Assign these to member data self.mean_per_video and 
        self.std_per_video, respectively.

        Returns 
        ------------------
        None
        
        """
        self.mean_per_video = np.mean(self.data)
        
        num_samples = len(self.data)
        if num_samples == 1:
            self.std_per_video = 0
        else:
            # We can optimize std dev computationsince we've already 
            # calculated the mean above
            self.std_per_video = np.sqrt \
                            (
                                (1/(num_samples-1)) * 
                                sum((self.data - self.mean_per_video)**2)
                            )


