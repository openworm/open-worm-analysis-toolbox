# -*- coding: utf-8 -*-
"""

The current processing approach is to take a set of features from an experiment
and to summarize each of these features as a binned data set (i.e. a histogram) 
where for each bin of a given width the # of values occupying each bin is
logged.

The histogram manager holds histograms for all computed features. Besides
helping to instantiate these histograms, it also holds any information
that is common to all of the histograms (e.g. experiment sources). By holding
the histograms it serves as a nice entry point for the histograms, rather than 
just having a list of histograms.

Formerly SegwormMatlabClasses/+seg_worm/+stats/@hist/manager.m

"""
import h5py
import numpy as np
import copy
import six # For compatibility between Python 2 and 3 in case we have to revert

from .histogram import Histogram
from . import specs
from .. import utils
from ..features.worm_features import WormFeatures


class HistogramManager(object):
    """
    
    Equivalent to seg_worm.stats.hist.manager class
    
    Attributes
    ----------    
    hists: list
    
    """
    def __init__(self, feature_path_or_object_list):
        """
        Parameters
        ----------
        feature_path_or_object_list: list of strings or feature objects
            Full paths to all feature files making up this histogram, or
            their in-memory object equivalents.
        
        """
        #DEBUG: just for fun
        print("Number of feature files passed into the hisotgram manager:", len(feature_path_or_object_list))
        
        #For each ...
        hist_cell_array = []        
        
        # Loop over all feature files and get histogram objects for each
        for feature_path_or_object in feature_path_or_object_list:
            worm_features = None
            
            if isinstance(feature_path_or_object, six.string_types):
                # If we have a string, it's a filepath to an HDF5 feature file
                feature_file = h5py.File(feature_path_or_object, 'r')
                #worm_features = feature_file["worm"]
                worm_features = WormFeatures.from_disk(feature_path_or_object)
                feature_file.close()
            else:
                # Otherwise the worm features have been passed directly
                # as an instance of WormFeatures (we hope)
                worm_features = feature_path_or_object

            #TODO: Need to add on info to properties 
            #worm_features.info -> obj.info

            hist_cell_array.append(self.init_histograms(worm_features))

        # At this point hist_cell_array is a list, with one element for 
        # each video.
        # Each element is a numpy array of 700+ Histogram instances.
        # Here we merge them and we assign to self.hists a numpy array
        # of 700+ Histogram instances, for the merged video
        self.hists = HistogramManager.merge_histograms(hist_cell_array)
        #self.hists = hist_cell_array[0]   # DEBUG: remove this and replace with above


    def init_histograms(self, worm_features):
        """
        For a given set of worm features, prepare a 1D array of Histogram 
        instances, consisting of, in order:
        - The movement histograms
        - The "simple" histograms
        - The event histograms

        Parameters
        ------------------
        worm_features : An h5py group instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology, 
            path, locomotion, in an h5py group.
            (seg_worm.features or strut) This may truly be a feature
            object or the old structure. Both have the same format.
            -@JimHokanson

        Notes
        ------------------
        Formerly:
        hist_objs = seg_worm.stats.hist.manager.initObjects(feature_file_paths)

        Potential Improvements
        ------------------
        - We could optimize the histogram calculation data for motion data
        - For event data, we should remove partial events (events that 
          start at the first frame or end at the last frame)
          
        """

        # Movement histograms
        m_hists = self.__movement_histograms(worm_features, 
                                             specs.MovementSpecs.getSpecs())
        
        # Simple histograms
        s_hists = self.__simple_histograms(worm_features, 
                                           specs.SimpleSpecs.getSpecs())
        
        # Event histograms
        
        # :/ HACK: - @JimHokanson
        # Just get the size from the size of one of the pieces of data
        num_samples = len(worm_features.morphology.length)
        
        e_hists = self.__event_histograms(worm_features, 
                                          specs.EventSpecs.getSpecs(), 
                                          num_samples)

        # Put all these histograms together into one single-dim numpy array.
        return np.hstack((m_hists, s_hists, e_hists))


    ###########################################################################
    ## THREE FUNCTIONS TO CONVERT DATA TO HISTOGRAMS:
    ## __simple_histograms, __movement_histograms, __event_histograms
    ###########################################################################    
       
    def __simple_histograms(self, worm_features, specs):
        """
        Compute simple histograms
        
        Parameters
        ---------------------
        worm_features : An h5py group instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology, 
            path, locomotion, in an h5py group.
        specs: A list of SimpleSpecs instances
        
        Returns
        --------------------        
        A list of Histogram instances, one for each of the simple features
        
        """
        return [self.create_histogram(utils.filter_non_numeric(
                                          specs[iSpec].getData(worm_features)), 
                                      specs[iSpec], 
                                      'simple', 'all', 'all') 
                for iSpec in range(len(specs))]

    
    def __movement_histograms(self, worm_features, specs):
        """
        For movement features, we compute either 4 or 16 histogram objects,
        depending on whether or not the feature can be signed. If the data is
        not signed, then we compute 4. If it is, we compute 16. We compute
        histograms for when the motion of the midbody is:
            - going forward
            - going backward
            - paused
            - all 3 of the above combined

        If signed, this also gets computed on feature data that is:
            - positive
            - negative
            - absolute value of data
            - both positive and negative

        By combining the motion of the midbody with the sign of the data, we
        get 16 different possible combinations for each feature specification.

        Parameters
        -------------------------
        worm_features : An h5py group instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology, 
            path, locomotion, in an h5py group.
        specs: A list of MovementSpecs instances

        Returns
        --------------------        
        A list of Histogram instances, one for each of the movement features

        Notes
        -------------------------
        Formerly m_hists = h_computeMHists(worm_features, specs)

        We could significantly reduce the amount of binning done in this
        function - @JimHokanson

        """
        motion_modes = worm_features.locomotion.motion_mode
        
        num_frames = len(motion_modes)
        
        indices_use_mask = {}
        indices_use_mask["all"]      = np.ones(num_frames, dtype=bool)
        indices_use_mask["forward"]  = motion_modes == 1
        indices_use_mask["backward"] = motion_modes == -1
        indices_use_mask["paused"]   = motion_modes == 0

        # NOTE: motion types refers to the motion of the worm's midbody
        motion_types = ['all', 'forward', 'paused', 'backward']
        data_types = ['all', 'absolute', 'positive', 'negative']

        movement_histograms = []

        for cur_spec in specs:

            cur_data = cur_spec.getData(worm_features)

            good_data_mask = ~utils.get_non_numeric_mask(cur_data).flatten()
            
            # Now let's create 16 histograms, for each element of
            # (motion_types x data_types)
            
            for cur_motion_type in motion_types:
                if (good_data_mask.size != indices_use_mask[cur_motion_type].size):
                    import pdb
                    pdb.set_trace()
                
                #assert(good_data_mask.size == \
                 #      indices_use_mask[cur_motion_type].size)

                cur_mask = indices_use_mask[cur_motion_type] & good_data_mask
                assert(isinstance(cur_data, np.ndarray))
                assert(isinstance(cur_mask, np.ndarray))
                assert(cur_data.size == cur_mask.size)

                temp_data = cur_data[cur_mask]

                # Create the histogram for the case where we consider all
                # numeric data                
                all_hist = self.create_histogram(temp_data,
                                                 cur_spec, 
                                                 'motion',
                                                 cur_motion_type, 
                                                 data_types[0])

                movement_histograms.append(all_hist)

                if cur_spec.is_signed:
                    
                    # Histogram for the data made absolute
                    # TODO: This could be improved by merging results 
                    #       from positive and negative - @JimHokanson
                    abs_hist = self.create_histogram(abs(temp_data),
                                                     cur_spec,
                                                     'motion',
                                                     cur_motion_type,
                                                     data_types[1])
                                        
                    # TODO: To get a speed-up, we could avoid reliance on 
                    # create_histogram.  Instead, we could take the 
                    # positive and negative aspects of the object 
                    # that included all data. - @JimHokanson
                    # (see the SegWormMatlabClasses version of this to see
                    #  how this could be done)
                    
                    # Histogram for just the positive data
                    pos_hist  = self.create_histogram(
                                    temp_data[temp_data >= 0],
                                    cur_spec,
                                    'motion',
                                    cur_motion_type,
                                    data_types[2])
                    
                    # Histogram for just the negative data
                    neg_hist  = self.create_histogram(
                                    temp_data[temp_data <= 0], 
                                    cur_spec, 
                                    'motion', 
                                    cur_motion_type, 
                                    data_types[3])
                    
                    # Append our list with these histograms
                    movement_histograms.append(abs_hist)
                    movement_histograms.append(pos_hist)
                    movement_histograms.append(neg_hist)
        
        return movement_histograms


    def __event_histograms(self, worm_features, specs, num_samples):
        """
        Compute event histograms.  We produce four histograms for each
        specification in specs, for:
        - all data
        - absolute data
        - positive data
        - negative data

        Parameters
        ---------------------
        worm_features : A WormFeatures instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology, 
            path, locomotion, in an h5py group.
        specs: a list of EventSpecs instances
        num_samples: int
            number of samples

        Returns
        --------------------        
        A list of Histogram instances, one for each of the event features

        """
        temp_hists = []

        for cur_specs in specs:
            
            cur_data = cur_specs.getData(worm_features, num_samples)
            
            # Remove the NaN and Inf entries
            cur_data = utils.filter_non_numeric(cur_data)

            # Calculate the first histogram, on all the data.
            temp_hists.append(self.create_histogram(cur_data, cur_specs, 'event', 'all', 'all'))

            # If the data is signed, we calculate three more histograms:
            # - On an absolute version of the data, 
            # - On only the positive data, and 
            # - On only the negative data.
            if cur_specs.is_signed:
                if cur_data is None:
                    #TODO: This is a bit opaque and should be clarified
                    #The call to create_histograms() just returns None, so we put together a bunch of None's
                    #in a list and append, rather than calling the function 3x
                    temp_hists = temp_hists + [None, None, None]
                else:
                    temp_hists.append(self.create_histogram(abs(cur_data), cur_specs, 'event', 'all', 'absolute'))
                    positive_mask = cur_data > 0
                    negative_mask = cur_data < 0
                    temp_hists.append(self.create_histogram(cur_data[positive_mask], cur_specs, 'event', 'all', 'positive'))
                    temp_hists.append(self.create_histogram(cur_data[negative_mask], cur_specs, 'event', 'all', 'negative'))

        return temp_hists


    def create_histogram(self, data, specs, hist_type, motion_type, data_type):
        """
        Create individual Histogram instance.
        The only thing this does beyond the Histogram constructor is
        to check if the data is empty.
        
        Parameters
        ------------------
        data: numpy array
        specs: instance of Specs class
        hist_type:
        motion_type:
        data_type:

        Returns
        ------------------
        An instance of the Histogram class, prepared with
        the data provided.

        Notes
        ------------------
        Formerly:
        function obj = h__createIndividualObject(self, data, specs, 
                                                 hist_type, motion_type, 
                                                 data_type)
        
        """
        if data is None or not isinstance(data, np.ndarray) or data.size == 0:
            return None
        else:
            return Histogram(data, specs, hist_type, motion_type, data_type)


    def merge_histograms_michael(hist_cell_array):
        """
        I don't understand Jim's merge_histograms method.  Here's my version.
        """
        pass
        #hist_cell_array = np.arra


    @staticmethod
    def merge_histograms(hist_cell_array):
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
        always setting bin edges at multiples of the bin_width. This was
        not done in the original Schafer Lab code.

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
        print("In Histogram.merge_histograms... # of histograms to merge:", len(hist_cell_array))

        hist_cell_array = np.array(hist_cell_array)

        num_videos_per_object = [obj.num_videos for obj 
                                 in hist_cell_array.flatten() 
                                 if obj is not None]

        if any(num_videos_per_object != \
               np.ones(len(num_videos_per_object))):
            raise Exception("Multiple videos per object not yet implemented")

        num_videos   = np.shape(hist_cell_array)[0]
        num_features = np.shape(hist_cell_array)[1]
        
        # temp_results will contain 700+ Histogram instances.
        temp_results = []

        # Go through each feature and create a merged histogram across
        # all the videos.
        for iFeature in range(num_features):
            
            video_array = hist_cell_array[:,iFeature]

            # This is @MichaelCurrie's kludge to step over features that
            # for some reason didn't get their histograms populated on all
            # videos.
            none_video_found = False
            for video in video_array:
                if video is None:
                    print("found a None video in the feature list")
                    none_video_found = True
            if none_video_found:
                continue
            
            # Create an output object with same meta properties
            final_obj   =  copy.copy(video_array[0])
            # This underlying data, which was just for the FIRST video, 
            # will not be correct after the object is made to contain 
            # the merged histogram information:
            final_obj.data = None   

            # Align all bins
            # ---------------------------------------------------------------
            num_bins = [x.num_bins for x in video_array]
            first_bin_midpoints = [x.first_bin_midpoint for x in video_array]
            min_bin_midpoint = min(first_bin_midpoints)
            max_bin_midpoint = max([x.last_bin_midpoint for x in video_array])
            
            cur_bin_width = final_obj.bin_width
            new_bin_midpoints = np.arange(min_bin_midpoint, 
                                          max_bin_midpoint+cur_bin_width, 
                                          step=cur_bin_width)
            
            # Colon operator was giving warnings about non-integer indices :/
            # - @JimHokanson
            start_indices = (first_bin_midpoints - min_bin_midpoint) / \
                             cur_bin_width
            start_indices = start_indices.round()
            end_indices   = start_indices + num_bins
            
            new_counts = np.zeros((num_videos, len(new_bin_midpoints)))
            
            for iVideo in range(num_videos):
                cur_start = start_indices[iVideo]
                if not np.isnan(cur_start):
                    cur_end   = end_indices[iVideo]
                    new_counts[iVideo, cur_start:cur_end] = \
                                                video_array[iVideo].counts
            
            # Update final properties
            # Note that each of these is now no longer a scalar as in the
            # single-video case; it is now a numpy array
            # ---------------------------------------------------------------
            final_obj.bin_midpoints  = new_bin_midpoints
            final_obj.counts         = new_counts
            final_obj.num_samples    = [x.num_samples for x in video_array]
            final_obj.mean_per_video = [x.mean_per_video for x in video_array]
            final_obj.std_per_video  = [x.std_per_video for x in video_array]
            final_obj.pdf            = sum(final_obj.counts, 0) / \
                                       sum(final_obj.num_samples)
            
            # Hold onto final object for output
            temp_results.append(final_obj)
        
        return temp_results
