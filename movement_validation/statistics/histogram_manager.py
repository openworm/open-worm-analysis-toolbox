# -*- coding: utf-8 -*-
"""


Formerly SegwormMatlabClasses / +seg_worm / +stats / @hist / manager.m

"""
import h5py
import numpy as np
import six # For compatibility between Python 2 and 3 in case we have to revert

from .histogram import Histogram
from . import specs
from .. import utils
from ..features.worm_features import WormFeatures


class HistogramManager(object):
    """
    Equivalent to seg_worm.stats.hist.manager class
    
    """
    def __init__(self, feature_path_or_object_list):
        """
        Parameters
        -------------------------
        feature_path_or_object_list: list of strings or feature objects
            Full paths to all feature files making up this histogram, or
            their in-memory object equivalents
        
        """
        #DEBUG: just for fun
        print("Number of feature files passed:", len(feature_path_or_object_list))
        
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

            # %TODO: Need to add on info to properties 
            # %worm_features.info -> obj.info

            hist_cell_array.append(self.init_histograms(worm_features))

        self.hists = HistogramManager.merge_histograms(hist_cell_array)


    def init_histograms(self, worm_features):
        """
        For a given set of worm features, prepare a 2D array of Histogram 
        instances, organized into three columns:
        - A column for the movement histograms
        - A column for the "simple" histograms
        - A column for the event histograms

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
        m_hists = self.h_computeMHists(worm_features, specs.MovementSpecs.getSpecs())
        
        # Simple histograms
        s_hists = self.h_computeSHists(worm_features, specs.SimpleSpecs.getSpecs())
        
        # Event histograms
        
        # :/ HACK: - @JimHokanson
        # Just get the size from the size of one of the pieces of data
        num_samples = len(worm_features.morphology.length)
        
        e_hists = self.h_computeEHists(worm_features, 
                                       specs.EventSpecs.getSpecs(), 
                                       num_samples)

        # Put all these histograms together into one matrix        
        return np.hstack((m_hists, s_hists, e_hists))


    ###########################################################################
    ## THREE FUNCTIONS TO CONVERT DATA TO HISTOGRAMS:
    ## h_computeMHists, h_computeSHists, and h_computeEHists
    ###########################################################################    
    
    def h_computeMHists(self, worm_features, specs):
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
        get 16 different possible combinations

        Parameters
        -------------------------
        worm_features : An h5py group instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology, 
            path, locomotion, in an h5py group.
        specs: A list of MovementSpecs instances

        Notes
        -------------------------
        Formerly m_hists = h_computeMHists(worm_features, specs)

        We could significantly reduce the amount of binning done in this
        function - @JimHokanson

        """
        pass
        """
        #---------------------------------------------------------
        motion_modes = worm_features.locomotion.motion_mode
        
        n_frames = len(motion_modes)
        
        indices_use_mask = {...
            true(1,n_frames) ...
            motion_modes == 1 ...
            motion_modes == 0 ...
            motion_modes == -1}
        
        #NOTE: motion types refers to the motion of the worm's midbody
        motion_types = {'all' 'forward'     'paused'    'backward'}
        data_types   = {'all' 'absolute'    'positive'  'negative'}
        #---------------------------------------------------------
        
        all_hist_objects = cell(1, config.MAX_NUM_HIST_OBJECTS)
        hist_count = 0
        
        n_specs = len(specs)
        
        for iSpec in range(n_specs):
            
            cur_specs = specs[iSpec]
            
            cur_data = cur_specs.getData(worm_features)
            
            good_data_mask = ~h__getFilterMask(cur_data)
            
            for iMotion in range(4):
                cur_motion_type = motion_types[iMotion]
                
                hist_count += 1
                temp_data  = cur_data[indices_use_mask{iMotion} & good_data_mask]
                
                all_obj = self.create_histogram(temp_data, cur_specs, 'motion', cur_motion_type, data_types[0])
                all_hist_objects[hist_count] = all_obj

                if cur_specs.is_signed:
                    
                    # TODO: This could be improved by merging results 
                    #       from positive and negative - @JimHokanson
                    all_hist_objects{hist_count+1} = create_histogram(abs(temp_data),cur_specs,'motion',cur_motion_type,data_types{2});
                    
                    
                    # NOTE: To get a speed up, we don't rely on 
                    # create_histogram.  Instead we take the 
                    # positive and negative aspects of the object 
                    # that included all data.
                    
                    # Positive object ----------------------------------------
                    pos_obj  = h__getObject(0,cur_specs,'motion',cur_motion_type,data_types{3});
                    
                    I_pos = find(all_obj.bins > 0 & all_obj.counts > 0,1);
                    
                    if ~isempty(I_pos):
                        pos_obj.bins      = all_obj.bins(I_pos:end)
                        pos_obj.counts    = all_obj.counts(I_pos:end)
                        pos_obj.n_samples = sum(pos_obj.counts)
                        
                        pos_obj.compute_statistics(pos_obj,temp_data(temp_data > 0))
                    
                    # Negative object ----------------------------------------
                    neg_obj  = self.h__getObject(0, cur_specs, 'motion', cur_motion_type, data_types[3])
                    
                    I_neg = find(all_obj.bins < 0 & all_obj.counts > 0, 
                                 1, 
                                 'last')
                    
                    if ~isempty(I_neg):
                        neg_obj.bins      = all_obj.bins(1:I_neg)
                        neg_obj.counts    = all_obj.counts(1:I_neg)
                        neg_obj.n_samples = sum(neg_obj.counts)
                        neg_obj.compute_statistics(temp_data(temp_data < 0))
                    
                    # Final assignments -------------------------------------
                    all_hist_objects[hist_count+2] = pos_obj
                    all_hist_objects[hist_count+3] = neg_obj
                    hist_count += 3
        
        return [all_hist_objects[:hist_count]]
        """

        

    def h_computeSHists(self, worm_features, specs):
        """
        Compute simple histograms
        
        Parameters
        ---------------------
        worm_features : An h5py group instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology, 
            path, locomotion, in an h5py group.
        specs: A list of SimpleSpecs instances
        
        """
        """ RESTORE THE FILTER ONCE THE DATA IS AVIALALBE:
        return [self.create_histogram(utils.filter_non_numeric(
                                          specs[iSpec].getData(worm_features)), 
                                      specs[iSpec], 
                                      'simple', 'all', 'all') 
                for iSpec in range(len(specs))]
        """
        return [self.create_histogram(specs[iSpec].getData(worm_features), 
                                      specs[iSpec], 
                                      'simple', 'all', 'all') 
        for iSpec in range(len(specs))]


    def h_computeEHists(self, worm_features, specs, num_samples):
        """
        Compute event histograms

        Parameters
        ---------------------
        worm_features : A WormFeatures instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology, 
            path, locomotion, in an h5py group.
        specs: a list of EventSpecs instances
        num_samples: int
            number of samples

        """
        temp_hists = []
        """ TODO: @MichaelCurrie, please finish this method!
        """
        for iSpec in range(len(specs)):
            cur_specs = specs[iSpec]
            
            cur_data = cur_specs.getData(worm_features, num_samples)
            
            # TODO: RESTORE THIS LINE ONCE DATA IS AVAILABLE:
            cur_data = utils.filter_non_numeric(cur_data)

            # Calculate the first histogram, on all the data.
            temp_hists.append(self.create_histogram(cur_data, cur_specs, 'event', 'all', 'all'))

            # If the data is signed, we calculate three more histograms:
            # - On an absolute version of the data, 
            # - On only the positive data, and 
            # - On only the negative data.
            if cur_specs.is_signed:
                temp_hists.append(self.create_histogram(abs(cur_data), cur_specs, 'event', 'all', 'absolute'))
                positive_mask = cur_data > 0
                negative_mask = cur_data < 0
                if len(positive_mask) != len(cur_data) or len(negative_mask) != len(cur_data):
                    print("uh oh")
                temp_hists.append(self.create_histogram(cur_data[positive_mask], cur_specs, 'event', 'all', 'positive'))
                temp_hists.append(self.create_histogram(cur_data[negative_mask], cur_specs, 'event', 'all', 'negative'))
        """
        """
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
        if data == None or not isinstance(data, np.ndarray) or data.size == 0:
            return None
        else:
            return Histogram(data, specs, hist_type, motion_type, data_type)


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
        print("In Histogram.merge_histograms... # of histograms to merge:", len(hist_cell_array))

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
            
            cur_bin_width = final_obj.bin_width
            new_bins       = min_bin:cur_bin_width:max_bin
            
            # Colon operator was giving warnings about non-integer indices :/
            # - @JimHokanson
            start_indices = round((start_bins - min_bin)./cur_bin_width + 1);
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
        
