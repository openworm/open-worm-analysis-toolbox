# -*- coding: utf-8 -*-
"""


Formerly SegwormMatlabClasses / +seg_worm / +stats / @hist / manager.m

"""
import h5py
import numpy as np
import six # For compatibility between Python 2 and 3 in case we have to revert

from .histogram import Histogram
from . import specs
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
        print("number of feature files passed:", len(feature_path_or_object_list))
        
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

            hist_cell_array.append(self.init_objects(worm_features))

        self.hists = Histogram.merge_objects(hist_cell_array)


    def init_objects(self, worm_features):
        """
        This is essentially the constructor code.  Originally, @JimHokanson
        moved it here to "avoid the indenting".

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
        
        # :/ HACK  - @JimHokanson
        # TODO: replace with:  (once you fix the fact the worm_features is currently a <Closed HDF5 group>)
        # num_samples = len(worm_features["morphology"]["length"].value)
        num_samples = 26994
        
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
        motion_modes = worm_features.locomotion.motion.mode
        
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
                
                all_obj = self.h__createIndividualObject(temp_data, cur_specs, 'motion', cur_motion_type, data_types[0])
                all_hist_objects[hist_count] = all_obj

                if cur_specs.is_signed:
                    
                    # TODO: This could be improved by merging results 
                    #       from positive and negative - @JimHokanson
                    all_hist_objects{hist_count+1} = h__createIndividualObject(abs(temp_data),cur_specs,'motion',cur_motion_type,data_types{2});
                    
                    
                    # NOTE: To get a speed up, we don't rely on 
                    # h__createIndividualObject.  Instead we take the 
                    # positive and negative aspects of the object 
                    # that included all data.
                    
                    # Positive object ----------------------------------------
                    pos_obj  = h__getObject(0,cur_specs,'motion',cur_motion_type,data_types{3});
                    
                    I_pos = find(all_obj.bins > 0 & all_obj.counts > 0,1);
                    
                    if ~isempty(I_pos):
                        pos_obj.bins      = all_obj.bins(I_pos:end)
                        pos_obj.counts    = all_obj.counts(I_pos:end)
                        pos_obj.n_samples = sum(pos_obj.counts)
                        
                        h__computeStats(pos_obj,temp_data(temp_data > 0))
                    
                    # Negative object ----------------------------------------
                    neg_obj  = self.h__getObject(0, cur_specs, 'motion', cur_motion_type, data_types[3])
                    
                    I_neg = find(all_obj.bins < 0 & all_obj.counts > 0, 
                                 1, 
                                 'last')
                    
                    if ~isempty(I_neg):
                        neg_obj.bins      = all_obj.bins(1:I_neg)
                        neg_obj.counts    = all_obj.counts(1:I_neg)
                        neg_obj.n_samples = sum(neg_obj.counts)
                        self.h__computeStats(neg_obj, 
                                             temp_data(temp_data < 0))
                    
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
        pass
        """ TODO
        return [self.h__createIndividualObject(self.h__filterData(specs[iSpec].getData(worm_features)), 
                                               specs[iSpec], 
                                               'simple', 'all', 'all') 
                for iSpec in range(len(specs))]
        """

    def h_computeEHists(self, worm_features, specs, num_samples):
        """
        Compute event histograms

        Parameters
        ---------------------
        worm_features : An h5py group instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology, 
            path, locomotion, in an h5py group.
        specs: a list of EventSpecs instances
        num_samples: int
            number of samples

        """
        pass
        """ TODO
        num_specs = len(specs)
        temp_hists = []
        
        for iSpec in range(num_specs):
            cur_specs = specs[iSpec]
            
            cur_data = cur_specs.getData(worm_features, num_samples)
            
            cur_data = self.h__filterData(cur_data)

            # Calculate the first histogram, on all the data.
            temp_hists.append(self.h__createIndividualObject(cur_data,cur_specs,'event','all','all'))

            # If the data is signed, we calculate three more histograms:
            # - On an absolute version of the data, 
            # - On only the positive data, and 
            # - On only the negative data.
            if cur_specs.is_signed:
                temp_hists.append(self.h__createIndividualObject(abs(cur_data),cur_specs,'event','all','absolute'))
                positive_mask = cur_data > 0
                negative_mask = cur_data < 0
                temp_hists.append(self.h__createIndividualObject(cur_data(positive_mask),cur_specs,'event','all','positive'))
                temp_hists.append(self.h__createIndividualObject(cur_data(negative_mask),cur_specs,'event','all','negative'))

        return temp_hists
        """


    ###########################################################################
    ## SIX "PRIVATE" METHODS   (as private as Python gets, anyway)
    ## h__getObject
    ## h__filterData
    ## h__getFilterMask
    ## h__computeStats
    ## h__initMeta
    ## h__computeBinInfo
    ###########################################################################
    def h__createIndividualObject(self, data, specs, hist_type, motion_type, data_type):
        pass
        """ TODO: translate
        o = self.h__getObject(len(data), specs, hist_type, motion_type, data_type)
        
        if o.num_samples == 0:
            return None
        
        # Compute the histogram counts for all the data
        [o.bins, edges] = self.h__computeBinInfo(data, o.resolution)
        
        # Remove the extra bin at the end (for overflow)
        o.counts = np.histogram(data, edges)[:-1]

        o.pdf = o.counts / sum(o.counts)

        # Compute stats
        self.h__computeStats(data)
        """        
    
    def h__getObject(self, n_samples, specs, hist_type, motion_type, data_type):
        pass

    
    def h__filterData(self, data):
        """
        Filter the data
        
        Parameters
        ------------------
        data: 
        
        """
        # TODO: implement this function, currently the data doesn't get filtered
        return data  

        
    def h__getFilterMask(self, data):
        """
        Get the filter mask
        
        Parameters
        ------------------
        data: 
        
        """
        pass


    def h__computeStats(self, data):
        """
        Compute the stats
        
        Parameters
        ------------------
        data: 
        
        """
        pass


    def h__initMeta(self, specs, hist_type, motion_type, data_type):
        self.field            = specs.getLongField()
        self.feature_category = specs.feature_category
        self.resolution       = specs.resolution
        self.is_zero_bin      = specs.is_zero_bin
        self.is_signed        = specs.is_signed
        self.name             = specs.name
        self.short_name       = specs.short_name
        self.units            = specs.units
        
        self.hist_type   = hist_type
        self.motion_type = motion_type
        self.data_type   = data_type

    
    def h__computeBinInfo(self, data, resolution):
        """
        Compute histogram bin information
        
        Parameters
        ---------------------
        data:
        resolution:

        
        Formerly: 
        function [bins,edges] = h__computeBinInfo(data,resolution)
        """
        pass
    
