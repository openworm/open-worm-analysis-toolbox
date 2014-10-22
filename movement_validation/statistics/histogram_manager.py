# -*- coding: utf-8 -*-
"""


Formerly SegwormMatlabClasses / +seg_worm / +stats / @hist / manager.m

"""
import h5py
import numpy as np
import six # For compatibility between Python 2 and 3 in case we have to revert

from .histogram import Histogram
from . import specs
from .. import config

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
                worm_features = feature_file["worm"]
                feature_file.close()
            else:
                # Otherwise the worm features have been passed directly
                # as an in-memory HDF5 object
                worm_features = feature_path_or_object

            # %TODO: Need to add on info to properties 
            # %feature_obj.info -> obj.info

            hist_cell_array.append(self.init_objects(worm_features))

        self.hists = Histogram.merge_objects(hist_cell_array)


    def init_objects(self, feature_obj):
        """
        This is essentially the constructor code.  Originally, @JimHokanson
        moved it here to "avoid the indenting".

        Parameters
        ------------------
        feature_obj : (seg_worm.features or strut) This may truly be a feature
            object or the old structure. Both have the same format.

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
        m_hists = self.h_computeMHists(feature_obj, specs.MovementSpecs.getSpecs())
        
        # Simple histograms
        s_hists = self.h_computeSHists(feature_obj, specs.SimpleSpecs.getSpecs())
        
        # Event histograms
        
        # :/ HACK  - @JimHokanson
        # TODO: replace with num_samples = len(feature_obj.morphology.length)
        num_samples = 40
        
        e_hists = self.h_computeEHists(feature_obj, 
                                       specs.EventSpecs.getSpecs(), 
                                       num_samples)

        # Put all these histograms together into one matrix        
        return np.hstack((m_hists, s_hists, e_hists))

    ###########################################################################
    ## THREE FUNCTIONS TO CONVERT DATA TO HISTOGRAMS:
    ## h_computeMHists, h_computeSHists, and h_computeEHists
    ###########################################################################    
    
    def h_computeMHists(self, feature_obj, specs):
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
        feature_obj:
        specs:

        Notes
        -------------------------
        Formerly m_hists = h_computeMHists(feature_obj, specs)

        We could significantly reduce the amount of binning done in this
        function - @JimHokanson

        """
        pass
        """
        #---------------------------------------------------------
        motion_modes = feature_obj.locomotion.motion.mode
        
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
            
            cur_data = cur_specs.getData(feature_obj)
            
            good_data_mask = ~h__getFilterMask(cur_data)
            
            for iMotion in range(4):
                cur_motion_type = motion_types[iMotion]
                
                hist_count += 1
                temp_data  = cur_data[indices_use_mask{iMotion} & good_data_mask]
                
                all_obj = self.h__createIndividualObject(temp_data, cur_specs, 'motion', cur_motion_type, data_types[0])
                all_hist_objects[hist_count] = all_obj

                if cur_specs.is_signed:
                    
                    #TODO: This could be improved by merging results 
                    # from positive and negative - @JimHokanson
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

        

    def h_computeSHists(self, feature_obj, specs):
        pass

    def h_computeEHists(self, feature_obj, specs, num_samples):
        pass

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
        
    
    def h__getObject(self, n_samples, specs, hist_type, motion_type, data_type):
        pass
    
    def h__filterData(self, data):
        pass

    def h__getFilterMask(self, data):
        pass

    def h__computeStats(self, data):
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
        Formerly: 
        function [bins,edges] = h__computeBinInfo(data,resolution)
        """
        pass
    
