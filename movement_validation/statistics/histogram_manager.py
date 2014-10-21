# -*- coding: utf-8 -*-
"""


Formerly SegwormMatlabClasses / +seg_worm / +stats / @hist / manager.m

"""
import h5py
import numpy as np
import six # For compatibility between Python 2 and 3 in case we have to revert

from .histogram import Histogram

class HistogramManager:
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
        %
        %   hist_objs = seg_worm.stats.hist.manager.initObjects(feature_file_paths)
        %
        %   INPUTS
        %   =======================================
        %   feature_obj : (seg_worm.features or strut) This may truly be a feature
        %       object or the old structure. Both have the same format.
        %
        %   This is essentially the constructor code. I moved it in here to avoid
        %   the indenting.
        %
        %   Improvements
        %   -----------------------------------------------------------------------
        %   - We could optimize the histogram calculation data for motion data
        %   - for event data, we should remove partial events (events that start at
        %   the first frame or end at the last frame)
        """
        
        # Movement histograms - DONE
        # TODO: replace None with seg_worm.stats.movement_specs.getSpecs
        m_hists = self.h_computeMHists(feature_obj, None)
        
        # Simple histograms - DONE
        # TODO: replace None with seg_worm.stats.movement_specs.getSpecs
        s_hists = self.h_computeSHists(feature_obj, None)
        
        # Event histograms - DONE
        
        # :/ HACK  - @JimHokanson
        # TODO: replace with num_samples = len(feature_obj.morphology.length)
        num_samples = 40
        
        # TODO: replace None with seg_worm.stats.movement_specs.getSpecs
        e_hists = self.h_computeEHists(feature_obj, None, num_samples);
        
        hist_objs = np.hstack((m_hists, s_hists, e_hists))
    
        return hist_objs    

    ###########################################################################
    ## THREE FUNCTIONS TO CONVERT DATA TO HISTOGRAMS:
    ## h_computeMHists, h_computeSHists, and h_computeEHists
    ###########################################################################    
    
    def h_computeMHists(self, feature_obj, specs):
        pass

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
    
