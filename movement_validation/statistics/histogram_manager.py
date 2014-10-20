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
        
    def h_computeMHists(self, feature_obj, specs):
        pass

    def h_computeSHists(self, feature_obj, specs):
        pass

    def h_computeEHists(self, feature_obj, specs, num_samples):
        pass

        
        
def mergeObjects(hist_cell_array):
    """
    This is a placeholder.  It will eventually be turned into a static
    method for the class "hist".  See the mergeObjects method of
    +seg_worm/+stats/@hist in SegwormMatlabClasses
    https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/%40hist/hist.m
    

    
    """    
    
    # DEBUG: this is just a placeholder; instead of merging it just returns
    #        the first feature set
    return hist_cell_array[0]

        
