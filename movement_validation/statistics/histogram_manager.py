# -*- coding: utf-8 -*-
"""
Equivalent to seg_worm.stats.hist.manager class

"""
import h5py
import six # For compatibility between Python 2 and 3 in case we have to revert

class HistogramManager:
    def __init__(self, feature_path_or_object_list):
        """
        Parameters
        -------------------------
        feature_path_or_object_list: list of strings or feature objects
            Full paths to all feature files making up this histogram, or
            their in-memory object equivalents
        
        """
        print("number of feature files passed:", len(feature_path_or_object_list))
        
        hist_cell_array = []        
        
        # Loop over all feature files and get histogram objects for each
    #        num_videos = len(feature_file_list)
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

            hist_cell_array.append(worm_features)            
        
        # TODO: 
        """
                %Merge the objects from each file
        %--------------------------------------------------------------------------
        obj.hists = seg_worm.stats.hist.mergeObjects(hist_cell_array); 
        """