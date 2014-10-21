# -*- coding: utf-8 -*-
"""
This is the Python port of 
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/specs.m
and its subclass,
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/movement_specs.m
"""
import os
import csv


class Specs(object):
    """
    
    Notes
    ------------------
    Formerly seg_worm.stats.specs
    
    """
    def __init__(self):
       feature_field = None
       feature_category = None
       resolution = None
       is_zero_bin = None
       is_signed = None
       name = None
       short_name = None
       units = None


    @property
    def long_field(self):
        """
        Formerly getLongField

        """
        return self.feature_field

    
    @staticmethod
    def getObjectsHelper(csv_path, class_function_handle):
        """


        Parameters
        ----------------------
        csv_path:
        class_function_handle:


        Notes
        ---------------------
        Formerly function objs = seg_worm.stats.specs.getObjectsHelper(csv_path,class_function_handle,prop_names,prop_types)
        
        The inherited objects can give relatively simple
        instructions on how their properties should be interpreted
        from their CSV specification file.

        TODO: 
        It would be nice to do the reading and object construction in 
        here but Matlab is awkward for dynamic object creation 
        - @JimHokanson
        """
        stats_instances = []    

        # See below comment above prop_types
        data_types = {1: str, 2: float, 3: int, 4: bool}

        with open(csv_path) as feature_metadata_file:
            feature_metadata = csv.DictReader(feature_metadata_file)
            # The first row of the CSV file contains the field names.
            
            # The second row of the CSV file contains information about 
            # what kind of data is held in each column:
            #    1 = str
            #    2 = float
            #    3 = int
            #    4 = bool
            #   (this mapping was recorded above in data_types)
            field_data_types = feature_metadata.__next__()
    
            # The third to last rows of the CSV file contain the feature
            # metadata.  Let's now create a stats_instance for each
            # of these rows, initializing them with the row's metadata.
            for row in feature_metadata:
                stats_instance = class_function_handle()
                
                for field in row:
                    # Blank values are given the value None
                    value = None
                    if(row[field] != ''):
                        # Here we are dynamically casting the element 
                        # to the correct data type of the field,
                        # which was recorded in the prop_types dictionary.
                        data_type = data_types[int(field_data_types[field])]
                        value = data_type(row[field])
                    # Dynamically assign the field's value to the 
                    # member data element of the same name in the object
                    setattr(stats_instance, field, value)
                
                stats_instances.append(stats_instance)
            
        return stats_instances    
    


    
class MovementSpecs(Specs):
    """
    %
    %   Class:
    %   seg_worm.stats.movement_specs
    %
    %   This class specifies how to treat each movement related feature for
    %   histogram processing.
    %
    %
    %   Access via static method:
    %   seg_worm.stats.movement_specs.getSpecs()
    %
    %   See Also:
    %   seg_worm.stats.hist.createHistograms
    %
    %   TODO:
    %   - might need to incorporate seg_worm.w.stats.wormStatsInfo
    %   - remove is_time_series entry ...
    """

    def __init__(self):
        self.index = None
        self.is_time_series = None# TODO: This can be removed
        #%feature_category
        #%resolution
        #%is_zero_bin %This might not be important
        #%is_signed   %I think this dictates having 4 or 16 events ...
        #%        name
        #%        short_name
        #%        units

    @staticmethod
    def getSpecs():
        """
        Formerly objs = getSpecs()
        %seg_worm.stats.movement_specs.getSpecs();

        """
        csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'feature_metadata',
                                'movement_features.csv')
        
        # Return a list of MovementSpecs instances, one instance for each
        # row in the csv_path CSV file.  Each row represents a feature. 
        return Specs.getObjectsHelper(csv_path, MovementSpecs)
    
