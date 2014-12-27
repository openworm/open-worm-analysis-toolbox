# -*- coding: utf-8 -*-
"""

Instances of these classes define how a feature should be quantized into a
histogram as well as some additional informaton (see csv files and class definitions)

The raw information is actually located in csv files in:

    movement_validation/statistics/feature_metadata

These classes instantiate each row of these files as instances.

This is the Python port of:
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/specs.m
and its subclasses:
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/movement_specs.m
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/simple_specs.m
https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/event_specs.m

This module defines the following classes:
Specs
MovementSpecs(Specs)
SimpleSpecs(Specs)
EventSpecs(Specs)

"""
import os
import csv
import numpy as np

from .. import utils

class Specs(object):
    """
    
    Attributes
    ----------
    long_field : string
        
    
    
    Notes
    -----
    Formerly seg_worm.stats.specs in SegwormMatlabClasses
    
    """
    def __init__(self):
        """
        This initialization method does nothing.  To instantiate, you need
        to call the static factory method Specs.specs_factory
        
        """
        pass

    def __repr__(self):
        return utils.print_object(self)

    @property
    def long_field(self):
        """
        Give the "long" version of the instance's name.

        Returns
        -------
        string
        A '.' delimited concatenation of feature_field and sub_field.

        """
        value = self.feature_field

        if hasattr(self, 'sub_field') and \
           self.sub_field != None and self.sub_field != '':
            value += '.' + self.sub_field

        return value
    
    
    def getData(self, worm_features):
        """
        Drill down into the nested data structure of worm_features to obtain
        the numpy array with the data specific to this specification.

        Parameters
        ----------
        worm_features: A WormFeatures instance
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology, 
            path, locomotion, in an h5py group.

        Returns
        -------
        A numpy array

        """
        data = worm_features
        # Call getattr as many times as is necessary, to dynamically 
        # access a potentially nested field.
        # e.g. if self.feature_field = 'posture.coils', we'll need to call
        #      getattr twice, first on 'posture', and second on 'coils'.
        for cur_feature_field in self.feature_field.split('.'):
            if not hasattr(data, cur_feature_field):
                import pdb
                pdb.set_trace()
                raise Exception("The WormFeatures instance passed does " + 
                                "not have the feature: " + cur_feature_field + 
                                ". Its full name is " + self.long_field)
            data = getattr(data, cur_feature_field)

        return data

    
    @staticmethod
    def specs_factory(csv_path, class_function_handle):
        """
        Factory for creating Specs subclasses for every extended feature
        in a CSV file

        Parameters
        ----------
        csv_path: string
            The path to a CSV file that has a list of extended features
        class_function_handle: A class inheriting from Stats

        Returns
        -------
        list
            A list of instances of the Stats subclass provided by 
            class_function_handle, with each item in the list corresponding 
            to a row in the CSV file at the provided csv_path.

        Notes
        -----
        Formerly function objs = seg_worm.stats.specs.getObjectsHelper( ...
            csv_path,class_function_handle,prop_names,prop_types)
        
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
            field_data_types = next(feature_metadata)
    
            # The third to last rows of the CSV file contain the feature
            # metadata.  Let's now create a stats_instance for each
            # of these rows, initializing them with the row's metadata.
            for row in feature_metadata:
                # Dynamically create an instance of the right kind 
                # of class
                stats_instance = class_function_handle()
                
                for field in row:
                    # Blank values are given the value None
                    value = None
                    if(row[field] != ''):
                        # Here we are dynamically casting the element 
                        # to the correct data type of the field,
                        # which was recorded in the prop_types dictionary.
                        data_type = data_types[int(field_data_types[field])]
                        if data_type == bool:
                            # We must handle bool as a separate case because
                            # bool('0') = True.  To correct this, we must 
                            # first cast to int: e.g. bool(int('0')) = False
                            value = bool(int(row[field]))
                        else:
                            value = data_type(row[field])
                    # Dynamically assign the field's value to the 
                    # member data element of the same name in the object
                    setattr(stats_instance, field, value)

                # Only append this row to our list if there is 
                # actually a name.  If not it's likely just a blank row.
                if stats_instance.feature_field:
                    stats_instances.append(stats_instance)
            
        return stats_instances    


class SimpleSpecs(Specs):
    """
    %
    %   Class:
    %   seg_worm.stats.simple_specs
    %
    """
    def __init__(self):
        pass
    
    
    @staticmethod
    def getSpecs():
        """    
        Formerly function objs = getSpecs()
            %
            %
            %   s_specs = seg_worm.stats.simple_specs.getSpecs();
            %
            %
        """
        csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'feature_metadata',
                                'simple_features.csv')
        
        # Return a list of SimpleSpecs instances, one instance for each
        # row in the csv_path CSV file.  Each row represents a feature. 
        return Specs.specs_factory(csv_path, SimpleSpecs)


class MovementSpecs(Specs):
    """

    This class specifies how to treat each movement related feature when doing
    histogram processing.

    Attributes
    ----------
    feature_field :
    old_feature_field :
    index : 
    feature_category :
    is_time_series :
    bin_width :
    is_zero_bin :
    is_signed :
    name :
    short_name :
    units : 

    Created via static method, getSpecs()

    %From Matlab comments:
    %
    %   TODO:
    %   - might need to incorporate seg_worm.w.stats.wormStatsInfo
    %   - remove is_time_series entry ...
    """

    def __init__(self):
        pass


    def getData(self, worm_features):
        """
        Parameters
        ----------
        worm_features : movement_validation.features.WormFeatures
            All the feature data calculated for a single worm video.
            Arranged heirarchically into categories:, posture, morphology, 
            path, locomotion.        
            
        Notes
        -----------------------
        Formerly data = getData(obj,feature_obj)
        
        """
        data = super(MovementSpecs,self).getData(worm_features)
        
        # NOTE: We can't filter data here because the data is 
        #       filtered according to the value of the data, not 
        #       according to the velocity of the midbody
        
        if self.index != None and data != None:
            # This is for eigenprojections, i.e. for instances when 
            # self.feature_field = 'posture.eigen_projection'
            # In these cases the data is stored as a num_frames x 6 numpy 
            # array.  We use self.index to identify which of the 6 eigenworms
            # we are looking for projections of, for this particular feature.
            data = data[self.index,:]
            # So now our data has shape num_frames instead of [6, num_frames]

        return data


    @staticmethod
    def getSpecs():
        """
        Formerly objs = getSpecs()
        %seg_worm.stats.movement_specs.getSpecs();
        
        Returns
        ---------------------

        """
        csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'feature_metadata',
                                'movement_features.csv')
        
        # Return a list of MovementSpecs instances, one instance for each
        # row in the csv_path CSV file.  Each row represents a feature. 
        return Specs.specs_factory(csv_path, MovementSpecs)
    

class EventSpecs(Specs):
    """

    Notes
    --------------------------
    Formerly seg_worm.stats.event_specs

    """
    def __init__(self):
        pass

   
    @staticmethod
    def getSpecs():
        """    
        Formerly function objs = getSpecs()
            %
            %
            %   s_specs = seg_worm.stats.event_specs.getSpecs();
            %
            %
        """
        csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'feature_metadata',
                                'event_features.csv')
      
        # Return a list of MovementSpecs instances, one instance for each
        # row in the csv_path CSV file.  Each row represents a feature. 
        return Specs.specs_factory(csv_path, EventSpecs)

    
    def getData(self, worm_features, num_samples):
        """

        Parameters
        ---------------------
        worm_features: A WormFeatures instance
            All the feature data calculated for a single worm video.
            Arranged hierarchically into categories:
               - posture
               - morphology,
               - path
               - locomotion, in an h5py group.
        num_samples: int
            Number of samples (i.e. number of frames in the video)

        Returns
        ---------------------
        
        #https://github.com/JimHokanson/SegwormMatlabClasses/blob/master/%2Bseg_worm/%2Bstats/event_specs.m#L55

        Notes
        ---------------------
        Formerly  SegwormMatlabClasses / +seg_worm / +stats / event_specs.m
                  function data = getData(obj,feature_obj,n_samples)

        """

        # For example, self.feature_field might be
        #   locomotion.motion_events.forward,
            # and self.sub_field might be
            #   time_between_events or distance_between_events, etc.
            # Basically we want to drill to the bottom of the nested
            # heirarchy of data in worm_features.


        #JAH: This will fail in Python 2.7
        #???? super(Specs).getData(worm_features)

        parent_data = super(EventSpecs,self).getData(worm_features)

        #JAH: The Matlab code would use an empty structure.
        #Rather than just having an empty class, all events have a property 'is_null' which
        #indicates if the event class is fully populated or if there are no events for the video.

        if parent_data is None or parent_data.is_null:
            return None

        if self.sub_field is not None:

            data = getattr(parent_data, self.sub_field)

            if self.is_signed:
                negate_mask = getattr(parent_data, self.signed_field)
                if len(negate_mask) == 1 and negate_mask == True:
                    # Handle the case where data is just one value,
                    # a scalar, rather than a numpy array
                    data *= -1
                elif len(negate_mask) == len(data):
                    # Our mask size perfectly matches the data size
                    # e.g. for event_durations
                    data[negate_mask] *= -1
                elif len(negate_mask) == len(data) + 1:
                    # Our mask is one larger than the data size
                    # e.g. for time_between_events
                    # DEBUG: Are we masking the right entry here?
                    #        should we perhaps be using
                    #        negate_mask[:-1] instead?
                    data[negate_mask[1:]] *= -1
                else:
                    raise Exception("For the signed_field " +
                                    self.signed_field + " and the data " +
                                    self.long_field + ", " +
                                    "len(negate_mask) is not the same " +
                                    "size or one smaller as len(data), " +
                                    "as required.")

            if self.remove_partial_events:
                # Remove the starting and ending event if it's right
                # up against the edge of the data, since we can't be
                # sure that the video captured the full extent of the
                # event
                start_frames = parent_data.start_frames
                end_frames   = parent_data.end_frames

                remove_mask = np.empty(len(data), dtype=bool)*False

                if start_frames[0] == 0:
                    remove_mask[:end_frames[0]+1] = True

                if end_frames[-1] == num_samples:
                    remove_mask[start_frames[-1]:] = True

                # Remove all entries corresponding to True
                # in the remove_mask
                try:
                    data = data[~remove_mask]
                except:
                    import pdb
                    pdb.set_trace()
                
        else:
            import pdb
            pdb.set_trace()
            raise Exception("The WormFeature contains no data for " + self.long_field)
        
        if data.size == 0 and self.make_zero_if_empty:
            data = 0
        
        return data
