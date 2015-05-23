# -*- coding: utf-8 -*-
"""
BasicWorm, WormPartition, JSON_Serializer

Credit to Christopher R. Wagner at 
http://robotfantastic.org/serializing-python-data-to-json-some-edge-cases.html    
for the following six functions:
isnamedtuple
serialize
restore
data_to_json
json_to_data
nested_equal

"""

import numpy as np
import warnings
import copy
import h5py

import json
from collections import namedtuple, Iterable, OrderedDict

from . import utils


class JSON_Serializer():
    """
    A class that can save all of its attributes to a JSON file, or 
    load them from a JSON file.
    
    """
    def __init__(self):
        pass
    
    def save_to_JSON(self, JSON_path):
        serialized_data = data_to_json(list(self.__dict__.items()))

        with open(JSON_path, 'w') as outfile:
            outfile.write(serialized_data)

    def load_from_JSON(self, JSON_path):
        with open(JSON_path, 'r') as infile:
            serialized_data = infile.read()

        member_list = json_to_data(serialized_data)
        
        for member in member_list:
            setattr(self, member[0], member[1])


class BasicWorm(JSON_Serializer):
    """
    Encapsulates the notion of a worm contour data that might have
    been obtained from a computer vision operation 
    
    * Skeleton or contour can be None but not both.
    * We don't assume the contour or skeleton points are evenly spaced,
    but we do assume they are in order as you walk along the skeleton
    or contour.   
    
    Attributes
    ----------
    skeleton : [ndarray] or ndarray 
        This input can either be a list, composed of skeletons for each 
            frame or a singular ndarray. Shapes for these would be:
                - [n_frames] list with elements [2,n_points] OR
                - [n_points,2,n_frames]
            In the first case n_points is variable (for MRC around 200) and
            in the second case n_points is obviously fixed, and currently
            should be at 49.
            Missing frames in the first case should be identified by None.    
    
    is_stage_movement :
    is_valid : 


    
    Metadata Attributes
    -------------------    
    plate_wireframe_video_key : string
        ???? What is this ???    
    
    
    
    #TODO: Rewrite as attributes following skeleton above
    The data consists of 4 numpy arrays:
    - Of shape (k,2,n):
        skeleton    
    - Of shape (m,2,n):
        contour
        
    ???? - why are we holding onto these????
    - Of shape (2,n):
        head
        tail
    where:
        k is the number of skeleton points per frame
        m is the number of contour points per frame
        n is the number of frames
        
    """

    def __init__(self, basic_worm=None):
        """
        Populates an empty basic worm.
        
        If copy is specified, this becomes a copy constructor. This was
        implemented for the NormalizedWorm constructor.
        
        """
        print("in BasicWorm constructor")
        if basic_worm is not None:
#            self.contour = np.empty([], dtype=float)
#            self.skeleton = None
#            self.head = np.empty([], dtype=float)
#            self.tail = np.empty([], dtype=float)
#            self.ventral_mode = None
#            self.plate_wireframe_video_key = None
#        else:
            # Copy constructor
            #self.contour = copy.deepcopy(basic_worm.contour)
            self.skeleton = copy.deepcopy(basic_worm.skeleton)
            self.vulva_contour = copy.deepcopy(basic_worm.vulva_contour)
            self.non_vulva_contour = copy.deepcopy(basic_worm.non_vulva_contour)
            
            #self.head = copy.deepcopy(basic_worm.head)
            #self.tail = copy.deepcopy(basic_worm.tail)
            
            #TODO: We need to work this out            
            #self.ventral_mode = basic_worm.ventral_mode
            self.plate_wireframe_video_key = basic_worm.plate_wireframe_video_key


    def validate(self):
        """
        Checks array lengths, etc. to ensure that this is a valid instance
        and no further problems will arise if further processing is attempted
        on this instance
        
        """
        
        #TODO: This needs to be rewritten ...
        
        #TODO: Make num_frames an attribute
        if self.contour is not None:
            num_frames = np.shape(self.contour)[2]
        else:
            num_frames = np.shape(self.skeleton)[2]

        if self.contour is not None:
            if np.shape(self.contour)[2] != num_frames:
                return False
            
        if self.skeleton is not None:
            if np.shape(self.skeleton)[2] != num_frames:
                return False

        if np.shape(self.head)[1] != num_frames:
            return False

        if np.shape(self.tail)[1] != num_frames:
            return False

        if self.ventral_mode not in ('CW', 'CCW', 'X'):
            return False

        return True

    
    @classmethod
    def from_skeleton_factory(cls, skeleton):
        """
        A factory method taking the simplest possible input: just a skeleton.
        Assumes 0th point is head, n-1th point is tail. No contour.
        
        Parameters
        ----------
        skeleton : [ndarray] or ndarray 
            See definition in class documentation.
        
        """
        
        bw = cls()
        bw.load_skeleton(skeleton) 
        bw.vulva_contour = None
        bw.non_vulva_contour = None

        if len(np.shape(skeleton)) != 3 or np.shape(skeleton)[1] != 2:
            raise Exception("Provided skeleton must have shape (n_points,2,n_frames)")

        bw.skeleton = skeleton

        bw.plate_wireframe_video_key = 'Simple Skeleton'        
        
        return bw
    

    
#    def load_skeleton(self,skeleton):    
#        """
#        
#        See Also
#        --------
#        from_skeleton_factory
#        
#        """
#        #TODO: First check for ndarray or list, if ndarray use skeleton.shape
#        if len(np.shape(skeleton)) != 3 or np.shape(skeleton)[1] != 2:
#            raise Exception("Provided skeleton must have shape (n_points,2,n_frames)")
#
#        #TODO: We need to handle the list case
#        
#        
#        #Why do we have this????
#        self.tail = skeleton[0,:,:]
#        self.head = skeleton[-1,:,:]


    
    @classmethod
    def from_schafer_file_factory(cls,data_file_path):

        self = cls()        
    
        #Would 'with()' be more appropriate here ???
        h = h5py.File(data_file_path, 'r')

        #These are all HDF5 'references'
        all_vulva_contours_refs = h['all_vulva_contours'].value
        all_non_vulva_contours_refs = h['all_non_vulva_contours'].value
        all_skeletons_refs = h['all_skeletons'].value
                
        is_stage_movement = utils._extract_time_from_disk(h,'is_stage_movement')
        is_valid = utils._extract_time_from_disk(h,'is_valid')

        all_skeletons = []
        all_vulva_contours = []
        all_non_vulva_contours = []

        for valid_frame,iFrame in zip(is_valid,range(is_valid.size)):
            if valid_frame:
                all_skeletons.append(h[all_skeletons_refs[iFrame][0]].value) 
                all_vulva_contours.append(h[all_vulva_contours_refs[iFrame][0]].value)
                all_non_vulva_contours.append(h[all_non_vulva_contours_refs[iFrame][0]].value)
            else:
                all_skeletons.append(None) 
                all_vulva_contours.append(None) 
                all_non_vulva_contours.append(None)           
                
        self.is_stage_movement = is_stage_movement
        self.is_valid = is_valid
        self.skeleton = None
        #self.all_skeletons = all_skeletons
        self.vulva_contour = all_vulva_contours
        self.non_vulva_contour = all_non_vulva_contours 
        self.plate_wireframe_video_key = 'Cheeseburger? WTF is this?'
        h.close()   
        
        return self
        
    
    def __repr__(self):
        return utils.print_object(self)    
    
    
    
class WormPartition():
    def __init__(self):
        print("In WormPartition initializer")
        # These are RANGE values, so the last value is not inclusive
        self.worm_partitions = {'head': (0, 8),
                                'neck': (8, 16),
                                'midbody':  (16, 33),
                                'old_midbody_velocity': (20, 29),
                                'hips':  (33, 41),
                                'tail': (41, 49),
                                # refinements of ['head']
                                'head_tip': (0, 4),
                                'head_base': (4, 8),    # ""
                                # refinements of ['tail']
                                'tail_base': (40, 45),
                                'tail_tip': (45, 49),   # ""
                                'all': (0, 49),
                                # neck, midbody, and hips
                                'body': (8, 41)}

        self.worm_partition_subsets = {'normal': ('head', 'neck', 'midbody', 'hips', 'tail'),
                                       'first_third': ('head', 'neck'),
                                       'second_third': ('midbody',),
                                       'last_third': ('hips', 'tail'),
                                       'all': ('all',)}

    def get_partition_subset(self, partition_type):
        """ 
        There are various ways of partitioning the worm's 49 points.
        this method returns a subset of the worm partition dictionary

        TODO: This method still is not obvious to me. Also, we should move
        these things to a separate class.

        Parameters
        ---------------------------------------
        partition_type: string
          e.g. 'head'

        Usage
        ---------------------------------------
        For example, to see the mean of the head and the mean of the neck, 
        use the partition subset, 'first_third', like this:

        nw = NormalizedWorm(....)

        width_dict = {k: np.mean(nw.get_partition(k), 0) for k in ('head', 'neck')}

        OR, using self.worm_partition_subsets,

        s = nw.get_paritition_subset('first_third')
        # i.e. s = {'head':(0,8), 'neck':(8,16)}

        width_dict = {k: np.mean(nw.get_partition(k), 0) for k in s.keys()}

        Notes
        ---------------------------------------    
        Translated from get.ALL_NORMAL_INDICES in SegwormMatlabClasses / 
        +seg_worm / @skeleton_indices / skeleton_indices.m

        """

        # parition_type is assumed to be a key for the dictionary
        # worm_partition_subsets
        p = self.worm_partition_subsets[partition_type]

        # return only the subset of partitions contained in the particular
        # subset of interest, p.
        return {k: self.worm_partitions[k] for k in p}


    def get_subset_partition_mask(self, name):
        """
        Returns a boolean mask - for working with arrays given a partition.
        
        """
        keys = self.worm_partition_subsets[name]
        mask = np.zeros(49, dtype=bool)
        for key in keys:
            mask = mask | self.partition_mask(key)

        return mask


    def partition_mask(self, partition_key):
        """
        Returns a boolean numpy array corresponding to the partition requested.

        """
        mask = np.zeros(49, dtype=bool)
        slice_val = self.worm_partitions[partition_key]
        mask[slice(*slice_val)] = True
        return mask


    def get_partition(self, partition_key, data_key='skeletons',
                      split_spatial_dimensions=False):
        """    
        Retrieve partition of a measurement of the worm, that is, across all
        available frames but across only a subset of the 49 points.

        Parameters
        ---------------------------------------    
        partition_key: string
          The desired partition.  e.g. 'head', 'tail', etc.

          #TODO: This should be documented better 

          INPUT: a partition key, and an optional data key.
            If split_spatial_dimensions is True, the partition is returned 
            separated into x and y
          OUTPUT: a numpy array containing the data requested, cropped to just
                  the partition requested.
                  (so the shape might be, say, 4xn if data is 'angles')

        data_key: string  (optional)
          The desired measurement (default is 'skeletons')

        split_spatial_dimensions: bool    (optional)
          If True, the partition is returned separated into x and y

        Returns
        ---------------------------------------    
        A numpy array containing the data requested, cropped to just
        the partition requested.
        (so the shape might be, say, 4xn if data is 'angles')

        Notes
        ---------------------------------------    
        Translated from get.ALL_NORMAL_INDICES in SegwormMatlabClasses / 
        +seg_worm / @skeleton_indices / skeleton_indices.m

        """
        # We use numpy.split to split a data_dict element into three, cleaved
        # first by the first entry in the duple worm_partitions[partition_key],
        # and second by the second entry in that duple.

        # Taking the second element of the resulting list of arrays, i.e. [1],
        # gives the partitioned component we were looking for.
        part = self.worm_partitions[partition_key]

        worm_attribute_values = getattr(self, data_key)
        if(worm_attribute_values.size != 0):
            # Let's suppress the warning about zero arrays being reshaped
            # since that's irrelevant since we are only looking at the 
            # non-zero array in the middle i.e. the 2nd element i.e. [1]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=FutureWarning)
                partition = np.split(worm_attribute_values,
                                 part)[1]
            if(split_spatial_dimensions):
                return partition[:, 0, :], partition[:, 1,:]
            else:
                return partition
        else:
            return None



def isnamedtuple(obj):
    """
    Heuristic check if an object is a namedtuple.

    """
    return isinstance(obj, tuple) \
           and hasattr(obj, "_fields") \
           and hasattr(obj, "_asdict") \
           and callable(obj._asdict)

def serialize(data):
    """
    """

    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [serialize(val) for val in data]
    if isinstance(data, OrderedDict):
        return {"py/collections.OrderedDict":
                [[serialize(k), serialize(v)] for k, v in data.items()]}
    if isnamedtuple(data):
        return {"py/collections.namedtuple": {
            "type":   type(data).__name__,
            "fields": list(data._fields),
            "values": [serialize(getattr(data, f)) for f in data._fields]}}
    if isinstance(data, dict):
        if all(isinstance(k, str) for k in data):
            return {k: serialize(v) for k, v in data.items()}
        return {"py/dict": [[serialize(k), serialize(v)] for k, v in data.items()]}
    if isinstance(data, tuple):
        return {"py/tuple": [serialize(val) for val in data]}
    if isinstance(data, set):
        return {"py/set": [serialize(val) for val in data]}
    if isinstance(data, np.ndarray):
        return {"py/numpy.ndarray": {
            "values": data.tolist(),
            "dtype":  str(data.dtype)}}
    raise TypeError("Type %s not data-serializable" % type(data))

def restore(dct):
    """
    """

    if "py/dict" in dct:
        return dict(dct["py/dict"])
    if "py/tuple" in dct:
        return tuple(dct["py/tuple"])
    if "py/set" in dct:
        return set(dct["py/set"])
    if "py/collections.namedtuple" in dct:
        data = dct["py/collections.namedtuple"]
        return namedtuple(data["type"], data["fields"])(*data["values"])
    if "py/numpy.ndarray" in dct:
        data = dct["py/numpy.ndarray"]
        return np.array(data["values"], dtype=data["dtype"])
    if "py/collections.OrderedDict" in dct:
        return OrderedDict(dct["py/collections.OrderedDict"])
    return dct

def data_to_json(data):
    """
    """

    return json.dumps(serialize(data))

def json_to_data(s):
    """
    """

    return json.loads(s, object_hook=restore)



def nested_equal(v1, v2):
    """
    Compares two complex data structures.

    This handles the case where numpy arrays are leaf nodes.

    """
    if isinstance(v1, str) or isinstance(v2, str):
        return v1 == v2
    if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
        return np.array_equal(v1, v2)
    if isinstance(v1, dict) and isinstance(v2, dict):
        return nested_equal(v1.items(), v2.items())
    if isinstance(v1, Iterable) and isinstance(v2, Iterable):
        return all(nested_equal(sub1, sub2) for sub1, sub2 in zip(v1, v2))
    return v1 == v2
