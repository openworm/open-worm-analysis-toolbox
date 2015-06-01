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
import matplotlib.pyplot as plt

import json
from collections import namedtuple, Iterable, OrderedDict

from . import utils
from .pre_features import WormParsing

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


class UnorderedWorm(JSON_Serializer):
    """
    Encapsulates the notion of worm contour or skeleton data that might have
    been obtained from a computer vision operation

    * We don't assume the contour or skeleton points are evenly spaced,
    but we do assume they are in order as you walk along the skeleton.

    * We DON'T assume that the head and the tail are at points 0 and -1, 
    respectively - hence the use of the word "unordered" in the name of this 
    class.
    
    * We don't assume that there is the same number of contour points 
    in each frame.  This means we can't use a simple ndarray representation
    for the contour.  Instead, we use a list of single-dimension numpy 
    arrays.

    
    """
    def __init__(self, other):
        attributes = ['unordered_contour', 'unordered_skeleton', 
                      'head', 'tail']
        
        if other is None:
            for a in attributes:
                setattr(self, a, None)
                
            self.plate_wireframe_video_key = "Unordered Worm"           
        else:
            # Copy constructor
            for a in attributes:
                setattr(self, a, copy.deepcopy(getattr(other, a)))

            self.plate_wireframe_video_key = other.plate_wireframe_video_key
    
    
    @classmethod
    def from_skeleton_factory(cls, skeleton, head=None, tail=None):
        """
        A factory method taking the simplest possible input: just a skeleton.
        Assumes 0th point is head, n-1th point is tail. No contour.
        
        Parameters
        ----------
        skeleton : list of ndarray or ndarray 
            If ndarray, we are in the simpler "homocardinal" case
            If list of ndarray, each frame can have a varying number of points
        head: ndarray containing the position of the head.
        tail: ndarray containing the position of the tail.
        
        """
        uow = cls()


        #if len(np.shape(skeleton)) != 3 or np.shape(skeleton)[1] != 2:        
        #   raise Exception("Provided skeleton must have "
        #                   "shape (n_points,2,n_frames)")
        uow.skeleton = skeleton

        uow.plate_wireframe_video_key = 'Simple Skeleton'
        if tail is None:
            uow.tail = skeleton[0,:,:]
        else:
            uow.tail = tail

        if head is None:
            uow.head = skeleton[-1,:,:]
        else:
            uow.head = head

        #TODO: First check for ndarray or list, if ndarray use skeleton.shape
        #if len(np.shape(skeleton)) != 3 or np.shape(skeleton)[1] != 2:
        #   raise Exception("Provided skeleton must have "
        #                   "shape (n_points,2,n_frames)")

        #TODO: We need to handle the list case
        
        return uow

    @classmethod
    def from_contour_factory(cls, contour, head=None, tail=None):
        pass


    def ordered_vulva_contour(self):
        """
        Return the vulva side of the ordered heterocardinal contour.
        
        i.e. with tail at position -1 and head at position 0.
        
        """
        # TODO
        pass


    def ordered_non_vulva_contour(self):
        """
        Return the non-vulva side of the ordered heterocardinal contour.
        
        i.e. with tail at position -1 and head at position 0.
        
        """
        # TODO
        pass

    def ordered_skeleton(self):
        """
        Return the ordered skeleton.
        
        i.e. with tail at position -1 and head at position 0.
        
        """
        # TODO
        pass


class BasicWorm(JSON_Serializer):
    """
    A worm's skeleton and contour, not necessarily "normalized" to 49 points,
    and possibly heterocardinal (i.e. possibly with a varying number of 
    points per frame).

    Attributes
    ----------
    h_skeleton : [ndarray] or ndarray 
        This input can either be a list, composed of skeletons for each 
            frame or a singular ndarray. Shapes for these would be:
                - [n_frames] list with elements [2,n_points] OR
                - [n_points,2,n_frames]
            In the first case n_points is variable (for MRC around 200) and
            in the second case n_points is obviously fixed, and currently
            should be at 49.
            Missing frames in the first case should be identified by None.    
    h_vulva_contour : The vulva side of the contour.  See skeleton.
    h_non_vulva_contour : The vulva side of the contour.  See skeleton.

    ?????    
    is_stage_movement :
    is_valid : 
    ????

    Metadata Attributes
    -------------------    
    plate_wireframe_video_key : string
        This is the foreign key to the metadata table in the database
        giving plate information like lab name, time of recording, etc.

    """
    def __init__(self, other=None):
        attributes = ['_h_skeleton', '_h_vulva_contour', 
                      '_h_non_vulva_contour']
        
        if other is None:
            for a in attributes:
                setattr(self, a, None)
                
            self.plate_wireframe_video_key = "Basic Worm"
        else:
            # Copy constructor
            for a in attributes:
                setattr(self, a, copy.deepcopy(getattr(other, a)))

            self.plate_wireframe_video_key = other.plate_wireframe_video_key

    
    @classmethod
    def from_schafer_file_factory(cls, data_file_path):
        bw = cls()
    
        # Would 'with()' be more appropriate here ???
        h = h5py.File(data_file_path, 'r')

        # These are all HDF5 'references'
        all_vulva_contours_refs     = h['all_vulva_contours'].value
        all_non_vulva_contours_refs = h['all_non_vulva_contours'].value
        all_skeletons_refs          = h['all_skeletons'].value
                
        is_stage_movement = utils._extract_time_from_disk(h,
                                                          'is_stage_movement')
        is_valid = utils._extract_time_from_disk(h, 'is_valid')

        all_skeletons = []
        all_vulva_contours = []
        all_non_vulva_contours = []

        for valid_frame, iFrame in zip(is_valid, range(is_valid.size)):
            if valid_frame:
                all_skeletons.append(
                            h[all_skeletons_refs[iFrame][0]].value) 
                all_vulva_contours.append(
                            h[all_vulva_contours_refs[iFrame][0]].value)
                all_non_vulva_contours.append(
                            h[all_non_vulva_contours_refs[iFrame][0]].value)
            else:
                all_skeletons.append(None) 
                all_vulva_contours.append(None) 
                all_non_vulva_contours.append(None)           
                
        bw.is_stage_movement = is_stage_movement
        bw.is_valid = is_valid
        
        # We purposely ignore the saved skeleton information contained
        # in the BasicWorm, preferring to derive it ourselves.        
        bw.remove_precalculated_skeleton()
        #bw.h_skeleton = all_skeletons

        bw._h_vulva_contour = all_vulva_contours
        bw._h_non_vulva_contour = all_non_vulva_contours 
        bw.plate_wireframe_video_key = 'Loaded from Schafer File'
        h.close()   
        
        return bw


    @classmethod
    def from_h_skeleton_factory(cls, h_skeleton, extrapolate_contour=False):
        """
        Factory method         
        
        Optionally tries to extrapolate a contour from just the skeleton data.
        
        """
        bw = cls()
        
        if extrapolate_contour:
            # TODO: extrapolate the bw.h_vulva_contour and 
            #       bw.h_non_vulva_contour from bw._h_skeleton
            # for now we just make the contour on both sides = skeleton
            # which makes for a one-dimensional worm.
            bw.h_vulva_contour = h_skeleton
            bw.h_non_vulva_contour = h_skeleton

        return bw

    @property
    def h_vulva_contour(self):
        return self._h_vulva_contour

    @h_vulva_contour.setter
    def h_vulva_contour(self, x):
        self._h_vulva_contour = x
        self.remove_precalculated_skeleton()

    @property
    def h_non_vulva_contour(self):
        return self._h_non_vulva_contour

    @h_non_vulva_contour.setter
    def h_non_vulva_contour(self, x):
        self._h_non_vulva_contour = x
        self.remove_precalculated_skeleton()
        
    def remove_precalculated_skeleton(self):
        """
        Removes the precalculated self._h_skeleton, if it exists.
        
        This is typically called if we've potentially changed something, 
        i.e. if we've loaded new values for self.h_vulva_contour or 
        self.h_non_vulva contour.
        
        In these cases we must be sure to delete h_skeleton, since it is 
        derived from vulva_contour and non_vulva_contour.
        
        It will be recalculated if it's ever asked for.
        """
        try:        
            del(self._h_skeleton)
        except NameError:
            pass
    
    @property
    def h_skeleton(self):
        """
        If self._h_skeleton has been defined, then return it.

        Otherwise, try to extrapolate it from the contour.
        
        Note: This method does not have an obvious use case.  The normal 
        pipeline is to call NormalizedWorm.from_BasicWorm_factory, which will
        calculate a skeleton.

        """
        try:
            return self._h_skeleton
        except AttributeError:
            # Extrapolate skeleton from contour
            # TODO: improve this: for now
            self._h_skeleton = \
                WormParsing.computeWidths(self.h_vulva_contour, 
                                          self.h_non_vulva_contour)[1]
            
            return self._h_skeleton            
    

    def plot_frame(self, frame_index):
        """
        Plot the contour and skeleton the worm for one of the frames.
        
        Parameters
        ----------------
        frame_index: int
            The desired frame # to plot.
        
        """
        vc = self.h_vulva_contour[frame_index]
        nvc = self.h_non_vulva_contour[frame_index]
        skeleton_x = self.h_skeleton[frame_index][0]
        skeleton_y = self.h_skeleton[frame_index][1]
        
        plt.scatter(vc[0,:], vc[1,:])
        plt.scatter(nvc[0,:], nvc[1,:])
        plt.scatter(skeleton_x, skeleton_y)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    
    def __repr__(self):
        return utils.print_object(self)    
    
    
    
class WormPartition():
    def __init__(self):
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
