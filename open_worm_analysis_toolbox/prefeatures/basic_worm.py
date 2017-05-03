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

from .. import config, utils
from .pre_features import WormParsing
from .video_info import VideoInfo

#%%


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

#%%


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
                      'head', 'tail', 'video_info']

        if other is None:
            for a in attributes:
                setattr(self, a, None)

        else:
            # Copy constructor
            for a in attributes:
                setattr(self, a, copy.deepcopy(getattr(other, a)))

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

        # if len(np.shape(skeleton)) != 3 or np.shape(skeleton)[1] != 2:
        #   raise Exception("Provided skeleton must have "
        #                   "shape (n_points,2,n_frames)")
        uow.skeleton = skeleton

        if tail is None:
            uow.tail = skeleton[0, :, :]
        else:
            uow.tail = tail

        if head is None:
            uow.head = skeleton[-1, :, :]
        else:
            uow.head = head

        # TODO: First check for ndarray or list, if ndarray use skeleton.shape
        # if len(np.shape(skeleton)) != 3 or np.shape(skeleton)[1] != 2:
        #   raise Exception("Provided skeleton must have "
        #                   "shape (n_points,2,n_frames)")

        # TODO: We need to handle the list case

        return uow

    @classmethod
    def from_contour_factory(cls, contour, head=None, tail=None):
        pass

    def ordered_ventral_contour(self):
        """
        Return the vulva side of the ordered heterocardinal contour.

        i.e. with tail at position -1 and head at position 0.

        """
        # TODO
        pass

    def ordered_dorsal_contour(self):
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

#%%


class BasicWorm(JSON_Serializer):
    """
    A worm's skeleton and contour, not necessarily "normalized" to 49 points,
    and possibly heterocardinal (i.e. possibly with a varying number of
    points per frame).

    Attributes
    ----------
    h_skeleton : list, where each element is a numpy array of shape (2,k_i)
        Each element of the list is a frame.
        Where k_i is the number of skeleton points in frame i.
        The first axis of the numpy array, having len 2, is the x and y.
         Missing frames should be identified by None.
    h_ventral_contour:   Same type and shape as skeleton (see above)
        The vulva side of the contour.
    h_dorsal_contour: Same type and shape as skeleton (see above)
        The non-vulva side of the contour.
    video_info : An instance of the VideoInfo class.
                 (contains metadata attributes of the worm video)

    """

    def __init__(self, other=None):
        attributes = ['_h_skeleton', '_h_ventral_contour',
                      '_h_dorsal_contour']

        if other is None:
            for a in attributes:
                setattr(self, a, None)

            self.video_info = VideoInfo()
        else:
            # Copy constructor
            for a in attributes:
                setattr(self, a, copy.deepcopy(getattr(other, a)))

    @classmethod
    def from_schafer_file_factory(cls, data_file_path):
        bw = cls()

        with h5py.File(data_file_path, 'r') as h:
            # These are all HDF5 'references'
            all_ventral_contours_refs = h['all_vulva_contours'].value
            all_dorsal_contours_refs = h['all_non_vulva_contours'].value
            all_skeletons_refs = h['all_skeletons'].value

            is_stage_movement = utils._extract_time_from_disk(
                h, 'is_stage_movement')
            is_valid = utils._extract_time_from_disk(h, 'is_valid')

            all_skeletons = []
            all_ventral_contours = []
            dorsal_contour = []

            for valid_frame, iFrame in zip(is_valid, range(is_valid.size)):
                if valid_frame:
                    all_skeletons.append(
                        h[all_skeletons_refs[iFrame][0]].value)
                    all_ventral_contours.append(
                        h[all_ventral_contours_refs[iFrame][0]].value)
                    dorsal_contour.append(
                        h[all_dorsal_contours_refs[iFrame][0]].value)
                else:
                    all_skeletons.append(None)
                    all_ventral_contours.append(None)
                    dorsal_contour.append(None)

        # Video Metadata
        is_stage_movement = is_stage_movement.astype(bool)
        is_valid = is_valid.astype(bool)

        # A kludge, we drop frames in is_stage_movement that are in excess
        # of the number of frames in the video.  It's unclear why
        # is_stage_movement would be longer by 1, which it was in our
        # canonical example.
        is_stage_movement = is_stage_movement[0:len(all_skeletons)]

        # 5. Derive frame_code from the two pieces of data we have,
        #    is_valid and is_stage_movement.
        bw.video_info.frame_code = (1 * is_valid +
                                    2 * is_stage_movement +
                                    100 * ~(is_valid | is_stage_movement))

        # We purposely ignore the saved skeleton information contained
        # in the BasicWorm, preferring to derive it ourselves.
        bw.__remove_precalculated_skeleton()
        #bw.h_skeleton = all_skeletons

        bw._h_ventral_contour = all_ventral_contours
        bw._h_dorsal_contour = dorsal_contour

        return bw

    @classmethod
    def from_contour_factory(cls, ventral_contour, dorsal_contour):
        """
        Return a BasicWorm from a normalized ventral_contour and dorsal_contour

        Parameters
        ---------------
        ventral_contour: numpy array of shape (49,2,n)
        dorsal_contour: numpy array of shape (49,2,n)

        Returns
        ----------------
        BasicWorm object

        """
        
        
        if not isinstance(ventral_contour, (list,tuple)):
            # we need to change the data from a (49,2,n) array to a list of (2,49)
            assert(np.shape(ventral_contour) == np.shape(dorsal_contour))
            assert ventral_contour.shape[1] == 2
            h_ventral_contour = WormParsing._h_array2list(ventral_contour)
            h_dorsal_contour = WormParsing._h_array2list(dorsal_contour)
        else:
            h_ventral_contour = ventral_contour
            h_dorsal_contour = dorsal_contour

        # Here I am checking that the contour missing frames are aligned. 
        # I prefer to populate the frame_code in normalized worm.
        assert all( v == d for v,d in zip(h_ventral_contour, h_dorsal_contour) if v is None or d is None)

        # Having converted our normalized contour to a heterocardinal-type
        # contour that just "happens" to have all its frames with the same
        # number of skeleton points, we can just call another factory method
        # and we are done:

        bw = cls()
        bw.h_ventral_contour = h_ventral_contour
        bw.h_dorsal_contour = h_dorsal_contour

        return bw

    @classmethod
    def from_skeleton_factory(cls, skeleton, extrapolate_contour=False):
        if not extrapolate_contour:
            '''
            Construct the object using only the skeletons without contours. 
            This is a better default because the contour interpolation will produce a fake contour.
            '''
            bw = cls()

            #other option will be to give a list of None, but this make more obvious when there is a mistake
            bw.h_ventral_contour = None 
            bw.h_dorsal_contour = None
            if isinstance(skeleton,  (list,tuple)):
                bw._h_skeleton =  skeleton
            else:
                assert skeleton.shape[1] == 2
                bw._h_skeleton =  WormParsing._h_array2list(skeleton)
            return bw

        else:

            """
            Derives a contour from the skeleton
            THIS PART IS BUGGY, THE INTERPOLATION WORKS ONLY IN A LIMITED NUMBER OF CASES
            TODO: right now the method creates the bulge entirely in the y-axis,
                  across the x-axis.  Instead the bulge should be rotated to
                  apply across the head-tail orientation.

            TODO: the bulge should be more naturalistic than the simple sine wave
                  currently used.


            """
            # Make ventral_contour != dorsal_contour by making them "bulge"
            # in the middle, in a basic simulation of what a real worm looks like
            bulge_x = np.zeros((config.N_POINTS_NORMALIZED))
            # Create the "bulge"
            x = np.linspace(0, 1, config.N_POINTS_NORMALIZED)
            bulge_y = np.sin(x * np.pi) * 50

            # Shape is (49,2,1):
            bulge_frame1 = np.rollaxis(np.dstack([bulge_x, bulge_y]),
                                       axis=0, start=3)
            # Repeat the bulge across all frames:
            num_frames = skeleton.shape[2]
            bulge_frames = np.repeat(bulge_frame1, num_frames, axis=2)

            # Apply the bulge above and below the skeleton
            ventral_contour = skeleton + bulge_frames
            dorsal_contour = skeleton - bulge_frames

            # Now we are reduced to the contour factory case:
            return  BasicWorm.from_contour_factory(ventral_contour, dorsal_contour)

    
    @property
    def h_ventral_contour(self):
        return self._h_ventral_contour

    @h_ventral_contour.setter
    def h_ventral_contour(self, x):
        self._h_ventral_contour = x
        self.__remove_precalculated_skeleton()

    @property
    def h_dorsal_contour(self):
        return self._h_dorsal_contour

    @h_dorsal_contour.setter
    def h_dorsal_contour(self, x):
        self._h_dorsal_contour = x
        self.__remove_precalculated_skeleton()

    def __remove_precalculated_skeleton(self):
        """
        Removes the precalculated self._h_skeleton, if it exists.

        This is typically called if we've potentially changed something,
        i.e. if we've loaded new values for self.h_ventral_contour or
        self.h_non_vulva contour.

        In these cases we must be sure to delete h_skeleton, since it is
        derived from ventral_contour and dorsal_contour.

        It will be recalculated if it's ever asked for.

        """
        try:
            del(self._h_skeleton)
        except AttributeError:
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
            self._h_widths, self._h_skeleton = \
            WormParsing.compute_skeleton_and_widths(self.h_ventral_contour, self.h_dorsal_contour)
            #how can i call _h_widths???

            return self._h_skeleton

    def plot_frame(self, frame_index):
        """
        Plot the contour and skeleton the worm for one of the frames.

        Parameters
        ----------------
        frame_index: int
            The desired frame # to plot.

        """
        vc = self.h_ventral_contour[frame_index]
        dc = self.h_dorsal_contour[frame_index]
        s = self.h_skeleton[frame_index]

        plt.scatter(vc[0, :], vc[1, :])
        plt.scatter(dc[0, :], dc[1, :])
        plt.scatter(s[0, :], s[1, :])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def validate(self):
        """
        Validate that self is a well-defined BasicWorm instance.

        """
        assert(len(self.h_ventral_contour) == len(self.h_dorsal_contour))

    def __repr__(self):
        return utils.print_object(self)

    def __eq__(self, other):
        """
        Compare this BasicWorm against another.

        """
        attribute_list = ['h_ventral_contour', 'h_dorsal_contour',
                          'h_skeleton', 'video_info']

        return utils.compare_attributes(self, other, attribute_list)


#%%
class WormPartition():

    def __init__(self):
        # These are RANGE values, so the last value is not inclusive
        self.worm_partitions = {'head': (0, 8),
                                'neck': (8, 16),
                                'midbody': (16, 33),
                                'old_midbody_velocity': (20, 29),
                                'hips': (33, 41),
                                'tail': (41, 49),
                                # refinements of ['head']
                                'head_tip': (0, 3),
                                'head_base': (5, 8),    # ""
                                # refinements of ['tail']
                                'tail_base': (41, 44),
                                'tail_tip': (46, 49),   # ""
                                'all': (0, 49),
                                # neck, midbody, and hips
                                'body': (8, 41)}

        self.worm_partition_subsets = {
            'normal': (
                'head', 'neck', 'midbody', 'hips', 'tail'), 'first_third': (
                'head', 'neck'), 'second_third': (
                'midbody',), 'last_third': (
                    'hips', 'tail'), 'all': (
                        'all',)}

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

        # Return only the subset of partitions contained in the particular
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
        if(len(worm_attribute_values) != 0):
            # Let's suppress the warning about zero arrays being reshaped
            # since that's irrelevant since we are only looking at the
            # non-zero array in the middle i.e. the 2nd element i.e. [1]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=FutureWarning)
                partition = np.split(worm_attribute_values,
                                     part)[1]
            if(split_spatial_dimensions):
                return partition[:, 0, :], partition[:, 1, :]
            else:
                return partition
        else:
            return None


#%%

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
            "type": type(data).__name__,
            "fields": list(data._fields),
            "values": [serialize(getattr(data, f)) for f in data._fields]}}
    if isinstance(data, dict):
        if all(isinstance(k, str) for k in data):
            return {k: serialize(v) for k, v in data.items()}
        return {"py/dict": [[serialize(k), serialize(v)]
                            for k, v in data.items()]}
    if isinstance(data, tuple):
        return {"py/tuple": [serialize(val) for val in data]}
    if isinstance(data, set):
        return {"py/set": [serialize(val) for val in data]}
    if isinstance(data, np.ndarray):
        return {"py/numpy.ndarray": {
            "values": data.tolist(),
            "dtype": str(data.dtype)}}
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
