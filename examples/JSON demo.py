# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:29:15 2015

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
import sys, os, copy
import numpy as np
import scipy

import json
from collections import namedtuple, Iterable, OrderedDict
import warnings

# We must add .. to the path so that we can perform the 
# import of movement_validation while running this as 
# a top-level script (i.e. with __name__ = '__main__')
sys.path.append('..') 
from movement_validation import config, user_config
 
 
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
    
    The data consists of 4 numpy arrays:
    - Of shape (k,2,n):
        skeleton    
    - Of shape (m,2,n):
        contour
    - Of shape (2,n):
        head
        tail
    where:
        k is the number of skeleton points per frame
        m is the number of contour points per frame
        n is the number of frames
    
    Note: skeleton or contour can be None but not both.
    Note: we don't assume the contour or skeleton points are evenly spaced,
          but we do assume they are in order as you walk along the skeleton
          or contour.
    
    Also, some metadata:
        - PlateWireframeVideoKey
    
    """

    def __init__(self, basic_worm=None):
        """
        Populates an empty basic worm.
        If copy is specified, this becomes a copy constructor.
        
        """
        print("in BasicWorm constructor")
        if not basic_worm:
            self.contour = np.empty([], dtype=float)
            self.skeleton = None
            self.head = np.empty([], dtype=float)
            self.tail = np.empty([], dtype=float)
            self.ventral_mode = None
            self.plate_wireframe_video_key = None
        else:
            # Copy constructor
            self.contour = copy.deepcopy(basic_worm.contour)
            self.skeleton = copy.deepcopy(basic_worm.skeleton)
            self.head = copy.deepcopy(basic_worm.head)
            self.tail = copy.deepcopy(basic_worm.tail)
            self.ventral_mode = basic_worm.ventral_mode
            self.plate_wireframe_video_key = \
                basic_worm.plate_wireframe_video_key


    def validate(self):
        """
        Checks array lengths, etc. to ensure that this is a valid instance
        and no further problems will arise if further processing is attempted
        on this instance
        
        """
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
        
        """
        if len(np.shape(skeleton)) != 3 or np.shape(skeleton)[1] != 2:
            raise Exception("Provided skeleton must have shape (m,2,n)")

        bw = cls()
        bw.skeleton = skeleton
        bw.contour = None
        bw.tail = skeleton[0,:,:]
        bw.head = skeleton[-1,:,:]
        bw.plate_wireframe_video_key = 'Simple Skeleton'

        return bw
    

class WormPartition():
    def __init__(self):
        pass


class NormalizedWorm(BasicWorm, WormPartition):
    """
    Encapsulates the notion of a worm's elementary measurements, scaled
    (i.e. "normalized") to 49 points along the length of the worm.

    The data consists of 13 Numpy arrays (where n is the number of frames):
    - Of shape (49,2,n):
        vulva_contour
        non_vulva_contour
        skeleton
    - Of shape (49,n):
        angles        
        in_out_touches
        widths
    - Of shape (n):
        length
        head_area
        tail_area
        vulva_area
        non_vulva_area
        segmentation_status   (not used in further processing)
        frame_code            (not used in further processing)

    Also, some metadata:
        plate_wireframe_video_key
        
    """

    def __init__(self, normalized_worm=None):
        """
        Populates an empty normalized worm.
        If copy is specified, this becomes a copy constructor.
        
        """
        print("in NormalizedWorm consructor")
        if not normalized_worm:
            super(NormalizedWorm, self).__init__()
            self.angles = np.array([], dtype=float)

            # DEBUG: (Note from @MichaelCurrie:)
            # This should be set by the normalized worm file, since each
            # worm subjected to an experiment is manually examined to find the
            # vulva so the ventral mode can be determined.  Here we just set
            # the ventral mode to a default value as a stopgap measure
            self.ventral_mode = config.DEFAULT_VENTRAL_MODE

        else:
            super(NormalizedWorm, self).__init__(normalized_worm)
            self.angles = copy.deepcopy(normalized_worm.angles)


    @classmethod
    def from_BasicWorm_factory(cls, basic_worm):
        nw = NormalizedWorm()        
        
        # Call BasicWorm's copy constructor:
        super(NormalizedWorm, nw).__init__(basic_worm)

        # TODO: We should probably validate that the worm is valid before
        #       calculating pre-features.
        
        nw.calculate_pre_features()
        
        return nw

    @classmethod
    def from_schafer_file_factory(cls, data_file_path):
        """
        Load full Normalized Worm data from the Schafer File

        data_file_path: the path to the MATLAB file
        
        These files were created at the Schafer Lab in a format used 
        prior to MATLAB's switch to HDF5, which they did in MATLAB version 7.3.


        """
        nw = cls()
        nw.plate_wireframe_video_key = 'Schafer'
        
        if(not os.path.isfile(data_file_path)):
            raise Exception("Data file not found: " + data_file_path)
        else:
            data_file = scipy.io.loadmat(data_file_path,
                                         # squeeze unit matrix dimensions:
                                         squeeze_me=True,
                                         # force return numpy object
                                         # array:
                                         struct_as_record=False)

            # All the action is in data_file['s'], which is a numpy.ndarray where
            # data_file['s'].dtype is an array showing how the data is structured.
            # it is structured in precisely the order specified in data_keys
            # below

            staging_data = data_file['s']

            # NOTE: These are aligned to the order in the files.
            # these will be the keys of the dictionary data_dict
            data_keys = [
                # this just contains a string for where to find the
                # eigenworm file.  we do not use this, however, since
                # the eigenworm postures are universal to all worm files,
                # so the file is just stored in the /features directory
                # of the source code, and is loaded at the features 
                # calculation step
                'EIGENWORM_PATH',
                # a string of length n, showing, for each frame of the video:
                # s = segmented
                # f = segmentation failed
                # m = stage movement
                # d = dropped frame
                # n??? - there is reference in some old code to this
                # after loading this we convert it to a numpy array.
                'segmentation_status',
                # shape is (1 n), see comments in
                # seg_worm.parsing.frame_errors
                'frame_codes',
                'vulva_contours',     # shape is (49, 2, n) integer
                'non_vulva_contours',  # shape is (49, 2, n) integer
                'skeletons',          # shape is (49, 2, n) integer
                'angles',             # shape is (49, n) integer (degrees)
                'in_out_touches',     # shpe is (49, n)
                'lengths',            # shape is (n) integer
                'widths',             # shape is (49, n) integer
                'head_areas',         # shape is (n) integer
                'tail_areas',         # shape is (n) integer
                'vulva_areas',        # shape is (n) integer
                'non_vulva_areas',    # shape is (n) integer
                'x',                  # shape is (49, n) integer
                'y']                  # shape is (49, n) integer

            # Here I use powerful python syntax to reference data elements of s
            # dynamically through built-in method getattr
            # that is, getattr(s, x)  works syntactically just like s.x,
            # only x is a variable, so we can do a list comprehension with it!
            for key in data_keys:
                setattr(nw, key, getattr(staging_data, key))

            # We don't need the eigenworm path here, as it's the same
            # for all worm files.
            del(nw.EIGENWORM_PATH)
            # x and y are redundant since that information is already 
            # in "skeletons"
            del(nw.x)
            del(nw.y)
            
            # Now for something pedantic: only use plural nouns for
            # those measurements taken along multiple points per frame
            # for those with just one data point per frame, it should be 
            # singular.
            # i.e. plural for numpy arrays of shape (49, n)
            #     singular for numpy arrays of shape (n)
            # and singular for numpy arrays of shape (49, 2, n)
            # (where n is the number of frames)

            nw.skeleton = nw.skeletons
            nw.vulva_contour = nw.vulva_contours
            nw.non_vulva_contour = nw.non_vulva_contours
            del(nw.skeletons)
            del(nw.vulva_contours)
            del(nw.non_vulva_contours)
            nw.length = nw.lengths
            nw.head_area = nw.head_areas
            nw.tail_area = nw.tail_areas
            nw.vulva_area = nw.vulva_areas
            nw.non_vulva_area = nw.non_vulva_areas
            nw.frame_code = nw.frame_codes
            del(nw.lengths)
            del(nw.head_areas)
            del(nw.tail_areas)
            del(nw.vulva_areas)
            del(nw.non_vulva_areas)
            del(nw.frame_codes)

            # Let's change the string of length n to a numpy array of single
            # characters of length n, to be consistent with the other data
            # structures
            nw.segmentation_status = np.array(list(nw.segmentation_status))
            
            return nw

    def get_BasicWorm(self):
        """
        Return an instance of BasicWorm containing this instance of 
        NormalizedWorm's basic data.

        """
        bw = BasicWorm()
        bw.head = np.copy(self.skeleton[0,:,:])
        bw.tail = np.copy(self.skeleton[-1,:,:])
        bw.skeleton = np.copy(self.skeleton)
        # We have to reverse the contour points of the non_vulva_contour
        # so that the tail end picks up where the tail end of vulva_contour
        # left off:
        bw.contour = np.copy(np.concatenate((self.vulva_contour, 
                                             self.non_vulva_contour[::-1,:,:]), 
                                            axis=0))
        bw.ventral_mode = 'CW'
        
        return bw
    
    def calculate_pre_features(self):
        """
        Calculate "pre-features" given basic information about the worm.
        
        1. If contour is specified, normalize it to 98 points evenly split 
           between head and tail
        2. If skeleton is specified, normalize it to 49 points
        3. Calculate vulva_contour and non_vulva_contour from contour 
           (preferably) or that's not available, skeleton 
           (use an approximation in this case)
        4. Calculate angles, in_out_touches, widths for each skeleton point
           and each frame
        5. Calculate length, head_area, tail_area, vulva_area, non_vulva area
           for each frame 

        """
        print("Calculating pre-features")
        # TODO
        
        # 1. If contour is specified, normalize it to 98 points evenly split 
        #   between head and tail        
        # calculating the "hemiworms" requires stepping through frame-by-frame
        # since we cannot assume that the number of points between head
        # and tail and between tail and head remains constant between frames.
        pass



    
    def validate(self):
        """
        Checks array lengths, etc. to ensure that this is a valid instance
        and no further problems will arise if further processing is attempted
        on this instance

        """
        return True


        
def main():
    warnings.filterwarnings('error')
    
    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)

    JSON_path = os.path.join(base_path, 'test.JSON')

    b = BasicWorm()
    #b.contour[0] = 100.2
    #b.metadata['vulva'] = 'CCW'
    b.save_to_JSON(JSON_path)

    c = BasicWorm()
    c.load_from_JSON(JSON_path)
    print(c.contour)

    dat = get_nw()
    dat.save_to_JSON(JSON_path)

    

def get_nw():
    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)

    schafer_file_path = os.path.join(
        base_path, 'example_video_norm_worm.mat')

    return NormalizedWorm.from_schafer_file_factory(schafer_file_path)


def main2():
    JSON_path0 = 'C:\\Users\\Michael\\Dropbox\\INBOX\\test0.JSON'
    JSON_path = 'C:\\Users\\Michael\\Dropbox\\INBOX\\test.JSON'
    
    data_array = np.arange(10)
    x = data_array    

    # METHOD 1
    serialized_data = json.dumps(data_array.tolist())
    with open(JSON_path0, 'w') as outfile:
        outfile.write(serialized_data)

    with open(JSON_path0, 'r') as infile:
        y = json.loads(infile.read())
    
    print("x:", str(type(x)), x)
    print("y:", str(type(y)), y)
    print("x==y:", np.array_equal(x,y))

    
    # METHOD 2
    #http://robotfantastic.org/serializing-python-data-to-json-some-edge-cases.html
    serialized_data = json.dumps(data_array.tolist())
    y = json.loads(serialized_data)



    print("x:", str(type(x)), x)
    print("y:", str(type(y)), y)
    print("x==y:", np.array_equal(x,y))
        

    b = BasicWorm()    
    b_as_list = list(b.__dict__.items())
    serialized_data = data_to_json(b_as_list)
    print(serialized_data)

    with open(JSON_path, 'w') as outfile:
        outfile.write(serialized_data)

    with open(JSON_path, 'r') as infile:
        y = json.loads(infile.read())


    

    # TODO: make sure you can read it back!!!
    # TODO: then you just need to make the specification and you are done!

    # TODO: somehow replace the equality check with using nested_equal
    #       on the serializations of the two objects?? maybe.

    #print(json.dumps(b))
    #b_encoded = jsonpickle.encode(b, unpicklable=False)
    #print(b_encoded)





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


if __name__ == '__main__':
    main()
    
    
    
    
    